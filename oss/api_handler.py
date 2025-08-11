#!/usr/bin/env python3

from operator import ge
import secrets
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler
from typing import (
    Any,
    Literal,
)

from huggingface_hub import scan_cache_dir

from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.utils import common_prefix_len


from .cache import PromptCache
from .model import ModelProvider
from .utils import (
    process_message_content,
    sequence_overlap,
    stopping_criteria,
)
from openai_harmony import (
    HarmonyEncodingName,
    Role,
    load_harmony_encoding,
    StreamableParser,
) 

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: PromptCache | None = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache or PromptCache()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        endpoints = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        if self.path not in endpoints:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode())
        except json.JSONDecodeError as e:
            logging.error(f"JSONDecodeError: {
                          e} - Raw body: {raw_body.decode()}")
            # Set appropriate headers based on streaming requirement
            if self.stream:
                self._set_stream_headers(400)
                self.wfile.write(
                    f"data: {json.dumps(
                        {'error': f'Invalid JSON in request body: {e}'})}\n\n".encode()
                )
            else:
                self._set_completion_headers(400)
                self.wfile.write(
                    json.dumps(
                        {"error": f"Invalid JSON in request body: {e}"}).encode()
                )
            return

        indent = "\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Incoming Request Body: {
                      json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.requested_draft_model = self.body.get(
            "draft_model", "default_model")
        self.num_draft_tokens = self.body.get(
            "num_draft_tokens", self.model_provider.cli_args.num_draft_tokens
        )
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get(
                "max_tokens", self.model_provider.cli_args.max_tokens
            )
        self.temperature = self.body.get(
            "temperature", self.model_provider.cli_args.temp
        )
        self.top_p = self.body.get("top_p", self.model_provider.cli_args.top_p)
        self.top_k = self.body.get("top_k", self.model_provider.cli_args.top_k)
        self.min_p = self.body.get("min_p", self.model_provider.cli_args.min_p)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get(
            "repetition_context_size", 20)
        self.xtc_probability = self.body.get("xtc_probability", 0.0)
        self.xtc_threshold = self.body.get("xtc_threshold", 0.0)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.validate_model_parameters()

        # Load the model if needed
        try:
            self.model, self.tokenizer = self.model_provider.load(
                self.requested_model,
                self.adapter,
                self.requested_draft_model,
            )
        except:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Get stop id sequences, if provided
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(
            stop_words, str) else stop_words
        stop_id_sequences = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Send header type
        (
            self._set_stream_headers(200)
            if self.stream
            else self._set_completion_headers(200)
        )

        # Call endpoint specific method
        prompt = endpoints[self.path]()
        self.handle_completion(prompt, stop_id_sequences)

    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a non-negative integer")

        if not isinstance(self.min_p, (float, int)) or self.min_p < 0 or self.min_p > 1:
            raise ValueError("min_p must be a float between 0 and 1")

        if not isinstance(self.num_draft_tokens, int) or self.num_draft_tokens < 0:
            raise ValueError("num_draft_tokens must be a non-negative integer")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {self.logprobs:,}"
            )

        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError(
                "repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {
                    int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")
        if not (
            isinstance(self.xtc_probability, float)
            and 0.00 <= self.xtc_probability <= 1.00
        ):
            raise ValueError(
                f"xtc_probability must be a float between 0.00 and 1.00")
        if not (
            isinstance(self.xtc_threshold,
                       float) and 0.00 <= self.xtc_threshold <= 0.50
        ):
            raise ValueError(
                f"xtc_threshold must be a float between 0.00 and 0.5")
        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")

    def convert_custom_format_to_json(self, data: dict[str, str]) -> dict[str, int | str | dict[str, str | Any]]:
        """
        Parses a custom-tagged string and converts it into a specific JSON format.

        Args:
            data: A dict containing the custom function and parameter.

        Returns:
            A JSON formatted string, or an empty string if parsing fails.
        """
        try:
            if data.get("current_recipient", "").startswith("functions."):
                func_name = data["current_recipient"].split(".", 1)[1].split("<", 1)[0]
            else:
                return None

            unique_id_suffix = secrets.token_hex(6)
            function_call_id = f"fc_{unique_id_suffix}"
            call_id = f"call_{unique_id_suffix}"

            output_data = {
                "index": 0,
                "id": function_call_id,
                "call_id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": data["current_content"] or "{}"
                },
            }

            return output_data

        except Exception as e:
            print(f"An error occurred during conversion: {e}")
            return None

    def generate_response(
        self,
        text: str,
        finish_reason: Literal["length", "stop"] | None,
        prompt_token_count: int | None = None,
        completion_token_count: int | None = None,
        token_logprobs: list[float] | None = None,
        top_tokens: list[dict[int, float]] | None = None,
        tokens: list[int] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (int | None): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (int | None): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            token_logprobs (list[float] | None): The log probabilities per token,
              in token order.
            top_tokens (list[dict[int, float]] | None): List of dictionaries mapping
              tokens to logprobs for the top N tokens at each token position.
            tokens (list[int] | None): List of tokens to return with logprobs structure
            tool_calls (list[str] | None): List of tool calls.

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs or []
        top_logprobs = top_tokens or []
        tool_calls = tool_calls or []

        # Static response
        response = {
            "id": self.request_id,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                },
            ],
        }

        if token_logprobs or top_logprobs or tokens:
            response["choices"][0]["logprobs"] = {
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "tokens": tokens,
            }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {
                "role": "assistant",
                "content": text,
                "tool_calls": list(filter(lambda x: x is not None, [self.convert_custom_format_to_json(tool or {}) for tool in tool_calls])),
            }
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            raise ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def reset_prompt_cache(self, prompt):
        """Resets the prompt cache and associated state.

        Args:
            prompt (list[int]): The tokenized new prompt which will populate the
                reset cache.
        """
        logging.debug(f"*** Resetting cache. ***")
        self.prompt_cache.model_key = self.model_provider.model_key
        self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)
        if self.model_provider.draft_model is not None:
            self.prompt_cache.cache += make_prompt_cache(
                self.model_provider.draft_model
            )
        self.prompt_cache.tokens = list(prompt)  # Cache the new prompt fully

    def get_prompt_cache(self, prompt):
        """
        Determines the portion of the prompt that needs processing by comparing
        it to the cached prompt and attempting to reuse the common prefix.

        This function updates the internal prompt cache state (tokens and model cache)
        based on the comparison. If a common prefix exists, it attempts to trim
        the model cache (if supported) to match the common prefix length, avoiding
        recomputation.

        Args:
            prompt (list[int]): The tokenized new prompt.

        Returns:
            list[int]: The suffix of the prompt that actually needs to be processed
                       by the model. This will be the full prompt if the cache is
                       reset or cannot be effectively used.
        """
        cache_len = len(self.prompt_cache.tokens)
        prompt_len = len(prompt)
        com_prefix_len = common_prefix_len(self.prompt_cache.tokens, prompt)

        # Leave at least one token in the prompt
        com_prefix_len = min(com_prefix_len, len(prompt) - 1)

        # Condition 1: Model changed or no common prefix at all. Reset cache.
        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or com_prefix_len == 0
        ):
            self.reset_prompt_cache(prompt)

        # Condition 2: Common prefix exists and matches cache length. Process suffix.
        elif com_prefix_len == cache_len:
            logging.debug(
                f"*** Cache is prefix of prompt (cache_len: {cache_len}, prompt_len: {
                    prompt_len}). Processing suffix. ***"
            )
            prompt = prompt[com_prefix_len:]
            self.prompt_cache.tokens.extend(prompt)

        # Condition 3: Common prefix exists but is shorter than cache length. Attempt trim.
        elif com_prefix_len < cache_len:
            logging.debug(
                f"*** Common prefix ({com_prefix_len}) shorter than cache ({
                    cache_len}). Attempting trim. ***"
            )

            if can_trim_prompt_cache(self.prompt_cache.cache):
                num_to_trim = cache_len - com_prefix_len
                logging.debug(f"    Trimming {num_to_trim} tokens from cache.")
                trim_prompt_cache(self.prompt_cache.cache, num_to_trim)
                self.prompt_cache.tokens = self.prompt_cache.tokens[:com_prefix_len]
                prompt = prompt[com_prefix_len:]
                self.prompt_cache.tokens.extend(prompt)
            else:
                logging.debug(f"    Cache cannot be trimmed. Resetting cache.")
                self.reset_prompt_cache(prompt)

        # This case should logically not be reached if com_prefix_len <= cache_len
        else:
            logging.error(
                f"Unexpected cache state: com_prefix_len ({com_prefix_len}) > cache_len ({
                    cache_len}). Resetting cache."
            )
            self.reset_prompt_cache(prompt)

        logging.debug(f"Returning {len(prompt)} tokens for processing.")
        return prompt

    def handle_completion(
        self,
        prompt: list[int],
        stop_id_sequences: list[list[int]],
    ):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (list[int]): The tokenized prompt.
            stop_id_sequences (list[list[int]]): A list of stop words passed
              to the stopping_criteria function
        """
        tokens = []
        finish_reason = "length"
        stop_sequence_suffix = None
        if self.stream:
            self.end_headers()
            logging.debug(f"Starting stream:")
        else:
            logging.debug(f"Starting completion:")
        token_logprobs = []
        top_tokens = []

        prompt = self.get_prompt_cache(prompt)

        text = ""
        tic = time.perf_counter()
        sampler = make_sampler(
            self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            xtc_probability=self.xtc_probability,
            xtc_threshold=self.xtc_threshold,
            xtc_special_tokens=[
                self.tokenizer.eos_token_id,
                self.tokenizer.encode("\n"),
            ],
        )
        logits_processors = make_logits_processors(
            self.logit_bias,
            self.repetition_penalty,
            self.repetition_context_size,
        )

        tool_calls = []
        in_tool_call = False
        tool_call = None
        segment = ""
        stream = StreamableParser(encoding, role=Role.ASSISTANT)
 
        for gen_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            prompt_cache=self.prompt_cache.cache,
            draft_model=self.model_provider.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            try:
                stream.process(gen_response.token)
                tokens.append(gen_response.token)

                print(f"xxxxxx stream {stream.current_role} {stream.last_content_delta} {stream.current_channel} {stream.current_content_type} {stream.current_recipient}")
                if (
                    self.tokenizer.has_tool_calling
                    and stream.current_recipient
                    and stream.current_recipient.startswith("function")
                    and in_tool_call != True
                ):
                    in_tool_call = True
                elif in_tool_call:
                    if gen_response.token == 200012:
                        tool_calls.append(tool_call)
                        tool_call = None
                        in_tool_call = False
                    else:
                        tool_call = {
                            "current_role": stream.current_role,
                            "current_channel": stream.current_channel,
                            "current_recipient": stream.current_recipient,
                            "current_content": stream.current_content
                        }
                else:
                    if stream.current_content:
                        segment += stream.last_content_delta or ""
                        text += stream.last_content_delta or ""

                token = gen_response.token
                logprobs = gen_response.logprobs
                if self.logprobs > 0:
                    sorted_indices = mx.argpartition(-logprobs,
                                                     kth=self.logprobs - 1)
                    top_indices = sorted_indices[: self.logprobs]
                    top_logprobs = logprobs[top_indices]
                    top_token_info = zip(top_indices.tolist(),
                                         top_logprobs.tolist())
                    top_tokens.append(tuple(top_token_info))

                token_logprobs.append(logprobs[token].item())

                if gen_response.token == 200002:
                    finish_reason = "stop"
                    break

                if self.stream and not in_tool_call:
                    if any(
                        (
                            sequence_overlap(stream.tokens, sequence)
                            for sequence in stop_id_sequences
                        )
                    ):
                        continue
                    elif segment or tool_calls:
                        response = self.generate_response(
                            segment, None, tool_calls=tool_calls
                        )
                        self.wfile.write(
                            f"data: {json.dumps(response)}\n\n".encode())
                        self.wfile.flush()
                        segment = ""
                        tool_calls = []
                if gen_response.token == 200012:
                    finish_reason = "stop"
                    break

            except Exception as e:
                print(f"error: {e}" )

        self.prompt_cache.tokens.extend(stream.tokens)

        if gen_response.finish_reason is not None:
            finish_reason = gen_response.finish_reason

        logging.debug(f"Prompt: {gen_response.prompt_tps:.3f} tokens-per-sec")
        logging.debug(
            f"Generation: {gen_response.generation_tps:.3f} tokens-per-sec")
        logging.debug(f"Peak memory: {gen_response.peak_memory:.3f} GB")

        if self.stream:
            response = self.generate_response(
                segment, finish_reason, tool_calls=tool_calls
            )
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            if self.stream_options is not None and self.stream_options["include_usage"]:
                original_prompt_length = (
                    len(self.prompt_cache.tokens) - len(tokens) + len(prompt)
                )
                response = self.completion_usage_response(
                    original_prompt_length, len(tokens)
                )
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()
        else:
            response = self.generate_response(
                text,
                finish_reason,
                len(prompt),
                len(tokens),
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls,
            )
            response_json = json.dumps(response).encode()
            indent = "\t"  # Backslashes can't be inside of f-strings
            logging.debug(f"Outgoing Response: {
                          json.dumps(response, indent=indent)}")

            # Send an additional Content-Length header when it is known
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

    def completion_usage_response(
        self,
        prompt_token_count: int | None = None,
        completion_token_count: int | None = None,
    ):
        response = {
            "id": self.request_id,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        return response

    def handle_chat_completions(self) -> list[int]:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        messages = body["messages"]
        tools = body.get("tools") or []
        convo = process_message_content(messages, tools)
        prompt = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)

        return prompt

    def handle_text_completions(self) -> list[int]:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return self.tokenizer.encode(self.body["prompt"])

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        if self.path.startswith("/v1/models"):
            self.handle_models_request()
        elif self.path == "/health":
            self.handle_health_check()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_health_check(self):
        """
        Handle a GET request for the /health endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        self.wfile.write('{"status": "ok"}'.encode())
        self.wfile.flush()

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        files = ["config.json", "model.safetensors.index.json",
                 "tokenizer_config.json"]

        parts = self.path.split("/")
        filter_repo_id = None
        if len(parts) > 3:
            filter_repo_id = "/".join(parts[3:])

        def probably_mlx_lm(repo):
            if repo.repo_type != "model":
                return False
            if "main" not in repo.refs:
                return False
            if filter_repo_id is not None and repo.repo_id != filter_repo_id:
                return False
            file_names = {f.file_path.name for f in repo.refs["main"].files}
            return all(f in file_names for f in files)

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()
