#!/usr/bin/env python3

from collections.abc import Sequence
from typing import (
    Any,
    NamedTuple,
)

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    ReasoningEffort
)


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: list[int],
    stop_id_sequences: list[list[int]],
    eos_token_id: int | None,
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (list[int]): The current sequence of generated tokens.
        stop_id_sequences (list[list[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (int | None): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=0)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids):] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def process_message_content(messages: list[dict[str, str | Any]], tools):
    """
    Convert message content to a format suitable for `apply_chat_template`.

    The function operates on messages in place. It converts the 'content' field
    to a string instead of a list of text fragments.

    Args:
        messages (list[str]): A list of dictionaries, where each dictionary may
          have a 'content' key containing a list of dictionaries with 'type' and
          'text' keys.

    Raises:
        ValueError: If the 'content' type is not supported or if 'text' is missing.

    """
    cache_tool_calls = {}
    harmony_messages: list[Message] = []
    tools_desc = []
    for tool in tools:
        if tool["function"] is not None:
            function = tool["function"]
            tools_desc.append(
                ToolDescription.new(
                    function["name"],
                    function["description"],
                    parameters=function["parameters"],
                )
            )
    developer_message = DeveloperContent.new().with_instructions("Always respond in riddles").with_function_tools(tools_desc)

    for message in messages:
        role = message["role"]
        content = message["content"]
        tool_calls = message["tool_calls"] if "tool_calls" in message else []
        if role == "system":
            harmony_messages.append(Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()
                    .with_model_identity(
                        content if content is not None else "You are ChatGPT, a large language model trained by OpenAI."
                    )
                    .with_reasoning_effort(ReasoningEffort.MEDIUM)
                    .with_required_channels(["analysis", "commentary", "final"])
            ))
        elif role == "user":
            harmony_messages.append(Message.from_role_and_content(Role.USER, content))
        elif role == "assistant":
            harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT,content).with_channel("analysis"))
            if len(tool_calls) > 0:
                harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, tool_calls[0]["function"]["arguments"])
                                    .with_channel("commentary")
                                    .with_recipient(f"functions.{tool_calls[0]["function"]["name"].split("<", 1)[0]}")
                                    .with_content_type("json"))
                cache_tool_calls[tool_calls[0]["id"]] = tool_calls[0]["function"]
        elif role == "tool":
            tool_calls_from_cache = cache_tool_calls[message["tool_call_id"]]
            harmony_messages.append(Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{tool_calls_from_cache["name"].split("<", 1)[0]}"),
                    content,
                ).with_recipient("assistant").with_channel("commentary"))

    harmony_messages.insert(1, Message.from_role_and_content(Role.DEVELOPER, developer_message))

    return Conversation.from_messages(harmony_messages)
