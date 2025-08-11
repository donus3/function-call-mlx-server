#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

from mlx_lm.utils import load


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None
        self.draft_model = None

        # Preload the default model if it is provided
        if self.cli_args.model is not None:
            self.load("default_model", draft_model_path="default_model")

    def _validate_model_path(self, model_path: str):
        model_path = Path(model_path)
        if model_path.exists() and not model_path.is_relative_to(Path.cwd()):
            raise RuntimeError(
                "Local models must be relative to the current working dir."
            )

    # Added in adapter_path to load dynamically
    def load(self, model_path, adapter_path=None, draft_model_path=None):
        if self.model_key == (model_path, adapter_path, draft_model_path):
            return self.model, self.tokenizer

        # Remove the old model if it exists.
        self.model = None
        self.tokenizer = None
        self.model_key = None
        self.draft_model = None

        # Building tokenizer_config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        if model_path == "default_model":
            if self.cli_args.model is None:
                raise ValueError(
                    "A model path has to be given as a CLI argument or in the HTTP request"
                )
            model, tokenizer = load(
                self.cli_args.model,
                adapter_path=(
                    adapter_path if adapter_path else self.cli_args.adapter_path
                ),  # if the user doesn't change the model but adds an adapter path
                tokenizer_config=tokenizer_config,
            )
        else:
            self._validate_model_path(model_path)
            model, tokenizer = load(
                model_path, adapter_path=adapter_path, tokenizer_config=tokenizer_config
            )

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = (model_path, adapter_path, draft_model_path)
        self.model = model
        self.tokenizer = tokenizer

        def validate_draft_tokenizer(draft_tokenizer):
            # Check if tokenizers are compatible
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                logging.warning(
                    "Draft model tokenizer does not match model tokenizer. Speculative decoding may not work as expected."
                )

        # Load draft model if specified
        if (
            draft_model_path == "default_model"
            and self.cli_args.draft_model is not None
        ):
            self.draft_model, draft_tokenizer = load(self.cli_args.draft_model)
            validate_draft_tokenizer(draft_tokenizer)

        elif draft_model_path is not None and draft_model_path != "default_model":
            self._validate_model_path(draft_model_path)
            self.draft_model, draft_tokenizer = load(draft_model_path)
            validate_draft_tokenizer(draft_tokenizer)
        return self.model, self.tokenizer
