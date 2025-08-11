#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import (
    Any,
)


@dataclass
class PromptCache:
    cache: list[Any] = field(default_factory=list)
    model_key: tuple[str, str | None,  str | None]  = ("", None, None)
    tokens: list[int] = field(default_factory=list)
