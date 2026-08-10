from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from .contracts import AgentAction


class ModelConfigurationError(ValueError):
    pass


class ModelActionError(RuntimeError):
    pass


class StructuredActionModel(Protocol):
    model_name: str

    def complete_action(self, *, system_prompt: str, user_prompt: str) -> AgentAction: ...


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60.0
    max_tokens: int = 800
    max_attempts: int = 2

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ModelConfigurationError("base_url must be HTTP(S)")
        if not self.model or not self.api_key:
            raise ModelConfigurationError("model and api_key must not be empty")
        if self.timeout_seconds <= 0 or self.max_tokens <= 0 or self.max_attempts <= 0:
            raise ModelConfigurationError("model limits must be positive")

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "CODING_AGENT_API_KEY",
        timeout_seconds: float = 60.0,
        max_tokens: int = 800,
    ) -> OpenAICompatibleConfig:
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ModelConfigurationError(f"missing API key environment variable: {api_key_env}")
        return cls(base_url, model, api_key, timeout_seconds, max_tokens)


class OpenAICompatibleActionModel:
    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self.config = config
        self.model_name = config.model

    def complete_action(self, *, system_prompt: str, user_prompt: str) -> AgentAction:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                value = json.loads(content)
                if not isinstance(value, dict):
                    raise ModelActionError("model action must be a JSON object")
                return AgentAction.from_dict(value)
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                ModelActionError,
            ) as exc:
                last_error = exc
                if attempt < self.config.max_attempts:
                    time.sleep(0.2 * attempt)
        raise ModelActionError(f"model failed to produce a valid action: {type(last_error).__name__}: {last_error}")
