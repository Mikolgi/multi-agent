from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from app.config import AppConfig


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._base_url = config.base_url.removesuffix("/v1")
        self._client = httpx.Client(timeout=config.request_timeout)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}"
        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": stream,
            "think": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_predict,
            },
        }

        if not stream:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")

        chunks: list[str] = []
        with self._client.stream(
            "POST",
            f"{self._base_url}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                message = json.loads(line)
                chunk = message.get("response", "")
                if chunk:
                    chunks.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)
                if message.get("done"):
                    break
        return "".join(chunks)
