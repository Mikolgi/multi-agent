from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from app.config import AppConfig


class LangfuseObserver:
    """Optional Langfuse tracing for agent runs.

    It is disabled by default so the local demo works without a Langfuse server.
    Enable with LANGFUSE_ENABLED=1 plus LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY.
    """

    def __init__(self, config: AppConfig) -> None:
        self._enabled = bool(
            config.langfuse_enabled
            and config.langfuse_public_key
            and config.langfuse_secret_key
        )
        self._client: Any | None = None
        if self._enabled:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host,
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def run_span(self, name: str, payload: dict[str, Any]) -> Iterator[None]:
        if not self._client:
            with nullcontext():
                yield
            return

        with self._client.start_as_current_observation(
            name=name,
            as_type="agent",
            input=payload,
            metadata={"component": "multi-agent-orchestrator"},
        ):
            yield

    @contextmanager
    def agent_span(self, name: str, payload: dict[str, Any]) -> Iterator[None]:
        if not self._client:
            with nullcontext():
                yield
            return

        with self._client.start_as_current_observation(
            name=name,
            as_type="agent",
            input=payload,
            metadata={"component": "sub-agent"},
        ):
            yield

    def flush(self) -> None:
        if self._client:
            self._client.flush()
