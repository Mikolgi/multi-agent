from __future__ import annotations

import os
import sys
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

from app.config import AppConfig


class LangfuseObserver:
    """Optional Langfuse tracing for agent runs.

    Langfuse is disabled by default so the local demo keeps working without
    an external/self-hosted Langfuse instance. Enable it with:

    LANGFUSE_ENABLED=1
    LANGFUSE_PUBLIC_KEY=...
    LANGFUSE_SECRET_KEY=...
    LANGFUSE_BASE_URL=http://localhost:3001

    The observer intentionally sends compact metadata instead of full resumes
    and vacancies to reduce the risk of leaking personal data into traces.
    """

    def __init__(self, config: AppConfig) -> None:
        self._enabled = bool(
            config.langfuse_enabled
            and config.langfuse_public_key
            and config.langfuse_secret_key
        )
        self._client: Any | None = None
        self._last_error: str | None = None

        if not self._enabled:
            return

        try:
            # Langfuse Python SDK v4 reads config from env via get_client().
            # LANGFUSE_HOST is also set for compatibility with older SDK versions.
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", config.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_BASE_URL", config.langfuse_host)
            os.environ.setdefault("LANGFUSE_HOST", config.langfuse_host)

            try:
                from langfuse import get_client

                self._client = get_client()
            except Exception:
                # Compatibility fallback for older Langfuse SDK versions.
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    host=config.langfuse_host,
                )
        except Exception as exc:
            # Observability must never prevent the application from starting.
            self._enabled = False
            self._client = None
            self._last_error = str(exc)

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @contextmanager
    def run_span(self, name: str, payload: dict[str, Any]) -> Iterator[Any | None]:
        with self._observation(
            name=name,
            as_type="agent",
            payload=payload,
            metadata={"component": "multi-agent-orchestrator"},
        ) as observation:
            yield observation

    @contextmanager
    def agent_span(self, name: str, payload: dict[str, Any]) -> Iterator[Any | None]:
        with self._observation(
            name=name,
            as_type="agent",
            payload=payload,
            metadata={"component": "sub-agent"},
        ) as observation:
            yield observation

    @contextmanager
    def _observation(
        self,
        name: str,
        as_type: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any | None]:
        """Safely create a Langfuse observation.

        Important: if the application code inside the `with` block fails, that
        exception must propagate. Only Langfuse SDK errors are swallowed.
        """
        if not self.enabled:
            with nullcontext(None) as observation:
                yield observation
            return

        try:
            manager = self._client.start_as_current_observation(
                name=name,
                as_type=as_type,
                input=payload,
                metadata=metadata or {},
            )
            observation = manager.__enter__()
        except Exception as exc:
            self._last_error = str(exc)
            with nullcontext(None) as observation:
                yield observation
            return

        try:
            yield observation
        except BaseException:
            exc_info = sys.exc_info()
            try:
                manager.__exit__(*exc_info)
            except Exception as exc:
                self._last_error = str(exc)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception as exc:
                self._last_error = str(exc)

    def update_observation(
        self,
        observation: Any | None,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Update a Langfuse observation if the SDK supports it."""
        if not self.enabled:
            return

        payload: dict[str, Any] = {}
        if output is not None:
            payload["output"] = output
        if metadata:
            payload["metadata"] = metadata
        if level:
            payload["level"] = level
        if status_message:
            payload["status_message"] = status_message

        if not payload:
            return

        try:
            if observation is not None and hasattr(observation, "update"):
                observation.update(**payload)
                return
        except Exception as exc:
            self._last_error = str(exc)

        try:
            if hasattr(self._client, "update_current_observation"):
                self._client.update_current_observation(**payload)
        except Exception as exc:
            self._last_error = str(exc)

    def flush(self) -> None:
        """Flush buffered traces in short-lived CLI runs."""
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception as exc:
            self._last_error = str(exc)
