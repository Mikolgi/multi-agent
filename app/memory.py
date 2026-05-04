from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MemoryEntry:
    task: str
    mode: str
    planner_output: str | None
    executor_output: str
    critic_output: str | None

    def as_text(self) -> str:
        parts = [
            f"Задача: {self.task}",
            f"Режим: {self.mode}",
        ]
        if self.planner_output:
            parts.append(f"План и анализ: {self.planner_output}")
        parts.append(f"Результат: {self.executor_output}")
        if self.critic_output:
            parts.append(f"Критика: {self.critic_output}")
        return "\n".join(parts)


class Mem0MemoryStore:
    """Long-term memory via mem0 with a local JSONL audit log for demo/debug."""

    def __init__(
        self,
        archive_path: Path,
        ollama_base_url: str,
        llm_model: str,
        qdrant_path: Path,
        history_db_path: Path,
        collection_name: str,
        embedder_model: str,
        embedding_dims: int,
        user_id: str,
        agent_id: str,
    ) -> None:
        self._archive_path = archive_path
        self._ollama_base_url = ollama_base_url.removesuffix("/v1")
        self._llm_model = llm_model
        self._qdrant_path = qdrant_path
        self._history_db_path = history_db_path
        self._collection_name = collection_name
        self._embedder_model = embedder_model
        self._embedding_dims = embedding_dims
        self._user_id = user_id
        self._agent_id = agent_id
        self._client: Any | None = None
        self._last_error: str | None = None
        self._archive_bootstrapped = False

        self._archive_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._archive_path.exists():
            self._archive_path.write_text("", encoding="utf-8")

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def add(self, entry: MemoryEntry, run_id: str | None = None) -> None:
        try:
            client = self._memory()
        except Exception as exc:
            client = None
            self._last_error = str(exc)

        with self._archive_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        if client is None:
            return

        try:
            client.add(
                messages=entry.as_text(),
                user_id=self._user_id,
                agent_id=self._agent_id,
                run_id=run_id,
                metadata={"mode": entry.mode, "task": entry.task},
                infer=False,
            )
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)

    def clear(self) -> None:
        self._archive_path.write_text("", encoding="utf-8")
        try:
            self._memory().delete_all(user_id=self._user_id, agent_id=self._agent_id)
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)

    def list_recent(self, limit: int = 5) -> list[MemoryEntry]:
        entries = self._load_archive()
        if limit <= 0:
            return entries
        return entries[-limit:]

    def search(self, query: str, limit: int = 3) -> list[str]:
        if limit <= 0 or not query.strip():
            return []

        try:
            result = self._memory().search(
                query=query,
                top_k=limit,
                filters={"user_id": self._user_id, "agent_id": self._agent_id},
                threshold=0.01,
            )
            self._last_error = None
        except Exception as exc:
            self._last_error = str(exc)
            return []

        hits: list[str] = []
        for item in result.get("results", []):
            memory = item.get("memory") if isinstance(item, dict) else None
            if memory:
                hits.append(str(memory))
        return hits

    def format_for_prompt(self, query: str, limit: int = 3) -> str:
        if limit <= 0:
            return ""
        hits = self.search(query=query, limit=limit)
        if not hits:
            return ""

        blocks = []
        for index, hit in enumerate(hits, start=1):
            blocks.append(f"Память mem0 #{index}\n{hit}")
        return "\n\n".join(blocks)

    def _memory(self) -> Any:
        if self._client is not None:
            return self._client

        os.environ.setdefault("MEM0_TELEMETRY", "False")
        logging.getLogger("mem0").setLevel(logging.ERROR)
        from mem0 import Memory

        self._qdrant_path.mkdir(parents=True, exist_ok=True)
        self._history_db_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self._llm_model,
                    "ollama_base_url": self._ollama_base_url,
                    "temperature": 0.1,
                    "max_tokens": 1000,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self._embedder_model,
                    "ollama_base_url": self._ollama_base_url,
                    "embedding_dims": self._embedding_dims,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(self._qdrant_path),
                    "collection_name": self._collection_name,
                    "embedding_model_dims": self._embedding_dims,
                    "on_disk": True,
                },
            },
            "history_db_path": str(self._history_db_path),
        }
        self._client = Memory.from_config(config)
        self._bootstrap_archive_to_mem0()
        return self._client

    def _bootstrap_archive_to_mem0(self) -> None:
        if self._archive_bootstrapped or self._client is None:
            return
        self._archive_bootstrapped = True
        for entry in self._load_archive():
            self._client.add(
                messages=entry.as_text(),
                user_id=self._user_id,
                agent_id=self._agent_id,
                metadata={"mode": entry.mode, "task": entry.task},
                infer=False,
            )

    def _load_archive(self) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        if not self._archive_path.exists():
            return entries
        with self._archive_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entries.append(MemoryEntry(**json.loads(line)))
        return entries


LocalMemoryStore = Mem0MemoryStore
