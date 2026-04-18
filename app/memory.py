from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class MemoryEntry:
    task: str
    mode: str
    planner_output: str | None
    executor_output: str
    critic_output: str | None


class LocalMemoryStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    def add(self, entry: MemoryEntry) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def clear(self) -> None:
        self._path.write_text("", encoding="utf-8")

    def list_recent(self, limit: int = 5) -> list[MemoryEntry]:
        entries = self._load()
        if limit <= 0:
            return entries
        return entries[-limit:]

    def search(self, query: str, limit: int = 3) -> list[MemoryEntry]:
        entries = self._load()
        if not entries:
            return []

        query_terms = self._tokenize(query)
        ranked = sorted(
            entries,
            key=lambda entry: self._score(entry, query_terms),
            reverse=True,
        )
        return [entry for entry in ranked[:limit] if self._score(entry, query_terms) > 0]

    def format_for_prompt(self, query: str, limit: int = 3) -> str:
        hits = self.search(query=query, limit=limit)
        if not hits:
            return ""

        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            parts = [
                f"Память #{index}",
                f"Запрос: {hit.task}",
                f"Режим: {hit.mode}",
            ]
            if hit.planner_output:
                parts.append(f"Анализ: {hit.planner_output}")
            parts.append(f"Ответ: {hit.executor_output}")
            if hit.critic_output:
                parts.append(f"Критика: {hit.critic_output}")
            blocks.append("\n".join(parts))
        return "\n\n".join(blocks)

    def _load(self) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                entries.append(MemoryEntry(**payload))
        return entries

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w{3,}", text.lower(), flags=re.UNICODE))

    def _score(self, entry: MemoryEntry, query_terms: set[str]) -> int:
        haystack = " ".join(
            part
            for part in (
                entry.task,
                entry.planner_output or "",
                entry.executor_output,
                entry.critic_output or "",
            )
            if part
        )
        entry_terms = self._tokenize(haystack)
        return len(query_terms & entry_terms)
