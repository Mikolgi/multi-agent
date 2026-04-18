from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AgentTrace:
    agent: str
    duration_ms: int
    output_chars: int
    streamed: bool
    status: str = "ok"
    error: str | None = None


@dataclass
class RunTrace:
    run_id: str
    started_at: str
    finished_at: str
    mode: str
    model: str
    objective: str
    status: str
    total_duration_ms: int
    candidate_profile_chars: int
    vacancy_chars: int
    history_items: int
    memory_context_chars: int
    agents: list[AgentTrace] = field(default_factory=list)
    error: str | None = None


class ObservabilityStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def record_run(self, trace: RunTrace) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(trace)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def list_recent(self, limit: int = 10) -> list[dict]:
        items = self._load_all()
        if limit <= 0:
            return list(reversed(items))
        return list(reversed(items[-limit:]))

    def get_run(self, run_id: str) -> dict | None:
        for item in self._load_all():
            if item.get("run_id") == run_id:
                return item
        return None

    def list_errors(self, limit: int = 20) -> list[dict]:
        errors = [item for item in self.list_recent(limit=0) if item.get("status") == "error"]
        if limit <= 0:
            return errors
        return errors[:limit]

    def summary_stats(self) -> dict:
        runs = self._load_all()
        if not runs:
            return {
                "total_runs": 0,
                "ok_runs": 0,
                "error_runs": 0,
                "avg_total_duration_ms": 0,
                "slowest_run": None,
                "agent_stats": {},
                "mode_stats": {},
            }

        ok_runs = sum(1 for item in runs if item.get("status") == "ok")
        error_runs = sum(1 for item in runs if item.get("status") == "error")
        avg_total_duration_ms = round(
            sum(int(item.get("total_duration_ms", 0)) for item in runs) / len(runs), 1
        )
        slowest_run = max(runs, key=lambda item: int(item.get("total_duration_ms", 0)))

        agent_groups: dict[str, list[int]] = {}
        for item in runs:
            for agent in item.get("agents", []):
                agent_groups.setdefault(agent["agent"], []).append(int(agent["duration_ms"]))

        agent_stats = {
            agent: {
                "count": len(durations),
                "avg_duration_ms": round(sum(durations) / len(durations), 1),
                "max_duration_ms": max(durations),
            }
            for agent, durations in sorted(agent_groups.items())
        }

        mode_groups: dict[str, int] = {}
        for item in runs:
            mode = item.get("mode", "unknown")
            mode_groups[mode] = mode_groups.get(mode, 0) + 1

        return {
            "total_runs": len(runs),
            "ok_runs": ok_runs,
            "error_runs": error_runs,
            "avg_total_duration_ms": avg_total_duration_ms,
            "slowest_run": {
                "run_id": slowest_run.get("run_id"),
                "duration_ms": int(slowest_run.get("total_duration_ms", 0)),
                "mode": slowest_run.get("mode"),
                "status": slowest_run.get("status"),
            },
            "agent_stats": agent_stats,
            "mode_stats": mode_groups,
        }

    def clear(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("", encoding="utf-8")

    def summary(self, limit: int = 5) -> str:
        runs = self.list_recent(limit=limit)
        if not runs:
            return "Логи observability пусты."

        blocks: list[str] = []
        for item in runs:
            agents = ", ".join(
                f"{agent['agent']}={agent['duration_ms']}мс"
                for agent in item.get("agents", [])
            )
            lines = [
                f"run_id: {item.get('run_id', '-')}",
                f"status: {item.get('status', '-')}",
                f"mode: {item.get('mode', '-')}",
                f"duration: {item.get('total_duration_ms', 0)}мс",
                f"objective: {item.get('objective', '')[:120]}",
            ]
            if agents:
                lines.append(f"agents: {agents}")
            if item.get("error"):
                lines.append(f"error: {item['error']}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def _load_all(self) -> list[dict]:
        if not self._path.exists():
            return []

        items: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            items.append(json.loads(line))
        return items
