from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    api_key: str = os.getenv("OLLAMA_API_KEY", "ollama")
    model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    request_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "300"))
    max_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "384"))
    skills_dir: Path = project_root / "skills"
    session_profile_path: Path = project_root / "data" / "session_profile.md"
    session_vacancy_path: Path = project_root / "data" / "session_vacancy.md"
    memory_path: Path = project_root / "data" / "memory.jsonl"
    memory_limit: int = int(os.getenv("MEMORY_LIMIT", "3"))
    observability_path: Path = Path(
        os.getenv("OBSERVABILITY_PATH", str(project_root / "logs" / "runs.jsonl"))
    )
