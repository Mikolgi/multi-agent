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
    memory_backend: str = os.getenv("MEMORY_BACKEND", "mem0")
    memory_user_id: str = os.getenv("MEMORY_USER_ID", "resume-copilot-user")
    memory_agent_id: str = os.getenv("MEMORY_AGENT_ID", "resume-copilot")
    mem0_qdrant_path: Path = Path(
        os.getenv("MEM0_QDRANT_PATH", str(project_root / "data" / "mem0_qdrant"))
    )
    mem0_history_db_path: Path = Path(
        os.getenv("MEM0_HISTORY_DB_PATH", str(project_root / "data" / "mem0_history.db"))
    )
    mem0_collection_name: str = os.getenv("MEM0_COLLECTION_NAME", "resume_copilot_memory")
    mem0_embedder_model: str = os.getenv("MEM0_EMBEDDER_MODEL", "nomic-embed-text")
    mem0_embedding_dims: int = int(os.getenv("MEM0_EMBEDDING_DIMS", "768"))
    memory_limit: int = int(os.getenv("MEMORY_LIMIT", "3"))
    observability_path: Path = Path(
        os.getenv("OBSERVABILITY_PATH", str(project_root / "logs" / "runs.jsonl"))
    )
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_enabled: bool = os.getenv("LANGFUSE_ENABLED", "0") == "1"
