from __future__ import annotations

from pathlib import Path


class SkillRegistry:
    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir

    def get(self, role: str) -> str:
        skill_path = self._skills_dir / f"{role}.md"
        if not skill_path.exists():
            return ""
        return skill_path.read_text(encoding="utf-8").strip()
