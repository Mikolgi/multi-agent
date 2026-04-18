from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    candidate_profile: str = ""
    vacancy_text: str = ""
    history: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.candidate_profile = ""
        self.vacancy_text = ""
        self.history.clear()


@dataclass
class ResumeRequest:
    objective: str
    candidate_profile: str
    vacancy_text: str = ""
    conversation_history: list[str] = field(default_factory=list)

    def memory_query(self) -> str:
        return "\n".join(
            part
            for part in (
                self.objective,
                self.candidate_profile,
                self.vacancy_text,
            )
            if part.strip()
        )

    def prompt_context(self) -> str:
        blocks = [
            f"Цель:\n{self.objective}",
            f"Профиль кандидата:\n{self.candidate_profile}",
        ]
        if self.vacancy_text.strip():
            blocks.append(f"Вакансия:\n{self.vacancy_text}")
        if self.conversation_history:
            blocks.append(
                "История диалога:\n" + "\n\n".join(self.conversation_history[-6:])
            )
        return "\n\n".join(blocks)
