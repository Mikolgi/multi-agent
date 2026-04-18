from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from app.config import AppConfig
from app.domain import ResumeRequest
from app.llm import LLMClient
from app.memory import LocalMemoryStore, MemoryEntry
from app.observability import AgentTrace, ObservabilityStore, RunTrace
from app.skills import SkillRegistry


@dataclass
class AgentResult:
    mode: str
    objective: str
    profile_analysis: str | None
    resume_draft: str
    vacancy_match: str | None
    critic_output: str | None
    run_id: str | None = None
    total_duration_ms: int | None = None
    agent_durations_ms: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        sections = [f"РЕЖИМ: {self.mode}", f"ЦЕЛЬ: {self.objective}"]
        if self.run_id:
            sections.append(f"RUN ID: {self.run_id}")
        if self.total_duration_ms is not None:
            sections.append(f"ВРЕМЯ: {self.total_duration_ms} мс")
        if self.agent_durations_ms:
            timings = ", ".join(
                f"{agent}={duration}мс"
                for agent, duration in self.agent_durations_ms.items()
            )
            sections.append(f"АГЕНТЫ: {timings}")
        sections.append("")
        if self.profile_analysis:
            sections.extend(("АНАЛИТИК ПРОФИЛЯ", self.profile_analysis, ""))
        sections.extend(("РЕДАКТОР РЕЗЮМЕ", self.resume_draft, ""))
        if self.vacancy_match:
            sections.extend(("СОПОСТАВЛЕНИЕ С ВАКАНСИЕЙ", self.vacancy_match, ""))
        if self.critic_output:
            sections.extend(("КРИТИК", self.critic_output))
        return "\n".join(sections).strip()


class BaseAgent:
    def __init__(
        self,
        llm: LLMClient,
        role: str,
        system_prompt: str,
        skill_text: str = "",
    ) -> None:
        self._llm = llm
        self.role = role
        self.system_prompt = system_prompt
        self.skill_text = skill_text

    def run(
        self,
        objective: str,
        request_context: str,
        context: str = "",
        memory_context: str = "",
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        prompt_parts = [f"Роль: {self.role}", f"Цель: {objective}", request_context]
        if self.skill_text:
            prompt_parts.append(f"Навык:\n{self.skill_text}")
        if memory_context:
            prompt_parts.append(f"Память:\n{memory_context}")
        if context:
            prompt_parts.append(f"Контекст:\n{context}")
        prompt_parts.append(
            "Отвечай только на русском языке. Будь конкретным, не выдумывай факты и явно помечай пробелы в данных."
        )
        return self._llm.generate(
            system_prompt=self.system_prompt,
            user_prompt="\n\n".join(prompt_parts),
            stream=stream,
            on_chunk=on_chunk,
        )


class MultiAgentSystem:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._llm = LLMClient(config)
        self._skills = SkillRegistry(config.skills_dir)
        self._memory = LocalMemoryStore(config.memory_path)
        self._observability = ObservabilityStore(config.observability_path)

        self._single_agent = BaseAgent(
            llm=self._llm,
            role="resume-assistant",
            system_prompt=(
                "Ты ассистент по резюме. Составляй чистый, фактический и ATS-friendly "
                "черновик резюме по данным кандидата и тексту вакансии. Всегда отвечай на русском."
            ),
            skill_text=self._skills.get("resume-assistant"),
        )
        self._profile_analyzer = BaseAgent(
            llm=self._llm,
            role="profile-analyzer",
            system_prompt=(
                "Ты аналитик профиля. Выделяй факты, пригодные для резюме, "
                "устраняй неясности и структурируй опыт кандидата. Всегда отвечай на русском."
            ),
            skill_text=self._skills.get("profile-analyzer"),
        )
        self._resume_writer = BaseAgent(
            llm=self._llm,
            role="resume-writer",
            system_prompt=(
                "Ты редактор резюме. Пиши краткий, сильный и ATS-friendly черновик "
                "резюме, адаптированный под цель и вакансию. Всегда отвечай на русском."
            ),
            skill_text=self._skills.get("resume-writer"),
        )
        self._vacancy_matcher = BaseAgent(
            llm=self._llm,
            role="vacancy-matcher",
            system_prompt=(
                "Ты агент сопоставления с вакансией. Адаптируй черновик под вакансию, "
                "усиливай релевантный опыт и отмечай пробелы по требованиям. Всегда отвечай на русском."
            ),
            skill_text=self._skills.get("vacancy-matcher"),
        )
        self._critic = BaseAgent(
            llm=self._llm,
            role="critic",
            system_prompt=(
                "Ты критик резюме. Проверяй текст на ясность, доказательность, "
                "ATS-friendly стиль и релевантность вакансии. Всегда отвечай на русском."
            ),
            skill_text=self._skills.get("critic"),
        )

    def clear_memory(self) -> None:
        self._memory.clear()

    def memory_summary(self, limit: int = 5) -> str:
        items = self._memory.list_recent(limit=limit)
        if not items:
            return "Память пуста."

        blocks: list[str] = []
        for index, item in enumerate(items, start=1):
            lines = [
                f"Запись #{index}",
                f"Запрос: {item.task}",
                f"Режим: {item.mode}",
            ]
            if item.planner_output:
                lines.append(f"Анализ: {item.planner_output[:200]}")
            lines.append(f"Ответ: {item.executor_output[:200]}")
            if item.critic_output:
                lines.append(f"Критика: {item.critic_output[:200]}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def clear_observability(self) -> None:
        self._observability.clear()

    def observability_summary(self, limit: int = 5) -> str:
        return self._observability.summary(limit=limit)

    def recent_runs(self, limit: int = 10) -> list[dict]:
        return self._observability.list_recent(limit=limit)

    def run_summary(self) -> dict:
        return self._observability.summary_stats()

    def run_errors(self, limit: int = 20) -> list[dict]:
        return self._observability.list_errors(limit=limit)

    def get_run(self, run_id: str) -> dict | None:
        return self._observability.get_run(run_id)

    def _run_agent(
        self,
        label: str,
        agent: BaseAgent,
        objective: str,
        request_context: str,
        context: str = "",
        memory_context: str = "",
        stream: bool = False,
    ) -> tuple[str, AgentTrace]:
        started_at = time.perf_counter()

        if stream:
            sys.stdout.write(f"{label}\n")
            sys.stdout.flush()

        try:
            output = agent.run(
                objective=objective,
                request_context=request_context,
                context=context,
                memory_context=memory_context,
                stream=stream,
                on_chunk=self._write_chunk if stream else None,
            )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            raise RuntimeError(
                f"Агент {agent.role} завершился ошибкой после {duration_ms}мс: {exc}"
            ) from exc

        if stream:
            if output and not output.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.write("\n")
            sys.stdout.flush()

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        trace = AgentTrace(
            agent=agent.role,
            duration_ms=duration_ms,
            output_chars=len(output),
            streamed=stream,
        )
        return output, trace

    @staticmethod
    def _write_chunk(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def run(self, request: ResumeRequest, mode: str = "multi", stream: bool = False) -> AgentResult:
        run_id = uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        request_context = request.prompt_context()
        memory_context = self._memory.format_for_prompt(
            query=request.memory_query(),
            limit=self._config.memory_limit,
        )
        traces: list[AgentTrace] = []

        try:
            if mode == "single":
                resume_draft, trace = self._run_agent(
                    label="АССИСТЕНТ ПО РЕЗЮМЕ",
                    agent=self._single_agent,
                    objective=request.objective,
                    request_context=request_context,
                    memory_context=memory_context,
                    stream=stream,
                )
                traces.append(trace)
                result = AgentResult(
                    mode=mode,
                    objective=request.objective,
                    profile_analysis=None,
                    resume_draft=resume_draft,
                    vacancy_match=None,
                    critic_output=None,
                )
            else:
                profile_analysis, trace = self._run_agent(
                    label="АНАЛИТИК ПРОФИЛЯ",
                    agent=self._profile_analyzer,
                    objective=request.objective,
                    request_context=request_context,
                    memory_context=memory_context,
                    stream=stream,
                )
                traces.append(trace)

                resume_draft, trace = self._run_agent(
                    label="РЕДАКТОР РЕЗЮМЕ",
                    agent=self._resume_writer,
                    objective=request.objective,
                    request_context=request_context,
                    context=profile_analysis,
                    memory_context=memory_context,
                    stream=stream,
                )
                traces.append(trace)

                vacancy_match, trace = self._run_agent(
                    label="СОПОСТАВЛЕНИЕ С ВАКАНСИЕЙ",
                    agent=self._vacancy_matcher,
                    objective=request.objective,
                    request_context=request_context,
                    context=(
                        f"Анализ профиля:\n{profile_analysis}\n\n"
                        f"Текущий черновик:\n{resume_draft}"
                    ),
                    memory_context=memory_context,
                    stream=stream,
                )
                traces.append(trace)

                critic_output, trace = self._run_agent(
                    label="КРИТИК",
                    agent=self._critic,
                    objective="Проверь пакет резюме на качество и риски.",
                    request_context=request_context,
                    context=(
                        f"Анализ профиля:\n{profile_analysis}\n\n"
                        f"Черновик резюме:\n{resume_draft}\n\n"
                        f"Сопоставление с вакансией:\n{vacancy_match}"
                    ),
                    memory_context=memory_context,
                    stream=stream,
                )
                traces.append(trace)

                result = AgentResult(
                    mode=mode,
                    objective=request.objective,
                    profile_analysis=profile_analysis,
                    resume_draft=resume_draft,
                    vacancy_match=vacancy_match,
                    critic_output=critic_output,
                )

            total_duration_ms = int((time.perf_counter() - started_perf) * 1000)
            result.run_id = run_id
            result.total_duration_ms = total_duration_ms
            result.agent_durations_ms = {
                item.agent: item.duration_ms for item in traces
            }

            self._memory.add(
                MemoryEntry(
                    task=result.objective,
                    mode=result.mode,
                    planner_output=result.profile_analysis,
                    executor_output=result.resume_draft,
                    critic_output=result.critic_output,
                )
            )
            self._observability.record_run(
                RunTrace(
                    run_id=run_id,
                    started_at=started_at.isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    mode=mode,
                    model=self._config.model,
                    objective=request.objective,
                    status="ok",
                    total_duration_ms=total_duration_ms,
                    candidate_profile_chars=len(request.candidate_profile),
                    vacancy_chars=len(request.vacancy_text),
                    history_items=len(request.conversation_history),
                    memory_context_chars=len(memory_context),
                    agents=traces,
                )
            )
            return result
        except Exception as exc:
            total_duration_ms = int((time.perf_counter() - started_perf) * 1000)
            self._observability.record_run(
                RunTrace(
                    run_id=run_id,
                    started_at=started_at.isoformat(),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    mode=mode,
                    model=self._config.model,
                    objective=request.objective,
                    status="error",
                    total_duration_ms=total_duration_ms,
                    candidate_profile_chars=len(request.candidate_profile),
                    vacancy_chars=len(request.vacancy_text),
                    history_items=len(request.conversation_history),
                    memory_context_chars=len(memory_context),
                    agents=traces,
                    error=str(exc),
                )
            )
            raise
