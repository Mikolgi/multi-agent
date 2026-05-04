from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.config import AppConfig
from app.domain import ResumeRequest
from app.langfuse_observability import LangfuseObserver
from app.llm import LLMClient
from app.memory import Mem0MemoryStore, MemoryEntry
from app.observability import AgentTrace, ObservabilityStore, RunTrace
from app.skills import SkillRegistry


ROUTE_PROFILE_ANALYZER = "profile_analyzer"
ROUTE_RESUME_WRITER = "resume_writer"
ROUTE_VACANCY_MATCHER = "vacancy_matcher"
ROUTE_CRITIC = "critic"
ROUTE_END = "end"

ALLOWED_ROUTE_NODES = (
    ROUTE_PROFILE_ANALYZER,
    ROUTE_RESUME_WRITER,
    ROUTE_VACANCY_MATCHER,
    ROUTE_CRITIC,
)

ROUTE_ALIASES = {
    "profile-analyzer": ROUTE_PROFILE_ANALYZER,
    "profile_analyzer": ROUTE_PROFILE_ANALYZER,
    "profile": ROUTE_PROFILE_ANALYZER,
    "analyzer": ROUTE_PROFILE_ANALYZER,
    "resume-writer": ROUTE_RESUME_WRITER,
    "resume_writer": ROUTE_RESUME_WRITER,
    "writer": ROUTE_RESUME_WRITER,
    "vacancy-matcher": ROUTE_VACANCY_MATCHER,
    "vacancy_matcher": ROUTE_VACANCY_MATCHER,
    "matcher": ROUTE_VACANCY_MATCHER,
    "critic": ROUTE_CRITIC,
    "reviewer": ROUTE_CRITIC,
}

DEFAULT_FULL_ROUTE = [
    ROUTE_PROFILE_ANALYZER,
    ROUTE_RESUME_WRITER,
    ROUTE_VACANCY_MATCHER,
    ROUTE_CRITIC,
]


@dataclass
class RouteDecision:
    route: list[str]
    reason: str
    risks: list[str] = field(default_factory=list)
    raw_output: str = ""


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
    route: list[str] = field(default_factory=list)
    supervisor_reason: str | None = None
    supervisor_risks: list[str] = field(default_factory=list)

    def render(self) -> str:
        sections = [f"РЕЖИМ: {self.mode}", f"ЦЕЛЬ: {self.objective}"]
        if self.run_id:
            sections.append(f"RUN ID: {self.run_id}")
        if self.route:
            sections.append(f"МАРШРУТ: {' -> '.join(self.route)}")
        if self.supervisor_reason:
            sections.append(f"ПРИЧИНА МАРШРУТА: {self.supervisor_reason}")
        if self.supervisor_risks:
            sections.append("РИСКИ: " + "; ".join(self.supervisor_risks))
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
        if self.resume_draft:
            sections.extend(("РЕДАКТОР РЕЗЮМЕ", self.resume_draft, ""))
        if self.vacancy_match:
            sections.extend(("СОПОСТАВЛЕНИЕ С ВАКАНСИЕЙ", self.vacancy_match, ""))
        if self.critic_output:
            sections.extend(("КРИТИК", self.critic_output))
        return "\n".join(sections).strip()


class AgentState(TypedDict, total=False):
    request: ResumeRequest
    mode: str
    stream: bool
    request_context: str
    memory_context: str
    supervisor_plan: str
    supervisor_reason: str
    supervisor_risks: list[str]
    route: list[str]
    route_index: int
    profile_analysis: str
    resume_draft: str
    vacancy_match: str
    critic_output: str
    traces: list[AgentTrace]


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
            prompt_parts.append(f"Долгосрочная память:\n{memory_context}")
        if context:
            prompt_parts.append(f"Контекст:\n{context}")
        prompt_parts.append(
            "Отвечай только на русском языке. Будь конкретным, не выдумывай факты "
            "и явно отмечай пробелы в данных."
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
        self._memory = Mem0MemoryStore(
            archive_path=config.memory_path,
            ollama_base_url=config.base_url,
            llm_model=config.model,
            qdrant_path=config.mem0_qdrant_path,
            history_db_path=config.mem0_history_db_path,
            collection_name=config.mem0_collection_name,
            embedder_model=config.mem0_embedder_model,
            embedding_dims=config.mem0_embedding_dims,
            user_id=config.memory_user_id,
            agent_id=config.memory_agent_id,
        )
        self._observability = ObservabilityStore(config.observability_path)
        self._langfuse = LangfuseObserver(config)

        self._supervisor = BaseAgent(
            llm=self._llm,
            role="supervisor",
            system_prompt=(
                "Ты главный агент-маршрутизатор в архитектуре Supervisor + Sub-agents. "
                "Ты не решаешь задачу сам, а выбираешь, какие подчиненные агенты нужны. "
                "Верни только валидный JSON без markdown, пояснений до или после JSON."
            ),
            skill_text=(
                "Доступные route nodes:\n"
                "- profile_analyzer: извлекает и структурирует факты из профиля кандидата.\n"
                "- resume_writer: пишет или переписывает ATS-friendly резюме.\n"
                "- vacancy_matcher: сопоставляет профиль/резюме с вакансией и дает адаптацию.\n"
                "- critic: проверяет резюме/ответ на качество, риски, галлюцинации и пробелы.\n\n"
                "Выбирай только нужных агентов:\n"
                "- Если пользователь просит только проверить/оценить резюме, route должен быть [\"critic\"].\n"
                "- Если пользователь просит только разобрать профиль/вытащить навыки, route должен быть [\"profile_analyzer\"].\n"
                "- Если пользователь просит написать резюме без вакансии, обычно route: [\"profile_analyzer\", \"resume_writer\"].\n"
                "- Если пользователь просит написать и сразу проверить резюме, добавь critic.\n"
                "- Если пользователь просит понять соответствие вакансии, route: [\"profile_analyzer\", \"vacancy_matcher\"].\n"
                "- Если пользователь просит адаптировать резюме под вакансию, route: [\"profile_analyzer\", \"resume_writer\", \"vacancy_matcher\"].\n"
                "- Если нужен полный качественный pipeline, route: [\"profile_analyzer\", \"resume_writer\", \"vacancy_matcher\", \"critic\"].\n\n"
                "Формат ответа:\n"
                "{\n"
                "  \"route\": [\"profile_analyzer\", \"resume_writer\"],\n"
                "  \"reason\": \"коротко почему выбран именно этот маршрут\",\n"
                "  \"risks\": [\"короткие риски\"]\n"
                "}"
            ),
        )
        self._single_agent = BaseAgent(
            llm=self._llm,
            role="resume-assistant",
            system_prompt=(
                "Ты single-agent ассистент по резюме. Сам анализируешь профиль, пишешь "
                "резюме и проверяешь результат, если это нужно по запросу."
            ),
            skill_text=self._skills.get("resume-assistant"),
        )
        self._profile_analyzer = BaseAgent(
            llm=self._llm,
            role="profile-analyzer",
            system_prompt=(
                "Ты аналитик профиля. Выделяй подтвержденные факты, навыки, опыт, "
                "достижения и пробелы. Не пиши готовое резюме."
            ),
            skill_text=self._skills.get("profile-analyzer"),
        )
        self._resume_writer = BaseAgent(
            llm=self._llm,
            role="resume-writer",
            system_prompt=(
                "Ты редактор резюме. Пиши краткий, сильный и ATS-friendly черновик, "
                "адаптированный под цель пользователя. Не выдумывай факты."
            ),
            skill_text=self._skills.get("resume-writer"),
        )
        self._vacancy_matcher = BaseAgent(
            llm=self._llm,
            role="vacancy-matcher",
            system_prompt=(
                "Ты агент сопоставления с вакансией. Сравнивай профиль/резюме с вакансией, "
                "усиливай релевантные места и явно отмечай недостающие требования."
            ),
            skill_text=self._skills.get("vacancy-matcher"),
        )
        self._critic = BaseAgent(
            llm=self._llm,
            role="critic",
            system_prompt=(
                "Ты критик резюме. Проверяй ясность, фактичность, доказательность, "
                "ATS-friendly структуру, соответствие запросу и риск галлюцинаций."
            ),
            skill_text=self._skills.get("critic"),
        )

        self._single_graph = self._build_single_graph()
        self._multi_graph = self._build_multi_graph()

    def clear_memory(self) -> None:
        self._memory.clear()

    def memory_summary(self, limit: int = 5) -> str:
        items = self._memory.list_recent(limit=limit)
        blocks: list[str] = []
        if self._memory.last_error:
            blocks.append(f"mem0 warning: {self._memory.last_error}")
        if not items:
            blocks.append("Память пуста.")
            return "\n\n".join(blocks)

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

    def _build_single_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("resume_assistant", self._single_node)
        graph.add_edge(START, "resume_assistant")
        graph.add_edge("resume_assistant", END)
        return graph.compile()

    def _build_multi_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node(ROUTE_PROFILE_ANALYZER, self._profile_node)
        graph.add_node(ROUTE_RESUME_WRITER, self._resume_node)
        graph.add_node(ROUTE_VACANCY_MATCHER, self._vacancy_node)
        graph.add_node(ROUTE_CRITIC, self._critic_node)

        route_targets = {
            ROUTE_PROFILE_ANALYZER: ROUTE_PROFILE_ANALYZER,
            ROUTE_RESUME_WRITER: ROUTE_RESUME_WRITER,
            ROUTE_VACANCY_MATCHER: ROUTE_VACANCY_MATCHER,
            ROUTE_CRITIC: ROUTE_CRITIC,
            ROUTE_END: END,
        }
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self._next_route_node, route_targets)
        for node in ALLOWED_ROUTE_NODES:
            graph.add_conditional_edges(node, self._next_route_node, route_targets)
        return graph.compile()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        output, trace = self._run_agent(
            label="SUPERVISOR ROUTER",
            agent=self._supervisor,
            objective=(
                "Выбери маршрут выполнения для запроса. "
                "Верни только JSON с полями route, reason, risks."
            ),
            request_context=state["request_context"],
            memory_context=state.get("memory_context", ""),
            stream=False,
        )
        decision = self._parse_route_decision(output, state)
        return self._with_trace(
            state,
            trace,
            supervisor_plan=self._format_supervisor_plan(decision),
            supervisor_reason=decision.reason,
            supervisor_risks=decision.risks,
            route=decision.route,
            route_index=0,
        )

    def _single_node(self, state: AgentState) -> AgentState:
        output, trace = self._run_agent(
            label="АССИСТЕНТ ПО РЕЗЮМЕ",
            agent=self._single_agent,
            objective=state["request"].objective,
            request_context=state["request_context"],
            memory_context=state.get("memory_context", ""),
            stream=state.get("stream", False),
        )
        return self._with_trace(state, trace, resume_draft=output)

    def _profile_node(self, state: AgentState) -> AgentState:
        output, trace = self._run_agent(
            label="АНАЛИТИК ПРОФИЛЯ",
            agent=self._profile_analyzer,
            objective=state["request"].objective,
            request_context=state["request_context"],
            context=state.get("supervisor_plan", ""),
            memory_context=state.get("memory_context", ""),
            stream=state.get("stream", False),
        )
        return self._with_trace(
            state,
            trace,
            **self._advance_route(state, profile_analysis=output),
        )

    def _resume_node(self, state: AgentState) -> AgentState:
        context = (
            f"План supervisor:\n{state.get('supervisor_plan', '')}\n\n"
            f"Анализ профиля:\n{state.get('profile_analysis', '')}"
        )
        output, trace = self._run_agent(
            label="РЕДАКТОР РЕЗЮМЕ",
            agent=self._resume_writer,
            objective=state["request"].objective,
            request_context=state["request_context"],
            context=context,
            memory_context=state.get("memory_context", ""),
            stream=state.get("stream", False),
        )
        return self._with_trace(
            state,
            trace,
            **self._advance_route(state, resume_draft=output),
        )

    def _vacancy_node(self, state: AgentState) -> AgentState:
        context = (
            f"План supervisor:\n{state.get('supervisor_plan', '')}\n\n"
            f"Анализ профиля:\n{state.get('profile_analysis', '')}\n\n"
            f"Черновик резюме:\n{state.get('resume_draft', '')}"
        )
        output, trace = self._run_agent(
            label="СОПОСТАВЛЕНИЕ С ВАКАНСИЕЙ",
            agent=self._vacancy_matcher,
            objective=state["request"].objective,
            request_context=state["request_context"],
            context=context,
            memory_context=state.get("memory_context", ""),
            stream=state.get("stream", False),
        )
        return self._with_trace(
            state,
            trace,
            **self._advance_route(state, vacancy_match=output),
        )

    def _critic_node(self, state: AgentState) -> AgentState:
        context = (
            f"План supervisor:\n{state.get('supervisor_plan', '')}\n\n"
            f"Анализ профиля:\n{state.get('profile_analysis', '')}\n\n"
            f"Черновик резюме:\n{state.get('resume_draft', '')}\n\n"
            f"Сопоставление с вакансией:\n{state.get('vacancy_match', '')}"
        )
        output, trace = self._run_agent(
            label="КРИТИК",
            agent=self._critic,
            objective=state["request"].objective,
            request_context=state["request_context"],
            context=context,
            memory_context=state.get("memory_context", ""),
            stream=state.get("stream", False),
        )
        return self._with_trace(
            state,
            trace,
            **self._advance_route(state, critic_output=output),
        )

    def _next_route_node(self, state: AgentState) -> str:
        route = state.get("route") or DEFAULT_FULL_ROUTE
        route_index = state.get("route_index", 0)
        if route_index >= len(route):
            return ROUTE_END

        next_node = route[route_index]
        if next_node not in ALLOWED_ROUTE_NODES:
            return ROUTE_END
        return next_node

    @staticmethod
    def _advance_route(state: AgentState, **updates: object) -> dict[str, object]:
        updates["route_index"] = state.get("route_index", 0) + 1
        return updates

    def _parse_route_decision(self, raw_output: str, state: AgentState) -> RouteDecision:
        try:
            payload = self._extract_json_object(raw_output)
            normalized_route = self._normalize_route(payload.get("route", []))
            validated_route = self._validate_route(route=normalized_route, state=state)
            if not validated_route:
                raise ValueError("empty route")

            risks_value = payload.get("risks", [])
            risks = (
                [str(item).strip() for item in risks_value if str(item).strip()]
                if isinstance(risks_value, list)
                else []
            )
            return RouteDecision(
                route=validated_route,
                reason=str(
                    payload.get("reason") or "Supervisor выбрал маршрут по запросу."
                ).strip(),
                risks=risks,
                raw_output=raw_output,
            )
        except Exception:
            fallback = self._fallback_route(state)
            return RouteDecision(
                route=fallback,
                reason=(
                    "Supervisor вернул невалидный JSON или некорректный route, "
                    "поэтому применен fallback по ключевым словам запроса."
                ),
                risks=[
                    "Маршрут выбран fallback-логикой; стоит проверить supervisor output в Langfuse."
                ],
                raw_output=raw_output,
            )

    @staticmethod
    def _extract_json_object(raw_output: str) -> dict:
        text = raw_output.strip()
        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            text = fenced.group(1)
        else:
            json_object = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if json_object:
                text = json_object.group(0)

        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Supervisor response is not a JSON object")
        return payload

    @staticmethod
    def _normalize_route(route: object) -> list[str]:
        if not isinstance(route, list):
            return []

        normalized: list[str] = []
        for item in route:
            key = str(item).strip().lower().replace("-", "_")
            canonical = ROUTE_ALIASES.get(key)
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    @staticmethod
    def _validate_route(route: list[str], state: AgentState) -> list[str]:
        request = state["request"]
        has_vacancy = bool(request.vacancy_text.strip())
        validated = [node for node in route if node in ALLOWED_ROUTE_NODES]
        if not has_vacancy:
            validated = [
                node for node in validated
                if node != ROUTE_VACANCY_MATCHER
            ]
        return validated

    def _fallback_route(self, state: AgentState) -> list[str]:
        request = state["request"]
        objective = request.objective.lower()
        has_vacancy = bool(request.vacancy_text.strip())
        has_profile = bool(request.candidate_profile.strip())

        critique_words = (
            "проверь",
            "проверить",
            "оцени",
            "оценить",
            "критик",
            "критика",
            "ревью",
            "ошиб",
            "риск",
            "качество",
        )
        analysis_words = (
            "проанализ",
            "разбери",
            "извлеки",
            "вытащи",
            "факты",
            "навыки",
            "профиль",
        )
        match_words = (
            "подхожу",
            "соответств",
            "сравни",
            "сопостав",
            "match",
            "ваканс",
        )
        write_words = (
            "черновик",
            "собери",
            "составь",
            "напиши",
            "сделай",
            "создай",
            "адаптир",
            "улучши",
            "подправ",
            "перепиши",
        )

        wants_critique = any(word in objective for word in critique_words)
        wants_analysis = any(word in objective for word in analysis_words)
        wants_match = any(word in objective for word in match_words)
        wants_write = any(word in objective for word in write_words)
        if (
            "резюме" in objective
            and not wants_critique
            and not wants_analysis
            and not wants_match
        ):
            wants_write = True

        if wants_critique and not wants_write and not wants_match and not wants_analysis:
            return [ROUTE_CRITIC]

        if wants_analysis and not wants_write and not wants_match:
            return [ROUTE_PROFILE_ANALYZER]

        if wants_match and has_vacancy and not wants_write:
            route = [ROUTE_VACANCY_MATCHER]
            if has_profile:
                route.insert(0, ROUTE_PROFILE_ANALYZER)
            if wants_critique:
                route.append(ROUTE_CRITIC)
            return route

        if wants_write and has_vacancy:
            route = [ROUTE_RESUME_WRITER, ROUTE_VACANCY_MATCHER]
            if has_profile:
                route.insert(0, ROUTE_PROFILE_ANALYZER)
            if wants_critique:
                route.append(ROUTE_CRITIC)
            return route

        if wants_write:
            route = [ROUTE_RESUME_WRITER]
            if has_profile:
                route.insert(0, ROUTE_PROFILE_ANALYZER)
            if wants_critique:
                route.append(ROUTE_CRITIC)
            return route

        if has_vacancy and has_profile:
            return [ROUTE_PROFILE_ANALYZER, ROUTE_VACANCY_MATCHER]

        return [ROUTE_CRITIC]

    @staticmethod
    def _format_supervisor_plan(decision: RouteDecision) -> str:
        risks = decision.risks or [
            "Не выдумывать факты, которых нет в профиле, вакансии, истории или памяти."
        ]
        return "\n".join(
            [
                "Supervisor route decision",
                f"Маршрут: {' -> '.join(decision.route)}",
                f"Причина: {decision.reason}",
                "Риски:",
                *[f"- {risk}" for risk in risks],
            ]
        )

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

        agent_observation = None
        try:
            with self._langfuse.agent_span(
                name=agent.role,
                payload={
                    "objective": objective,
                    "has_memory": bool(memory_context),
                    "context_chars": len(context),
                    "request_context_chars": len(request_context),
                    "memory_context_chars": len(memory_context),
                    "stream": stream,
                },
            ) as agent_observation:
                output = agent.run(
                    objective=objective,
                    request_context=request_context,
                    context=context,
                    memory_context=memory_context,
                    stream=stream,
                    on_chunk=self._write_chunk if stream else None,
                )
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                self._langfuse.update_observation(
                    agent_observation,
                    output={
                        "output_preview": output[:1200],
                        "output_chars": len(output),
                    },
                    metadata={
                        "status": "ok",
                        "agent": agent.role,
                        "duration_ms": duration_ms,
                    },
                )
        except Exception as exc:
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            self._langfuse.update_observation(
                agent_observation,
                metadata={
                    "status": "error",
                    "agent": agent.role,
                    "duration_ms": duration_ms,
                },
                level="ERROR",
                status_message=str(exc),
            )
            raise RuntimeError(
                f"Агент {agent.role} завершился ошибкой после {duration_ms}мс: {exc}"
            ) from exc

        if stream:
            if output and not output.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.write("\n")
            sys.stdout.flush()

        return output, AgentTrace(
            agent=agent.role,
            duration_ms=duration_ms,
            output_chars=len(output),
            streamed=stream,
        )

    @staticmethod
    def _with_trace(
        state: AgentState,
        trace: AgentTrace,
        **updates: object,
    ) -> AgentState:
        traces = [*state.get("traces", []), trace]
        return {"traces": traces, **updates}

    @staticmethod
    def _write_chunk(chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def run(
        self,
        request: ResumeRequest,
        mode: str = "multi",
        stream: bool = False,
    ) -> AgentResult:
        run_id = uuid4().hex[:12]
        started_at = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        request_context = request.prompt_context()
        memory_context = self._memory.format_for_prompt(
            query=request.memory_query(),
            limit=self._config.memory_limit,
        )
        traces: list[AgentTrace] = []
        run_observation = None

        try:
            with self._langfuse.run_span(
                name="resume-copilot-run",
                payload={
                    "run_id": run_id,
                    "mode": mode,
                    "model": self._config.model,
                    "objective": request.objective,
                    "candidate_profile_chars": len(request.candidate_profile),
                    "vacancy_chars": len(request.vacancy_text),
                    "history_items": len(request.conversation_history),
                    "memory_context_chars": len(memory_context),
                    "stream": stream,
                },
            ) as run_observation:
                initial_state: AgentState = {
                    "request": request,
                    "mode": mode,
                    "stream": stream,
                    "request_context": request_context,
                    "memory_context": memory_context,
                    "traces": [],
                }
                final_state = (
                    self._single_graph.invoke(initial_state)
                    if mode == "single"
                    else self._multi_graph.invoke(initial_state)
                )
                traces = final_state.get("traces", [])
                result = AgentResult(
                    mode=mode,
                    objective=request.objective,
                    profile_analysis=final_state.get("profile_analysis"),
                    resume_draft=final_state.get("resume_draft", ""),
                    vacancy_match=final_state.get("vacancy_match"),
                    critic_output=final_state.get("critic_output"),
                    route=final_state.get("route", []),
                    supervisor_reason=final_state.get("supervisor_reason"),
                    supervisor_risks=final_state.get("supervisor_risks", []),
                )
                total_duration_ms = int((time.perf_counter() - started_perf) * 1000)
                result.run_id = run_id
                result.total_duration_ms = total_duration_ms
                result.agent_durations_ms = {
                    item.agent: item.duration_ms for item in traces
                }
                self._langfuse.update_observation(
                    run_observation,
                    output={
                        "resume_preview": result.resume_draft[:1200],
                        "vacancy_match_preview": (result.vacancy_match or "")[:1200],
                        "critic_preview": (result.critic_output or "")[:1200],
                    },
                    metadata={
                        "status": "ok",
                        "run_id": run_id,
                        "total_duration_ms": total_duration_ms,
                        "route": result.route,
                        "supervisor_reason": result.supervisor_reason,
                        "agents": [trace.agent for trace in traces],
                    },
                )

            self._memory.add(
                MemoryEntry(
                    task=result.objective,
                    mode=result.mode,
                    planner_output=self._planner_memory(final_state),
                    executor_output=self._executor_memory(result),
                    critic_output=result.critic_output,
                ),
                run_id=run_id,
            )
            self._record_run(
                run_id=run_id,
                started_at=started_at,
                mode=mode,
                request=request,
                status="ok",
                total_duration_ms=total_duration_ms,
                memory_context=memory_context,
                traces=traces,
            )
            self._langfuse.flush()
            return result
        except Exception as exc:
            total_duration_ms = int((time.perf_counter() - started_perf) * 1000)
            self._langfuse.update_observation(
                run_observation,
                metadata={
                    "status": "error",
                    "run_id": run_id,
                    "total_duration_ms": total_duration_ms,
                },
                level="ERROR",
                status_message=str(exc),
            )
            self._record_run(
                run_id=run_id,
                started_at=started_at,
                mode=mode,
                request=request,
                status="error",
                total_duration_ms=total_duration_ms,
                memory_context=memory_context,
                traces=traces,
                error=str(exc),
            )
            self._langfuse.flush()
            raise

    @staticmethod
    def _planner_memory(final_state: AgentState) -> str | None:
        parts = [
            final_state.get("supervisor_plan"),
            final_state.get("profile_analysis"),
        ]
        text = "\n\n".join(part for part in parts if part)
        return text or None

    @staticmethod
    def _executor_memory(result: AgentResult) -> str:
        return (
            result.resume_draft
            or result.vacancy_match
            or result.profile_analysis
            or result.critic_output
            or ""
        )

    def _record_run(
        self,
        run_id: str,
        started_at: datetime,
        mode: str,
        request: ResumeRequest,
        status: str,
        total_duration_ms: int,
        memory_context: str,
        traces: list[AgentTrace],
        error: str | None = None,
    ) -> None:
        self._observability.record_run(
            RunTrace(
                run_id=run_id,
                started_at=started_at.isoformat(),
                finished_at=datetime.now(timezone.utc).isoformat(),
                mode=mode,
                model=self._config.model,
                objective=request.objective,
                status=status,
                total_duration_ms=total_duration_ms,
                candidate_profile_chars=len(request.candidate_profile),
                vacancy_chars=len(request.vacancy_text),
                history_items=len(request.conversation_history),
                memory_context_chars=len(memory_context),
                agents=traces,
                error=error,
            )
        )
