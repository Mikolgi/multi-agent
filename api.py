from __future__ import annotations

from typing import Annotated

from bootstrap import ensure_user_site_on_path

ensure_user_site_on_path()

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.agents import MultiAgentSystem
from app.config import AppConfig
from app.domain import ResumeRequest

app = FastAPI(title="Resume Copilot API", version="0.2.0")
system = MultiAgentSystem(AppConfig())


class RunRequest(BaseModel):
    objective: str = Field(..., description="Цель генерации резюме.")
    candidate_profile: str = Field(..., description="Факты и сведения о кандидате.")
    vacancy_text: str = Field(default="", description="Текст целевой вакансии.")
    mode: str = Field(default="multi", pattern="^(single|multi)$")


class RunResponse(BaseModel):
    mode: str
    objective: str
    profile_analysis: str | None = None
    resume_draft: str
    vacancy_match: str | None = None
    critic_output: str | None = None
    run_id: str | None = None
    total_duration_ms: int | None = None
    agent_durations_ms: dict[str, int] = Field(default_factory=dict)


class AgentTraceResponse(BaseModel):
    agent: str
    duration_ms: int
    output_chars: int
    streamed: bool
    status: str
    error: str | None = None


class RunLogEntry(BaseModel):
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
    agents: list[AgentTraceResponse]
    error: str | None = None


class SlowestRunResponse(BaseModel):
    run_id: str | None = None
    duration_ms: int | None = None
    mode: str | None = None
    status: str | None = None


class AgentStatsResponse(BaseModel):
    count: int
    avg_duration_ms: float
    max_duration_ms: int


class RunSummaryResponse(BaseModel):
    total_runs: int
    ok_runs: int
    error_runs: int
    avg_total_duration_ms: float
    slowest_run: SlowestRunResponse | None = None
    agent_stats: dict[str, AgentStatsResponse] = Field(default_factory=dict)
    mode_stats: dict[str, int] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_agents(payload: RunRequest) -> RunResponse:
    request = ResumeRequest(
        objective=payload.objective,
        candidate_profile=payload.candidate_profile,
        vacancy_text=payload.vacancy_text,
    )
    result = system.run(request=request, mode=payload.mode)
    return RunResponse(
        mode=result.mode,
        objective=result.objective,
        profile_analysis=result.profile_analysis,
        resume_draft=result.resume_draft,
        vacancy_match=result.vacancy_match,
        critic_output=result.critic_output,
        run_id=result.run_id,
        total_duration_ms=result.total_duration_ms,
        agent_durations_ms=result.agent_durations_ms,
    )


@app.get("/runs/recent", response_model=list[RunLogEntry])
def recent_runs(
    limit: Annotated[int, Query(ge=1, le=200)] = 10,
    status: Annotated[str | None, Query(pattern="^(ok|error)$")] = None,
    mode: Annotated[str | None, Query(pattern="^(single|multi)$")] = None,
    q: str | None = None,
) -> list[RunLogEntry]:
    items = system.recent_runs(limit=limit)
    if status:
        items = [item for item in items if item.get("status") == status]
    if mode:
        items = [item for item in items if item.get("mode") == mode]
    if q:
        needle = q.lower()
        items = [
            item
            for item in items
            if needle in item.get("objective", "").lower()
            or needle in item.get("run_id", "").lower()
        ]
    return [RunLogEntry(**item) for item in items]


@app.get("/runs/summary", response_model=RunSummaryResponse)
def runs_summary() -> RunSummaryResponse:
    return RunSummaryResponse(**system.run_summary())


@app.get("/runs/errors", response_model=list[RunLogEntry])
def run_errors(limit: Annotated[int, Query(ge=1, le=200)] = 20) -> list[RunLogEntry]:
    return [RunLogEntry(**item) for item in system.run_errors(limit=limit)]


@app.get("/runs/{run_id}", response_model=RunLogEntry)
def run_details(run_id: str) -> RunLogEntry:
    item = system.get_run(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunLogEntry(**item)


@app.get("/observability", response_class=HTMLResponse)
def observability_dashboard() -> str:
    return """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Resume Copilot Observability</title>
  <style>
    body { margin: 0; font-family: Segoe UI, sans-serif; background: #f5f0e6; color: #1d2420; }
    main { width: min(1180px, calc(100% - 32px)); margin: 28px auto; }
    h1 { font-size: clamp(34px, 5vw, 56px); margin: 0 0 8px; letter-spacing: -0.04em; }
    .lead { color: #5d675f; margin-bottom: 24px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
    .card, .panel { background: #fffaf0; border: 1px solid #d8cfbe; border-radius: 18px; padding: 16px; box-shadow: 0 8px 22px rgba(31,37,33,.06); }
    .label { color: #667067; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .value { font-size: 28px; font-weight: 700; margin-top: 6px; }
    .layout { display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; margin-top: 14px; align-items: start; }
    .filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    input, select, button { border: 1px solid #d8cfbe; border-radius: 999px; padding: 10px 12px; font: inherit; background: white; }
    button { background: #0f766e; color: white; border-color: #0f766e; cursor: pointer; }
    .runs { display: grid; gap: 10px; max-height: 650px; overflow: auto; }
    .run { background: white; border: 1px solid #d8cfbe; border-radius: 14px; padding: 12px; cursor: pointer; }
    .run:hover, .run.active { border-color: #0f766e; }
    .top { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 8px; }
    .badge { border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 700; }
    .ok { background: #dcfce7; color: #166534; }
    .error { background: #fee2e2; color: #991b1b; }
    .meta { color: #667067; font-size: 13px; display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; background: rgba(31,37,33,.05); border-radius: 12px; padding: 12px; }
    .agent { display: flex; justify-content: space-between; border-bottom: 1px dashed #d8cfbe; padding: 9px 0; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Observability</h1>
  <p class="lead">Локальный dashboard по запускам: статусы, задержки, ошибки, agent traces и метрики.</p>
  <section id="summary" class="cards"></section>
  <section class="layout">
    <div class="panel">
      <div class="filters">
        <input id="q" placeholder="Поиск по run_id/objective" />
        <select id="status"><option value="">Все статусы</option><option>ok</option><option>error</option></select>
        <select id="mode"><option value="">Все режимы</option><option>single</option><option>multi</option></select>
        <button id="refresh">Обновить</button>
      </div>
      <div id="runs" class="runs"></div>
    </div>
    <div class="panel">
      <h2>Детали запуска</h2>
      <div id="details">Выбери запуск слева.</div>
    </div>
  </section>
</main>
<script>
const summary = document.querySelector('#summary');
const runs = document.querySelector('#runs');
const details = document.querySelector('#details');
const q = document.querySelector('#q');
const statusFilter = document.querySelector('#status');
const modeFilter = document.querySelector('#mode');
let active = null;

function card(label, value) {
  return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

async function loadSummary() {
  const data = await fetch('/runs/summary').then(r => r.json());
  summary.innerHTML = [
    card('Runs', data.total_runs),
    card('OK', data.ok_runs),
    card('Errors', data.error_runs),
    card('Avg duration', `${data.avg_total_duration_ms} мс`),
  ].join('');
}

async function loadRuns() {
  const params = new URLSearchParams({limit: '50'});
  if (q.value.trim()) params.set('q', q.value.trim());
  if (statusFilter.value) params.set('status', statusFilter.value);
  if (modeFilter.value) params.set('mode', modeFilter.value);
  const data = await fetch(`/runs/recent?${params}`).then(r => r.json());
  runs.innerHTML = data.length ? data.map(item => `
    <article class="run ${item.run_id === active ? 'active' : ''}" data-id="${item.run_id}">
      <div class="top"><strong>${item.run_id}</strong><span class="badge ${item.status}">${item.status}</span></div>
      <div>${item.objective}</div>
      <div class="meta"><span>${item.mode}</span><span>${item.total_duration_ms} мс</span><span>${item.model}</span></div>
    </article>
  `).join('') : '<p>Запусков нет.</p>';
  runs.querySelectorAll('.run').forEach(node => node.onclick = () => loadRun(node.dataset.id));
}

async function loadRun(id) {
  active = id;
  const item = await fetch(`/runs/${id}`).then(r => r.json());
  details.innerHTML = `
    <p><strong>run_id:</strong> ${item.run_id}</p>
    <p><strong>status:</strong> ${item.status}</p>
    <p><strong>mode:</strong> ${item.mode}</p>
    <p><strong>duration:</strong> ${item.total_duration_ms} мс</p>
    <h3>Агенты</h3>
    ${(item.agents || []).map(agent => `
      <div class="agent"><span>${agent.agent}</span><strong>${agent.duration_ms} мс</strong></div>
    `).join('') || '<p>Нет данных.</p>'}
    <h3>Objective</h3>
    <pre>${item.objective || ''}</pre>
    <h3>Error</h3>
    <pre>${item.error || 'Ошибок нет.'}</pre>
  `;
  await loadRuns();
}

document.querySelector('#refresh').onclick = async () => { await loadSummary(); await loadRuns(); };
q.onkeydown = e => { if (e.key === 'Enter') document.querySelector('#refresh').click(); };
loadSummary().then(loadRuns);
</script>
</body>
</html>"""
