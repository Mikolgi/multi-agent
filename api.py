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

app = FastAPI(title="Local Multi-Agent API", version="0.1.0")
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
    payload = system.run_summary()
    return RunSummaryResponse(**payload)


@app.get("/runs/errors", response_model=list[RunLogEntry])
def run_errors(limit: Annotated[int, Query(ge=1, le=200)] = 20) -> list[RunLogEntry]:
    items = system.run_errors(limit=limit)
    return [RunLogEntry(**item) for item in items]


@app.get("/runs/{run_id}", response_model=RunLogEntry)
def run_details(run_id: str) -> RunLogEntry:
    item = system.get_run(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunLogEntry(**item)


@app.get("/observability", response_class=HTMLResponse)
def observability_dashboard() -> str:
    title = "Resume Copilot Observability"
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f3efe4;
      --panel: #fffaf0;
      --ink: #1f2521;
      --muted: #5f6b63;
      --accent: #0f766e;
      --accent-2: #b45309;
      --line: #d8cfbe;
      --ok: #166534;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(180,83,9,0.12), transparent 24%),
        linear-gradient(180deg, #f8f4ea 0%, var(--bg) 100%);
    }}
    .wrap {{
      width: min(1280px, calc(100% - 32px));
      margin: 28px auto 48px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(32px, 5vw, 54px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .lead {{
      color: var(--muted);
      margin: 0 0 24px;
      max-width: 760px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .cards {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .main {{
      grid-template-columns: minmax(360px, 1.2fr) minmax(320px, 0.8fr);
      align-items: start;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(31,37,33,0.06);
      backdrop-filter: blur(8px);
    }}
    .card-value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .card-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 14px;
    }}
    input, select, button {{
      font: inherit;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 10px 14px;
      background: white;
      color: var(--ink);
    }}
    button {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      cursor: pointer;
    }}
    .run-list {{
      display: grid;
      gap: 10px;
      max-height: 680px;
      overflow: auto;
      padding-right: 4px;
    }}
    .run-item {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.8);
      cursor: pointer;
      transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }}
    .run-item:hover {{
      transform: translateY(-1px);
      border-color: var(--accent);
      box-shadow: 0 8px 16px rgba(15,118,110,0.10);
    }}
    .run-item.active {{
      border-color: var(--accent);
      outline: 2px solid rgba(15,118,110,0.14);
    }}
    .run-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .ok {{ background: rgba(22,101,52,0.10); color: var(--ok); }}
    .error {{ background: rgba(185,28,28,0.10); color: var(--bad); }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 10px;
    }}
    .objective {{
      margin: 0;
      font-weight: 600;
      line-height: 1.35;
    }}
    .details pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(31,37,33,0.04);
      border-radius: 14px;
      padding: 14px;
      margin: 0;
      max-height: 420px;
      overflow: auto;
    }}
    .agent-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 0;
      border-bottom: 1px dashed var(--line);
    }}
    .agent-row:last-child {{ border-bottom: none; }}
    .detail-muted {{
      color: var(--muted);
      font-size: 14px;
    }}
    .section-title {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .pillbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .pill {{
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(15,118,110,0.08);
      color: var(--ink);
      font-size: 13px;
    }}
    @media (max-width: 960px) {{
      .main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Observability</h1>
    <p class="lead">
      Просмотр запусков мультиагентной системы: статусы, тайминги, ошибки,
      детали по агентам и агрегированные метрики без чтения JSONL вручную.
    </p>

    <section id="summary" class="grid cards"></section>

    <section class="grid main" style="margin-top: 16px;">
      <div class="panel">
        <div class="filters">
          <input id="query" placeholder="Поиск по run_id или objective" />
          <select id="status">
            <option value="">Все статусы</option>
            <option value="ok">ok</option>
            <option value="error">error</option>
          </select>
          <select id="mode">
            <option value="">Все режимы</option>
            <option value="single">single</option>
            <option value="multi">multi</option>
          </select>
          <button id="refresh">Обновить</button>
        </div>
        <div id="runs" class="run-list"></div>
      </div>

      <div class="panel details">
        <h2 class="section-title">Детали запуска</h2>
        <div id="details-empty" class="detail-muted">Выбери запуск слева.</div>
        <div id="details-content" hidden>
          <div class="pillbar" id="detail-meta"></div>
          <h3 class="section-title" style="margin-top: 18px;">Тайминги агентов</h3>
          <div id="detail-agents"></div>
          <h3 class="section-title" style="margin-top: 18px;">Objective</h3>
          <pre id="detail-objective"></pre>
          <h3 class="section-title" style="margin-top: 18px;">Ошибка</h3>
          <pre id="detail-error"></pre>
        </div>
      </div>
    </section>
  </div>

  <script>
    const summaryRoot = document.getElementById('summary');
    const runsRoot = document.getElementById('runs');
    const detailsEmpty = document.getElementById('details-empty');
    const detailsContent = document.getElementById('details-content');
    const detailMeta = document.getElementById('detail-meta');
    const detailAgents = document.getElementById('detail-agents');
    const detailObjective = document.getElementById('detail-objective');
    const detailError = document.getElementById('detail-error');
    const queryInput = document.getElementById('query');
    const statusSelect = document.getElementById('status');
    const modeSelect = document.getElementById('mode');
    const refreshButton = document.getElementById('refresh');

    let activeRunId = null;

    function card(label, value) {{
      return `<div class="panel"><div class="card-label">${{label}}</div><div class="card-value">${{value}}</div></div>`;
    }}

    async function loadSummary() {{
      const data = await fetch('/runs/summary').then(r => r.json());
      const slowest = data.slowest_run ? `${{data.slowest_run.duration_ms}} мс` : '—';
      const slowestAgent = Object.entries(data.agent_stats || {{}})
        .sort((a, b) => b[1].avg_duration_ms - a[1].avg_duration_ms)[0];
      summaryRoot.innerHTML = [
        card('Всего запусков', data.total_runs),
        card('Успешные', data.ok_runs),
        card('Ошибки', data.error_runs),
        card('Среднее время', `${{data.avg_total_duration_ms}} мс`),
        card('Самый медленный запуск', slowest),
        card('Самый медленный агент', slowestAgent ? `${{slowestAgent[0]}} (${{
          slowestAgent[1].avg_duration_ms
        }} мс)` : '—'),
      ].join('');
    }}

    function renderRuns(items) {{
      if (!items.length) {{
        runsRoot.innerHTML = '<div class="detail-muted">Запусков по текущему фильтру нет.</div>';
        return;
      }}
      runsRoot.innerHTML = items.map(item => {{
        const agentPreview = (item.agents || []).map(agent =>
          `${{agent.agent}}=${{agent.duration_ms}}мс`
        ).join(', ');
        return `
          <article class="run-item ${{item.run_id === activeRunId ? 'active' : ''}}" data-run-id="${{item.run_id}}">
            <div class="run-top">
              <strong>${{item.run_id}}</strong>
              <span class="badge ${{item.status}}">${{item.status}}</span>
            </div>
            <p class="objective">${{item.objective}}</p>
            <div class="meta">
              <span>mode: ${{item.mode}}</span>
              <span>duration: ${{item.total_duration_ms}} мс</span>
              <span>model: ${{item.model}}</span>
            </div>
            <div class="meta">
              <span>${{agentPreview || 'агентов нет'}}</span>
            </div>
          </article>
        `;
      }}).join('');

      runsRoot.querySelectorAll('.run-item').forEach(node => {{
        node.addEventListener('click', () => loadRun(node.dataset.runId));
      }});
    }}

    async function loadRuns() {{
      const params = new URLSearchParams();
      params.set('limit', '50');
      if (queryInput.value.trim()) params.set('q', queryInput.value.trim());
      if (statusSelect.value) params.set('status', statusSelect.value);
      if (modeSelect.value) params.set('mode', modeSelect.value);
      const data = await fetch(`/runs/recent?${{params.toString()}}`).then(r => r.json());
      renderRuns(data);
      if (!activeRunId && data.length) {{
        loadRun(data[0].run_id);
      }}
    }}

    async function loadRun(runId) {{
      const item = await fetch(`/runs/${{runId}}`).then(r => r.json());
      activeRunId = runId;
      detailMeta.innerHTML = [
        `<span class="pill">run_id: ${{item.run_id}}</span>`,
        `<span class="pill">status: ${{item.status}}</span>`,
        `<span class="pill">mode: ${{item.mode}}</span>`,
        `<span class="pill">duration: ${{item.total_duration_ms}} мс</span>`,
        `<span class="pill">profile chars: ${{item.candidate_profile_chars}}</span>`,
        `<span class="pill">vacancy chars: ${{item.vacancy_chars}}</span>`,
        `<span class="pill">memory chars: ${{item.memory_context_chars}}</span>`,
      ].join('');
      detailAgents.innerHTML = (item.agents || []).map(agent => `
        <div class="agent-row">
          <div>
            <strong>${{agent.agent}}</strong>
            <div class="detail-muted">streamed: ${{agent.streamed}} | output chars: ${{agent.output_chars}}</div>
          </div>
          <div>
            <strong>${{agent.duration_ms}} мс</strong>
          </div>
        </div>
      `).join('') || '<div class="detail-muted">Нет данных по агентам.</div>';
      detailObjective.textContent = item.objective || '';
      detailError.textContent = item.error || 'Ошибок нет.';
      detailsEmpty.hidden = true;
      detailsContent.hidden = false;
      await loadRuns();
    }}

    refreshButton.addEventListener('click', async () => {{
      await loadSummary();
      await loadRuns();
    }});

    queryInput.addEventListener('keydown', event => {{
      if (event.key === 'Enter') refreshButton.click();
    }});

    loadSummary().then(loadRuns);
  </script>
</body>
</html>"""
