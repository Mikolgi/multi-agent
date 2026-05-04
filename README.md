# Resume Copilot

Локальная мультиагентная система для создания и адаптации резюме под вакансию. Проект построен как Sub-agents архитектура: главный supervisor-agent координирует подчиненных агентов, а выполнение графа описано через `LangGraph`.

## Стек

- LLM: `qwen3.5:4b`
- Локальный inference: `Ollama`
- Оркестрация агентов: `LangGraph`
- Память: `mem0` + локальный `Qdrant` on disk
- Observability: JSONL traces, FastAPI dashboard, Grafana + Loki + Promtail, optional `Langfuse`
- API: `FastAPI`
- HTTP client: `httpx`

## Архитектура

В режиме `multi` работает граф:

```text
START -> supervisor -> profile-analyzer -> resume-writer -> vacancy-matcher -> critic -> END
```

`supervisor` является главным агентом. Он анализирует запрос, профиль, вакансию, историю сессии и долгосрочную память, затем формирует план для sub-agents. Остальные агенты stateless относительно себя: они получают нужный контекст от supervisor и общего state графа.

Роли:

- `supervisor` - координирует маршрут и фиксирует риски.
- `profile-analyzer` - извлекает факты из профиля кандидата.
- `resume-writer` - пишет ATS-friendly черновик резюме.
- `vacancy-matcher` - проверяет соответствие вакансии и усиливает релевантные места.
- `critic` - оценивает качество результата и находит риски.
- `resume-assistant` - single-agent baseline для сравнения.

## Память

Есть два слоя:

- Session state: профиль, вакансия и история текущего чата.
- Long-term memory: `mem0` хранит прошлые результаты в локальном Qdrant, а `data/memory.jsonl` остается audit-log для демонстрации.

Для `mem0` используется embedding-модель Ollama `nomic-embed-text`. Перед первым запуском лучше выполнить:

```powershell
ollama pull nomic-embed-text
```

Если модель не подтянута заранее, `mem0` может попытаться скачать ее при первом обращении к памяти.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py --chat
```

Обычный запуск:

```powershell
.\.venv\Scripts\python.exe main.py --no-stream "Собери резюме под backend intern вакансию"
```

Single-agent baseline:

```powershell
.\.venv\Scripts\python.exe main.py --mode single --no-stream "Сделай краткий черновик резюме"
```

Команды в чате:

- `/set-profile` - сохранить профиль кандидата в `data/session_profile.md`.
- `/set-vacancy` - сохранить вакансию в `data/session_vacancy.md`.
- `/show` - показать session state.
- `/history-clear` - очистить историю текущего чата.
- `/session-clear` - очистить профиль, вакансию и историю.
- `/memory-show` - показать последние записи долгосрочной памяти.
- `/memory-clear` - очистить `mem0` и audit-log.
- `/runs-show` - показать последние observability-логи.
- `/runs-clear` - очистить observability-логи.
- `/exit` или `/quit` - выйти.

## HTTP API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Полезные страницы:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/observability`
- `http://127.0.0.1:8000/runs/summary`
- `http://127.0.0.1:8000/runs/recent`

## Контейнеризация

Приложение можно запускать в Docker, при этом Ollama остается на хосте:

```powershell
docker compose -f ops/app/docker-compose.yml up -d --build
```

Контейнер ходит к Ollama через `http://host.docker.internal:11434/v1`.

## Grafana + Loki

```powershell
docker compose -f ops/observability/docker-compose.yml up -d
```

После запуска:

- Grafana: `http://127.0.0.1:3000`
- login/password: `admin` / `admin`
- Dashboard: `Resume Copilot Observability`

Promtail читает `logs/runs.jsonl`, Loki хранит логи, Grafana показывает runs, errors, latency и raw traces.

## Langfuse

Langfuse SDK встроен опционально. Для включения:

```powershell
$env:LANGFUSE_ENABLED="1"
$env:LANGFUSE_PUBLIC_KEY="..."
$env:LANGFUSE_SECRET_KEY="..."
$env:LANGFUSE_HOST="http://localhost:3000"
```

Если переменные не заданы, Langfuse не используется, а локальная observability продолжает работать.

## Evals

```powershell
.\.venv\Scripts\python.exe evals/run_evals.py
```

Результаты сохраняются в `evals/results.json`.

## Переменные окружения

- `OLLAMA_BASE_URL` - default `http://localhost:11434/v1`.
- `OLLAMA_MODEL` - default `qwen3.5:4b`.
- `OLLAMA_TEMPERATURE` - default `0.3`.
- `OLLAMA_TIMEOUT` - default `300`.
- `OLLAMA_NUM_PREDICT` - default `384`.
- `MEMORY_LIMIT` - сколько релевантных записей из mem0 добавлять в prompt.
- `MEM0_EMBEDDER_MODEL` - default `nomic-embed-text`.
- `MEM0_QDRANT_PATH` - default `data/mem0_qdrant`.
- `MEM0_HISTORY_DB_PATH` - default `data/mem0_history.db`.
- `OBSERVABILITY_PATH` - default `logs/runs.jsonl`.
