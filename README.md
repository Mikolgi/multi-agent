# Resume Copilot

Локальная мультиагентная система для создания и адаптации резюме под IT-вакансии.
Работает поверх Ollama и использует orchestrator, ролевых агентов, markdown-skills,
долгосрочную память и базовый слой observability.

## Стек

- LLM: `qwen3.5:4b`
- Локальный инференс: `Ollama`
- HTTP client: `httpx`
- API-слой: `FastAPI`

## Структура проекта

- `main.py` - CLI-вход
- `api.py` - HTTP API
- `bootstrap.py` - подхват user site-packages при проблемах окружения
- `app/config.py` - runtime-конфиг
- `app/llm.py` - клиент к локальной LLM
- `app/domain.py` - доменные сущности и session state
- `app/orchestrator.py` - orchestrator и ролевые агенты
- `app/memory.py` - постоянная локальная память
- `app/observability.py` - structured observability-логи
- `app/skills.py` - загрузчик markdown-skills
- `skills/*.md` - skill-файлы агентов
- `data/session_profile.md` - текущий профиль кандидата для chat-режима
- `data/session_vacancy.md` - текущая вакансия для chat-режима
- `data/memory.jsonl` - история прогонов
- `logs/runs.jsonl` - observability-логи по каждому запуску

## Запуск из CLI

```powershell
.\.venv\Scripts\python.exe main.py "Собери резюме под backend intern вакансию"
```

Одиночный режим:

```powershell
.\.venv\Scripts\python.exe main.py --mode single "Сделай краткий черновик резюме"
```

Интерактивный чат:

```powershell
.\.venv\Scripts\python.exe main.py --chat
```

В чате доступны команды:

- `/show` - показать текущие профиль, вакансию и историю
- `/session-clear` - очистить session state: профиль, вакансию и историю чата
- `/set-profile` - обновить профиль кандидата и сохранить его на диск
- `/set-vacancy` - обновить вакансию и сохранить ее на диск
- `/memory-show` - показать последние записи долгосрочной памяти
- `/memory-clear` - очистить долгосрочную память
- `/runs-show` - показать последние observability-логи
- `/runs-clear` - очистить observability-логи
- `/exit` или `/quit` - выйти

Свои файлы:

```powershell
.\.venv\Scripts\python.exe main.py `
  --candidate-file my_profile.md `
  --vacancy-file my_vacancy.md `
  "Собери резюме под backend intern вакансию"
```

## Запуск HTTP API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Потом открыть: 

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/runs/recent`
- `http://127.0.0.1:8000/runs/summary`
- `http://127.0.0.1:8000/observability`

## Контейнер приложения

Приложение можно запускать отдельно в Docker. `Ollama` при этом пока остается на хосте.

Сборка образа:

```powershell
docker build -t resume-copilot-app .
```

Запуск через compose:

```powershell
docker compose -f ops/app/docker-compose.yml up -d --build
```

После запуска:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/observability`

Что важно:

- контейнер ходит к локальной `Ollama` через `http://host.docker.internal:11434/v1`
- `data/` и `logs/` примонтированы с хоста, поэтому память и observability сохраняются вне контейнера

Остановка:

```powershell
docker compose -f ops/app/docker-compose.yml down
```

## Grafana + Loki

Внешний стек observability лежит в `ops/observability/`.

Запуск:

```powershell
docker compose -f ops/observability/docker-compose.yml up -d
```

После запуска:

- `http://127.0.0.1:3000` - Grafana
- логин: `admin`
- пароль: `admin`
- datasource `Loki` и dashboard `Resume Copilot Observability` будут подхвачены автоматически

Что он читает:

- `logs/runs.jsonl` через `Promtail`

Остановка:

```powershell
docker compose -f ops/observability/docker-compose.yml down
```

## Evals

```powershell
.\.venv\Scripts\python.exe evals/run_evals.py
```

Результаты сохраняются в:

- `evals/results.json`

## Переменные окружения

- `OLLAMA_BASE_URL` - default `http://localhost:11434/v1`
- `OLLAMA_API_KEY` - default `ollama`
- `OLLAMA_MODEL` - default `qwen3.5:4b`
- `OLLAMA_TEMPERATURE` - default `0.3`
- `OLLAMA_TIMEOUT` - timeout запроса в секундах, по умолчанию `300`
- `OLLAMA_NUM_PREDICT` - максимум генерируемых токенов на один вызов агента, по умолчанию `384`
- `MEMORY_LIMIT` - сколько релевантных прошлых прогонов подмешивать в промпт
- `OBSERVABILITY_PATH` - путь до JSONL-лога по запускам, по умолчанию `logs/runs.jsonl`

## Архитектура

- `orchestrator` выбирает путь выполнения и сохраняет результаты в память
- `profile-analyzer` извлекает структурированные факты из профиля кандидата
- `resume-writer` пишет основной черновик резюме
- `vacancy-matcher` адаптирует черновик под вакансию и показывает пробелы по требованиям
- `critic` проверяет итог на ясность, доказательность и ATS-friendly качество
- `session state` хранит текущий профиль кандидата, текущую вакансию и историю текущего диалога
- профиль и вакансия для session state сохраняются в `data/session_profile.md` и `data/session_vacancy.md`
- `long-term memory` хранит прошлые прогоны в `data/memory.jsonl` и достает релевантные записи по lexical overlap
- `observability` пишет в `logs/runs.jsonl` run_id, status, total duration, поагентные тайминги, ошибки и базовые метаданные запуска
- `observability API` дает `recent`, `summary`, `errors` и `run details`, а `/observability` показывает это в браузере
