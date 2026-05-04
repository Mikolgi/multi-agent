# Langfuse для Resume Copilot

Langfuse здесь подключен как optional LLM observability.
Если переменные окружения не заданы, приложение продолжает работать как раньше:
JSONL traces, FastAPI dashboard, Grafana + Loki не отключаются.

## Что будет попадать в Langfuse

Один запуск Resume Copilot превращается в trace:

```text
resume-copilot-run
  ├─ supervisor
  ├─ profile-analyzer
  ├─ resume-writer
  ├─ vacancy-matcher
  └─ critic
```

В trace пишутся:

- run_id;
- режим `single` / `multi`;
- модель;
- цель пользователя;
- длительность всего запуска;
- длительность каждого агента;
- размер контекста, памяти, профиля и вакансии;
- preview результата агента;
- ошибки, если агент упал.

Полный профиль и полная вакансия специально не отправляются в Langfuse, чтобы не светить персональные данные кандидата.

## Вариант 1. Langfuse Cloud

1. Создай проект в Langfuse Cloud.
2. В настройках проекта создай API keys.
3. В PowerShell выставь переменные:

```powershell
$env:LANGFUSE_ENABLED="1"
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
$env:LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

Для другого региона используй URL из своего Langfuse проекта.

## Вариант 2. Локальный Langfuse через Docker

Официальный быстрый вариант — поднять Langfuse из их репозитория:

```powershell
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

По умолчанию Langfuse открывается на:

```text
http://localhost:3000
```

Но в этом проекте Grafana тоже использует порт `3000`. Поэтому есть два варианта:

1. Не запускать Grafana одновременно с Langfuse.
2. Переназначить порт Langfuse на `3001:3000` в docker-compose.yml Langfuse и потом использовать:

```powershell
$env:LANGFUSE_BASE_URL="http://localhost:3001"
```

## Запуск приложения с Langfuse

После того как Langfuse поднят и ключи созданы:

```powershell
$env:LANGFUSE_ENABLED="1"
$env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
$env:LANGFUSE_SECRET_KEY="sk-lf-..."
$env:LANGFUSE_BASE_URL="http://localhost:3001"

.\.venv\Scripts\python.exe main.py --no-stream "Собери резюме под backend intern вакансию"
```

Или chat-режим:

```powershell
.\.venv\Scripts\python.exe main.py --chat
```

## Где смотреть traces

Открой Langfuse UI:

```text
http://localhost:3001
```

Дальше:

```text
Project → Tracing / Traces → resume-copilot-run
```

Внутри trace будут вложенные observations по агентам.

## Запуск FastAPI

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Потом дерни API или используй Swagger:

```text
http://127.0.0.1:8000/docs
```

Каждый вызов multi/single pipeline тоже должен появиться в Langfuse.

## Работают ли старые метрики и Grafana

Да. Langfuse ничего не заменяет в текущем проекте.

Остаются рабочими:

- `logs/runs.jsonl`;
- `/observability`;
- `/runs/summary`;
- `/runs/recent`;
- Grafana + Loki + Promtail.

Langfuse — это дополнительный слой observability именно для LLM/agent traces.

## Если traces не появляются

Проверь:

```powershell
echo $env:LANGFUSE_ENABLED
echo $env:LANGFUSE_PUBLIC_KEY
echo $env:LANGFUSE_SECRET_KEY
echo $env:LANGFUSE_BASE_URL
```

Проверь, что Langfuse UI открывается в браузере.

Если приложение запущено в Docker, а Langfuse на host machine, для приложения URL должен быть:

```text
http://host.docker.internal:3001
```

а не `http://localhost:3001`, потому что `localhost` внутри контейнера указывает на сам контейнер.
