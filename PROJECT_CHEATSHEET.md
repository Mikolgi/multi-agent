# Шпаргалка по проекту Resume Copilot

Этот файл нужен для подготовки к защите. В нем кратко, но достаточно подробно описано, как устроен проект, почему выбраны конкретные технологии и как отвечать на возможные вопросы преподавателя.

## 1. Идея проекта

Мы сделали локальную мультиагентную систему для создания, адаптации и проверки резюме под вакансию.

Предметная область: помощь кандидату при поиске работы. Пользователь задает профиль, вакансию и цель, например:

- составить резюме;
- адаптировать резюме под вакансию;
- проверить резюме;
- оценить соответствие вакансии;
- вытащить сильные/слабые стороны профиля.

Система не просто вызывает одну LLM. Она использует агентную архитектуру: главный supervisor-agent выбирает маршрут, а специализированные sub-agents выполняют только нужные шаги.

## 2. Главный стек

- Python 3.10.
- Ollama как локальный inference server.
- Qwen3.5 4B как локальная LLM.
- LangGraph как framework для графа агентов.
- mem0 как framework долгосрочной памяти.
- Qdrant local on disk как vector store для mem0.
- nomic-embed-text как embedding-модель через Ollama.
- FastAPI как HTTP API и локальный dashboard.
- JSONL logs как базовый observability store.
- Grafana + Loki + Promtail как внешний стек просмотра логов.
- Langfuse как optional LLM-native tracing.
- Docker для контейнеризации приложения.

## 3. Почему выбрали Qwen3.5 4B

Главный критерий был локальный запуск на доступном железе. Большие модели могут давать лучшее качество, но они медленнее и требуют больше RAM/VRAM. Модель 4B является компромиссом: она достаточно легкая, работает через Ollama и подходит для демонстрационного проекта.

Формулировка для защиты:

> Мы выбрали Qwen3.5 4B, потому что проект должен работать локально. Нам был важен баланс между качеством, скоростью и ограничениями железа. Более крупные модели потенциально качественнее, но на текущей машине они работали бы медленнее или нестабильно.

## 4. Что такое LLM и агент

LLM - это большая языковая модель, которая генерирует текст по контексту. Она сама по себе не знает, какую роль выполняет в приложении.

Агент - это LLM, обернутая в роль, инструкцию, контекст, память и место в workflow. В нашем проекте агент имеет:

- role;
- system prompt;
- skill-файл;
- входной context;
- доступ к LLMClient;
- trace в observability.

Коротко:

> LLM - это модель генерации. Агент - это прикладной исполнитель с ролью, инструкциями и участием в workflow.

## 5. Архитектура проекта

Основной паттерн: Supervisor + Sub-agents.

Главный agent:

- `supervisor`;
- анализирует запрос;
- выбирает route;
- не решает задачу сам;
- возвращает JSON с маршрутом.

Sub-agents:

- `profile_analyzer` - анализирует профиль кандидата;
- `resume_writer` - пишет или переписывает резюме;
- `vacancy_matcher` - сопоставляет резюме/профиль с вакансией;
- `critic` - проверяет качество, риски, галлюцинации и пробелы;
- `resume_assistant` - single-agent baseline для сравнения.

## 6. Dynamic Routing

Важная часть проекта: маршрут не фиксированный.

Supervisor может выбрать разные routes:

```json
{
  "route": ["critic"],
  "reason": "Пользователь просит только проверить резюме",
  "risks": ["Не переписывать резюме без запроса"]
}
```

или:

```json
{
  "route": ["profile_analyzer", "resume_writer", "vacancy_matcher"],
  "reason": "Нужно адаптировать резюме под вакансию",
  "risks": ["Не выдумывать опыт под требования вакансии"]
}
```

То есть если пользователь просит только проверку, не запускаются writer и matcher. Если просит адаптацию под вакансию, запускается более длинный маршрут.

## 7. Как работает LangGraph

LangGraph - это framework для построения LLM/agent workflow в виде графа.

В графе есть:

- nodes - шаги выполнения;
- edges - переходы между шагами;
- conditional edges - переходы по условию;
- state - общий объект состояния между узлами.

У нас LangGraph используется так:

```text
START -> supervisor -> conditional route -> selected sub-agents -> END
```

Ключевая логика:

1. Создается `AgentState`.
2. LangGraph запускает `supervisor`.
3. Supervisor возвращает route.
4. Conditional edge выбирает следующий node.
5. Каждый sub-agent обновляет state.
6. После последнего node граф завершается.

Почему это важно:

- это не ручной набор `if/else`;
- архитектура явно описана как граф;
- можно объяснить workflow на диаграмме;
- легко добавлять новых агентов;
- есть общий state выполнения.

## 8. Что такое AgentState

`AgentState` - это краткосрочное состояние одного запуска графа.

В нем лежит:

- исходный `ResumeRequest`;
- `request_context`;
- `memory_context`;
- route от supervisor;
- индекс текущего шага route;
- outputs агентов;
- traces.

Пример:

```python
{
    "request": ResumeRequest(...),
    "request_context": "...",
    "memory_context": "...",
    "route": ["profile_analyzer", "resume_writer"],
    "route_index": 0,
    "profile_analysis": "...",
    "resume_draft": "...",
    "traces": [...]
}
```

Важно:

> AgentState - это не долгосрочная память. Это рабочее состояние одного запуска.

## 9. Как работает полный flow

Sequence flow:

1. Пользователь пишет запрос.
2. CLI или API создает `ResumeRequest`.
3. Система читает профиль, вакансию и историю текущей сессии.
4. Формируется `memory_query`.
5. mem0 ищет релевантную долгосрочную память.
6. Формируется `memory_context`.
7. LangGraph запускает supervisor.
8. Supervisor выбирает route.
9. LangGraph запускает только выбранных sub-agents.
10. Результат сохраняется в mem0 и JSONL logs.
11. Пользователь получает ответ, route, run_id и timings.

## 10. Память

У нас два слоя памяти.

### Краткосрочная память

Это session state текущего чата:

- `candidate_profile`;
- `vacancy_text`;
- `history`.

Команды:

- `/set-profile`;
- `/set-vacancy`;
- `/show`;
- `/history-clear`;
- `/session-clear`.

Профиль и вакансия сохраняются на диск:

- `data/session_profile.md`;
- `data/session_vacancy.md`.

### Долгосрочная память

Долгосрочная память реализована через `mem0`.

Состав:

- `mem0` - framework памяти;
- `Qdrant` - vector store;
- `nomic-embed-text` - embedding-модель;
- `data/mem0_qdrant` - локальные векторы;
- `data/mem0_history.db` - служебная история mem0;
- `data/memory.jsonl` - audit-log для демонстрации.

Как работает:

1. Система берет цель, профиль и вакансию.
2. Собирает `memory_query`.
3. mem0 ищет похожие прошлые записи.
4. Найденные записи добавляются в prompt как `memory_context`.
5. После ответа новый результат сохраняется в mem0.

Важно:

> Настоящая semantic memory - это mem0 + Qdrant. JSONL нужен как человекочитаемый audit-log.

## 11. Что такое embeddings

Embeddings - это числовые векторы, которые кодируют смысл текста.

Пример:

- "backend на Go";
- "Golang developer";
- "микросервисы на Go".

Эти фразы могут быть близки по смыслу, даже если слова разные. Embedding-модель превращает их в векторы, а Qdrant ищет похожие векторы.

Зачем это нужно:

- лучше, чем поиск по точным словам;
- помогает доставать релевантные прошлые записи;
- снижает необходимость хранить всю историю в prompt.

## 12. Почему выбрали mem0

Мы выбрали mem0, потому что это готовый memory framework для AI-агентов из списка подходящих инструментов.

Плюсы:

- agent memory abstraction;
- semantic retrieval;
- интеграция с vector store;
- локальный режим через Qdrant;
- подходит для персональной памяти кандидата.

Минусы:

- нужна embedding-модель;
- первый запуск может быть медленнее;
- требуется следить за устареванием памяти;
- пока нет удобного UI для редактирования конкретных memories.

Почему не просто JSON:

> JSON подходит для audit-log, но не является полноценной semantic memory. mem0 дает retrieval по смыслу, а не только хранение строк.

## 13. Skills

Skills - это markdown-файлы с инструкциями для конкретных агентов.

Файлы:

- `skills/profile-analyzer.md`;
- `skills/resume-writer.md`;
- `skills/vacancy-matcher.md`;
- `skills/critic.md`;
- `skills/resume-assistant.md`.

Зачем они нужны:

- отделяют поведение агента от Python-кода;
- делают систему понятнее;
- позволяют менять инструкции без изменения orchestration;
- являются артефактом для защиты.

Что отвечать:

> Skill - это специализированная инструкция/навык агента. У нас каждый агент получает свой skill-файл, который уточняет, как выполнять роль и какие ограничения соблюдать.

## 14. Observability

У нас несколько уровней observability.

### JSONL logs

Основной локальный лог:

- `logs/runs.jsonl`.

Туда пишется:

- `run_id`;
- `started_at`;
- `finished_at`;
- `mode`;
- `model`;
- `objective`;
- `status`;
- `total_duration_ms`;
- `candidate_profile_chars`;
- `vacancy_chars`;
- `history_items`;
- `memory_context_chars`;
- agents traces;
- errors.

### FastAPI Dashboard

Endpoint:

- `/observability`.

Он показывает последние запуски, ошибки, длительность, детали агентов.

### Grafana + Loki + Promtail

Promtail читает `logs/runs.jsonl`, отправляет в Loki, Grafana показывает dashboard.

### Langfuse

Langfuse подключается опционально через env:

```powershell
$env:LANGFUSE_ENABLED="1"
$env:LANGFUSE_PUBLIC_KEY="..."
$env:LANGFUSE_SECRET_KEY="..."
$env:LANGFUSE_BASE_URL="http://localhost:3001"
```

Langfuse нужен для LLM-native traces:

- span запуска;
- span каждого агента;
- metadata route;
- latency;
- previews output;
- errors.

## 15. Alerting

Сейчас полноценного alerting нет.

Честный ответ:

> У нас реализована база observability: логи, traces, dashboard и Grafana/Loki. Полноценные alerts пока не настроены. Следующий шаг - Grafana alert rules по error rate, latency и падениям запусков.

Возможные alert rules:

- `error_runs > 0`;
- `total_duration_ms > threshold`;
- `agent duration > threshold`;
- нет новых успешных запусков;
- Langfuse trace с error status.

## 16. Evals

Evals нужны, чтобы оценивать не только один красивый ответ, а поведение системы на наборе кейсов.

Что можно оценивать:

- следование инструкции;
- отсутствие галлюцинаций;
- наличие нужных разделов резюме;
- соответствие вакансии;
- качество route supervisor;
- latency;
- ошибки.

У нас evals базовые. Это честно можно сказать как limitation.

Формулировка:

> Мы сделали базовый eval layer, но понимаем, что для production его нужно расширять: добавлять больше кейсов, LLM-as-judge, human review и отдельные метрики для route accuracy.

## 17. Как оценивать LLM и Agentic System

LLM оценивается по:

- качеству текста;
- фактичности;
- следованию prompt;
- скорости;
- стабильности;
- склонности к галлюцинациям.

Agentic System оценивается шире:

- правильно ли supervisor выбрал route;
- нужны ли были все запущенные агенты;
- корректно ли передается state;
- использовалась ли память;
- есть ли ошибки;
- сколько занял весь pipeline;
- улучшается ли результат после sub-agents;
- можно ли отследить выполнение через traces.

## 18. Контейнеризация и изоляция

У нас приложение контейнеризируется через Docker, но Ollama остается на host machine.

Почему так нормально:

- Ollama - это тяжелый локальный inference server;
- модель уже установлена на машине;
- контейнер приложения может ходить к Ollama через host;
- для учебного проекта важно показать изоляцию приложения, а не обязательно упаковывать модель.

Изолированные среды бывают:

- venv;
- Docker;
- VM;
- sandbox;
- cloud runtime.

У нас:

- `.venv` для Python;
- Docker для приложения;
- Ollama отдельно на host.

## 19. C4 и Sequence

Для C4 можно описывать так:

System:

- Resume Copilot.

Containers:

- CLI / Chat;
- FastAPI API;
- LangGraph Orchestrator;
- Sub-agents;
- LLMClient;
- mem0 Memory;
- Observability Store;
- Ollama;
- Grafana/Loki;
- Langfuse.

Sequence:

```text
User -> CLI/API -> MultiAgentSystem -> mem0 -> LangGraph -> Supervisor -> selected agents -> Ollama -> memory/observability -> User
```

## 20. Red flags и как мы их закрыли

### Редфлаг: несколько single agents без общей системы

Закрыто.

У нас есть Supervisor + Sub-agents и общий LangGraph state. Агенты не существуют отдельно, они связаны маршрутом и общим state.

### Редфлаг: нет routing/handoffs/choreography

Закрыто.

Supervisor делает dynamic routing. LangGraph conditional edges запускают только выбранные nodes.

### Редфлаг: не использовали framework

Закрыто.

Используется LangGraph как orchestration framework.

### Редфлаг: не использовали memory framework

Закрыто.

Используется mem0 + Qdrant.

### Редфлаг: выбрали модель без понимания

Закрыто.

Qwen3.5 4B выбрана из-за локального запуска и ограничений железа.

### Редфлаг: нет skills

Закрыто.

Есть отдельные markdown skills для каждого агента.

### Редфлаг: нет observability

Частично/в целом закрыто.

Есть JSONL logs, API dashboard, Grafana/Loki/Promtail, optional Langfuse. Не хватает только полноценного alerting.

## 21. Что честно признать как ограничения

- Модель локальная и небольшая, поэтому качество хуже, чем у больших cloud LLM.
- Supervisor иногда может вернуть невалидный JSON, поэтому есть fallback route.
- Evals базовые, не production-grade.
- Alerting пока не настроен полноценно.
- UI для редактирования долгосрочной памяти пока отсутствует.
- Если профиль устарел, память может вернуть старый контекст.
- Multi-agent pipeline медленнее single-agent, потому что делает несколько LLM-вызовов.

Важно: ограничения не ломают проект, если их честно объяснять.

## 22. Каверзные вопросы и ответы

### Почему вообще нужна мультиагентность?

Потому что задача распадается на разные роли: анализ профиля, написание резюме, сопоставление с вакансией и критика. Один агент может сделать все, но специализированные агенты дают более управляемый процесс и лучше объясняются через traces.

### Почему supervisor, а не фиксированная цепочка?

Фиксированная цепочка всегда гоняет всех агентов, даже если нужен только critic. Supervisor выбирает маршрут динамически, поэтому система гибче и дешевле по времени.

### Что если supervisor ошибется?

Есть fallback route по ключевым словам. Также route логируется в observability/Langfuse, поэтому ошибку можно увидеть и улучшить prompt или правила fallback.

### Почему critic не всегда запускается?

Потому что задача может быть только на извлечение фактов или сопоставление. Но если нужен quality gate, supervisor может добавить critic. Это осознанный dynamic routing, а не обязательная фиксированная цепочка.

### Почему vacancy_matcher убирается без вакансии?

Без текста вакансии matcher будет додумывать требования. Чтобы снизить галлюцинации, route validation удаляет vacancy_matcher, если вакансии нет.

### Почему memory не равна history?

History - это текущий диалог. Memory - долгосрочное хранилище между запусками. History нужна для локального контекста, memory - для поиска релевантного прошлого опыта.

### Почему mem0, а не просто vector database напрямую?

Vector database - это только хранилище. mem0 дает agent memory abstraction: add/search/delete, работу с пользователем/agent_id/run_id и удобную интеграцию в агентный pipeline.

### Почему LangGraph, а не CrewAI/AutoGen?

LangGraph лучше подходит для явного stateful graph workflow и conditional routing. CrewAI удобен для role-based collaboration, AutoGen - для conversational agents, но нам нужен был контролируемый граф с supervisor route.

### Почему не используем OpenAI-compatible route Ollama?

Мы используем native Ollama `/api/generate`, потому что он проще для локального Ollama, поддерживает stream и параметр `think: false`. OpenAI-compatible route тоже возможен, но для текущей локальной системы native endpoint достаточно прямой и контролируемый.

### Почему система может быть медленной?

Multi-agent режим делает несколько LLM-вызовов: supervisor плюс выбранные agents. Локальная 4B модель на CPU/GPU пользователя может отвечать медленно. Это trade-off локальности и приватности.

### Что мониторим?

Run status, total duration, agent durations, output chars, profile/vacancy/history/memory sizes, errors, route и список запущенных агентов.

### Что делать, если память начала мешать?

Можно очистить `/memory-clear`. Дальше можно улучшить систему: добавить memory delete by id, TTL, versioning профиля и просмотр retrieved memories.

## 23. Команды для демонстрации

Запуск чата:

```powershell
.\.venv\Scripts\python.exe main.py --chat
```

Single baseline:

```powershell
.\.venv\Scripts\python.exe main.py --mode single --no-stream "Составь резюме"
```

Multi dynamic routing:

```powershell
.\.venv\Scripts\python.exe main.py --mode multi --no-stream "Проверь мое резюме"
```

API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

Dashboard:

```text
http://127.0.0.1:8000/observability
```

Grafana/Loki:

```powershell
docker compose -f ops/observability/docker-compose.yml up -d
```

App container:

```powershell
docker compose -f ops/app/docker-compose.yml up -d --build
```

## 24. Короткая финальная формулировка проекта

> Мы реализовали локальную мультиагентную систему для работы с резюме. Архитектура построена как Supervisor + Sub-agents. Supervisor через LangGraph динамически выбирает route, а специализированные агенты выполняют только нужные шаги: анализ профиля, написание резюме, сопоставление с вакансией и критику. Система использует Qwen3.5 4B через Ollama, долгосрочную память mem0 + Qdrant, session state для текущего чата, skills в markdown-файлах, observability через JSONL, FastAPI dashboard, Grafana/Loki и optional Langfuse.
