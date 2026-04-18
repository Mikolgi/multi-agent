from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bootstrap import ensure_user_site_on_path

ensure_user_site_on_path()

from app.agents import MultiAgentSystem
from app.config import AppConfig
from app.domain import ResumeRequest, SessionState


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Локальная мультиагентная система для создания и адаптации резюме."
    )
    parser.add_argument(
        "objective",
        nargs="?",
        default="Составь краткий ATS-friendly черновик резюме под IT-вакансию.",
        help="Цель для системы.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="multi",
        help="Запуск одного агента или полного мультиагентного пайплайна.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Переопределить модель из конфига.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Отключить потоковый вывод в терминале.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=None,
        help="Путь до markdown-файла с профилем кандидата.",
    )
    parser.add_argument(
        "--vacancy-file",
        type=Path,
        default=None,
        help="Путь до markdown-файла с вакансией.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Интерактивный режим диалога с системой.",
    )
    return parser


def read_optional_file(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip(), encoding="utf-8")


def clear_text_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def read_multiline_block(header: str) -> str:
    print(header)
    print("Вставь текст и закончи ввод строкой /end")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "/end":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_chat(
    system: MultiAgentSystem,
    config: AppConfig,
    mode: str,
    stream: bool,
    session: SessionState,
) -> int:
    print(
        "Чат-режим запущен.\n"
        "Пиши запросы по резюме.\n"
        "Команды: /show, /session-clear, /memory-show, /memory-clear, "
        "/runs-show, /runs-clear, /history-clear, /set-profile, /set-vacancy, /exit, /quit."
    )

    while True:
        try:
            objective = input("\nТы> ").strip()
        except EOFError:
            print()
            return 0

        if not objective:
            continue
        if objective in {"/exit", "/quit"}:
            return 0
        if objective == "/show":
            print("\nSession state\n")
            print("Профиль кандидата:\n")
            print(session.candidate_profile or "[пусто]")
            print(f"\nФайл профиля: {config.session_profile_path}")
            print("\nВакансия:\n")
            print(session.vacancy_text or "[пусто]")
            print(f"\nФайл вакансии: {config.session_vacancy_path}")
            print("\nИстория чата:\n")
            if session.history:
                print("\n\n".join(session.history[-4:]))
            else:
                print("[пусто]")
            continue
        if objective == "/session-clear":
            session.clear()
            clear_text_file(config.session_profile_path)
            clear_text_file(config.session_vacancy_path)
            print("Session state очищен: профиль, вакансия и история чата удалены.")
            continue
        if objective == "/memory-show":
            print("\n" + system.memory_summary())
            continue
        if objective == "/memory-clear":
            system.clear_memory()
            print("Long-term memory очищена.")
            continue
        if objective == "/runs-show":
            print("\n" + system.observability_summary())
            continue
        if objective == "/runs-clear":
            system.clear_observability()
            print(f"Логи observability очищены: {config.observability_path}")
            continue
        if objective == "/history-clear":
            session.history.clear()
            print("История текущего чата очищена.")
            continue
        if objective == "/set-profile":
            session.candidate_profile = read_multiline_block("Новый профиль кандидата:")
            write_text_file(config.session_profile_path, session.candidate_profile)
            session.history.clear()
            print(
                f"Профиль кандидата обновлен и сохранен в {config.session_profile_path}. "
                "История чата очищена."
            )
            continue
        if objective == "/set-vacancy":
            session.vacancy_text = read_multiline_block("Новый текст вакансии:")
            write_text_file(config.session_vacancy_path, session.vacancy_text)
            session.history.clear()
            print(
                f"Вакансия обновлена и сохранена в {config.session_vacancy_path}. "
                "История чата очищена."
            )
            continue

        request = ResumeRequest(
            objective=objective,
            candidate_profile=session.candidate_profile,
            vacancy_text=session.vacancy_text,
            conversation_history=session.history,
        )
        result = system.run(request=request, mode=mode, stream=stream)
        if not stream:
            print(result.render())
        else:
            print(
                f"[run_id={result.run_id} duration={result.total_duration_ms}мс]"
            )
        session.history.append(f"Пользователь: {objective}")
        session.history.append(f"Ассистент:\n{result.render()}")


def main() -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()

    config = AppConfig()
    if args.model:
        config.model = args.model

    candidate_profile = read_optional_file(args.candidate_file) or read_optional_file(
        config.session_profile_path
    )
    vacancy_text = read_optional_file(args.vacancy_file) or read_optional_file(
        config.session_vacancy_path
    )
    request = ResumeRequest(
        objective=args.objective,
        candidate_profile=candidate_profile,
        vacancy_text=vacancy_text,
    )
    session = SessionState(
        candidate_profile=candidate_profile,
        vacancy_text=vacancy_text,
    )

    system = MultiAgentSystem(config)
    if args.chat:
        return run_chat(
            system=system,
            config=config,
            mode=args.mode,
            stream=not args.no_stream,
            session=session,
        )

    result = system.run(
        request=request,
        mode=args.mode,
        stream=not args.no_stream,
    )
    if args.no_stream:
        print(result.render())
    else:
        print(f"[run_id={result.run_id} duration={result.total_duration_ms}мс]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
