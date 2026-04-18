from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap import ensure_user_site_on_path

ensure_user_site_on_path()

from app.config import AppConfig
from app.domain import ResumeRequest
from app.orchestrator import AgentResult, MultiAgentSystem


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    score: float
    checks: dict[str, bool]
    details: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local evals for Resume Copilot.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/cases.json"),
        help="Path to eval case definitions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results.json"),
        help="Path to write eval results.",
    )
    parser.add_argument(
        "--max-predict",
        type=int,
        default=192,
        help="Maximum generated tokens per agent call during evals.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N cases. 0 means all cases.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_all(text: str, keywords: list[str]) -> bool:
    haystack = text.lower()
    return all(keyword.lower() in haystack for keyword in keywords)


def contains_any(text: str, keywords: list[str]) -> bool:
    haystack = text.lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def contains_none(text: str, keywords: list[str]) -> bool:
    haystack = text.lower()
    return all(keyword.lower() not in haystack for keyword in keywords)


def has_resume_structure(text: str) -> bool:
    sections = ("опыт", "навык", "образ", "проект", "summary", "профиль")
    return sum(section in text.lower() for section in sections) >= 2


def evaluate_case(case: dict[str, Any], system: MultiAgentSystem) -> CaseResult:
    request = ResumeRequest(
        objective=case["objective"],
        candidate_profile=case["candidate_profile"],
        vacancy_text=case["vacancy_text"],
    )
    result = system.run(request=request, mode=case.get("mode", "multi"), stream=False)
    return score_result(case=case, result=result)


def score_result(case: dict[str, Any], result: AgentResult) -> CaseResult:
    resume_text = result.resume_draft or ""
    match_text = result.vacancy_match or ""
    critic_text = result.critic_output or ""
    combined = "\n".join((resume_text, match_text, critic_text))

    checks = {
        "resume_structure": has_resume_structure(resume_text),
        "resume_keywords": contains_all(
            resume_text, case.get("required_resume_keywords", [])
        ),
        "match_keywords": contains_any(
            combined, case.get("required_match_keywords", [])
        ),
        "critic_keywords": contains_any(
            critic_text, case.get("required_critic_keywords", [])
        ),
        "no_forbidden_keywords": contains_none(
            combined, case.get("forbidden_keywords", [])
        ),
    }
    score = sum(checks.values()) / len(checks)
    return CaseResult(
        case_id=case["id"],
        passed=all(checks.values()),
        score=score,
        checks=checks,
        details={
            "objective": result.objective,
            "profile_analysis": result.profile_analysis,
            "resume_draft": result.resume_draft,
            "vacancy_match": result.vacancy_match,
            "critic_output": result.critic_output,
        },
    )


def build_system(max_predict: int) -> MultiAgentSystem:
    config = AppConfig()
    config.max_predict = max_predict
    config.memory_limit = 0
    config.memory_path = PROJECT_ROOT / "evals" / "tmp_memory.jsonl"
    return MultiAgentSystem(config)


def write_results(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": {
            "cases": len(results),
            "passed": sum(1 for item in results if item.passed),
            "avg_score": round(sum(item.score for item in results) / len(results), 3)
            if results
            else 0.0,
        },
        "results": [
            {
                "case_id": item.case_id,
                "passed": item.passed,
                "score": item.score,
                "checks": item.checks,
                "details": item.details,
            }
            for item in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(results: list[CaseResult]) -> None:
    print("EVAL SUMMARY")
    for item in results:
        status = "PASS" if item.passed else "FAIL"
        print(f"- {item.case_id}: {status} ({item.score:.2f})")
        for check_name, value in item.checks.items():
            marker = "ok" if value else "bad"
            print(f"  {marker}: {check_name}")


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]
    system = build_system(max_predict=args.max_predict)
    results = [evaluate_case(case=case, system=system) for case in cases]
    write_results(args.output, results)
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
