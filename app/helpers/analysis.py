from collections.abc import Mapping
from typing import cast


def _normalize_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = cast(list[object], value)
        return [
            str(item) for item in items if isinstance(item, (str, int, float, bool))
        ]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _normalize_score(value: object) -> float | None:
    try:
        if value is None:
            return None
        if not isinstance(value, (str, int, float)):
            return None
        return float(value)
    except Exception:
        return None


def normalize_analysis_payload(
    result: Mapping[str, object],
) -> dict[str, str | float | None | list[str]]:
    return {
        "skill_gaps": _normalize_list(result.get("skill_gaps")),
        "job_match_score": _normalize_score(result.get("job_match_score")),
        "improvement_suggestions": _normalize_list(
            result.get("improvement_suggestions")
        ),
        "stack": _normalize_list(result.get("stack")),
        "raw_ai_output": str(result.get("raw_ai_output", "")),
    }
