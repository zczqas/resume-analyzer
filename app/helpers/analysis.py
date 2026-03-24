

def _normalize_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        return [value]
    return [str(value)]


def _normalize_score(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def normalize_analysis_payload(result: dict[str, str | float | None]) -> dict[str, str | float | None | list[str]]:
    return {
        "skill_gaps": _normalize_list(result.get("skill_gaps")),
        "job_match_score": _normalize_score(result.get("job_match_score")),
        "improvement_suggestions": _normalize_list(result.get("improvement_suggestions")),
        "stack": _normalize_list(result.get("stack")),
        "raw_ai_output": str(result.get("raw_ai_output", "")),
    }
