import json
import os
import re
from typing import cast

import openai
from dotenv import load_dotenv

_ = load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")


def _format_context_block(retrieved_context: list[dict[str, object]]) -> str:
    if not retrieved_context:
        return "No retrieved job context available."

    lines: list[str] = []
    for idx, item in enumerate(retrieved_context, start=1):
        title = str(item.get("title", "")).strip() or f"Document {idx}"
        skills = str(item.get("skills", "")).strip()
        text = str(item.get("text", "")).strip()
        lines.append(f"[{idx}] Title: {title}")
        if skills:
            lines.append(f"Skills: {skills}")
        lines.append(f"Requirement: {text}")
        lines.append("")
    return "\n".join(lines).strip()


def _make_prompt(
    resume_text: str, job_requirement: str, retrieved_context: list[dict[str, object]]
) -> str:
    context_block = _format_context_block(retrieved_context)
    return (
        """Analyze this CV based on THIS job requirement.
        Job requirement query:
        """
        + job_requirement
        + """

        Retrieved job context (Top-K):
        """
        + context_block
        + """

        Instructions:
        - Use the retrieved job context as grounding for your evaluation.
        - Compare the CV against role expectations and required skills.

        Return ONLY valid JSON (no markdown, no explanation, no <think> tags) with keys:
        - skill_gaps (list of strings),
        - job_match_score (number 0-100),
        - improvement_suggestions (list of strings),
        - stack (list of strings).
        If information is missing, use empty lists and null score.
        Resume:
        """
        + resume_text
    )


def _strip_reasoning_blocks(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None


def _fallback_response() -> str:
    return json.dumps(
        {
            "skill_gaps": ["No AI key configured, fallback estimate used"],
            "job_match_score": 65,
            "improvement_suggestions": [
                "Set GROQ_API_KEY",
                "Include backend project details",
            ],
            "stack": ["Python", "FastAPI", "SQL"],
        }
    )


def _parse_response(text: str) -> dict[str, object]:
    cleaned_text = _strip_reasoning_blocks(text)
    parsed: dict[str, object]

    try:
        parsed = cast(dict[str, object], json.loads(cleaned_text))
    except Exception:
        try:
            candidate = _extract_first_json_object(cleaned_text)
            if not candidate:
                raise ValueError("No JSON object found")
            parsed = cast(dict[str, object], json.loads(candidate))
        except Exception:
            parsed = {
                "skill_gaps": [],
                "job_match_score": None,
                "improvement_suggestions": [],
                "stack": [],
            }

    parsed["raw_ai_output"] = text
    return parsed


def call_llm(
    resume_text: str,
    job_requirement: str,
    retrieved_context: list[dict[str, object]],
) -> dict[str, object]:
    prompt = _make_prompt(resume_text, job_requirement, retrieved_context)

    if not GROQ_API_KEY:
        return _parse_response(_fallback_response())

    client = openai.OpenAI(
        api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON API. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1200,
        )
    except Exception:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON without extra text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1200,
        )

    content = response.choices[0].message.content or "{}"
    return _parse_response(content)
