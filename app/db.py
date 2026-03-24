import sqlite3
import json
from pathlib import Path
from typing import Any

DB_FILE = Path(__file__).resolve().parent.parent / "data" / "analysis.db"
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

def get_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resume_text TEXT NOT NULL,
                skill_gaps TEXT,
                job_match_score REAL,
                improvement_suggestions TEXT,
                stack TEXT,
                raw_ai_output TEXT
            )
            """
        )
        conn.commit()


def _to_json_text(value: Any) -> str:
    if value is None:
        return "[]"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _from_json_text(value: Any) -> Any:
    if value is None:
        return []
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return [value] if value else []


def save_analysis(filename: str, resume_text: str, payload: dict[str, Any]) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO analysis (filename, created_at, resume_text, skill_gaps, job_match_score, improvement_suggestions, stack, raw_ai_output)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                resume_text,
                _to_json_text(payload.get("skill_gaps", [])),
                payload.get("job_match_score"),
                _to_json_text(payload.get("improvement_suggestions", [])),
                _to_json_text(payload.get("stack", [])),
                payload.get("raw_ai_output", ""),
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_all_analyses() -> list[dict[str, Any]]:
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM analysis ORDER BY created_at DESC")
        rows = cur.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["skill_gaps"] = _from_json_text(item.get("skill_gaps"))
            item["improvement_suggestions"] = _from_json_text(item.get("improvement_suggestions"))
            item["stack"] = _from_json_text(item.get("stack"))
            results.append(item)
        return results
