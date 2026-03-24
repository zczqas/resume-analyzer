# AI Resume Analyzer

Simple backend CV analyzer for backend developer role.

## Features
- Upload PDF resume
- Extract text (PyMuPDF)
- Call Groq via OpenAI-compatible API with prompt
- Return JSON:
  - skill_gaps
  - job_match_score
  - improvement_suggestions
  - stack
- Store analysis in SQLite

## Setup

1. Install dependencies with uv:
   - `uv sync`

2. Configure env:
   - copy `.env.example` to `.env`
   - set `GROQ_API_KEY`
   - optional: set `GROQ_MODEL`

3. Run server:
   - `uv run uvicorn app.main:app --reload`

## API

- `POST /analyze` form-data: `file` (PDF)
- `GET /analyses` list all stored results

## Notes
- If `GROQ_API_KEY` is not set, a built-in heuristic fallback is used.
