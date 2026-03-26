# AI Resume Analyzer

Simple backend CV analyzer with retrieval-augmented job context.

## Features
- Upload PDF resume
- Extract text (PyMuPDF)
- Retrieve relevant job requirements via vector search (ChromaDB + sentence-transformers)
- Call Groq via OpenAI-compatible API with grounded prompt context
- Return JSON:
  - skill_gaps
  - job_match_score
  - improvement_suggestions
  - stack
- Include retrieved context used for evaluation
- Store analysis in SQLite

## Architecture

1. CV PDF upload
2. Text extraction
3. Embedding
4. Vector DB lookup (job requirements dataset)
5. Top-K relevant docs
6. LLM prompt + retrieved context
7. Final analysis

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

- `POST /analyze` form-data:
   - `file` (PDF)
   - `job_requirement` (string, optional; falls back to resume text for retrieval)
   - `embedding_id` (string, required; must be a valid id returned from upload)
   - `top_k` (int, optional, default `3`, max `10`)
- `POST /job-requirements/upload` form-data:
   - `file` (JSONL or JSON list of requirement objects)
   - `embedding_name` (string, optional, default `job_requirements`)
   - `embedding_id` (string, optional; if omitted, auto-generated UUID is returned)
- `GET /analyses` list all stored results

## Notes
- If `GROQ_API_KEY` is not set, a built-in heuristic fallback is used.
- Job requirement corpus is loaded from `data/job_requirements.jsonl` into ChromaDB at `data/chroma`.
- Uploaded embedding id to collection mapping is stored in `data/embedding_registry.json`.
