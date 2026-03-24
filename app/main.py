from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app import db
from app.helpers.analysis import normalize_analysis_payload
from app.helpers.pdf import extract_pdf_text
from app.llms.groq_client import call_llm

app = FastAPI(title="Resume Analyzer", version="0.1")

db.init_db()


@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    try:
        resume_text = extract_pdf_text(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    ai_result = call_llm(resume_text)
    payload = normalize_analysis_payload(ai_result)

    analysis_id = db.save_analysis(file.filename, resume_text, payload)

    return JSONResponse(
        status_code=201,
        content={"id": analysis_id, **payload},
    )


@app.get("/analyses")
async def list_analyses() -> list[dict[str, Any]]:
    return db.get_all_analyses()
