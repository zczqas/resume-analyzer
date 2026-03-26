from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app import db
from app.helpers.analysis import normalize_analysis_payload
from app.helpers.pdf import extract_pdf_text
from app.helpers.retrieval import get_job_store, resolve_collection_name
from app.llms.groq_client import call_llm
from app.routers.job_requirements import router as job_requirements_router

app = FastAPI(title="Resume Analyzer", version="0.1")
app.include_router(job_requirements_router)

db.init_db()


@app.post("/analyze")
async def analyze_resume(
    file: Annotated[UploadFile, File(...)],
    embedding_id: Annotated[str, Form(...)],
    job_requirement: Annotated[str, Form()] = "",
    top_k: Annotated[int, Form()] = 3,
) -> JSONResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not embedding_id.strip():
        raise HTTPException(status_code=400, detail="embedding_id cannot be empty")

    file_bytes = await file.read()
    try:
        resume_text = extract_pdf_text(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    resolved_collection = resolve_collection_name(embedding_id)
    if not resolved_collection:
        raise HTTPException(status_code=404, detail="embedding_id not found")
    retrieval_query = job_requirement.strip() or resume_text
    store = get_job_store(collection_name=resolved_collection)
    retrieved_docs = store.search(retrieval_query, top_k=max(1, min(top_k, 10)))

    ai_result = call_llm(
        resume_text=resume_text,
        job_requirement=retrieval_query,
        retrieved_context=retrieved_docs,
    )
    payload = normalize_analysis_payload(ai_result)

    filename = file.filename or "uploaded_resume.pdf"
    persist_payload: dict[str, object] = {
        **payload,
        "job_requirement": retrieval_query,
        "embedding_name": embedding_id,
        "retrieved_context": retrieved_docs,
    }

    analysis_id = db.save_analysis(filename, resume_text, persist_payload)

    return JSONResponse(
        status_code=201,
        content={
            "id": analysis_id,
            **payload,
            "job_requirement": retrieval_query,
            "embedding_id": embedding_id,
            "retrieved_context": retrieved_docs,
        },
    )


@app.get("/analyses")
async def list_analyses() -> list[dict[str, object]]:
    return db.get_all_analyses()
