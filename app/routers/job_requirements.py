from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.helpers.retrieval import load_job_requirements_from_bytes

router = APIRouter(prefix="/job-requirements", tags=["job-requirements"])


@router.post("/upload")
async def upload_job_requirements(
    file: Annotated[UploadFile, File(...)],
    embedding_name: Annotated[str, Form()] = "job_requirements",
    embedding_id: Annotated[str, Form()] = "",
) -> JSONResponse:
    if not embedding_name.strip():
        raise HTTPException(status_code=400, detail="embedding_name cannot be empty")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Dataset file is required")

    file_bytes = await file.read()
    try:
        payload = load_job_requirements_from_bytes(
            file_bytes,
            embedding_name=embedding_name,
            embedding_id=embedding_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return JSONResponse(status_code=201, content=payload)
