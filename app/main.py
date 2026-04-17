"""
app/main.py
───────────
FastAPI application — medical RAG chatbot backend.

Endpoints:
  POST /chat        — ask a question
  GET  /health      — system health check
  GET  /sources     — peek at indexed chunks
  POST /ingest      — upload a PDF via the API
  DELETE /reset     — clear the vector store
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import (
    ChatRequest,
    ChatResponse,
    SourceDocument,
    HealthResponse,
    IngestResponse,
)
from app.graph import run_rag
from app.retriever import collection_count, reset_collection


_ingest_jobs: dict[str, dict[str, Any]] = {}
_ingest_jobs_lock = threading.Lock()


def _set_ingest_job(job_id: str, **updates: Any) -> None:
    with _ingest_jobs_lock:
        if job_id not in _ingest_jobs:
            _ingest_jobs[job_id] = {}
        _ingest_jobs[job_id].update(updates)


def _compute_progress_percent(stage: str, current: int, total: int) -> int:
    safe_total = max(total, 1)
    ratio = max(0.0, min(1.0, current / safe_total))
    if stage == "load":
        return int(ratio * 10)
    if stage == "chunk":
        return 10 + int(ratio * 20)
    if stage == "embed":
        return 30 + int(ratio * 69)
    if stage == "complete":
        return 100
    return int(ratio * 100)


def _run_ingest_job(job_id: str, tmp_path: Path, safe_name: str) -> None:
    from scripts.ingest import ingest_pdf_file

    def _progress(stage: str, current: int, total: int, message: str) -> None:
        _set_ingest_job(
            job_id,
            stage=stage,
            status="running",
            current=current,
            total=total,
            progress_pct=_compute_progress_percent(stage, current, total),
            message=message,
        )

    try:
        _set_ingest_job(
            job_id,
            status="running",
            progress_pct=0,
            stage="queued",
            message="Queued for processing",
            filename=safe_name,
        )
        chunks_added = ingest_pdf_file(str(tmp_path), progress_callback=_progress)
        _set_ingest_job(
            job_id,
            status="completed",
            progress_pct=100,
            stage="complete",
            message="Ingestion completed",
            chunks_added=chunks_added,
            total_chunks=collection_count(),
        )
    except Exception as exc:
        _set_ingest_job(
            job_id,
            status="failed",
            stage="failed",
            message="Ingestion failed",
            error=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "medical book chatbot powered by LangGraph + Ollama + ChromaDB."
        
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["UI"])
async def ui_home():
    """Serve the user-facing web UI."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns service status, model info, and how many chunks are indexed.
    Use this to confirm everything is working before chatting.
    """
    try:
        total = collection_count()
        return HealthResponse(
            status="ok",
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            collection=settings.chroma_collection_name,
            total_chunks=total,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Ask a medical question. The system retrieves relevant passages from
    your book, grades their relevance, generates an answer, and checks
    for hallucinations — all using local, free models.

    **Tip:** Make sure you've run `scripts/ingest.py` first!
    """
    if collection_count() == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No documents indexed yet. "
                "Run `python scripts/ingest.py --pdf data/your_book.pdf` first."
            ),
        )

    try:
        result = run_rag(request.question, request.language)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM error: {str(exc)}. Is Ollama running? (ollama serve)",
        )

    sources = [
        SourceDocument(
            content=s["content"],
            page=s.get("page"),
            source=s.get("source"),
        )
        for s in result["sources"]
    ]

    return ChatResponse(
        answer=result["answer"],
        language=request.language,
        sources=sources,
        grounded=result["grounded"],
        session_id=request.session_id or str(uuid.uuid4()),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF directly via the API (alternative to the CLI script).
    The file is chunked, embedded, and stored in ChromaDB.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    # Save upload to temp location
    safe_name = Path(file.filename).name
    tmp_path = Path("temp") / safe_name
    tmp_path.parent.mkdir(exist_ok=True)
    content = await file.read()
    tmp_path.write_bytes(content)

    # Lazy import to keep startup fast
    from scripts.ingest import ingest_pdf_file

    try:
        chunks_added = ingest_pdf_file(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        message=f"Successfully ingested '{safe_name}'",
        chunks_added=chunks_added,
        total_chunks=collection_count(),
    )


@app.post("/ingest/start", tags=["Ingestion"])
async def ingest_pdf_start(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Start asynchronous PDF ingestion and return a job ID for progress polling."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    safe_name = Path(file.filename).name
    job_id = str(uuid.uuid4())
    tmp_path = Path("temp") / f"{job_id}_{safe_name}"
    tmp_path.parent.mkdir(exist_ok=True)

    content = await file.read()
    tmp_path.write_bytes(content)

    _set_ingest_job(
        job_id,
        status="queued",
        progress_pct=0,
        stage="queued",
        current=0,
        total=1,
        message="Upload received",
        filename=safe_name,
    )

    background_tasks.add_task(_run_ingest_job, job_id, tmp_path, safe_name)

    return {"job_id": job_id, "status": "queued", "message": "Ingestion job started"}


@app.get("/ingest/status/{job_id}", tags=["Ingestion"])
async def ingest_pdf_status(job_id: str):
    """Get ingestion job status and progress percentage."""
    with _ingest_jobs_lock:
        job = _ingest_jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    return job


@app.get("/sources", tags=["System"])
async def get_sources(limit: int = 10):
    """
    Peek at the first N chunks stored in ChromaDB.
    Useful for verifying that ingestion worked correctly.
    """
    from app.retriever import get_vectorstore

    try:
        vs = get_vectorstore()
        results = vs._collection.get(limit=limit)
        chunks = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            chunks.append({
                "id": results["ids"][i],
                "content_preview": doc[:200],
                "metadata": meta,
            })
        return {"total_chunks": collection_count(), "sample": chunks}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/reset", tags=["System"])
async def reset():
    """
    ⚠️  Delete all indexed documents from ChromaDB.
    You will need to re-run ingestion after this.
    """
    try:
        reset_collection()
        return JSONResponse({"message": "Collection cleared successfully."})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
