import asyncio
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4
from fastapi import APIRouter, HTTPException
from app.database import get_client
from app.models.schemas import AnalyzeResponse, JobStatus
from app.services.clusterer import run_clustering_pipeline

router = APIRouter(prefix="/analyze", tags=["analyze"])
_executor = ProcessPoolExecutor(max_workers=1)


@router.post("", response_model=AnalyzeResponse, status_code=202)
async def trigger_analysis():
    job_id = str(uuid4())
    db = get_client()
    db.table("pipeline_jobs").insert({"id": job_id, "status": "pending"}).execute()

    # CPU-bound work MUST NOT run in the async event loop.
    # ProcessPoolExecutor offloads to a separate OS process; event loop stays free.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, run_clustering_pipeline, job_id)

    return AnalyzeResponse(job_id=job_id, status="pending")


@router.get("/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    db = get_client()
    result = db.table("pipeline_jobs").select("*").eq("id", job_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")
    row = result.data[0]
    return JobStatus(
        job_id=row["id"],
        status=row["status"],
        created_at=row.get("created_at"),
        completed_at=row.get("completed_at"),
        error=row.get("error"),
    )
