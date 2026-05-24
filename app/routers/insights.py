from fastapi import APIRouter, HTTPException, Query
from app.database import get_client
from app.models.schemas import InsightListItem, InsightDetail

router = APIRouter(prefix="/insights", tags=["insights"])


@router.get("", response_model=list[InsightListItem])
async def list_insights(severity: str | None = Query(default=None)):
    db = get_client()
    query = db.table("clusters").select("id,topic_label,percentage,severity,week_over_week").order("percentage", desc=True)
    if severity:
        query = query.eq("severity", severity.upper())
    result = query.execute()
    return result.data


@router.get("/{cluster_id}", response_model=InsightDetail)
async def get_insight(cluster_id: int):
    db = get_client()
    result = db.table("clusters").select("*").eq("id", cluster_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return result.data[0]
