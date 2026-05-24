from typing import Literal
from pydantic import BaseModel


# --- Shared primitives ---

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# --- POST /ingest ---

class ConversationIn(BaseModel):
    session_id: str
    messages: list[Message]
    metadata: dict = {}


class IngestResponse(BaseModel):
    id: str
    status: str = "stored"


# --- Cluster insights (LLM output + API response — one schema, two jobs) ---

class ClusterInsight(BaseModel):
    topic_label: str
    pm_insight: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    action: str
    sample_quote: str


class InsightListItem(BaseModel):
    id: int
    topic_label: str
    percentage: float
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    week_over_week: float | None = None


class InsightDetail(InsightListItem):
    pm_insight: str
    action: str
    sample_quote: str
    volume: int


# --- POST /analyze + GET /analyze/{job_id} ---

class AnalyzeResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
