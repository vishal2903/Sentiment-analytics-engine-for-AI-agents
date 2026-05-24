from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import insights, ingest, analyze


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Agnost Insight Engine",
    description="Sentiment analytics for conversational AI agents. Drop in conversation logs, get PM-ready insights.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(insights.router)
app.include_router(ingest.router)
app.include_router(analyze.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
