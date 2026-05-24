from fastapi import APIRouter, HTTPException
from app.database import get_client
from app.models.schemas import ConversationIn, IngestResponse
from app.services.embedder import get_embedder

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, status_code=201)
async def ingest_conversation(payload: ConversationIn):
    user_messages = [m.content for m in payload.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=422, detail="At least one user message required")

    user_text = " ".join(user_messages)
    embedder = get_embedder()
    vector = embedder.embed_batch([user_text])[0]

    assistant_messages = [m.content for m in payload.messages if m.role == "assistant"]
    row = {
        "session_id": payload.session_id,
        "user_message": user_text,
        "assistant_message": assistant_messages[0] if assistant_messages else None,
        "metadata": payload.metadata,
        "embedding": vector,
        "source": "api",
    }

    db = get_client()
    result = db.table("conversations").insert(row).execute()
    return IngestResponse(id=result.data[0]["id"])
