import json
from openai import OpenAI
from app.config import settings
from app.models.schemas import ClusterInsight


_PROMPT = """You are a product analytics assistant. Given a cluster of user conversations from an AI agent,
return a JSON object with exactly these fields:
- topic_label: short name for the topic (5 words max)
- pm_insight: one sentence a PM can act on, include percentage and trend
- severity: HIGH, MEDIUM, or LOW
- action: one concrete recommended action
- sample_quote: most representative user quote (verbatim, under 15 words)

Cluster stats: {stats}

Sample conversations (closest to cluster center):
{conversations}

Return valid JSON only. No markdown, no explanation."""


def sample_cluster_conversations(
    cluster_id: int, all_rows: list[dict], n: int = 12
) -> list[str]:
    import random
    cluster_rows = [r for r in all_rows if r["cluster_id"] == cluster_id]
    if not cluster_rows:
        return []
    sampled = random.sample(cluster_rows, min(n, len(cluster_rows)))
    return [r["user_message"] for r in sampled]


def generate_cluster_insight(
    conversations: list[str],
    stats: dict,
) -> ClusterInsight:
    client = OpenAI(api_key=settings.openai_api_key)
    conv_text = "\n".join(f"- {c}" for c in conversations)
    prompt = _PROMPT.format(stats=json.dumps(stats), conversations=conv_text)

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = json.loads(response.choices[0].message.content)
            return ClusterInsight(**raw)
        except Exception:
            if attempt == 1:
                raise
