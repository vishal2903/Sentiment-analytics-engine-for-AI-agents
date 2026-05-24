# Agnost Insight Engine

> Sentiment analytics for conversational AI agents.
> Drop in conversation logs → get PM-ready insights.
> *"22% of users are blocked on refunds — up 18% this week."*

Built for the Agnost Track A assignment by Vishal Sharma.

---

## What This Does

AI agents generate conversations. Most signal inside those conversations is invisible — users phrase feature requests as complaints, surface bugs without naming them, and drop off at the same friction points repeatedly. This engine ingests raw agent conversations, clusters them by semantic similarity, and surfaces structured PM-readable insights: topic, severity, trend, and recommended action.

---

## Pipeline

```
Raw conversations
      ↓  POST /ingest or scripts/01_ingest.py
Supabase (conversations table + pgvector)
      ↓  scripts/02_cluster.py
UMAP (1536-dim → 5-dim, cosine) → K-Means (K=27 clusters)
ARI validation against ground-truth labels
      ↓  scripts/03_label.py
GPT-4.1-mini (27 calls) → topic labels + PM insights
      ↓  stored in Supabase clusters table
GET /insights
```

**Key design rule:** all heavy computation runs offline. The API serves from pre-computed DB rows — zero LLM calls at request time. `GET /insights` returns in ~110ms.

---

## Algorithm & Time Complexity

| Step | Algorithm | Complexity |
|---|---|---|
| Batch embed | OpenAI API, 100/call | O(n/batch) |
| Dim reduction | UMAP cosine | O(n log n) |
| Clustering | K-Means | O(n · k · i) |
| ARI validation | Adjusted Rand Index | O(n) |
| Centroid sampling | L2 distance to mean | O(n_cluster · d) |
| ANN vector search | ivfflat index | O(log n) |
| GET /insights | SQL SELECT 27 rows | O(1) |

CPU-bound clustering runs in `ProcessPoolExecutor` — separate OS process, event loop stays free. `POST /analyze` returns 202 in <1 second.

---

## Setup (5 steps)

**1. Clone + install**
```bash
git clone https://github.com/vishal2903/Sentiment-analytics-engine-for-AI-agents.git
cd Sentiment-analytics-engine-for-AI-agents
pip install -r requirements.txt
```

**2. Set env vars**
```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

**3. Set up Supabase database**
- Create a free project at [supabase.com](https://supabase.com)
- Copy your project URL and anon key into `.env`
- In the SQL editor, run the contents of `schema.sql`

**4. Run pipeline (one time, ~10 minutes)**
```bash
python scripts/run_pipeline.py
```
Loads 26,872 conversations, embeds in batches, clusters, generates 27 PM insights.
Progress bars show status throughout.

**5. Start API**
```bash
uvicorn app.main:app
```
Swagger UI: http://localhost:8000/docs

> **Note:** Do not use `--reload` when testing `/analyze` — WatchFiles kills background jobs mid-UMAP.

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/insights` | List all 27 insights sorted by volume (optional `?severity=HIGH\|MEDIUM\|LOW`) |
| `GET` | `/insights/{id}` | Single insight — full detail with pm_insight, action, sample_quote |
| `POST` | `/ingest` | Add a conversation. Embeds + stores. Returns `{id, status}` |
| `POST` | `/analyze` | Trigger re-clustering. Returns `{job_id, status: pending}` in <1s. Non-blocking. |
| `GET` | `/analyze/{job_id}` | Poll job status: `pending → running → complete` |

---

## Sample Requests & Responses

**Ingest a conversation**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-001",
    "messages": [
      {"role": "user", "content": "I cant find where to cancel my order"},
      {"role": "assistant", "content": "I can help with that"}
    ]
  }'
```
```json
{"id": "f3a1c2d4-...", "status": "stored"}
```

**List insights**
```bash
curl http://localhost:8000/insights
```
```json
[
  {"id": 3, "topic_label": "Refund policy confusion", "percentage": 22.1, "severity": "HIGH", "week_over_week": 18.3},
  {"id": 7, "topic_label": "Order cancellation", "percentage": 15.4, "severity": "HIGH", "week_over_week": -2.1}
]
```

**Drill into one insight**
```bash
curl http://localhost:8000/insights/3
```
```json
{
  "id": 3,
  "topic_label": "Refund policy confusion",
  "pm_insight": "1 in 5 users is blocked on refunds — volume up 18% this week",
  "severity": "HIGH",
  "action": "Audit return policy copy on checkout page",
  "sample_quote": "I cant find where to return my item",
  "volume": 5931,
  "percentage": 22.1,
  "week_over_week": 18.3
}
```

**Trigger re-clustering**
```bash
curl -X POST http://localhost:8000/analyze
```
```json
{"job_id": "abc-123", "status": "pending"}
```
Returns in <1 second. Clustering runs in background.

**Poll job status**
```bash
curl http://localhost:8000/analyze/abc-123
```
```json
{"job_id": "abc-123", "status": "complete", "completed_at": "2026-05-24T18:30:00Z", "error": null}
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Dataset

Bitext Customer Support (HuggingFace) — 26,872 rows, 27 labeled intents, updated October 2025.
Ground-truth labels enable ARI validation (proves clusters track real intent, not noise).

---

## Architecture & Decisions

See [`REASONING.md`](REASONING.md) for every architectural decision with rejected alternatives.
See [`ARCHITECTURE.md`](ARCHITECTURE.md) for HLD diagrams, data model, and latency profile.

---

## What I'd Do With a Month

- Centroid-nearest sampling (vs random n=12) using offline cache — better GPT cluster representation with no extra DB cost
- GPU UMAP via RAPIDS cuML — eliminates spectral init fallback, 10× faster at 500k+ rows
- ARQ + Redis instead of ProcessPoolExecutor — persistent job queue, survives server restarts, retries on crash
- Silhouette score sweep to auto-tune K instead of fixed K=27
- Week-over-week trend computation (currently null in all rows)
- BERTopic with K-Means backend: dynamic topic discovery, drift detection, auto K
- Split LLM jobs: stable labels weekly, insight synthesis nightly
- Model benchmark harness: GPT-4.1-mini vs Claude vs Gemini 3.5 Flash
- Per-customer cluster isolation
- Trend anomaly detection: flag spikes above baseline for PM alerts
- Privacy hardening: PII redaction before embedding, retention controls
