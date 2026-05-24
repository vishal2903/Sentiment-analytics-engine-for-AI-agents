# Agnost Insight Engine

> Turn agent conversation logs into PM-ready insights.
> *"22% of users are stuck on refunds, up 18% this week."*

Built for the Agnost Track A assignment by Vishal Sharma.

---

## What This Does

AI agents generate thousands of conversations. Most of the signal inside them is invisible: users phrase feature requests as complaints, hit the same friction points repeatedly, and drop off without ever saying what broke. This engine ingests raw agent conversations, clusters them by semantic similarity, and surfaces structured insights a PM can act on: what topic is spiking, how severe, what to do about it.

---

## How It Works

```
Raw conversations
      |  POST /ingest or scripts/01_ingest.py
      v
Supabase (conversations + pgvector)    +    .cache/embeddings.npy
      |  scripts/02_cluster.py                      |
      |  (reads .cache/, never DB) ◄────────────────┘
      v
UMAP (1536-dim to 5-dim, cosine) + K-Means (K=27, ARI=0.67)
      |  scripts/03_label.py
      v
GPT-4.1-mini (27 calls) generates topic labels + PM insights
      |  stored in Supabase insights table
      v
GET /insights
```

Heavy computation runs offline. The API serves pre-computed DB rows: zero LLM calls at request time, ~110ms response.

---

## Algorithm & Complexity

| Step | Algorithm | Complexity |
|---|---|---|
| Batch embed | OpenAI API, 100/call | O(n/batch) |
| Dim reduction | UMAP cosine | O(n log n) |
| Clustering | K-Means | O(n · k · i) |
| ARI validation | Adjusted Rand Index | O(n) |
| Centroid sampling | L2 distance to mean (production path) | O(n_cluster · d) |
| ANN vector search | ivfflat index | O(log n) |
| GET /insights | SQL SELECT 27 rows | O(1) |

CPU-bound clustering runs in `ProcessPoolExecutor`: separate OS process, event loop stays free. `POST /analyze` returns 202 in under 1 second.

---

## Setup

**1. Clone + install**
```bash
git clone https://github.com/vishal2903/Sentiment-analytics-engine-for-AI-agents.git
cd Sentiment-analytics-engine-for-AI-agents
pip install -r requirements.txt
```

**2. Configure**
```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

**3. Set up Supabase**
- Create a free project at [supabase.com](https://supabase.com)
- Copy project URL and anon key into `.env`
- Run `schema.sql` in the SQL editor

**4. Run the pipeline** (one time, ~10 min)
```bash
python scripts/run_pipeline.py
```
Loads 26,872 conversations, embeds in batches of 100, clusters into 27 topics, generates PM insights. Progress bars throughout.

**5. Start the API**
```bash
uvicorn app.main:app
```
Swagger UI: http://localhost:8000/docs

> Do not use `--reload` when testing `/analyze`. WatchFiles kills background jobs mid-UMAP.

---

## API Reference

| Method | Endpoint | What it does |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/insights` | All 27 insights sorted by volume. Filter: `?severity=HIGH\|MEDIUM\|LOW` |
| `GET` | `/insights/{id}` | Full detail: pm_insight, action, sample_quote, trend |
| `POST` | `/ingest` | Add a conversation. Embeds + stores. Returns `{id, status}` |
| `POST` | `/analyze` | Trigger re-clustering. Returns `{job_id, status: pending}` in under 1s |
| `GET` | `/analyze/{job_id}` | Poll job status: `pending -> running -> complete` |

---

## Sample Requests

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
  {"id": 3, "topic_label": "Refund policy confusion", "percentage": 22.1, "severity": "HIGH"},
  {"id": 7, "topic_label": "Order cancellation", "percentage": 15.4, "severity": "HIGH"}
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
  "pm_insight": "1 in 5 users is blocked on refunds, volume up 18% this week",
  "severity": "HIGH",
  "action": "Audit return policy copy on checkout page",
  "sample_quote": "I cant find where to return my item",
  "volume": 5931,
  "percentage": 22.1
}
```

**Trigger re-clustering**
```bash
curl -X POST http://localhost:8000/analyze
```
```json
{"job_id": "abc-123", "status": "pending"}
```

**Poll job**
```bash
curl http://localhost:8000/analyze/abc-123
```
```json
{"job_id": "abc-123", "status": "complete", "completed_at": "2026-05-24T18:30:00Z", "error": null}
```

---

## Tests

```bash
pytest tests/ -v
```
11 tests, all passing.

---

## Dataset

Bitext Customer Support (HuggingFace): 26,872 rows, 27 labeled intents, updated October 2025. Ground-truth labels enable ARI validation (ARI=0.67, target >0.3), which proves clusters reflect real user intent rather than noise.

---

## Sample Output

Real pipeline output — 27 PM insights from 26,872 conversations — is in [`sample_output/insights.json`](sample_output/insights.json). No setup needed to see what the engine produces.

To hit the live API instead:
1. Get `.env` credentials (SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY)
2. `uvicorn app.main:app`
3. `curl http://localhost:8000/insights` or open http://localhost:8000/docs

---

## Docs

- [`REASONING.md`](REASONING.md): every architecture decision with rejected alternatives
- [`ARCHITECTURE.md`](ARCHITECTURE.md): data model, API reference, latency profile

---

## What I'd Build With a Month

- Centroid-nearest sampling instead of random n=12: better GPT cluster representation, vectors are already in the offline cache
- GPU UMAP via RAPIDS cuML: eliminates spectral init fallback, 10x faster at 500k+ rows
- ARQ + Redis instead of ProcessPoolExecutor: persistent job queue, survives server restarts, retries on crash
- Silhouette sweep to auto-tune K instead of fixed K=27
- Week-over-week trend computation (currently null in all rows)
- BERTopic: dynamic topic discovery, drift detection, no hardcoded K
- Split LLM jobs: stable labels weekly, insight synthesis nightly
- Model benchmark: GPT-4.1-mini vs Claude vs Gemini 3.5 Flash on label consistency and cost
- Per-customer cluster isolation
- Trend anomaly alerts for PMs
- PII redaction before embedding
