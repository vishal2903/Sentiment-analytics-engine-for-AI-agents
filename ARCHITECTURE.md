# Architecture: Sentiment Analytics Engine

## Core Principle

Heavy work runs offline: embedding, clustering, and LLM insight generation. The API serves pre-computed insights from Supabase. That keeps `GET /insights` fast, cheap, and predictable.

---

## Pipeline

```text
DATA SOURCES
  Agnost OTel-style SDK logs
  Bitext demo dataset
  Any agent SDK normalized to {session_id, messages[], metadata}
        |
        | POST /ingest
        v
INGESTION + EMBEDDING
  Pydantic validates schema
  text-embedding-3-small creates 1536-dim vectors
  EmbeddingProvider lets prod swap to Qwen3 or Gemini
        |
        | batch write
        v
SUPABASE POSTGRES + PGVECTOR
  conversations: raw content, metadata, embedding, cluster_id
  clusters: centroid, size, pct, trend
  insights: PM-ready labels, severity, action, quotes
        |
        | POST /analyze starts background job
        v
CLUSTERING JOB
  ProcessPoolExecutor keeps CPU work out of FastAPI event loop
  UMAP: 1536 dims -> 5 dims, cosine neighborhood preservation
  K-Means: K=27 demo, silhouette sweep in production
  ARI: validates against Bitext ground-truth intents
        |
        v
INSIGHT JOB
  Per cluster: 8 representative conversations + stats
  GPT-4.1 mini returns structured JSON
  Writes topic_label, pm_insight, severity, action, sample_quote
        |
        v
FASTAPI SERVE LAYER
  GET /insights and GET /insights/{id}
  Reads Supabase only, zero LLM at request time
```

---

## Data Model

```sql
conversations (
  id          uuid PRIMARY KEY,
  session_id  text,
  content     text,
  embedding   vector(1536),
  cluster_id  int,
  source      text,          -- bitext | agnost_otel | api
  metadata    jsonb,
  created_at  timestamptz
)

clusters (
  id          int PRIMARY KEY,
  centroid    vector(1536),
  size        int,
  pct         float,
  trend       float,
  updated_at  timestamptz
)

insights (
  id             uuid PRIMARY KEY,
  cluster_id     int REFERENCES clusters(id),
  topic_label    text,
  pm_insight     text,
  severity       text,       -- HIGH | MEDIUM | LOW
  action         text,
  sample_quote   text,
  week_over_week text,
  created_at     timestamptz
)
```

Why Postgres + pgvector:
- Embeddings and PM analytics live together.
- Percentages/trends are SQL aggregations, not vector-search-only problems.
- Supabase keeps weekend setup simple while preserving a credible production path.

---

## API Reference

| Method | Endpoint | Purpose | Behavior |
|---|---|---|---|
| `POST` | `/ingest` | Accept conversation JSON | Validates, embeds, stores |
| `POST` | `/analyze` | Trigger clustering run | Returns `202` + `job_id`; CPU work runs in process pool |
| `GET` | `/analyze/{job_id}` | Poll job status | `pending -> running -> complete -> failed` |
| `GET` | `/insights` | List PM insights | Sorted by volume/severity, served from DB |
| `GET` | `/insights/{id}` | Drill into one topic | Includes sample quotes and trend |

Example ingest body:

```json
{
  "session_id": "uuid",
  "messages": [
    {"role": "user", "content": "I can't find where to return my item"},
    {"role": "assistant", "content": "I can help you initiate a return..."}
  ],
  "metadata": {"category": "ORDER"}
}
```

Example insight response:

```json
{
  "id": "uuid",
  "topic_label": "Refund policy confusion",
  "percentage": 22.1,
  "severity": "HIGH",
  "week_over_week": "+18%",
  "pm_insight": "1 in 5 users is blocked on refunds; volume is up 18% this week",
  "action": "Audit return policy copy on checkout page",
  "sample_quote": "I cant find where to return my item"
}
```

---

## Component Choices

| Component | Demo | Production path | Reason |
|---|---|---|---|
| Ingestion | Pydantic schema | Same | One contract for Bitext, Agnost traces, and any agent SDK |
| Embedding | OpenAI `text-embedding-3-small` | Qwen3 or Gemini behind provider interface | Simple demo, low-friction swap later |
| Storage | Supabase pgvector | Same engine, larger instance or tenant split | SQL analytics + vector storage in one system |
| Reduction | UMAP, 5 dims, cosine | Same | Makes semantic distance usable before clustering |
| Clustering | K-Means K=27 | BERTopic + silhouette sweep | Demo is ARI-validatable; prod needs dynamic topics |
| Validation | ARI vs Bitext labels | Silhouette, cohesion, drift | Proves demo correctness, monitors prod health |
| LLM | GPT-4.1 mini | GPT-4.1 mini, Claude, or Gemini 3.5 Flash after benchmark | Structured PM synthesis, not the clustering engine |
| Jobs | ProcessPoolExecutor | ARQ + Redis, then Celery | Keep CPU work out of event loop; scale workers only when needed |
| API | FastAPI REST | REST first, GraphQL only for richer dashboards | Easy reviewer testing and clean ML/Python fit |

---

## Latency + Scaling

Hot path:

```text
GET /insights
  FastAPI routing:           ~1ms
  SQL over precomputed rows: <5ms
  Supabase network RTT:      ~50-100ms
  Total:                     ~100-110ms
```

Background path:

```text
POST /analyze
  API response:              <1s with job_id
  UMAP + K-Means:            ~60-90s on 27k rows
  LLM insight generation:    27 GPT-4.1 mini calls
```

Scale thresholds:

| Trigger | What breaks | Fix |
|---|---|---|
| >500k conversations | pgvector scans slow down | Add `ivfflat` index, tune probes |
| >5 customers with nightly jobs | Local process pool queues jobs behind each other | ARQ + Redis queues |
| >20 customers | Shared DB writes and tenant isolation get messy | Per-customer schema or DB split |
| >100 customers | Nightly clustering needs distributed workers | Celery + worker autoscaling |
| >1M conversations/customer | K-Means memory pressure | MiniBatch K-Means or BERTopic hierarchy |

---

## Month-Plan Architecture

```text
Agnost agent logs
      |
      v
FastAPI /ingest
      |
      v
Supabase pgvector, tenant-isolated
      |
      v
ARQ + Redis nightly jobs
  - BERTopic
  - silhouette sweep
  - drift detection
  - LLM insight synthesis
      |
      v
FastAPI REST, then GraphQL only if dashboard query shapes demand it
```
