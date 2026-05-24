# Architecture: Sentiment Analytics Engine

Heavy work runs offline. The API serves pre-computed rows. `GET /insights` never calls an LLM.

---

## System Diagram

```
  DATA IN
  Agent SDK · Bitext · Any {session_id, messages[]}
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  INGEST + EMBED  (01_ingest.py)                     │
  │  Pydantic → text-embedding-3-small → 1536-dim       │
  └──────────────┬──────────────────────────────────────┘
                 │                          │
    psycopg2     │                          │ numpy save
    Txn Pool     │                          │
    500-row batch│                          ▼
                 ▼                ┌──────────────────────┐
  ┌──────────────────────┐        │   OFFLINE CACHE      │
  │  SUPABASE            │        │   .cache/            │
  │  Postgres + pgvector │        │   embeddings.npy     │
  │                      │        │   158MB              │
  │  conversations       │        │   ML jobs read here  │
  │  clusters            │        │   never from DB      │
  │  insights            │        └──────────┬───────────┘
  │                      │                   │ read
  │  REST  → API layer   │                   ▼
  │  psycopg2 → pipeline │        ┌──────────────────────┐
  │                      │        │  CLUSTERING JOB      │
  │  ◄── cluster UPDATE ─┼────────│  ProcessPoolExecutor │
  │                      │        │  UMAP 1536 → 5 dims  │
  │                      │        │  K-Means K=27        │
  │                      │        │  ARI = 0.67          │
  │                      │        └──────────┬───────────┘
  │                      │                   │ sample n=12
  │  ◄── REST write ─────┼────────┐          ▼
  │  insights table      │        │ ┌──────────────────────┐
  └──────────┬───────────┘        └─│  LLM LABELING        │
             │                      │  27 × GPT-4.1-mini   │
             │ REST reads           │  topic · severity    │
             ▼                      │  action · quote      │
  ┌──────────────────────────────┐  └──────────────────────┘
  │  FASTAPI                     │
  │  POST /ingest   → embed+write│
  │  POST /analyze  → 202 <1s    │
  │  GET  /analyze/{id} → status │
  │  GET  /insights → ~110ms     │
  │  GET  /insights/{id}         │
  │  GET  /health                │
  └──────────────────────────────┘
```

---

## Data Access Layers

```
  ┌──────────────────────────────────────────────────────────┐
  │                    SUPABASE POSTGRES                     │
  └───────────────────────────┬──────────────────────────────┘
                              │
          ┌───────────────────┴────────────────────┐
          ▼                                        ▼
  ┌──────────────────┐                  ┌──────────────────────┐
  │  REST            │                  │  psycopg2            │
  │  API layer only  │                  │  Txn Pooler :6543    │
  │                  │                  │  Pipeline layer only │
  │  single-row ops  │                  │                      │
  │  /ingest write   │                  │  500-row batches     │
  │  /insights read  │                  │  cluster_id UPDATE   │
  └──────────────────┘                  └──────────────────────┘

  REST serializes vector(1536) as JSON (~15KB/row, 1000-row cap).
  psycopg2 writes binary: 27k rows, no cap, no overhead.
  Same DB. Two access paths matched to workload.
```

---

## Data Model

```sql
conversations (
  id          uuid PRIMARY KEY,
  session_id  text UNIQUE,
  content     text,
  embedding   vector(1536),
  cluster_id  int,
  source      text,        -- bitext | agnost_otel | api
  metadata    jsonb,
  created_at  timestamptz
)

clusters (
  id        int PRIMARY KEY,
  centroid  vector(1536),
  size      int,
  pct       float,
  trend     float
)

insights (
  id             uuid PRIMARY KEY,
  cluster_id     int REFERENCES clusters(id),
  topic_label    text,
  pm_insight     text,
  severity       text,     -- HIGH | MEDIUM | LOW
  action         text,
  sample_quote   text,
  week_over_week text
)
```

---

## API Reference

| Method | Endpoint | Behavior |
|---|---|---|
| `GET` | `/health` | `{status: ok}` |
| `GET` | `/insights` | 27 insights sorted by volume. Filter: `?severity=HIGH\|MEDIUM\|LOW` |
| `GET` | `/insights/{id}` | pm_insight, action, sample_quote, trend |
| `POST` | `/ingest` | Validate → embed → store. Returns `{id, status}` |
| `POST` | `/analyze` | Returns `{job_id, status: pending}` in <1s. Clustering runs in background. |
| `GET` | `/analyze/{job_id}` | `pending → running → complete → failed` |

---

## Latency

| Path | Breakdown | Total |
|---|---|---|
| `GET /insights` | routing ~1ms + SQL <5ms + network ~100ms | ~110ms |
| `POST /analyze` | API response only | <1s |
| Clustering job | UMAP + K-Means on 27k rows | ~60-90s |
| LLM labeling | 27 sequential GPT calls | ~68s |

---

## Scale Thresholds

| Trigger | Breaks | Fix |
|---|---|---|
| >500k rows | pgvector full scan slows | `ivfflat` index |
| >5 customers, nightly | Process pool queues jobs | ARQ + Redis |
| >20 customers | Tenant isolation gaps | Per-customer schema |
| >100 customers | Clustering needs distributed workers | Cloud Run Jobs |
| >1M rows/customer | K-Means memory ~3GB peak | MiniBatch K-Means |

---

## Month-Plan

```
  Agnost agent logs → FastAPI /ingest → Supabase (tenant-isolated)
        │
        ▼
  ARQ + Redis nightly jobs
  ├─ BERTopic (auto K, drift detection)
  ├─ Silhouette sweep K=10-50
  ├─ Centroid-nearest sampling n=20
  └─ Async LLM calls (~8s vs 68s sequential)
        │
        ▼
  FastAPI REST (GraphQL only if dashboard query shapes demand it)
```
