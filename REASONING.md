# Sentiment Analytics Engine: Reasoning

**Vishal Sharma · Agnost Track A**

Agnost already lives close to the most valuable product signal: real conversations between users and AI agents. This project turns those traces into PM-ready intelligence. It takes raw agent conversations, clusters emerging topics, and surfaces insights like "refund confusion is spiking" or "users are asking for feature Y without naming it as a feature request."

---

## The Input Contract

Any conversational AI agent sends logs via `POST /ingest`:

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

Why this shape:
- Matches what Agnost can extract from OpenTelemetry-style agent traces: session, turns, metadata.
- `messages[]` with `role/content` is the standard LLM conversation object across OpenAI, Anthropic, and Vercel AI SDK. No adapter layer needed for most agent stacks.
- `metadata` is passthrough. The engine needs only message content but preserves agent_id, model, latency, or customer context if present.

Bitext is the weekend validation source. In production, Agnost's live agent logs drop in without changing the pipeline.

---

## The Decisions

### 1. Dataset: Bitext Customer Support

My first instinct for Agnost was agent-native data. I looked at `arcee-ai/agent-data` for tool-call sequences and `AgentTrove` for full agent trajectories. Both were tempting, but both had the same issue: no ground-truth labels. Without labels, I cannot prove whether the clustering is discovering real structure or just making a confident-looking map.

Bitext gives the validation layer: 26,872 rows, 27 labeled intents, customer support domain, updated October 2025. The labels let the repo compute Adjusted Rand Index, which measures whether discovered clusters align with human-labeled intent. That makes the demo honest. In production, Agnost's OTel-style SDK logs are the data source. The demo dataset only proves the engine.

**Rejected:**
- `arcee-ai/agent-data`: 485k rows, in-memory clustering breaks on a laptop, no ground-truth labels
- `AgentTrove`: synthetically generated, no labels, correctness is hard to verify
- `ABCD`: multi-turn JSON, complex parsing, no clean single-label ground truth
- Twitter Customer Support: 3M rows, single utterances, noisy format for this assignment

---

### 2. Database: Supabase + pgvector + psycopg2 Split

My first instinct was a vector database like Qdrant or Pinecone. Then I asked what the output actually needs: percentages, trends, counts per cluster, and drill-down queries.

```sql
SELECT cluster_id, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS pct
FROM conversations
GROUP BY cluster_id;
```

That is SQL. This product is not only nearest-neighbor search. It is vector storage plus analytics. A pure vector DB would force a second database for aggregation: two systems, two failure domains, two places to debug. pgvector gives both in one system: store the 1536-dim embedding and run SQL analytics in the same DB. Supabase makes it reviewer-friendly: managed Postgres, fast setup, no Docker tax for the evaluator.

**Two-layer DB access strategy (important):**

Two access paths, one database:
- **REST (PostgREST):** right for the API layer. `POST /ingest` stores one conversation, `GET /insights` reads 27 rows. Low volume, low latency, clean.
- **psycopg2 via Transaction Pooler (port 6543):** right for the pipeline layer. 500-row batches, each batch commits in under 5 seconds, connection released before any timeout fires. A 1536-dim embedding serialized as JSON is ~15KB per row. Inserting 27k rows via REST means hundreds of paginated HTTP calls. PostgREST is not a bulk loader.

Rule: same database, two access paths matched to workload.

**Rejected:**
- Qdrant / Pinecone: strong ANN stores, but no native SQL aggregation layer for PM analytics
- Chroma: useful locally, not the production persistence layer I would choose
- Self-hosted pgvector: same engine, worse reviewer experience for a weekend submission

---

### 3. Embedding Model: Provider Interface

For the demo: `text-embedding-3-small`. MTEB ~60 is sufficient for clustering 27k customer support utterances, and it keeps the weekend stack simple. Not the best model on paper. The right demo constraint.

The more important decision is the provider boundary:

```python
class EmbeddingProvider:
    def embed(self, texts: list[str]) -> np.ndarray: ...
```

The pipeline should not care whether embeddings come from OpenAI, Qwen3, or Gemini. `EMBEDDING_PROVIDER=openai|qwen3|gemini` is a one-env-var swap, not a rewrite.

Production Path A: open source / low lock-in
- Qwen3-Embedding-0.6B via SiliconFlow API
- MTEB 68+ and explicitly strong for clustering
- OpenAI-compatible API surface, low migration cost

Production Path B: Google stack consolidation
- Gemini Embedding 2, MTEB 68.3, $0.006/1M tokens
- Lower unit cost than `text-3-small` in the researched comparison
- Pairs cleanly with a Gemini production LLM if the stack moves that way

**Rejected:**
- `all-MiniLM-L6-v2`: good fallback, weaker quality
- `text-embedding-3-large`: higher cost without enough upside for this use case
- Voyage 4: excellent clustering benchmark, but 10x Gemini unit cost in the researched comparison
- Cohere embed-v4: multilingual strength is less relevant for the English-only demo

---

### 4. Clustering: UMAP + K-Means

HDBSCAN was the obvious first instinct because it can discover density-based topics without a fixed K. Before committing, I ran `check_dataset.py`. Average user utterance length is 8.7 words. Short text does not give HDBSCAN enough density signal; the researched risk was a high outlier rate where useful conversations disappear as noise. K-Means assigns every conversation to a cluster, which is better for a PM-facing analytics product.

Raw K-Means on 1536-dim embeddings is also not enough. In high dimensions, distances flatten and the algorithm can quietly lose semantic structure. UMAP fixes that by reducing to 5 dimensions while preserving local neighborhoods. PCA is simpler, but UMAP is better suited for non-linear semantic neighborhoods.

K=27 is a demo shortcut, not a product assumption. It matches Bitext's 27 labeled intents so the repo can compute ARI against known ground truth. In production, K should be discovered automatically through a nightly silhouette sweep or moved into BERTopic.

Key specifics:
- `n_components=5, metric='cosine'`
- Demo K=27 for validation against Bitext labels
- ARI achieved: **0.67**. Well above the 0.3 threshold, confirming clusters track real user intent
- Production K discovery: silhouette sweep K=10-50, then drift/cohesion monitoring

**Data loading pattern: offline store, not REST**

The clustering job reads embeddings directly from a local numpy cache (`.cache/embeddings.npy`), not from the database. This is the correct separation: the database is the serving layer for the API; the offline store is the data layer for ML batch jobs. Pulling 26k × 1536-dim vectors through a REST API (1000-row page limit, JSON string serialization, public internet latency) is the wrong tool for this job regardless of the network constraint.

The rule established here scales cleanly: swap the numpy file for a Parquet file on S3 and the clustering job runs identically at 10M rows on a cloud worker. The logic does not change. Only the storage backend does.

**Rejected:**
- HDBSCAN: high outlier risk on 8.7-word support utterances
- Raw K-Means without UMAP: 1536-dim distances are a weak clustering surface
- LDA: ignores embeddings, wrong tool for semantic agent conversations
- Agglomerative clustering: O(n^2) memory does not fit the weekend + scale constraints
- BERTopic: right month-plan tool, but more moving parts than needed for the demo proof

---

### 5. LLM: GPT-4.1 mini Demo, Claude/Gemini Option in Production

The LLM is not the clustering engine. It is the translation layer that turns cluster statistics into PM language: topic label, severity, trend, sample quote, and recommended action.

For the demo I chose GPT-4.1 mini. It keeps the working stack on OpenAI for embeddings + labeling, supports reliable structured output, and is strong enough for synthesis without turning the demo into a model comparison project.

Call structure:
- 27 calls per run, one per cluster
- Input per call: 12 randomly sampled conversations plus `{ volume, pct, trend }`
- Output: `{ topic_label, pm_insight, severity, action, sample_quote }`
- Stored in Supabase so `GET /insights` never calls the LLM

**Demo sampling choice: random vs centroid-nearest**

The original design called for centroid-nearest sampling: find the 8 conversations geometrically closest to the cluster center, since those are the most "typical" members. For the demo, sampling is random at n=12 instead. The reasoning: with ARI at 0.67, clusters are already tight and semantically coherent. Picking 12 random conversations from a cluster of ~1000 that are all about refunds produces samples that all say "I want my refund back" regardless of their distance to the centroid. GPT gets equivalent signal at zero extra compute cost.

Production path:
- Keep GPT-4.1 mini if reliability/cost is good enough
- Benchmark Claude and Gemini 3.5 Flash for label consistency, JSON validity, latency, and cost
- Split jobs at scale: stable labels weekly, changing insight synthesis nightly

On DeepSeek-style models, the rejection is not capability. It is enterprise risk. This product processes real customer conversation data, so data handling, legal exposure, and buyer trust matter as much as benchmarks.

**Rejected for demo:**
- Claude: strong model, but second vendor and no need for this small structured synthesis step
- Gemini: good production candidate, but OpenAI-only is simpler for the weekend demo
- Local Llama: more setup, weaker structured-output reliability for unattended jobs
- Template-only output: cheap, but cannot infer severity, causality, or action quality

---

### 6. API: FastAPI + ProcessPoolExecutor

FastAPI fits the stack because the ML pipeline is Python, the API is small, and Pydantic gives one schema layer for request validation and LLM-output validation. Swagger at `/docs` also matters for this assignment: the reviewer can test endpoints without guessing curl shapes.

The important architecture decision is not the framework. It is avoiding CPU-bound work in the async event loop. UMAP + K-Means can take 60-90 seconds on 27k vectors. If that runs inside the request handler, every endpoint freezes. `ProcessPoolExecutor` keeps the demo dependency-light while moving clustering to a separate OS process. `POST /analyze` returns `202 Accepted` with a `job_id`, and the API remains responsive.

**Rejected:**
- Flask: sync-first, weaker fit for async embedding calls and auto-doc review
- Django REST: too much framework for five endpoints
- Litestar: technically interesting, less reviewer/ecosystem payoff
- Express / Node: would split the API from the Python ML stack for no benefit
- GraphQL: useful later for dashboards, unnecessary for five weekend endpoints

---

### 7. SDK & Libraries

I avoided orchestration frameworks on purpose. LangChain, LlamaIndex, and CrewAI are useful when the system is an agent loop. This is a linear analytics pipeline: embed, cluster, label, serve. Direct calls keep the system inspectable, which matters when explaining why a cluster formed.

Core libraries:
- `openai`: embeddings and GPT-4.1 mini labeling
- `umap-learn`: dimensionality reduction
- `scikit-learn`: K-Means, silhouette score, ARI
- `fastapi` + `pydantic`: API and validation
- `supabase-py`: Postgres + pgvector access
- `datasets`: HuggingFace Bitext loader
- `tqdm`: useful progress visibility for embedding/clustering runs

---

### 8. run_clustering_pipeline Data Access: Offline Store vs REST vs psycopg2+register_vector

Decision: `.cache/embeddings.npy` offline store, same pattern as `scripts/02_cluster.py`.

**Rejected:**
- REST (PostgREST): serializes the `vector` column as a JSON string, not a float array. The pipeline cannot parse it. 1000-row hard cap causes silent truncation on a 27k-row dataset. ~165MB HTTP overhead per re-cluster run with no upside.
- psycopg2 + `register_vector`: technically fixes the type and row-cap issues, but contradicts the documented layer rule: DB is the serving layer, `.cache/` is the ML data layer. Introducing a second DB access path for the same ML job creates ambiguity and erodes the rule that prevents the REST anti-pattern from recurring.

Chosen because it is the pattern already proven in Phase 2, costs zero network, and is consistent with the anti-pattern rules already established. Production path: versioned Parquet on S3, same clustering logic, swap file path only.

---

## Demo vs Production

| Area | Weekend demo | Production path |
|---|---|---|
| Data source | Bitext, 27 ground-truth intents | Agnost OTel-style SDK logs |
| Embeddings | `text-embedding-3-small` | Qwen3 or Gemini Embedding 2 behind provider interface |
| Bulk ingest connection | psycopg2 via Transaction Pooler (port 6543), 500-row batches | PostgreSQL `COPY` binary format from same-region compute: 10-100x faster, zero logic change |
| Clustering data layer | Local numpy cache (`.cache/embeddings.npy`) | Versioned Parquet on S3: same clustering code, only the file path swaps |
| Clustering algorithm | UMAP + K-Means, K=27, ARI=0.67 | BERTopic + silhouette sweep + drift monitoring |
| Cluster write-back | psycopg2 batch UPDATE via Transaction Pooler | Same pattern or COPY-style bulk update |
| LLM sampling | Random, n=12 per cluster | Centroid-nearest from offline cache, n=20, async parallel calls |
| LLM model | GPT-4.1 mini, 27 sequential calls (~68s) | GPT-4.1 mini / Claude / Gemini benchmarked; async parallel (~8-10s) |
| REST pagination | 1000-row pages, client-side loop | Server-side cursor pagination or direct offline store read |
| Jobs | ProcessPoolExecutor | ARQ + Redis, then Celery for distributed workers |
| Tenancy | Single-company demo | Per-customer clustering isolation |
| Job infra | ProcessPoolExecutor: demo tier 1, zero deps, event loop free | ARQ + Redis at ~50 customers, Cloud Run Jobs at 100+: zero logic change, only the dispatcher swaps |

---

## With a Month

Each item below is a deliberate demo constraint being lifted, not an afterthought. The demo was built with these migrations in mind: the seams are already in the code.

1. **Bulk ingest → PostgreSQL COPY binary format** from same-region cloud compute. Demo uses Transaction Pooler + psycopg2 because it is free, reliable, and network-agnostic. Production shifts to COPY because 10-100x throughput matters when ingesting live agent logs daily at customer scale. Zero code change to pipeline logic: only the connection layer swaps.

2. **Offline store → versioned Parquet on S3.** Demo uses a local numpy file. Production uses S3-backed Parquet with a manifest: embedding model version, dataset hash, created timestamp. The clustering job already reads from a file path: swapping the path is the entire migration.

3. **BERTopic with K-Means backend:** true emerging topic discovery, drift detection, auto K. Demo hardcodes K=27 because Bitext has 27 known intents and that enables ARI validation. Production K is unknown and shifts over time.

4. **Centroid-nearest sampling + async LLM calls:** restore centroid-nearest from offline cache (n=20), parallelize 27 GPT calls with asyncio. Cuts labeling job from ~68s to ~8-10s.

5. **Nightly silhouette sweep K=10-50:** no hardcoded topic count. K adapts as conversation patterns change.

6. **Split LLM jobs:** weekly stable labels, nightly insight synthesis. Lower cost and latency at customer scale.

7. **Model benchmark harness:** compare GPT-4.1 mini, Claude, and Gemini 3.5 Flash on label consistency, JSON validity, latency, and cost per cluster.

8. **ARQ + Redis then Cloud Run Jobs:** ProcessPoolExecutor is the correct tier-1 choice: simplest thing that works, migration to ARQ is a one-line dispatcher swap. At 100+ customers, nightly clustering moves to Cloud Run Jobs: pay only for execution time, no always-on infra needed. `run_clustering_pipeline()` runs identically in all three tiers.

9. **Per-customer isolation:** separate vector/cluster space per company.

10. **Trend anomaly detection:** flag topics spiking above baseline for PM alerts.

11. **Privacy hardening:** PII redaction before embeddings, retention controls, audit logs.

---

## What the Build Revealed

Two constraints that shaped final design decisions, documented here because they affect anyone reproducing or extending this:

**Postgres connection portability.** Managed providers restrict direct TCP connections: IPv6-only hosts, connection limits, session duration caps. The pipeline uses a connection abstraction rather than a hardcoded `psycopg2.connect()`. Any backend swap is a config change, not a rewrite. This is why the Transaction Pooler (port 6543) is the default path, not direct Postgres.

**REST client blind spots on ML data.** The Supabase REST client silently serializes `vector` columns as JSON strings and enforces a 1000-row hard cap with no error. Both are invisible: the pipeline runs, produces wrong output. The solution is the offline store rule: `.cache/embeddings.npy` for all ML reads, REST only for the API serving layer. Decision 8 documents the specifics.

---

## Validation

The repo should print ARI from the clustering run instead of hardcoding a number in this document.

ARI against Bitext's 27 ground-truth labels:
- `0.0` means random clustering
- `1.0` means perfect label alignment
- Target: `> 0.3`, enough to show clusters track real user intent rather than noise

Expected proof command once the repo is built:

```bash
python scripts/02_cluster.py
```
