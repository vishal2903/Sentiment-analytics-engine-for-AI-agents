-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Every ingested conversation stored here
CREATE TABLE IF NOT EXISTS conversations (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id        TEXT NOT NULL,
    user_message      TEXT NOT NULL,
    assistant_message TEXT,
    metadata          JSONB DEFAULT '{}',
    embedding         VECTOR(1536),
    cluster_id        INTEGER,
    source            TEXT DEFAULT 'api',
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- One row per cluster: stats + PM insight merged (no join needed at read time)
CREATE TABLE IF NOT EXISTS clusters (
    id             INTEGER PRIMARY KEY,
    topic_label    TEXT NOT NULL,
    pm_insight     TEXT NOT NULL,
    severity       TEXT CHECK (severity IN ('HIGH', 'MEDIUM', 'LOW')),
    action         TEXT,
    sample_quote   TEXT,
    volume         INTEGER,
    percentage     FLOAT,
    week_over_week FLOAT,
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);

-- Background job tracking for POST /analyze
CREATE TABLE IF NOT EXISTS pipeline_jobs (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status       TEXT DEFAULT 'pending'
                     CHECK (status IN ('pending', 'running', 'complete', 'failed')),
    triggered_by TEXT DEFAULT 'api',
    error        TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- ANN index: ivfflat cosine — O(log n) approx vs O(n) exact scan
-- lists=100 is optimal for ~27k rows (rule of thumb: sqrt(n))
CREATE INDEX IF NOT EXISTS conversations_embedding_idx
    ON conversations USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Cluster analytics lookups
CREATE INDEX IF NOT EXISTS conversations_cluster_id_idx ON conversations (cluster_id);
CREATE INDEX IF NOT EXISTS conversations_source_idx ON conversations (source);

-- Disable RLS: this is an analytics backend, not a multi-tenant user-data store.
-- Supabase enables RLS by default — anon key cannot insert without this.
ALTER TABLE conversations DISABLE ROW LEVEL SECURITY;
ALTER TABLE clusters DISABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_jobs DISABLE ROW LEVEL SECURITY;
