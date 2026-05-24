"""
Load Bitext -> embed -> bulk insert via psycopg2 Transaction Pooler (port 6543).

Why Transaction Pooler (not Session Pooler):
  Session Pooler (5432): kills long-running connections — bulk insert times out.
  Direct connection (db.*): IPv6-only — fails on IPv4 Windows networks.
  Transaction Pooler (6543): IPv4 proxied for free. Each batch = one short
  transaction. Commits in <5s. Pooler never times out.

Idempotency (2 layers):
  1. DB guard: skips if bitext rows already present.
  2. Disk cache: embeddings saved to .cache/ after first OpenAI call. Re-run = zero API cost.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from app.config import settings
from app.database import get_client
from app.services.embedder import get_embedder

CACHE_DIR = Path(__file__).parent.parent / ".cache"
EMBEDDINGS_FILE = CACHE_DIR / "embeddings.npy"
METADATA_FILE = CACHE_DIR / "metadata.json"

BATCH_SIZE = 500


def _connect():
    """Fresh connection from Transaction Pooler — no SET commands (not supported)."""
    return psycopg2.connect(settings.database_url, sslmode="require")


def _load_or_embed(rows: list) -> tuple[np.ndarray, list[dict]]:
    if EMBEDDINGS_FILE.exists() and METADATA_FILE.exists():
        print("[CACHE] Loading embeddings from disk — zero OpenAI cost")
        vecs = np.load(EMBEDDINGS_FILE)
        with open(METADATA_FILE) as f:
            meta = json.load(f)
        assert len(vecs) == len(rows), "Cache row count mismatch — delete .cache/ and re-run"
        return vecs, meta

    embedder = get_embedder()
    texts = [r["instruction"] for r in rows]
    all_vecs: list[list[float]] = []
    for i in tqdm(range(0, len(texts), settings.batch_size), desc="Embedding (OpenAI)"):
        all_vecs.extend(embedder.embed_batch(texts[i : i + settings.batch_size]))

    vecs = np.array(all_vecs, dtype=np.float32)
    meta = [{"intent": r["intent"], "category": r["category"]} for r in rows]
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_FILE, vecs)
    with open(METADATA_FILE, "w") as f:
        json.dump(meta, f)
    print(f"[CACHE] Saved to {CACHE_DIR}")
    return vecs, meta


def main() -> None:
    db = get_client()
    existing = db.table("conversations").select("id", count="exact").eq("source", "bitext").execute()
    existing_count = existing.count or 0
    if existing_count >= 26872:
        print(f"[OK] {existing_count} bitext rows already in DB — skipping")
        return
    if existing_count > 0:
        print(f"[PARTIAL] {existing_count} rows found — cleaning up before re-insert")
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 0")
            cur.execute("DELETE FROM conversations WHERE source = 'bitext'")
        conn.commit()
        conn.close()
        print("[OK] Partial rows deleted")

    print("Loading Bitext from HuggingFace...")
    rows = list(load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"))
    print(f"Loaded {len(rows)} conversations")

    vecs, meta = _load_or_embed(rows)

    records = [
        (
            f"bitext-{i}",
            rows[i]["instruction"],
            rows[i]["response"],
            json.dumps(meta[i]),
            vecs[i],
            "bitext",
        )
        for i in range(len(rows))
    ]

    print(f"Inserting {len(records)} rows via psycopg2 Transaction Pooler ({len(records)//BATCH_SIZE + 1} batches)...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Inserting"):
        conn = _connect()
        register_vector(conn)
        with conn.cursor() as cur:
            execute_values(
                cur,
                """INSERT INTO conversations
                   (session_id, user_message, assistant_message, metadata, embedding, source)
                   VALUES %s
                   ON CONFLICT (session_id) DO NOTHING""",
                records[i : i + BATCH_SIZE],
                template="(%s, %s, %s, %s::jsonb, %s::vector, %s)",
            )
        conn.commit()
        conn.close()

    total = db.table("conversations").select("id", count="exact").eq("source", "bitext").execute()
    print(f"[OK] {total.count} conversations stored")


if __name__ == "__main__":
    main()
