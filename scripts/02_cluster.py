"""
Load embeddings → UMAP reduction → K-Means clustering → write cluster_id back.
Prints ARI against Bitext ground-truth intent labels.
ARI > 0.3 = clusters track real user intent (random = 0.0, perfect = 1.0).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tqdm import tqdm
from app.config import settings
from app.database import get_client
from app.services.clusterer import reduce_dimensions, cluster_vectors, compute_ari


def main() -> None:
    db = get_client()

    # Production pattern: read from offline store (numpy cache), not REST API.
    # REST = for serving single rows. Cache = for batch ML jobs.
    import json as _json
    from pathlib import Path
    cache_dir = Path(__file__).parent.parent / ".cache"
    embeddings_file = cache_dir / "embeddings.npy"
    metadata_file = cache_dir / "metadata.json"

    if embeddings_file.exists() and metadata_file.exists():
        print("[CACHE] Loading embeddings from local cache (offline store)...")
        vectors = np.load(embeddings_file)
        with open(metadata_file) as f:
            meta_list = _json.load(f)
        ground_truth = [m.get("intent", "unknown") for m in meta_list]
        print(f"Loaded {len(vectors)} vectors from cache")

        # Still need DB row IDs to write cluster_id back — lightweight query
        print("Fetching row IDs from Supabase (IDs only, no vectors)...")
        rows = []
        page_size = 1000
        offset = 0
        while True:
            result = (
                db.table("conversations")
                .select("id")
                .eq("source", "bitext")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        ids = [r["id"] for r in rows]
        print(f"Fetched {len(ids)} row IDs")
    else:
        print("[FALLBACK] Cache missing — loading embeddings from Supabase (paginated)...")
        rows = []
        page_size = 1000
        offset = 0
        while True:
            result = (
                db.table("conversations")
                .select("id,embedding,metadata")
                .eq("source", "bitext")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data
            rows.extend(batch)
            print(f"  fetched {len(rows)} rows...", end="\r")
            if len(batch) < page_size:
                break
            offset += page_size
        print(f"\nLoaded {len(rows)} rows")
        ids = [r["id"] for r in rows]
        vectors = np.array(
            [_json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"] for r in rows],
            dtype=np.float32,
        )
        ground_truth = [r["metadata"].get("intent", "unknown") for r in rows]

    print("Running UMAP (1536-dim → 5-dim, cosine)...")
    reduced = reduce_dimensions(tqdm(vectors, desc="UMAP") if False else vectors)

    print(f"Clustering into K={settings.cluster_k} clusters...")
    labels = cluster_vectors(reduced)

    ari = compute_ari(labels, ground_truth)
    print(f"\n[ARI Score] {ari:.4f}  (target: > 0.3)")
    if ari < 0.3:
        print("[WARN] ARI below threshold — check UMAP params before proceeding")

    print("Writing cluster_ids to Supabase (psycopg2 batch UPDATE)...")
    import psycopg2
    from psycopg2.extras import execute_values
    pairs = [(ids[j], int(labels[j])) for j in range(len(ids))]
    batch_size = 500
    for i in tqdm(range(0, len(pairs), batch_size), desc="Updating"):
        conn = psycopg2.connect(settings.database_url, sslmode="require")
        with conn.cursor() as cur:
            execute_values(
                cur,
                """UPDATE conversations SET cluster_id = data.cluster_id
                   FROM (VALUES %s) AS data(id, cluster_id)
                   WHERE conversations.id = data.id::uuid""",
                pairs[i : i + batch_size],
                template="(%s, %s)",
            )
        conn.commit()
        conn.close()

    null_check = db.table("conversations").select("id", count="exact").eq("source", "bitext").is_("cluster_id", "null").execute()
    remaining = null_check.count or 0
    assert remaining == 0, f"[ERROR] {remaining} bitext rows still missing cluster_id"
    print(f"[OK] All {len(ids)} conversations have cluster_id")


if __name__ == "__main__":
    main()
