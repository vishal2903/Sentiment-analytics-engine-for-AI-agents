"""
For each cluster: sample 8 centroid-nearest conversations + stats -> GPT-4.1-mini -> PM insight.
Writes to clusters table. Pre-computes all insights so GET /insights costs zero LLM calls.

Idempotent: skips clusters that already have a row in the clusters table.
Re-run after partial failure only calls GPT for unlabeled clusters — no wasted API cost.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from tqdm import tqdm
from app.config import settings
from app.database import get_client
from app.services.labeler import sample_cluster_conversations, generate_cluster_insight


def main() -> None:
    db = get_client()

    print("Loading conversations from Supabase (paginated, text only)...")
    rows = []
    page_size = 1000
    offset = 0
    while True:
        result = (
            db.table("conversations")
            .select("id,cluster_id,user_message")
            .eq("source", "bitext")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = result.data
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    total = len(rows)

    cluster_counts = Counter(r["cluster_id"] for r in rows if r["cluster_id"] is not None)
    k = len(cluster_counts)
    print(f"Found {k} clusters across {total} conversations")

    existing = db.table("clusters").select("id").execute()
    labeled_ids = {r["id"] for r in existing.data}
    pending = {cid: vol for cid, vol in cluster_counts.items() if cid not in labeled_ids}
    if labeled_ids:
        print(f"[CACHE] {len(labeled_ids)} clusters already labeled — skipping (no API cost)")
    print(f"Labeling {len(pending)} clusters...")

    success = len(labeled_ids)
    for cluster_id, volume in tqdm(sorted(pending.items()), desc="Labeling clusters"):
        pct = round(volume / total * 100, 1)
        stats = {"volume": volume, "percentage": pct, "week_over_week": None}

        conversations = sample_cluster_conversations(cluster_id, rows, n=8)
        if not conversations:
            continue

        try:
            insight = generate_cluster_insight(conversations, stats)
            db.table("clusters").upsert({
                "id": cluster_id,
                "topic_label": insight.topic_label,
                "pm_insight": insight.pm_insight,
                "severity": insight.severity,
                "action": insight.action,
                "sample_quote": insight.sample_quote,
                "volume": volume,
                "percentage": pct,
                "week_over_week": None,
            }).execute()
            success += 1
        except Exception as exc:
            print(f"[ERROR] Cluster {cluster_id}: {exc}")

    print(f"\n[OK] {success}/{k} insights generated and stored")

    if success > 0:
        sample = db.table("clusters").select("*").order("percentage", desc=True).limit(3).execute()
        print("\n=== TOP 3 INSIGHTS BY VOLUME ===")
        for row in sample.data:
            print(f"  {row['percentage']:.1f}% [{row['severity']}] {row['topic_label']}")
            print(f"  {row['pm_insight']}")
            print(f"  Action: {row['action']}\n")


if __name__ == "__main__":
    main()
