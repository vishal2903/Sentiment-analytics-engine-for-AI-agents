"""
Orchestrator: runs 01 → 02 → 03 in sequence, validates each step before continuing.
Run once to populate Supabase. ~10 minutes total (embedding dominates).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib
import time


def run_step(module_path: str, label: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    start = time.time()
    mod = importlib.import_module(module_path)
    mod.main()
    elapsed = time.time() - start
    print(f"[OK] {label} completed in {elapsed:.0f}s")


def validate_phase_0() -> None:
    from app.config import settings
    from app.database import get_client
    get_client()
    print("[OK] Config + DB connection verified")


if __name__ == "__main__":
    print("Agnost Insight Engine — Full Pipeline")
    validate_phase_0()
    run_step("scripts.01_ingest", "Phase 1: Ingest + Embed (27k conversations)")
    run_step("scripts.02_cluster", "Phase 2: UMAP + K-Means + ARI Validation")
    run_step("scripts.03_label", "Phase 3: LLM Insight Generation (27 clusters)")
    print("\n[DONE] Pipeline complete. Run: uvicorn app.main:app --reload")
