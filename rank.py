"""
rank.py — Entrypoint: candidates.jsonl[.gz] -> submission.csv
Orchestrates Stages A through G of the pipeline.
Usage:
  python rank.py --candidates candidates.jsonl --out submission.csv
  python rank.py --candidates candidates.jsonl.gz --out submission.csv
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

from features import extract_features
from consistency_audit import train_isolation_forest, compute_suspicion
from scoring import compute_score
from select_top_k import select_top_k
from reasoning import generate_reasoning


def iter_candidates(path: Path):
    """Stage A: Stream-parse JSONL or JSONL.GZ line by line."""
    opener = gzip.open if path.suffix == ".gz" else open
    mode = "rt"
    with opener(path, mode, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def run(candidates_path: Path, out_path: Path, verbose: bool = True):
    t0 = time.time()

    # ----------------------------------------------------------------
    # Stage A + B: Ingest & extract features (two-pass for IsoForest)
    # Pass 1: extract features + collect consistency features for training
    # ----------------------------------------------------------------
    if verbose:
        print("[A+B] Ingesting and extracting features...", flush=True)

    all_features = []
    n = 0
    for candidate in iter_candidates(candidates_path):
        f = extract_features(candidate)
        all_features.append(f)
        n += 1
        if verbose and n % 10000 == 0:
            print("  processed {:,} candidates ({:.1f}s)".format(n, time.time()-t0), flush=True)

    if verbose:
        print("  total: {:,} candidates in {:.1f}s".format(n, time.time()-t0), flush=True)

    # ----------------------------------------------------------------
    # Stage D: Train Isolation Forest on consistency features
    # ----------------------------------------------------------------
    if verbose:
        print("[D] Training Isolation Forest...", flush=True)
    iso_model = train_isolation_forest(all_features, contamination=0.05)
    if verbose:
        print("  ISO forest trained in {:.1f}s".format(time.time()-t0), flush=True)

    # ----------------------------------------------------------------
    # Stage C + D + E: Score, audit, select top-100 (single pass over features)
    # ----------------------------------------------------------------
    if verbose:
        print("[C+D+E] Scoring and selecting top 100...", flush=True)

    def scored_stream(feature_list):
        for f in feature_list:
            susp = compute_suspicion(f, iso_model)
            sr   = compute_score(f, susp)
            yield {
                "candidate_id":    f["_candidate_id"],
                "score":           sr["score"],
                "features":        f,
                "score_result":    sr,
                "suspicion_result": susp,
            }

    top100 = select_top_k(scored_stream(all_features), k=100)

    if verbose:
        print("  top-100 selected in {:.1f}s".format(time.time()-t0), flush=True)

    # ----------------------------------------------------------------
    # Stage F: Reasoning generation
    # ----------------------------------------------------------------
    if verbose:
        print("[F] Generating reasoning...", flush=True)

    rows = []
    for rank, item in enumerate(top100, start=1):
        reasoning = generate_reasoning(
            item["features"], item["score_result"], item["suspicion_result"]
        )
        rows.append({
            "candidate_id": item["candidate_id"],
            "rank":         rank,
            "score":        round(item["score"], 6),
            "reasoning":    reasoning,
        })

    # ----------------------------------------------------------------
    # Stage G: Write CSV
    # ----------------------------------------------------------------
    if verbose:
        print("[G] Writing {}...".format(out_path), flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id","rank","score","reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time()-t0
    if verbose:
        print("Done in {:.1f}s. Top-5 candidates:".format(elapsed))
        for r in rows[:5]:
            print("  Rank {:3d} | {:12s} | score={:.4f} | {}".format(
                r["rank"], r["candidate_id"], r["score"], r["reasoning"][:80]))

    return rows


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out", default="submission.csv", help="Output CSV path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run(Path(args.candidates), Path(args.out), verbose=not args.quiet)


if __name__ == "__main__":
    main()
