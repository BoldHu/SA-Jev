"""Read-only BM25 mining timing; no API calls, dataset writes, or model downloads.

Run in a separate Colab terminal while the notebook kernel is busy. This uses
additional RAM and CPU; process CPU time helps expose CPU contention. Estimates
cover candidate scoring only, not state/trajectory construction or Drive writes.
"""
import argparse
import random
import re
from pathlib import Path
from time import perf_counter, process_time
from types import SimpleNamespace

import numpy as np
from rank_bm25 import BM25Okapi

from sa_jev.data import sample_training
from sa_jev.io import read_jsonl
from sa_jev.schema import Evidence, deduplicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--samples-per-dataset", type=int, default=3)
    args = parser.parse_args()
    if args.samples_per_dataset < 1:
        parser.error("--samples-per-dataset must be positive")
    print("Read-only benchmark. Extra CPU/RAM required; no API or data writes.", flush=True)
    # Keep only query metadata, not a second copy of all question evidence.
    questions = [SimpleNamespace(**{k: r[k] for k in ("id", "dataset", "split", "question")})
                 for r in read_jsonl(args.data / "questions.jsonl")]
    training = sample_training(questions, {"hotpotqa": 8000, "musique": 4000}, 17)
    if not training:
        raise ValueError("No training questions found")
    print(f"Selected {len(training):,} training questions; reading corpus...", flush=True)
    corpus = deduplicate(Evidence(**r) for r in read_jsonl(args.data / "corpus.jsonl"))
    tokenize = lambda s: re.findall(r"\w+", s.lower())
    print(f"Building BM25 on {len(corpus):,} documents...", flush=True)
    started = perf_counter()
    tokens = [tokenize(e.title + " " + e.text) for e in corpus]
    bm25 = BM25Okapi(tokens)
    del tokens
    print(f"Index ready in {perf_counter()-started:.1f}s; timing queries...", flush=True)
    rng = random.Random(2026)
    estimated_wall = estimated_cpu = 0.0
    for dataset in dict.fromkeys(q.dataset for q in training):
        population = [q for q in training if q.dataset == dataset]
        sample = rng.sample(population, min(args.samples_per_dataset, len(population)))
        walls, cpus = [], []
        for i, q in enumerate(sample, 1):
            wall, cpu = perf_counter(), process_time()
            # Same scoring, full argsort, top-64 and deduplication as hard_candidates.
            indices = np.argsort(-bm25.get_scores(tokenize(q.question)))[:64]
            candidates = deduplicate([corpus[j] for j in indices])[:64]
            cpus.append(process_time() - cpu)
            walls.append(perf_counter() - wall)
            print(f"{dataset} {i}/{len(sample)}: wall={walls[-1]:.3f}s, "
                  f"CPU={cpus[-1]:.3f}s, query_tokens={len(tokenize(q.question))}, "
                  f"candidates={len(candidates)}", flush=True)
        mean_wall, mean_cpu = sum(walls)/len(walls), sum(cpus)/len(cpus)
        estimated_wall += len(population) * mean_wall
        estimated_cpu += len(population) * mean_cpu
        print(f"{dataset}: N={len(population):,}, mean wall={mean_wall:.3f}s/query, "
              f"sample range={min(walls):.3f}..{max(walls):.3f}s", flush=True)
    print(f"\nSCORING FROM START: {estimated_wall/60:.1f} min at sampled wall speed; "
          f"{estimated_cpu/60:.1f} CPU-min of work.", flush=True)
    print("Small-sample estimate, NOT remaining time. Wall timing includes contention.\n"
          "CPU-min is not a guaranteed elapsed-time estimate. Excludes state/preference\n"
          "construction, validation, index setup and saving. Main process progress is unknown.", flush=True)


if __name__ == "__main__":
    main()
