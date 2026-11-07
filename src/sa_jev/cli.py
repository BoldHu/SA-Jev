from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from .io import read_json, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description="SA-Jev reproducible research commands")
    parser.add_argument("command", choices=["smoke", "normalize", "download", "cache-pools", "prepare-tasks", "overlap", "train", "main", "diagnostics", "efficiency", "robustness", "review-packets", "industrial-download"])
    parser.add_argument("--root", default="outputs/research")
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw")
    parser.add_argument("--variant", default="stage1")
    parser.add_argument("--method", default="openjev_frozen")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--no-qa", action="store_true")
    parser.add_argument("--small-k", action="store_true")
    args = parser.parse_args()
    env_file = Path(os.environ.get("SAJEV_ENV_FILE") or Path(__file__).resolve().parents[2] / ".env")
    if os.environ.get("SAJEV_ENV_FILE") or env_file.is_file():
        from .environment import load_environment
        load_environment(env_file)
    root = Path(args.root)
    if args.command == "smoke":
        from .smoke import run
        print(run(root, args.device, args.raw)); return
    from .experiments import load_config, cache_pools, train_variant, main_experiment, diagnostics_experiment, efficiency_experiment, robustness_experiment, read_questions
    config = load_config(args.config)
    from .preparation import download_sources, normalize_sources, prepare_tasks
    if args.command == "download": print(download_sources(root/"raw", args.limit))
    elif args.command == "normalize": print(normalize_sources(args.raw or root/"raw", root/"data", seed=args.seed)[1])
    elif args.command == "cache-pools": asyncio.run(cache_pools(root, config, args.limit))
    elif args.command == "prepare-tasks":
        print(prepare_tasks(root/"data", seed=args.seed, budget=config["data"]["budget"], counts=config["data"]["train_counts"],
              mine=config["data"]["mine_hard_candidates"], dense=config["data"]["dense_mining"], small_k=args.small_k))
    elif args.command == "overlap":
        from datasets import load_dataset
        from .data import overlap_audit
        from itertools import chain, islice
        public = load_dataset("shenjunhao/OpenJevData-140k", streaming=True)
        records = chain.from_iterable(public.values())
        if args.limit: records = islice(records, args.limit)
        result = overlap_audit(read_questions(root), records)
        result.update(question_fingerprint=read_json(root/"data/data_manifest.json")["question_fingerprint"],
                      complete_scan=args.limit is None, source="shenjunhao/OpenJevData-140k")
        write_json(root/"data/overlap_audit.json", result); print({k:v for k,v in result.items() if k!="matches"})
    elif args.command == "train": train_variant(root, config, args.variant, args.seed, args.max_steps, args.resume)
    elif args.command == "main": main_experiment(root, config, args.methods, [args.seed], args.limit, not args.no_qa)
    elif args.command == "diagnostics": diagnostics_experiment(root, config, args.method, args.limit or 50, args.seed)
    elif args.command == "efficiency": efficiency_experiment(root, config, args.method, args.limit or 50, args.seed)
    elif args.command == "robustness": robustness_experiment(root, config, args.methods or ("openjev_frozen", "stage1", "full"), args.seed)
    elif args.command == "review-packets":
        from .robustness import review_packets
        from .retrieval import PoolCache
        from .data import task_states
        questions = read_questions(root); cache = PoolCache(root/"data/pools")
        states = [x.state for q in questions if q.split=="heldout" for x in task_states(q, cache.read(q.id))]
        print(len(review_packets(questions, states, root/"data/perturbations_draft.jsonl", args.limit or 600)))
    elif args.command == "industrial-download":
        from .industrial import download_ocr
        download_ocr(root/"industrial")


if __name__ == "__main__": main()
