"""CPU-only repair of training candidate supervision; retrieved evaluation is untouched."""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from .io import read_json, read_jsonl, write_json
from .preparation import prepare_tasks


def repair_training_tasks(data_directory, *, counts=None, seed=17, budget=4, small_k=False):
    """Rebuild from original annotated contexts, with backup and no retrieval/API calls.

    Does not edit questions, split assignments, validation states, retrieved pools,
    indexes, overlap audit, or perturbations. Changed pools require new preference
    trajectories and invalidate previously bound training/reference/context caches.
    """
    directory = Path(data_directory)
    suffix = "_small_k" if small_k else ""
    names = [f"train_states{suffix}.jsonl", f"preferences{suffix}.jsonl", f"task_manifest{suffix}.json"]
    for name in [*names, "questions.jsonl", "validation_states.jsonl"]:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    old = read_json(directory / names[-1])
    # Never accidentally publish a differently sampled set under the old run.
    old_questions = {r["state"]["question_id"] for r in read_jsonl(directory / names[0])}
    work = directory / "task_repairs" / uuid4().hex[:12]
    staged, backup = work / "new", work / "backup"
    print(f"[repair] CPU only; no BM25, embeddings, retrieval or API. Workspace: {work}", flush=True)
    manifest = prepare_tasks(directory, counts=counts, seed=seed, budget=budget, mine=False,
                             small_k=small_k, output_directory=staged, rebuild_validation=False)
    new_questions = {r["state"]["question_id"] for r in read_jsonl(staged / names[0])}
    if new_questions != old_questions:
        raise ValueError("Repair changed training question selection; originals untouched. Use the original counts/seed.")
    if not manifest["rank_valid"]:
        raise ValueError("Repair still has no valid ranking states; originals untouched. Inspect original annotations.")
    report = {"protocol": "known_candidates_v1", "backup": str(backup),
              "before": old, "after": manifest,
              "evaluation": "Original validation/test candidate pools and validation states unchanged",
              "limitation": "Validation ranking may remain unavailable; do not gold-filter evaluation pools",
              "training": "Use new runs; old checkpoints, DPO reference and context views are not compatible"}
    write_json(work / "report.json", report)
    backup.mkdir(parents=True)
    moved = []
    try:
        for name in names:
            os.replace(directory / name, backup / name)
            moved.append(name)
            os.replace(staged / name, directory / name)
    except BaseException:
        for name in reversed(moved):
            os.replace(backup / name, directory / name)
        raise
    write_json(directory / f"task_repair{suffix}.json", report)
    print(f"[repair] Complete: rank-valid {old.get('rank_valid')} -> {manifest['rank_valid']}; "
          f"actual K: {manifest['actual_k_counts']}; backup: {backup}", flush=True)
    print("[repair] Validation/test are unchanged. Rerun the label audit before formal training.", flush=True)
    return report
