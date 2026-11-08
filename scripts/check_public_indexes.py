"""Actual public artifacts + small local encoder + native retrieval; no paid API or 4B weights."""
import asyncio
import json
import os
from pathlib import Path
import tempfile

from sa_jev.public_indexes import import_public_indexes, open_public_index, DATASETS, prepare_public_data, cache_public_pools
from sa_jev.io import read_json, write_json
from sa_jev.retrieval import PoolCache, retrieve_once
from sa_jev.schema import Evidence


async def main():
    import yaml
    assets = Path("outputs/public_index_inspection")
    root = Path(os.environ["SAJEV_PUBLIC_CHECK_ROOT"]) if os.environ.get("SAJEV_PUBLIC_CHECK_ROOT") else Path(tempfile.mkdtemp(prefix="public_", dir=assets))
    cfg = yaml.safe_load(Path("configs/public_lightrag.yaml").read_text())
    cfg["retrieval"]["embedding_device"] = "cpu"
    manifest = import_public_indexes(assets/"lightrag_indexes_N500.zip", root)
    calls = 0

    async def fixture_keywords(prompt, **kwargs):
        nonlocal calls
        calls += 1
        return json.dumps({"high_level_keywords": ["history"], "low_level_keywords": ["England", "United States"]})

    results = []
    for ds in DATASETS:
        rag = await open_public_index(root, ds, cfg, llm_func=fixture_keywords)
        try:
            registry = {k: Evidence(**v) for k,v in read_json(root/"index"/ds/"source_registry.json").items()}
            query = "What historical connections exist between England and the United States?"
            cache = PoolCache(root/"pools"/ds)
            p = await retrieve_once(rag, ds+":fixture", query, 16, cache, corpus_fingerprint=ds,
                                    chunk_registry=registry)
            assert p.candidates and all(e.id in registry for e in p.candidates)
            before = calls
            again = await retrieve_once(rag, ds+":fixture", query, 16, cache, corpus_fingerprint=ds,
                                        chunk_registry=registry)
            assert p == again and calls == before
            results.append({"dataset": ds, "returned_k": len(p.candidates), "source_traceability": True})
        finally:
            await rag.finalize_storages()
    # Full small pipeline with actual benchmark labels and real graph retrieval.
    cfg["data"]["pilot_train_counts"] = {"hotpotqa": 4, "musique": 4}
    cfg["data"]["main_k"] = 8
    cfg["retrieval"]["max_k"] = 16
    prepared = prepare_public_data(root, assets, Path("data/smoke_raw"), cfg, mode="pilot")
    await cache_public_pools(root, cfg, llm_func=fixture_keywords)
    from sa_jev.preparation import prepare_tasks
    tasks = prepare_tasks(root/"data", counts=cfg["data"]["pilot_train_counts"], mine=True)
    from sa_jev.model import tiny_model
    from sa_jev.schema import TaskExample, Question
    from sa_jev.io import read_jsonl
    from sa_jev.training import fit, TrainConfig, cache_reference
    from sa_jev.trajectories import PreferencePair
    from sa_jev.evaluation import evaluate
    from sa_jev.reporting import export_report, training_report
    import torch
    torch.set_num_threads(2)
    train = [TaskExample.from_dict(r) for r in read_jsonl(root/"data/train_states.jsonl")]
    val = [TaskExample.from_dict(r) for r in read_jsonl(root/"data/validation_states.jsonl")]
    pairs = [PreferencePair.from_dict(r) for r in read_jsonl(root/"data/preferences.jsonl")][:4]
    model = tiny_model("cpu")
    h1 = fit(model, train, val, TrainConfig(stage=1, max_steps=2, accumulation=2, validate_every=2, save_every=2), root/"tiny_stage1")
    training_report(h1, root/"tiny_stage1")
    reference = cache_reference(model, pairs, root/"tiny_reference.json")
    h2 = fit(model, train, val, TrainConfig(stage=2, max_steps=2, accumulation=2, validate_every=2, save_every=2), root/"tiny_stage2", pairs=pairs, reference=reference)
    training_report(h2, root/"tiny_stage2")
    qs = [Question.from_dict(r) for r in read_jsonl(root/"data/questions.jsonl") if r["split"] == "test"]
    pools = {q.id: PoolCache(root/"data/pools").read(q.id) for q in qs}
    rows = evaluate(model, qs, pools, root/"tiny_evaluation", name="TEST_ONLY_tiny_public", formal=False, budget=2)
    export_report(rows, root/"tiny_evaluation")
    assert all(r["status"] != "error" for r in rows)
    write_json(root/"verification.json", {"public_indexes": manifest["stats"], "retrieval": results,
               "data_splits": prepared["dataset_splits"], "task_counts": tasks,
               "stage1_steps": len(h1), "stage2_steps": len(h2), "evaluated_questions": len(rows),
               "external_api_calls": 0, "scope": "real public graphs, real MiniLM, fixture query keywords; not paper metrics"})
    print("Public index integration passed:", root/"verification.json")


if __name__ == "__main__":
    asyncio.run(main())
