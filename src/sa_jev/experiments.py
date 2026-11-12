"""Thin orchestration shared by CLI and Colab notebooks."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import gc
import os
import platform
import subprocess
import time
import importlib.metadata
import torch
import yaml
from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .schema import Question, TaskExample, Evidence, State, Pool
from .model import SAJev
from .training import TrainConfig, fit, cache_reference
from .trajectories import PreferencePair
from .retrieval import PoolCache, make_lightrag, ingest_corpus, retrieve_once
from .preparation import prepare_tasks
from .reporting import export_report, training_report, csv_rows
from .evaluation import OfficialJev, Answerer, evaluate, repeat_diagnostics


def load_config(path="configs/public_lightrag.yaml"):
    with Path(path).open(encoding="utf-8") as f: return yaml.safe_load(f)


def read_questions(root):
    return [Question.from_dict(x) for x in read_jsonl(Path(root)/"data/questions.jsonl")]


def environment_manifest(output, config=None):
    packages = {}
    for p in ("torch", "transformers", "peft", "fla-core", "lightrag-hku", "sentence-transformers", "datasets", "numpy"):
        try: packages[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError: packages[p] = None
    try: git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError: git = "uncommitted_workspace"
    manifest = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0,
                "git": git, "config": config}
    write_json(output, manifest); return manifest


async def cache_pools(root, config, limit=None):
    if config.get("protocol") == "public_lightrag_n500_v1":
        from .public_indexes import cache_public_pools
        return await cache_public_pools(root, config, limit)
    root = Path(root); data = root/"data"; cfg = config["retrieval"]
    corpus = [Evidence(**r) for r in read_jsonl(data/"corpus.jsonl")]
    questions = [q for q in read_questions(root) if q.split != "train"]
    if limit: questions = questions[:limit]
    args = {k:v for k,v in cfg.items() if k not in {"mode", "max_k"}}
    rag = await make_lightrag(root/"index", **args)
    try:
        binding = await ingest_corpus(rag, corpus, root/"index/corpus_manifest.json", args)
        cache = PoolCache(data/"pools_max")
        main = PoolCache(data/"pools")
        source_registry = {e.id:e for e in corpus}
        started = time.perf_counter()
        fresh = 0
        for i, q in enumerate(questions):
            cached = cache.path(q.id).exists()
            pool = await retrieve_once(rag, q.id, q.question, cfg["max_k"], cache, corpus_fingerprint=binding,
                                       mode=cfg["mode"], extra={"max_total_tokens": 180000, "max_entity_tokens": 20000, "max_relation_tokens": 20000},
                                       source_registry=source_registry)
            main.write(slice_pool(pool, config["data"]["main_k"]))
            fresh += int(not cached)
            elapsed = time.perf_counter() - started
            eta = (len(questions)-i-1) * elapsed / fresh if fresh else None
            eta_text = f"{eta/3600:.2f} h" if eta is not None else "unknown (cache only)"
            print(f"cached {i+1}/{len(questions)} actual K={len(pool.candidates)}; "
                  f"remaining query ETA={eta_text}", flush=True)
        write_json(data / "pool_cache_manifest.json", {
            "complete": limit is None, "questions": len(questions), "corpus_binding": binding,
            "question_fingerprint": read_json(data / "data_manifest.json")["question_fingerprint"],
            "config_fingerprint": digest({"retrieval": cfg, "main_k": config["data"]["main_k"]}),
        })
    finally: await rag.finalize_storages()


def slice_pool(pool, k):
    return Pool(pool.question_id, pool.candidates[:k], pool.retriever,
                {**pool.provenance, "parent_pool_hash": pool.fingerprint, "requested_k": k, "actual_k": min(k, len(pool.candidates))})


def latest(root, variant, seed):
    meta = read_json(Path(root)/f"runs/{variant}/seed_{seed}/latest.json")
    # Resolve relative to the current root; Drive may be mounted under another path.
    return Path(root)/f"runs/{variant}/seed_{seed}"/Path(meta["checkpoint"]).name


def load_method(root, config, method, seed):
    if method in {"static_jev", "dynamic_jev", "state_filtered_jev"}:
        policy = {"static_jev": "static", "dynamic_jev": "dynamic", "state_filtered_jev": "state_filtered"}[method]
        return OfficialJev(config["evaluation"]["jev_model"]), policy
    trainable = method not in {"retrieval_order", "openjev_frozen"}
    model = SAJev.from_pretrained(**config["model"], trainable=trainable)
    if trainable:
        variant = "full" if method == "no_state_update" else method
        model.load_trainable(latest(root, variant, seed)); model.requires_grad_(False)
    model.eval()
    return model, "retrieval" if method == "retrieval_order" else "no_state_update" if method == "no_state_update" else "dynamic"


def train_variant(root, config, variant="stage1", seed=17, max_steps=None, resume=None, validation_limit=None,
                  output_directory=None):
    from .public_indexes import require_public_question_protocol
    require_public_question_protocol(root, config)
    if variant not in {"stage1", "full", "dpo", "stability", "small_stage1", "small_k"} and not variant.startswith("stab_weight_"):
        raise ValueError("unknown training ablation")
    root = Path(root); data = root/"data"
    audit = read_json(data/"overlap_audit.json")
    manifest = read_json(data/"data_manifest.json")
    if max_steps is None and manifest.get("download", {}).get("formal") is False:
        raise ValueError("subset download is pilot-only; prepare the full root before formal training")
    if audit.get("question_fingerprint") != read_json(data/"data_manifest.json")["question_fingerprint"]:
        raise ValueError("OpenJevData overlap audit does not match these questions")
    if not audit.get("complete_scan"): raise ValueError("complete public OpenJevData audit is required before formal training")
    small = variant in {"small_stage1", "small_k"}
    suffix = "_small_k" if small else ""
    print(f"[train] Reading {data/f'train_states{suffix}.jsonl'} and validation states...", flush=True)
    train = [TaskExample.from_dict(x) for x in read_jsonl(data/f"train_states{suffix}.jsonl")]
    val = [TaskExample.from_dict(x) for x in read_jsonl(data/"validation_states.jsonl")]
    if validation_limit: val = val[:validation_limit]
    if not val: raise ValueError("build retrieved validation pools and prepare_tasks before training")
    print(f"[train] Loaded {len(train)} training states, {len(val)} validation states; "
          f"valid rank train={sum(x.labels.rank_valid for x in train)}, val={sum(x.labels.rank_valid for x in val)}", flush=True)
    stage = 1 if variant in {"stage1", "small_stage1"} else 2
    cfg = TrainConfig(**config["training"], stage=stage, seed=seed, max_steps=max_steps)
    if variant == "dpo": cfg.stability_weight = 0.
    if variant == "stability": cfg.dpo_weight = 0.
    if variant.startswith("stab_weight_"): cfg.stability_weight = float(variant.removeprefix("stab_weight_"))
    context_bank = None
    if stage == 2 and cfg.context_view_mode == "reviewed_context":
        from .context_views import ReviewedContexts
        context_bank = ReviewedContexts(data/f"training_contexts{suffix}.json", train,
                                        require_complete=max_steps is None)
    output = Path(output_directory) if output_directory is not None else root/f"runs/{variant}/seed_{seed}"
    if (output / "latest.json").exists() and resume is None:
        raise FileExistsError(f"Checkpoint already exists in {output}; use resume or a separate probe directory")
    environment_manifest(output/"environment.json", config)
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = SAJev.from_pretrained(**config["model"], trainable=True)
    reference = None; pairs = []
    if stage == 2:
        model.load_trainable(latest(root, "small_stage1" if small else "stage1", seed))
        if cfg.dpo_weight:
            pairs = [PreferencePair.from_dict(x) for x in read_jsonl(data/f"preferences{suffix}.jsonl")]
            reference = cache_reference(model, pairs, root/f"reference/seed_{seed}{suffix}.json")
    from .io import progress_phase
    with progress_phase(f"[train] Stage {stage} run; progress below, output={output}", interval=120):
        history = fit(model, train, val, cfg, output, pairs=pairs, reference=reference, resume=resume,
                      context_bank=context_bank)
    training_report(history, output)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return history


def main_experiment(root, config, methods=None, seeds=None, limit=None, qa=True, table_name="main"):
    root = Path(root); rows = []
    questions = [q for q in read_questions(root) if q.split == "test"]
    if limit:
        # Cap per dataset to retain cross-dataset testing during a pilot.
        questions = [q for dataset in sorted({q.dataset for q in questions}) for q in [r for r in questions if r.dataset == dataset][:limit]]
    cache = PoolCache(root/"data/pools"); pools = {q.id: cache.read(q.id) for q in questions}
    methods = methods or config["evaluation"]["main_methods"]
    for seed in seeds or config["seeds"]:
        answerer = Answerer(config["evaluation"]["answer_model"], root/"answers", seed=seed) if qa else None
        for method in methods:
            output = root/f"results/{table_name}/{method}/seed_{seed}"
            if method.endswith("_jev") and not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")):
                write_json(output/"unavailable.json", {"reason": "official Jev API credentials unavailable", "method": method}); continue
            model, policy = load_method(root, config, method, seed)
            rows.extend(evaluate(model, questions, pools, output, name=method, policy=policy,
                        budget=config["data"]["budget"], threshold=config["evaluation"]["threshold"], answerer=answerer, seed=seed))
            if isinstance(model, OfficialJev): write_json(output/"api_models.json", sorted(model.actual_models))
            del model; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    export_report(rows, root/f"results/{table_name}")
    # Paired bootstrap compares question families on identical question/seed sets.
    from .metrics import paired_bootstrap
    comparisons = []
    baseline = [r for r in rows if r["method"] == "openjev_frozen"]
    if baseline:
        for method in methods:
            other = [r for r in rows if r["method"] == method]
            if other and method != "openjev_frozen":
                for metric in ("support_recall", "qa_f1", "false_stop"):
                    comparisons.append({"method": method, "reference": "openjev_frozen", "metric": metric,
                                        **paired_bootstrap(other, baseline, metric)})
    csv_rows(root/f"results/{table_name}/paired_bootstrap.csv", comparisons)
    return rows


def efficiency_experiment(root, config, method="openjev_frozen", limit=50, seed=17):
    root = Path(root); cache = PoolCache(root/"data/pools_max")
    print(f"[efficiency] Preparing {method}; requested question limit={limit}", flush=True)
    questions = [q for q in read_questions(root) if q.split == "test"][:limit]
    print(f"[efficiency] {len(questions)} questions per K; K={config['evaluation']['k_values']}; loading model...", flush=True)
    model, policy = load_method(root, config, method, seed); rows = []
    print(f"[efficiency] Model ready on {model.device}", flush=True)
    train_manifest = root / "data/task_manifest.json"
    training_max_k = max(map(int, read_json(train_manifest).get("actual_k_counts", {})), default=0) if train_manifest.exists() else None
    if isinstance(model, SAJev):
        # Logical branch-set budget is a serializer guard, not the per-branch
        # backbone context. Declare the expansion explicitly for K extrapolation.
        model.serializer.max_decision_tokens = config["evaluation"]["efficiency_max_decision_tokens"]
    for k in config["evaluation"]["k_values"]:
        print(f"[efficiency] START K={k}: {len(questions)} questions; each may take up to {config['data']['budget']} READ decisions", flush=True)
        pools = {q.id: slice_pool(cache.read(q.id), k) for q in questions}
        measured = evaluate(model, questions, pools, root/f"results/efficiency/{method}/K{k}", name=method,
                            policy=policy, budget=config["data"]["budget"], seed=seed)
        for r in measured:
            r.update(requested_k=k, training_max_k=training_max_k,
                     extrapolation=(r["k"] > training_max_k) if training_max_k else None, exact_k=r["k"] == k,
                     max_decision_tokens=config["evaluation"]["efficiency_max_decision_tokens"] if isinstance(model, SAJev) else None)
        rows.extend(measured)
    export_report(rows, root/f"results/efficiency/{method}",
                  ("support_recall", "seconds", "logical_tokens", "executed_tokens", "peak_reserved_gib"), ("method", "requested_k", "exact_k"))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[efficiency] Complete: {root/f'results/efficiency/{method}'}", flush=True)
    return rows


def diagnostics_experiment(root, config, method="openjev_frozen", limit=50, seed=17):
    root = Path(root); cache = PoolCache(root/"data/pools")
    print(f"[diagnostics] Reading questions and cached pools; method={method}, limit={limit}", flush=True)
    questions = [q for q in read_questions(root) if q.split == "validation"][:limit]
    states = [State(q.id, q.group_id, q.question, cache.read(q.id), budget=config["data"]["budget"]) for q in questions]
    print(f"[diagnostics] Prepared {len(states)} states; loading model...", flush=True)
    model, _ = load_method(root, config, method, seed)
    print(f"[diagnostics] Model ready on {model.device}", flush=True)
    rows = repeat_diagnostics(model, states, config["evaluation"]["repeats"],
                              output=root/f"results/diagnostics/{method}")
    for row in rows: row["method"] = method
    export_report(rows, root/f"results/diagnostics/{method}", ("js", "top1_flip", "top5_change", "suf_flip", "seconds"), ("method",))
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"[diagnostics] Complete: {root/f'results/diagnostics/{method}'}", flush=True)
    return rows


def robustness_experiment(root, config, methods=("openjev_frozen", "stage1", "full"), seed=17, tag="robustness"):
    from .robustness import evaluate_robustness
    root = Path(root)
    packets = list(read_jsonl(root/"data/perturbations_reviewed.jsonl"))
    if not packets: raise ValueError("reviewed independent test is empty")
    groups = [q.group_id for q in read_questions(root) if q.split == "heldout"]
    rows = []
    for method in methods:
        model, _ = load_method(root, config, method, seed)
        rows.extend(evaluate_robustness(model, packets, groups, method, config["evaluation"]["threshold"]))
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    write_jsonl(root/f"results/{tag}/raw.jsonl", rows)
    export_report(rows, root/f"results/{tag}", ("js", "top1_flip", "top5_change", "suf_flip", "false_stop", "suf_correct", "next_support_correct", "ndcg"), ("method", "kind"))
    return rows
