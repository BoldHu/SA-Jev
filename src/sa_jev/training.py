"""Single-device training; frozen Stage-1 reference is cached before Stage 2."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random
import time
import torch
import numpy as np
from .io import digest, read_json, write_json, write_jsonl
from .losses import task_loss, stability_loss, dpo_loss
from .prompts import View
from .trajectories import trajectory_log_probability


@dataclass
class TrainConfig:
    stage: int = 1
    epochs: int = 1
    accumulation: int = 8
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = .01
    max_grad_norm: float = 1.
    dpo_weight: float = .2
    stability_weight: float = .1
    beta: float = .1
    tolerance: float = .05
    stability_warmup_steps: int = 100
    seed: int = 17
    max_steps: int | None = None
    validate_every: int = 100
    save_every: int = 100
    memory_limit_gib: float = 79.
    context_view_mode: str = "serialization"  # Legacy/control; experiment YAML uses reviewed_context.


def memory_stats(device):
    if torch.device(device).type != "cuda":
        return {"allocated_gib": 0., "reserved_gib": 0., "peak_allocated_gib": 0., "peak_reserved_gib": 0.}
    return {name: fn(device) / 1024**3 for name, fn in {
        "allocated_gib": torch.cuda.memory_allocated, "reserved_gib": torch.cuda.memory_reserved,
        "peak_allocated_gib": torch.cuda.max_memory_allocated,
        "peak_reserved_gib": torch.cuda.max_memory_reserved}.items()}


def synchronize(device):
    if torch.device(device).type == "cuda": torch.cuda.synchronize(device)


def model_fingerprint(model):
    import hashlib
    h = hashlib.sha256(model.base_identity.encode())
    # Include trainable Stage-1 weights and every decision-head parameter.
    for name, p in model.named_parameters():
        if p.requires_grad or name.startswith("core.head"):
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    h.update(str(model.temperature).encode())
    return h.hexdigest()


def cache_reference(model, pairs, path):
    identity = model_fingerprint(model)
    binding = digest({"model": identity, "pairs": [p.fingerprint for p in pairs],
                      "view": 0, "probability": "sum_READ_and_autonomous_STOP_v1"})
    path = Path(path)
    if path.exists():
        cache = read_json(path)
        if cache["binding"] != binding: raise ValueError("stale Stage-1 reference cache")
        return cache
    training = model.training; model.eval()
    values, started = {}, time.perf_counter()
    with torch.no_grad():
        for i, pair in enumerate(pairs):
            values[pair.fingerprint] = [float(trajectory_log_probability(model, t)) for t in (pair.chosen, pair.rejected)]
            if i % 10 == 0 or i == len(pairs) - 1:
                elapsed = time.perf_counter() - started
                print(f"reference {i+1}/{len(pairs)} ETA={elapsed/(i+1)*(len(pairs)-i-1):.1f}s", flush=True)
    model.train(training)
    cache = {"binding": binding, "stage1_fingerprint": identity, "values": values}
    write_json(path, cache)
    return cache


@torch.no_grad()
def validate(model, examples):
    if not examples: raise ValueError("validation requires held-out task states")
    if any(x.split != "validation" for x in examples):
        raise ValueError("only validation data may guide training")
    mode = model.training; model.eval()
    sums = {"rank": 0., "suf": 0., "rel": 0.}; counts = dict.fromkeys(sums, 0)
    started = last = time.perf_counter()
    print(f"[validation] START {len(examples)} states; missing ranking labels stay unavailable", flush=True)
    for index, x in enumerate(examples, 1):
        components, n = task_loss([model(x.state)], [x.labels])
        for key in sums:
            sums[key] += float(components[key]) * n[key]; counts[key] += n[key]
        now = time.perf_counter()
        if index == 1 or index == len(examples) or now-last >= 5:
            print(f"[validation] {index}/{len(examples)} ETA={(now-started)/index*(len(examples)-index):.1f}s "
                  f"valid_counts={counts} peak_reserved_gib={memory_stats(model.device)['peak_reserved_gib']:.2f}", flush=True)
            last = now
    model.train(mode)
    means = {k: sums[k] / counts[k] if counts[k] else None for k in sums}
    means["task"] = sum((means[k] or 0.) * w for k, w in (("rank", 1), ("suf", 1), ("rel", .2)))
    return {"validation": means, "validation_valid_counts": counts}


def fit(model, train, validation, config: TrainConfig, output, pairs=(), reference=None, resume=None,
        context_bank=None):
    if config.stage not in (1, 2) or not train or config.accumulation < 1:
        raise ValueError("invalid training configuration or empty data")
    if any(x.split != "train" for x in train): raise ValueError("training input contains held-out data")
    if {x.state.group_id for x in train} & {x.state.group_id for x in validation}:
        raise ValueError("question-family overlap between train and validation")
    use_dpo = config.stage == 2 and config.dpo_weight > 0
    if use_dpo and (not pairs or reference is None):
        raise ValueError("DPO requires real trajectory pairs and frozen Stage-1 reference cache")
    if use_dpo and any(p.fingerprint not in reference["values"] for p in pairs):
        raise ValueError("reference cache does not cover all pairs")
    if config.context_view_mode not in {"serialization", "reviewed_context"}:
        raise ValueError("unknown context view mode")
    reviewed = config.stage == 2 and config.context_view_mode == "reviewed_context"
    if reviewed and context_bank is None:
        raise ValueError("Stage 2 requires a reviewed context bank; run prepare_context_views first")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    memory_limit = config.memory_limit_gib
    if model.device.type == "cuda":
        total_gib = torch.cuda.get_device_properties(model.device).total_memory / 1024**3
        memory_limit = min(memory_limit, total_gib - 2.)
        print(f"[train] GPU total={total_gib:.2f} GiB; observed-memory stop threshold={memory_limit:.2f} GiB. "
              "This check does not prevent an allocation from raising CUDA OOM.", flush=True)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.json").exists() and resume is None:
        raise FileExistsError("run already has a checkpoint; use resume or a different output directory")
    write_json(output / "config.json", asdict(config))
    print(f"[train] Binding data/config: train={len(train)}, validation={len(validation)}, stage={config.stage}", flush=True)
    binding = digest({"states": train, "validation": validation, "pairs": pairs, "config": asdict(config),
                      "context_bank": context_bank.binding if reviewed else None})
    groups = []
    for head in (False, True):
        parameters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("core.head") == head]
        if parameters: groups.append({"params": parameters, "lr": config.head_learning_rate if head else config.learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    windows = math.ceil(len(train) / config.accumulation)
    total = min(windows * config.epochs, config.max_steps or windows * config.epochs)
    print(f"[train] {total} optimizer steps; accumulation={config.accumulation}; "
          f"validate_every={config.validate_every}, save_every={config.save_every}; "
          f"memory_limit={memory_limit:.2f} GiB; output={output}", flush=True)
    start_step, history = 0, []
    if resume:
        model.load_trainable(resume)
        saved = torch.load(Path(resume) / "optimizer.pt", map_location="cpu", weights_only=False)
        if saved["binding"] != binding: raise ValueError("resume dataset/config changed")
        optimizer.load_state_dict(saved["optimizer"]); start_step = saved["step"]
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"] is not None: torch.cuda.set_rng_state_all(saved["cuda_rng"])
        history = saved["history"]
    model.train()
    if model.device.type == "cuda": torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()

    def save(step):
        checkpoint = output / f"step_{step:06d}"
        print(f"[checkpoint] Saving step {step}: {checkpoint}", flush=True)
        model.save_trainable(checkpoint, {"stage": config.stage, "step": step, "data_binding": binding,
                             "reference_binding": reference["binding"] if use_dpo else None})
        torch.save({"optimizer": optimizer.state_dict(), "step": step, "binding": binding,
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "history": history}, checkpoint / "optimizer.pt")
        write_json(output / "latest.json", {"checkpoint": str(checkpoint.resolve()), "step": step})
        print(f"[checkpoint] Saved step {step}", flush=True)

    for step in range(start_step, total):
        epoch, offset = divmod(step, windows)
        indices = list(range(len(train))); random.Random(config.seed + epoch).shuffle(indices)
        batch = [train[i] for i in indices[offset*config.accumulation:(offset+1)*config.accumulation]]
        print(f"[train] START step {step+1}/{total}, epoch={epoch}, states={len(batch)}", flush=True)
        last_notice = time.perf_counter()
        # Denominators span the effective batch, not microbatches with different
        # numbers of valid labels. Each term has its own valid-sample denominator.
        counts = {"rank": sum(x.labels.rank_valid for x in batch),
                  "suf": sum(x.labels.suf_valid for x in batch),
                  "rel": sum(sum(x.labels.rel_valid) for x in batch)}
        secondary = []
        over_budget = 0
        for x in batch:
            view = None
            if config.stage == 2:
                view = context_bank.view(x, random) if reviewed else View(random.choice((1, 2)))
                if reviewed and view is not None and not model.view_within_budget(x.state, view):
                    view = None; over_budget += 1
            secondary.append(view)
        stab_counts = {
            "rank": sum(v is not None and x.labels.suf_valid and x.labels.gold_available and not x.labels.sufficient
                        and bool(x.labels.candidate_ids) for x, v in zip(batch, secondary)),
            "suf": sum(v is not None and x.labels.suf_valid and x.labels.gold_available for x, v in zip(batch, secondary))}
        optimizer.zero_grad(set_to_none=True)
        log = {"step": step + 1, "epoch": epoch, "rank": 0., "suf": 0., "rel": 0., "dpo": 0., "stability": 0.}
        log.update(paired_states=sum(v is not None for v in secondary),
                   context_pairs=sum(bool(v and v.context_suffix) for v in secondary),
                   context_over_budget=over_budget, unpaired_states=sum(v is None for v in secondary),
                   valid_stab_rank=stab_counts["rank"], valid_stab_suf=stab_counts["suf"])
        warmup = min(1., (step + 1) / max(1, config.stability_warmup_steps))
        for state_index, (x, second_view) in enumerate(zip(batch, secondary), 1):
            if state_index == 1 or time.perf_counter()-last_notice >= 30:
                print(f"[train] step {step+1}/{total}, state {state_index}/{len(batch)}: "
                      f"forward/backward K={len(x.state.candidates)}, id={x.state.question_id}", flush=True)
                last_notice = time.perf_counter()
            pa = model(x.state, View(0))
            views = [pa]
            if second_view is not None:
                views.append(model(x.state, second_view))
            loss = pa.log_h.sum() * 0.
            for p in views:
                c, n = task_loss([p], [x.labels])
                for key, weight in (("rank", 1.), ("suf", 1.), ("rel", .2)):
                    value = c[key] * n[key] / max(1, counts[key]) / len(views)
                    loss = loss + weight * value; log[key] += float(value.detach())
            if config.stage == 2 and config.stability_weight > 0 and second_view is not None:
                # Compute the two masked means separately to preserve their distinct denominators.
                pb = views[1]
                if set(pa.candidate_ids) != set(pb.candidate_ids):
                    raise ValueError("context view changed the effective candidate set")
                if x.labels.suf_valid and x.labels.gold_available and not x.labels.sufficient and pa.candidate_ids:
                    aligned = [pb.candidate_ids.index(eid) for eid in pa.candidate_ids]
                    value = torch.relu(torch.linalg.vector_norm(pa.p - pb.p[aligned]) - config.tolerance).square() / max(1, stab_counts["rank"])
                    loss = loss + config.stability_weight * warmup * value; log["stability"] += float(value.detach())
                if x.labels.suf_valid and x.labels.gold_available:
                    value = torch.relu(torch.abs(pa.h - pb.h) - config.tolerance).square() / max(1, stab_counts["suf"])
                    loss = loss + config.stability_weight * warmup * value; log["stability"] += float(value.detach())
            loss.backward()
            if time.perf_counter()-last_notice >= 30:
                print(f"[train] step {step+1}/{total}: completed state {state_index}/{len(batch)}; "
                      f"peak_reserved_gib={memory_stats(model.device)['peak_reserved_gib']:.2f}", flush=True)
                last_notice = time.perf_counter()
        if use_dpo:
            print(f"[train] step {step+1}/{total}: full-trajectory DPO forward/backward", flush=True)
            pair = pairs[(step + config.seed) % len(pairs)]
            ref = reference["values"][pair.fingerprint]
            chosen = trajectory_log_probability(model, pair.chosen)
            rejected = trajectory_log_probability(model, pair.rejected)
            value = dpo_loss(chosen, rejected, chosen.new_tensor(ref[0]), chosen.new_tensor(ref[1]), config.beta)
            (config.dpo_weight * value).backward(); log["dpo"] = float(value.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.max_grad_norm)
        if not torch.isfinite(grad_norm): raise FloatingPointError("non-finite gradient; refusing optimizer update")
        optimizer.step(); synchronize(model.device)
        elapsed = time.perf_counter() - started
        log.update(memory_stats(model.device), seconds=elapsed, eta_seconds=elapsed/(step-start_step+1)*(total-step-1),
                   grad_norm=float(grad_norm), valid_rank=counts["rank"], valid_suf=counts["suf"], valid_rel=counts["rel"])
        log["task"] = log["rank"] + log["suf"] + .2*log["rel"]
        log["total_loss"] = log["task"] + (config.dpo_weight*log["dpo"] + config.stability_weight*warmup*log["stability"] if config.stage == 2 else 0.)
        if log["peak_reserved_gib"] >= memory_limit:
            raise RuntimeError("measured memory exceeds budget; lower branch microbatch if above one or use a larger GPU; never split softmax")
        print(f"[train] DONE step {step+1}/{total}: task={log['task']:.5f} rank={log['rank']:.5f} "
              f"suf={log['suf']:.5f} rel={log['rel']:.5f} valid_rank={counts['rank']} "
              f"grad_norm={float(grad_norm):.5f} peak_reserved_gib={log['peak_reserved_gib']:.2f} "
              f"ETA={log['eta_seconds']:.1f}s (based on completed steps; validation/save time may add overhead)", flush=True)
        if (step+1) % config.validate_every == 0 or step+1 == total:
            log.update(validate(model, validation))
            log.update(memory_stats(model.device))
            elapsed = time.perf_counter() - started
            log.update(seconds=elapsed, eta_seconds=elapsed/(step-start_step+1)*(total-step-1))
            if log["peak_reserved_gib"] >= memory_limit:
                raise RuntimeError("validation exceeded the configured GPU memory budget")
        history.append(log); write_jsonl(output / "training.jsonl", history)
        print(log, flush=True)
        if (step+1) % config.save_every == 0 or step+1 == total: save(step+1)
    return history
