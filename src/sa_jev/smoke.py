"""A real backward/optimizer/checkpoint/evaluation run with an explicitly tiny test model."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import torch
from .schema import Evidence, Question, Pool, State
from .data import task_states, audit_splits
from .trajectories import build_preferences
from .model import tiny_model
from .training import TrainConfig, fit, cache_reference
from .evaluation import evaluate, repeat_diagnostics
from .reporting import export_report, training_report
from .io import write_json, write_jsonl, read_json


def fixture_questions():
    questions = []
    for i, split in enumerate(("train", "train", "validation", "test")):
        evidence = tuple(Evidence.make(f"Document {i}-{j}", text, f"fixture/{i}/{j}") for j, text in enumerate((
            f"Person {i} was born in City {i}.", f"City {i} is located in Country {i}.",
            f"Museum {i} displays paintings.", f"River {i} flows east.", f"Team {i} plays football.", f"Library {i} has maps.")))
        coverage = {e.id: (f"u{j}",) if j < 2 else () for j, e in enumerate(evidence)}
        questions.append(Question(f"fixture:{i}", f"fixture:{i}", "TEST_ONLY_fixture", split,
                         f"In which country was Person {i} born?", f"Country {i}", evidence,
                         ("u0", "u1"), coverage, {f"u{j}": evidence[j].text for j in range(2)},
                         tuple(e.id for e in evidence[2:])))
    return questions


def run(output="outputs/smoke", device="cpu", raw=None):
    torch.set_num_threads(2)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    questions = fixture_questions()
    if raw:
        from .preparation import load_raw
        from .data import hotpot, musique
        questions = []
        for name, adapter in (("hotpot_train.jsonl", hotpot), ("musique_ans_train.jsonl", musique)):
            records = list(load_raw(Path(raw)/name))[:4]
            for i, record in enumerate(records):
                q = adapter(record, split="train" if i<2 else "validation" if i==2 else "test")
                questions.append(q)
    pools = {q.id: Pool(q.id, q.evidence[:8], "TEST_ONLY_original_context", {"gold_injected": True}) for q in questions}
    states = [x for q in questions for x in task_states(q, pools[q.id], budget=3)]
    train = [x for x in states if x.split == "train"]
    val = [x for x in states if x.split == "validation"]
    pairs = build_preferences(questions, [x.state for x in train if not x.state.selected_ids], limit=6)
    audit_splits(questions, states, pairs)
    write_jsonl(output/"states.jsonl", states); write_jsonl(output/"pairs.jsonl", pairs)
    model = tiny_model(device)
    stage1 = TrainConfig(stage=1, accumulation=2, epochs=1, max_steps=2, validate_every=2, save_every=2,
                         learning_rate=.001, head_learning_rate=.001)
    history = fit(model, train, val, stage1, output/"stage1")
    training_report(history, output/"stage1")
    path = read_json(output/"stage1/latest.json")["checkpoint"]
    reloaded = tiny_model(device); reloaded.load_trainable(path)
    model.eval(); reloaded.eval()
    with torch.no_grad():
        a = model(train[0].state, include_rel=False); b = reloaded(train[0].state, include_rel=False)
        torch.testing.assert_close(a.log_p, b.log_p); torch.testing.assert_close(a.log_h, b.log_h)
    reference = cache_reference(reloaded, pairs, output/"reference.json")
    stage2 = TrainConfig(stage=2, accumulation=2, epochs=1, max_steps=2, validate_every=2, save_every=2,
                         stability_warmup_steps=1, learning_rate=.001, head_learning_rate=.001)
    history2 = fit(reloaded, train, val, stage2, output/"stage2", pairs=pairs, reference=reference)
    training_report(history2, output/"stage2")
    rows = evaluate(reloaded, [q for q in questions if q.split=="test"], pools, output/"evaluation",
                    name="TEST_ONLY_tiny", formal=False, budget=3)
    export_report(rows, output/"evaluation")
    diagnostics = repeat_diagnostics(reloaded, [val[0].state], repeats=2)
    export_report(diagnostics, output/"diagnostics", ("js", "top1_flip", "suf_flip", "seconds"), ("question_id",))
    summary = {"test_only": True, "model": "tiny causal encoder + actual upstream OpenJev DecisionHead/OptionSetInteractor",
               "data": "small real source records" if raw else "synthetic fixtures", "device": device,
               "train_states": len(train), "validation_states": len(val), "preference_pairs": len(pairs),
               "stage1_optimizer_steps": len(history), "stage2_optimizer_steps": len(history2),
               "checkpoint_roundtrip": True, "evaluation_rows": len(rows),
               "evaluation_errors": sum(r["status"] == "error" for r in rows),
               "peak_reserved_gib": max(r["peak_reserved_gib"] for r in history+history2),
               "not_verified": ["trained OpenJev-4B weights", "A100 80GB peak", "live LightRAG extraction", "official Jev API", "scientific performance"]}
    write_json(output/"verification.json", summary)
    return summary
