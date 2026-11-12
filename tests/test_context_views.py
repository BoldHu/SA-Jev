from dataclasses import asdict, replace
import json
import random

import pytest
import torch

from sa_jev.context_views import (CHECKS, ReviewedContexts, approved, prepare_context_views, safe_source)
from sa_jev.data import task_states
from sa_jev.io import digest, read_json, write_json, write_jsonl
from sa_jev.model import tiny_model
from sa_jev.prompts import View, requests, render_state
from sa_jev.schema import Evidence, Pool
from sa_jev.smoke import fixture_questions
from sa_jev.training import TrainConfig, cache_reference, fit
from sa_jev.trajectories import build_preferences


class FakeReviewer:
    identity = {"model": "TEST_ONLY_reviewer", "protocol": "test"}

    def __init__(self, reject=False):
        self.calls = 0
        self.reject = reject

    def __call__(self, payload):
        self.calls += 1
        return {"review_type": "llm", "model": "TEST_ONLY_reviewer", "finish_reason": "stop",
                "verdict": {**{k: True for k in CHECKS}, "no_new_support": not self.reject,
                            "confidence": "high", "rationale": "TEST ONLY: irrelevant birthstone exhibit."}}


@pytest.fixture
def prepared(tmp_path):
    qs = fixture_questions()
    extra = Evidence.make("Birthstone display", "A museum displays birthstones under glass. "
                          "The exhibit describes how artists polish the stones and illuminate their colours. "
                          "Its catalogue does not discuss any person's birthplace or nationality.", "fixture/context")
    qs[0] = replace(qs[0], evidence=qs[0].evidence + (extra,),
                    known_negative_ids=qs[0].known_negative_ids + (extra.id,))
    train = task_states(qs[0], Pool(qs[0].id, qs[0].evidence[:3], "fixture"), budget=2)
    val = task_states(qs[2], Pool(qs[2].id, qs[2].evidence[:3], "fixture"), budget=2)
    data = tmp_path / "data"
    write_jsonl(data / "questions.jsonl", qs)
    write_jsonl(data / "train_states.jsonl", train)
    config = {"evaluation": {"answer_model": "TEST_ONLY"}, "context_views": {"min_words": 5}}
    return tmp_path, qs, train, val, config, extra


def test_review_cache_source_and_same_labels(prepared):
    root, qs, train, val, cfg, extra = prepared
    before = (root / "data/train_states.jsonl").read_bytes()
    reviewer = FakeReviewer()
    result = prepare_context_views(root, cfg, reviewer=reviewer)
    assert result["accepted_questions"] == 1 and reviewer.calls == 1
    bank = ReviewedContexts(root / "data/training_contexts.json", train)
    v = bank.view(train[0], random.Random(17))
    assert v.context_suffix and v.style == 0
    clean = json.loads(render_state(train[0].state))
    augmented = json.loads(render_state(train[0].state, v))
    assert augmented.pop("context") == v.context_suffix
    clean.pop("context")
    assert augmented == clean
    assert (root / "data/train_states.jsonl").read_bytes() == before
    assert "TEST_ONLY_reviewer" not in render_state(train[0].state, v)
    assert bank.view(replace(train[0], state=replace(train[0].state, budget=99)), random) is None
    assert bank.view(replace(train[0], labels=replace(train[0].labels, sufficient=1.)), random) is None
    prepare_context_views(root, cfg, reviewer=reviewer)
    assert reviewer.calls == 1
    # Interrupt after per-review cache was saved: rebuilding aggregate uses that cache.
    (root / "data/training_contexts.json").unlink()
    prepare_context_views(root, cfg, reviewer=reviewer)
    assert reviewer.calls == 1
    choice, suf, rel = requests(train[0].state, v)
    assert v.context_suffix in json.loads(suf["state"])["context"]
    assert rel == requests(train[0].state)[2]  # Auxiliary relevance sees no context.


def test_reject_support_answers_pool_and_heldout(prepared):
    root, qs, train, val, cfg, extra = prepared
    q = qs[0]
    assert safe_source(q, train, extra, set(), 5, 384)
    for e in (q.evidence[0], q.evidence[2], replace(extra, text="Country 0 " + extra.text)):
        assert not safe_source(q, train, e, set(), 1, 384)
    from sa_jev.schema import normalize
    assert not safe_source(q, train, extra, {digest(normalize(extra.text))}, 5, 384)
    altered = [replace(train[0], state=replace(train[0].state, conditions="only after 2020"))]
    assert not safe_source(q, altered, extra, set(), 5, 384)
    reviewer = FakeReviewer(reject=True)
    result = prepare_context_views(root, cfg, reviewer=reviewer)
    assert result["accepted_questions"] == 0
    with pytest.raises(ValueError, match="No approved"):
        ReviewedContexts(root / "data/training_contexts.json", train)
    verdict = FakeReviewer()({})
    assert approved(verdict)
    assert not approved({**verdict, "finish_reason": "length"})
    assert not approved({**verdict, "verdict": {**verdict["verdict"], "no_new_support": "true"}})


def test_tampered_or_stale_bank_rejected(prepared):
    root, qs, train, val, cfg, extra = prepared
    prepare_context_views(root, cfg, reviewer=FakeReviewer())
    path = root / "data/training_contexts.json"
    bank = read_json(path)
    with pytest.raises(ValueError, match="different task"):
        ReviewedContexts(path, train[:-1])
    bank["rows"][0]["text"] += " Country 0"
    write_json(path, bank)
    with pytest.raises(ValueError, match="modified"):
        ReviewedContexts(path, train)


def test_incomplete_review_stops_batch_and_is_not_retried_silently(prepared):
    root, qs, train, val, cfg, extra = prepared
    reviewer = FakeReviewer()
    class Truncated(FakeReviewer):
        def __call__(self, payload):
            result = super().__call__(payload)
            return {**result, "finish_reason": "length", "verdict": {}}
    reviewer = Truncated()
    with pytest.raises(RuntimeError, match="truncated/malformed"):
        prepare_context_views(root, cfg, reviewer=reviewer)
    assert reviewer.calls == 1 and len(list((root / "data/context_reviews").glob("*.json"))) == 1
    with pytest.raises(RuntimeError, match="truncated/malformed"):
        prepare_context_views(root, cfg, reviewer=reviewer)
    assert reviewer.calls == 1


def test_real_two_stage_context_dpo_and_resume(prepared):
    root, qs, train, val, cfg, extra = prepared
    torch.set_num_threads(2)
    prepare_context_views(root, cfg, reviewer=FakeReviewer())
    bank = ReviewedContexts(root / "data/training_contexts.json", train)
    model = tiny_model()
    fit(model, train, val, TrainConfig(accumulation=2, max_steps=1), root / "stage1")
    stage1 = root / "stage1/step_000001"
    pairs = build_preferences(qs, [x.state for x in train if not x.state.selected_ids], limit=3)
    original_pairs = digest(pairs)
    reference = cache_reference(model, pairs, root / "reference.json")
    cfg2 = TrainConfig(stage=2, context_view_mode="reviewed_context", accumulation=2, state_batch_size=2,
                       max_steps=2, save_every=1, validate_every=2)
    history = fit(model, train, val, cfg2, root / "stage2", pairs=pairs, reference=reference, context_bank=bank)
    assert [h["context_pairs"] for h in history] == [2, 1]  # Final effective batch is short.
    assert [h["valid_stab_suf"] for h in history] == [2, 1]
    assert digest(pairs) == original_pairs
    serial = tiny_model(); serial.load_trainable(stage1)
    serial_history = fit(serial, train, val, replace(cfg2, state_batch_size=1), root / "serial_stage2",
                         pairs=pairs, reference=reference, context_bank=bank)
    for a, b in zip(history, serial_history):
        for key in ("task", "dpo", "stability", "total_loss"):
            assert a[key] == pytest.approx(b[key], abs=2e-6, rel=2e-5)
    for (name, a), (_, b) in zip(model.named_parameters(), serial.named_parameters()):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5, msg=name)
    resumed = tiny_model(); resumed.load_trainable(stage1)
    fit(resumed, train, val, cfg2, root / "resumed", pairs=pairs, reference=reference,
        context_bank=bank, resume=root / "stage2/step_000001")
    for (name, a), (_, b) in zip(model.named_parameters(), resumed.named_parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0, msg=name)


def test_budget_overflow_skips_pair_without_truncating(prepared):
    root, qs, train, val, cfg, extra = prepared
    prepare_context_views(root, cfg, reviewer=FakeReviewer())
    bank = ReviewedContexts(root / "data/training_contexts.json", train)
    model = tiny_model()
    v = bank.view(train[0], random)
    assert model.view_within_budget(train[0].state, v)
    assert not model.view_within_budget(train[0].state, View(0, "word " * 10000))
    model.view_within_budget = lambda s, v: False  # Simulate every appended view exceeding the budget.
    history = fit(model, train, val, TrainConfig(stage=2, context_view_mode="reviewed_context",
                  dpo_weight=0., accumulation=2, max_steps=1), root / "skip", context_bank=bank)
    assert history[0]["context_pairs"] == 0 and history[0]["context_over_budget"] == 2
    assert history[0]["stability"] == 0 and history[0]["valid_stab_suf"] == 0


def test_mixed_batch_counts_only_real_approved_pairs(prepared):
    root, qs, train, val, cfg, extra = prepared
    prepare_context_views(root, cfg, reviewer=FakeReviewer())
    path = root / "data/training_contexts.json"
    raw = read_json(path)
    raw["rows"][0]["example_keys"] = [digest(train[0])]
    write_json(path, raw)
    bank = ReviewedContexts(path, train)
    torch.set_num_threads(2)
    history = fit(tiny_model(), train, val, TrainConfig(stage=2, context_view_mode="reviewed_context",
                  dpo_weight=0., accumulation=len(train), state_batch_size=2, max_steps=1), root / "mixed", context_bank=bank)
    assert history[0]["context_pairs"] == 1 and history[0]["unpaired_states"] == len(train)-1
    assert history[0]["valid_stab_suf"] == history[0]["valid_stab_rank"] == 1
