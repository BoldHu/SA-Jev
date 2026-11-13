from copy import deepcopy
from dataclasses import asdict, replace

import pytest

from sa_jev.io import read_json, read_jsonl, write_jsonl
from sa_jev.perturbation_review import (CHECKS, EFFECTIVE_CHECKS, approved, complete,
                                       prepare_perturbation_reviews)
from sa_jev.robustness import review_packets, validate_packet
from sa_jev.schema import Evidence, Question, Pool, State
from sa_jev.prompts import requests


class FakeReviewer:
    identity = {"model": "TEST_ONLY", "protocol": "fixture"}

    def __init__(self, reject=False, malformed=False, fail_at=None):
        self.calls = 0
        self.reject = reject
        self.malformed = malformed
        self.fail_at = fail_at

    def __call__(self, payload):
        self.calls += 1
        if self.calls == self.fail_at: raise RuntimeError("TEST network interruption")
        def result(keys):
            return {**dict.fromkeys(keys, True), "confidence": "high", "rationale": "TEST ONLY audit"}
        verdict = {"perturbations": [result(CHECKS) for _ in range(3)], "effective": result(EFFECTIVE_CHECKS)}
        if self.reject: verdict["perturbations"][1]["no_new_support"] = False
        return {"review_type": "llm", "model": "TEST_ONLY", "finish_reason": "length" if self.malformed else "stop",
                "verdict": verdict, "raw_response": "TEST ONLY"}


@pytest.fixture
def prepared(tmp_path):
    evidence = tuple(Evidence.make(f"Page {i}", f"Distinct original paragraph number {i} with substantive text.", f"source/{i}")
                     for i in range(5))
    q = Question("q", "g", "fixture", "heldout", "What is the answer?", "SECRET_ANSWER",
                 evidence, ("u0", "u1"), {evidence[0].id: ("u0",), evidence[1].id: ("u1",)},
                 {"u0": evidence[0].text, "u1": evidence[1].text}, tuple(e.id for e in evidence[2:]))
    s = State(q.id, q.group_id, q.question, Pool(q.id, evidence[:2], "fixture"), budget=4)
    write_jsonl(tmp_path / "data/questions.jsonl", [q])
    packets = review_packets([q], [s, s.read(evidence[0].id)], tmp_path / "data/perturbations_draft.jsonl")
    cfg = {"evaluation": {"answer_model": "TEST_ONLY"}}
    return tmp_path, cfg, q, s, packets


def test_preview_full_cache_and_model_input_separation(prepared):
    root, cfg, q, s, packets = prepared
    reviewer = FakeReviewer()
    preview = prepare_perturbation_reviews(root, cfg, limit=1, reviewer=reviewer)
    assert preview["scope"] == "preview" and preview["accepted_packets"] == 1
    assert not (root / "data/perturbations_reviewed.jsonl").exists()
    report = prepare_perturbation_reviews(root, cfg, limit=None, reviewer=reviewer)
    assert reviewer.calls == 2 and report["cached_reviews"] == 1
    assert report["accepted_packets"] == 2 and report["accepted_questions"] == 1
    rows = list(read_jsonl(root / "data/perturbations_reviewed.jsonl"))
    clean, y, views, changed, yc = validate_packet(rows[0], [q.group_id])
    assert rows[0]["review_metadata"]["human_verified"] is False
    assert rows[0]["perturbations"][0]["reviewer"] == "llm:TEST_ONLY"
    for view in views:
        _, body, _ = requests(view)
        text = str(body)
        assert "SECRET_ANSWER" not in text and "TEST_ONLY" not in text and "review_key" not in text
        assert all(e.text not in text for e in s.candidates)
    checklist = read_json(root / "data/perturbations_reviewed.human_spotcheck.json")
    assert all(x["approved"] is None for x in checklist["samples"])
    prepare_perturbation_reviews(root, cfg, limit=None, reviewer=reviewer)
    assert reviewer.calls == 2


def test_rejection_and_malformed_never_export(prepared):
    root, cfg, *_ = prepared
    with pytest.raises(ValueError, match="No packets passed"):
        prepare_perturbation_reviews(root, cfg, limit=None, reviewer=FakeReviewer(reject=True))
    assert not (root / "data/perturbations_reviewed.jsonl").exists()
    bad = FakeReviewer(malformed=True)
    bad.identity = {"model": "TEST_ONLY_truncated"}
    with pytest.raises(RuntimeError, match="Incomplete/malformed"):
        prepare_perturbation_reviews(root, cfg, limit=None, reviewer=bad)
    assert bad.calls == 1
    with pytest.raises(RuntimeError, match="Incomplete/malformed"):
        prepare_perturbation_reviews(root, cfg, limit=None, reviewer=bad)
    assert bad.calls == 1  # Persistently bad responses do not consume more API calls.


def test_interrupted_review_resumes_per_packet(prepared):
    root, cfg, *_ = prepared
    reviewer = FakeReviewer(fail_at=2)
    with pytest.raises(RuntimeError, match="network interruption"):
        prepare_perturbation_reviews(root, cfg, limit=None, reviewer=reviewer)
    assert not (root / "data/perturbations_reviewed.jsonl").exists()
    reviewer = FakeReviewer()
    report = prepare_perturbation_reviews(root, cfg, limit=None, reviewer=reviewer)
    assert reviewer.calls == 1 and report["cached_reviews"] == 1


@pytest.mark.parametrize("change", ["labels", "candidate_copy", "family", "effective", "duplicate"])
def test_invalid_structure_rejected_before_api(prepared, change):
    root, cfg, q, s, packets = prepared
    p = deepcopy(packets[0])
    if change == "labels": p["clean_labels"]["sufficient"] = 1.
    if change == "candidate_copy":
        v = replace(s, context=q.evidence[0].title + "\n" + q.evidence[0].text)
        p["perturbations"][0]["state"] = asdict(v)
    if change == "family": p["split"] = "train"
    if change == "effective": p["effective"]["state"]["budget"] = 2
    if change == "duplicate": p["perturbations"][1] = deepcopy(p["perturbations"][0])
    write_jsonl(root / "data/perturbations_draft.jsonl", [p])
    reviewer = FakeReviewer()
    with pytest.raises(ValueError, match="No packets passed"):
        prepare_perturbation_reviews(root, cfg, limit=None, reviewer=reviewer)
    assert reviewer.calls == 0


def test_cache_binds_gold_and_export_preserved(prepared):
    root, cfg, q, s, packets = prepared
    reviewer = FakeReviewer()
    prepare_perturbation_reviews(root, cfg, limit=1, reviewer=reviewer)
    write_jsonl(root / "data/questions.jsonl", [replace(q, answer="OTHER_SECRET")])
    # Full review must not reuse the preview's audit under different gold.
    prepare_perturbation_reviews(root, cfg, limit=None, reviewer=reviewer)
    assert reviewer.calls == 3
    with pytest.raises(FileExistsError, match="preserve"):
        prepare_perturbation_reviews(root, cfg, limit=1, reviewer=reviewer)


def test_string_booleans_uncertain_and_missing_items_cannot_approve():
    review = FakeReviewer()({})
    assert approved(review)
    review["verdict"]["effective"]["confidence"] = "uncertain"
    assert complete(review) and not approved(review)
    review["verdict"]["effective"]["new_labels_correct"] = "true"
    assert not complete(review)
    review = FakeReviewer()({})
    review["verdict"]["perturbations"].pop()
    assert not complete(review)
