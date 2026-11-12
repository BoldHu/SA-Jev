from dataclasses import replace

import pytest
import torch

from sa_jev.data import training_pool, task_states
from sa_jev.io import read_json, read_jsonl, write_json, write_jsonl
from sa_jev.schema import Evidence, Pool, State, labels_for, TaskExample
from sa_jev.smoke import fixture_questions
from sa_jev.task_repair import repair_training_tasks
from sa_jev.trajectories import PreferencePair


def test_known_training_pool_keeps_distractors_and_full_rank_gradient():
    q = fixture_questions()[0]
    unknown = Evidence.make("External", "Possible alternative evidence without annotations.")
    q = replace(q, evidence=q.evidence + (unknown,))
    old = Pool(q.id, (q.evidence[0], q.evidence[1], unknown), "old_training")
    assert not labels_for(q, State(q.id, q.group_id, q.question, old)).rank_valid
    pool = training_pool(q, [unknown], k=64)
    assert len(pool.candidates) == 6
    assert unknown.id not in {e.id for e in pool.candidates}
    states = task_states(q, pool)
    y = states[0].labels
    assert y.rank_valid and sum(y.rank) == 1 and all(y.rel_valid)
    assert sum(r == 0 for r in y.relevant) == 4
    assert states[-1].labels.sufficient and not states[-1].labels.rank_valid
    from sa_jev.model import tiny_model
    from sa_jev.losses import task_loss
    torch.set_num_threads(2)
    model = tiny_model()
    prediction = model(states[0].state)
    loss, _ = task_loss([prediction], [y])
    loss["rank"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    torch.testing.assert_close(prediction.p.sum(), torch.tensor(1.))
    # The same unknown remains unknown in real retrieved evaluation.
    vq = replace(q, split="validation")
    assert not labels_for(vq, State(q.id, q.group_id, q.question, old)).rank_valid
    with pytest.raises(ValueError):
        training_pool(vq)


def setup_old_data(path):
    questions = fixture_questions()
    write_jsonl(path / "questions.jsonl", questions)
    states = []
    for q in questions[:2]:
        unknown = Evidence.make("External", "Unknown alternative path.")
        pool = Pool(q.id, q.evidence[:2] + (unknown,), "old_training")
        states.extend(task_states(q, pool))
    write_jsonl(path / "train_states.jsonl", states)
    write_jsonl(path / "preferences.jsonl", [])
    write_json(path / "task_manifest.json", {"rank_valid": 0})
    q = questions[2]
    unknown = Evidence.make("External", "Unknown evaluation evidence.")
    write_jsonl(path / "validation_states.jsonl", task_states(q, Pool(q.id, q.evidence + (unknown,), "fixed")))
    write_json(path / "pool_cache_manifest.json", {"sentinel": "must not change"})
    return {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


def test_repair_no_mining_or_eval_changes_and_replayable_preferences(tmp_path, monkeypatch):
    before = setup_old_data(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("must not mine or retrieve")
    monkeypatch.setattr("sa_jev.preparation.hard_candidates", forbidden)
    report = repair_training_tasks(tmp_path, counts={"TEST_ONLY_fixture": 2})
    assert report["after"]["rank_valid"] == 4
    assert report["after"]["validation_rank_valid"] == 0
    for name in ("questions.jsonl", "validation_states.jsonl", "pool_cache_manifest.json"):
        assert (tmp_path / name).read_bytes() == before[name]
    from pathlib import Path
    for name in ("train_states.jsonl", "preferences.jsonl", "task_manifest.json"):
        assert (Path(report["backup"]) / name).read_bytes() == before[name]
    for row in read_jsonl(tmp_path / "preferences.jsonl"):
        pair = PreferencePair.from_dict(row)
        pair.chosen.replay(); pair.rejected.replay()
        assert len(pair.chosen.initial.pool.candidates) == 6
    for row in read_jsonl(tmp_path / "train_states.jsonl"):
        assert all(TaskExample.from_dict(row).labels.rel_valid)


def test_repair_wrong_sample_does_not_publish(tmp_path):
    before = setup_old_data(tmp_path)
    with pytest.raises(ValueError, match="selection"):
        repair_training_tasks(tmp_path, counts={"TEST_ONLY_fixture": 1})
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
