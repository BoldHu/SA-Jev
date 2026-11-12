from pathlib import Path

import pytest
import torch

from sa_jev.data import task_states
from sa_jev.io import read_json, write_json, write_jsonl
from sa_jev.model import tiny_model
from sa_jev.schema import Pool
from sa_jev.smoke import fixture_questions
from sa_jev.training import TrainConfig, fit


def examples():
    qs = fixture_questions()
    train = [x for q in qs[:2] for x in task_states(q, Pool(q.id, q.evidence, "fixture"))]
    q = qs[2]
    val = task_states(q, Pool(q.id, q.evidence, "fixture"))
    return train, val


def test_training_reports_step_validation_and_checkpoint(tmp_path, capsys):
    torch.set_num_threads(2)
    train, val = examples()
    history = fit(tiny_model(), train, val,
                  TrainConfig(accumulation=2, max_steps=2, validate_every=1, save_every=1), tmp_path)
    out = capsys.readouterr().out
    for message in ("START step 1/2", "DONE step 2/2", "[validation] 3/3", "[checkpoint] Saved step 2", "ETA="):
        assert message in out
    assert len(history) == 2 and sum(x["valid_rank"] for x in history) > 0
    assert history[-1]["validation_valid_counts"]["rank"] == 2
    assert Path(read_json(tmp_path / "latest.json")["checkpoint"]).is_dir()


def test_probe_does_not_overwrite_or_load_formal_checkpoint(tmp_path, monkeypatch):
    from sa_jev import experiments as ex
    torch.set_num_threads(2)
    train, val = examples()
    data = tmp_path / "data"
    write_jsonl(data / "train_states.jsonl", train)
    write_jsonl(data / "validation_states.jsonl", val)
    write_json(data / "data_manifest.json", {"question_fingerprint": "fixture", "download": {"formal": True}})
    write_json(data / "overlap_audit.json", {"question_fingerprint": "fixture", "complete_scan": True})
    formal = tmp_path / "runs/stage1/seed_17/latest.json"
    write_json(formal, {"sentinel": "existing formal run"})
    before = formal.read_bytes()
    loaded = []
    def loader(**kwargs):
        loaded.append(True)
        return tiny_model()
    monkeypatch.setattr(ex.SAJev, "from_pretrained", loader)
    monkeypatch.setattr(ex, "environment_manifest", lambda *args: None)
    monkeypatch.setattr(ex, "training_report", lambda *args: None)
    cfg = {"model": {}, "training": {"accumulation": 2, "save_every": 1, "validate_every": 1}}
    probe = tmp_path / "runs/stage1_probe/test/seed_17"
    ex.train_variant(tmp_path, cfg, max_steps=1, output_directory=probe)
    assert formal.read_bytes() == before and (probe / "latest.json").is_file()
    with pytest.raises(FileExistsError):
        ex.train_variant(tmp_path, cfg, max_steps=1)
    assert len(loaded) == 1  # Detect an existing run before allocating another model.
