import pytest
import torch

from sa_jev.evaluation import repeat_diagnostics
from sa_jev.io import read_jsonl, write_json
from sa_jev.model import tiny_model
from sa_jev.schema import State, Pool, Evidence
from sa_jev.smoke import fixture_questions


def test_diagnostic_rows_survive_interrupt_and_match_predictions(tmp_path, capsys):
    torch.set_num_threads(2)
    q = fixture_questions()[2]
    state = State(q.id, q.group_id, q.question, Pool(q.id, q.evidence, "fixture"))
    core = tiny_model().eval()

    class InterruptAfterOne:
        device = core.device
        calls = 0
        def eval(self): return self
        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                assert len(list(read_jsonl(tmp_path / "raw.jsonl"))) == 1
                raise KeyboardInterrupt()
            return core(*args, **kwargs)

    with pytest.raises(KeyboardInterrupt):
        repeat_diagnostics(InterruptAfterOne(), [state], repeats=3, output=tmp_path)
    partial = list(read_jsonl(tmp_path / "raw.jsonl"))
    assert len(partial) == 1 and partial[0]["status"] == "ok"
    assert partial[0]["js"] == 0
    rows = repeat_diagnostics(core, [state], repeats=2, output=tmp_path)
    assert len(list(read_jsonl(tmp_path / "raw.jsonl"))) == 2
    assert all(r["js"] == 0 and r["top1_flip"] == 0 for r in rows)
    output = capsys.readouterr().out
    assert "START 1/3" in output and "DONE 1/3" in output and "ETA=" in output


def test_diagnostic_errors_are_visible_and_saved(tmp_path, capsys):
    q = fixture_questions()[2]
    state = State(q.id, q.group_id, q.question, Pool(q.id, q.evidence, "fixture"))
    class Broken:
        device = torch.device("cpu")
        def eval(self): return self
        def __call__(self, *args, **kwargs): raise RuntimeError("TEST_ONLY_kernel_failure")
    rows = repeat_diagnostics(Broken(), [state], repeats=1, output=tmp_path)
    assert rows[0]["status"] == "error"
    assert "TEST_ONLY_kernel_failure" in capsys.readouterr().out
    assert list(read_jsonl(tmp_path / "raw.jsonl"))[0]["status"] == "error"


def test_efficiency_uses_actual_training_k_for_extrapolation(tmp_path, monkeypatch):
    from sa_jev import experiments as ex
    from sa_jev.retrieval import PoolCache
    q = fixture_questions()[-1]
    es = tuple(Evidence.make(str(i), f"distinct evidence {i}") for i in range(21))
    PoolCache(tmp_path / "data/pools_max").write(Pool(q.id, es, "fixture"))
    write_json(tmp_path / "data/task_manifest.json", {"actual_k_counts": {"8": 5, "20": 3}})
    monkeypatch.setattr(ex, "read_questions", lambda root: [q])
    class Model:
        device = torch.device("cpu")
    monkeypatch.setattr(ex, "load_method", lambda *args: (Model(), "dynamic"))
    monkeypatch.setattr(ex, "export_report", lambda *args: None)
    monkeypatch.setattr(ex, "evaluate", lambda model, questions, pools, output, **kwargs:
                        [{"k": len(pools[q.id].candidates)}])
    rows = ex.efficiency_experiment(tmp_path, {"evaluation": {"k_values": [16, 32]}, "data": {"budget": 4}})
    assert rows[0]["extrapolation"] is False
    assert rows[1]["extrapolation"] is True and rows[1]["training_max_k"] == 20
    assert rows[1]["exact_k"] is False
