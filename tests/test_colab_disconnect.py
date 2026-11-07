import sys
from types import ModuleType, SimpleNamespace

import pytest

from sa_jev.environment import disconnect_after_preparation
from sa_jev.io import write_json


def fake_colab(monkeypatch, *, flush, unassign):
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(flush_and_unmount=flush)
    colab.runtime = SimpleNamespace(unassign=unassign)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_flush_precedes_release_and_marker_must_match_run(tmp_path, monkeypatch):
    calls = []
    fake_colab(monkeypatch, flush=lambda: calls.append("flush"), unassign=lambda: calls.append("unassign"))
    marker = {"status": "data_preparation_complete", "run_id": "previous-run"}
    write_json(tmp_path / "00_complete.json", marker)
    with pytest.raises(RuntimeError, match="has not completed"):
        disconnect_after_preparation(tmp_path, run_id="current-run")
    assert calls == []
    marker["run_id"] = "current-run"
    write_json(tmp_path / "00_complete.json", marker)
    assert disconnect_after_preparation(tmp_path, run_id="current-run")
    assert calls == ["flush", "unassign"]


def test_flush_failure_keeps_runtime(tmp_path, monkeypatch):
    calls = []

    def failed_flush():
        calls.append("flush")
        raise OSError("Drive upload failed")

    fake_colab(monkeypatch, flush=failed_flush, unassign=lambda: calls.append("unassign"))
    write_json(tmp_path / "00_complete.json", {"status": "data_preparation_complete", "run_id": "current"})
    with pytest.raises(OSError, match="Drive upload failed"):
        disconnect_after_preparation(tmp_path, run_id="current")
    assert calls == ["flush"]


def test_disabled_or_local_execution_never_releases(tmp_path, monkeypatch):
    def forbidden(): raise AssertionError("must not touch Colab")
    fake_colab(monkeypatch, flush=forbidden, unassign=forbidden)
    assert not disconnect_after_preparation(tmp_path, run_id="current", enabled=False)
    monkeypatch.delitem(sys.modules, "google.colab")
    assert not disconnect_after_preparation(tmp_path, run_id="current")
