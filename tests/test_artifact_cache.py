import hashlib
import json
from pathlib import Path

import pytest

from sa_jev.artifact_cache import prepare_openjev_artifact


REV = "a" * 40


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    root = tmp_path / "hub/models--fixture--OpenJev/snapshots" / REV
    root.mkdir(parents=True)
    data = {"backbone/model-00001-of-00003.safetensors": b"good shard one",
            "backbone/model-00002-of-00003.safetensors": b"good shard two",
            "backbone/model-00003-of-00003.safetensors": b"good shard three",
            "decision_head.safetensors": b"pretrained head",
            "calibration.json": b'{"temperature":1.0}'}
    for name, content in data.items():
        path = root / name; path.parent.mkdir(exist_ok=True); path.write_bytes(content)
    manifest = {"artifact_format_version": 1, "files": {
        name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in data.items()}}
    raw = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(raw)
    (root / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr("openjev.runtime.hub.resolve_artifact", lambda *a, **kw: str(root))
    calls = []
    def download(repo, name, *, revision, cache_dir, force_download):
        assert repo == "fixture/OpenJev" and revision == REV and force_download is True
        assert Path(cache_dir) == root.parent.parent.parent
        calls.append(name)
        (root / name).write_bytes(data[name])
        return str(root / name)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return root, data, calls


def test_only_broken_shard_redownloaded_and_preserved(artifact):
    root, data, calls = artifact
    name = "backbone/model-00003-of-00003.safetensors"
    (root / name).write_bytes(b"incomplete")
    manifest_before = (root / "manifest.json").read_bytes()
    first_before = (root / next(iter(data))).stat().st_mtime_ns
    assert prepare_openjev_artifact("fixture/OpenJev", REV) == str(root)
    assert calls == [name]
    assert next((root / "backbone").glob("*.invalid-*")).read_bytes() == b"incomplete"
    assert (root / "manifest.json").read_bytes() == manifest_before
    assert (root / next(iter(data))).stat().st_mtime_ns == first_before
    prepare_openjev_artifact("fixture/OpenJev", REV)
    assert calls == [name]


def test_same_size_corruption_repaired(artifact):
    root, data, calls = artifact
    name = "decision_head.safetensors"
    (root / name).write_bytes(b"x" * len(data[name]))
    prepare_openjev_artifact("fixture/OpenJev", REV)
    assert calls == [name] and (root / name).read_bytes() == data[name]


def test_missing_file_repaired(artifact):
    root, data, calls = artifact
    name = "backbone/model-00003-of-00003.safetensors"
    (root / name).unlink()
    prepare_openjev_artifact("fixture/OpenJev", REV)
    assert calls == [name]


def test_stage_verified_ordinary_files_and_reuse(artifact, tmp_path):
    root, data, calls = artifact
    local = tmp_path / "runtime"
    result = prepare_openjev_artifact("fixture/OpenJev", REV, local_directory=local)
    assert Path(result) == local and not calls
    for name, content in data.items():
        assert (local / name).read_bytes() == content and not (local / name).is_symlink()
    # Once verified locally, a later source failure must not invalidate the local copy.
    name = "backbone/model-00003-of-00003.safetensors"
    (root / name).write_bytes(b"bad Drive entry")
    prepare_openjev_artifact("fixture/OpenJev", REV, local_directory=local)
    assert not calls
    # If local copy also fails, repair the persistent source and re-stage it.
    (local / name).write_bytes(b"bad local entry")
    prepare_openjev_artifact("fixture/OpenJev", REV, local_directory=local)
    assert calls == [name] and (local / name).read_bytes() == data[name]


def test_retry_still_bad_stops(artifact, monkeypatch):
    root, data, calls = artifact
    name = "decision_head.safetensors"
    (root / name).write_bytes(b"bad")
    def fail(*a, **kw):
        calls.append(name)
        (root / name).write_bytes(b"still bad")
        return str(root / name)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail)
    with pytest.raises(RuntimeError, match="after one repair"):
        prepare_openjev_artifact("fixture/OpenJev", REV)
    assert calls == [name]


def test_bad_manifest_and_unsafe_paths_never_trigger_download(artifact):
    root, data, calls = artifact
    (root / "COMPLETE").write_text("wrong")
    with pytest.raises(ValueError, match="manifest/COMPLETE"):
        prepare_openjev_artifact("fixture/OpenJev", REV)
    m = json.loads((root / "manifest.json").read_text())
    m["files"]["../outside"] = next(iter(m["files"].values()))
    raw = json.dumps(m).encode(); (root / "manifest.json").write_bytes(raw)
    (root / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="invalid artifact manifest entry"):
        prepare_openjev_artifact("fixture/OpenJev", REV)
    assert not calls


def test_user_local_artifact_is_never_repaired_from_hub(artifact):
    root, data, calls = artifact
    (root / "decision_head.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="missing or wrong size"):
        prepare_openjev_artifact(str(root))
    assert not calls


def test_model_load_uses_verified_path(monkeypatch):
    from sa_jev.model import SAJev
    called = []
    monkeypatch.setattr("sa_jev.artifact_cache.prepare_openjev_artifact", lambda *a, **kw: "/verified/runtime")
    def loader(path, **kwargs):
        called.append(path)
        raise RuntimeError("TEST_ONLY_stop_before_model_allocation")
    monkeypatch.setattr("openjev.artifacts.load_decision_artifact", loader)
    with pytest.raises(RuntimeError, match="TEST_ONLY"):
        SAJev.from_pretrained("fixture/OpenJev", revision=REV)
    assert called == ["/verified/runtime"]
