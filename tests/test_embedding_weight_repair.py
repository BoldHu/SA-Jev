import hashlib
from pathlib import Path

import pytest

from sa_jev import embeddings


@pytest.fixture
def weight_fixture(tmp_path, monkeypatch):
    """Use tiny stand-in bytes; a separate integration check loads the real BGE."""
    good = b"known-good-test-weight"
    monkeypatch.setattr(embeddings, "BGE_WEIGHT_BYTES", len(good))
    monkeypatch.setattr(embeddings, "BGE_WEIGHT_SHA256", hashlib.sha256(good).hexdigest())
    path = (tmp_path / "models--BAAI--bge-small-en-v1.5" / "snapshots" /
            embeddings.BGE_REVISION / "model.safetensors")
    path.parent.mkdir(parents=True)
    return good, path


def test_intact_weight_needs_no_network(tmp_path, monkeypatch, weight_fixture):
    good, path = weight_fixture
    path.write_bytes(good)

    def forbidden(*args, **kwargs):
        raise AssertionError("valid cached weight must work offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    assert embeddings.ensure_bge_weights(tmp_path) == path


@pytest.mark.parametrize("bad", [b"version https://git-lfs.github.com/spec/v1\n", b"x" * 22])
def test_invalid_cache_repaired_without_touching_research_data(tmp_path, monkeypatch, weight_fixture, bad):
    good, path = weight_fixture
    path.write_bytes(bad)
    sentinel = tmp_path / "full" / "data" / "questions.jsonl"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep this research data", encoding="utf-8")
    calls = []

    def download(model, filename, *, revision, cache_dir, force_download):
        calls.append(force_download)
        assert model == embeddings.BGE_MODEL and revision == embeddings.BGE_REVISION
        assert filename == "model.safetensors" and Path(cache_dir) == tmp_path
        assert not path.exists(), "stale regular snapshot file must be moved before HF download"
        path.write_bytes(good)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert embeddings.ensure_bge_weights(tmp_path) == path
    assert calls == [True] and path.read_bytes() == good
    assert next(path.parent.glob("*.invalid-*")).read_bytes() == bad
    assert sentinel.read_text(encoding="utf-8") == "keep this research data"


def test_bad_redownload_fails_without_loading_or_looping(tmp_path, monkeypatch, weight_fixture):
    good, path = weight_fixture
    path.write_bytes(b"bad")
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs["force_download"])
        path.write_bytes(b"still bad")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(RuntimeError, match="still fails integrity validation"):
        embeddings.ensure_bge_weights(tmp_path)
    assert calls == [True]
