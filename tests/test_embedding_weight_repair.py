import hashlib
from pathlib import Path

import pytest

from sa_jev import embeddings


@pytest.fixture(params=["BGE", "MINILM"])
def weight_fixture(tmp_path, monkeypatch, request):
    """Use tiny stand-in bytes for both pinned encoders; integration loads real weights."""
    good = b"known-good-test-weight"
    prefix = request.param
    model = getattr(embeddings, prefix+"_MODEL")
    revision = getattr(embeddings, prefix+"_REVISION")
    monkeypatch.setattr(embeddings, prefix+"_WEIGHT_BYTES", len(good))
    monkeypatch.setattr(embeddings, prefix+"_WEIGHT_SHA256", hashlib.sha256(good).hexdigest())
    path = (tmp_path / ("models--"+model.replace("/", "--")) / "snapshots" /
            revision / "model.safetensors")
    path.parent.mkdir(parents=True)
    return good, path, model, revision


def test_intact_weight_needs_no_network(tmp_path, monkeypatch, weight_fixture):
    good, path, model, revision = weight_fixture
    path.write_bytes(good)

    def forbidden(*args, **kwargs):
        raise AssertionError("valid cached weight must work offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    assert embeddings.ensure_embedding_weights(model, revision, tmp_path) == path


@pytest.mark.parametrize("bad", [b"version https://git-lfs.github.com/spec/v1\n", None])
def test_invalid_cache_repaired_without_touching_research_data(tmp_path, monkeypatch, weight_fixture, bad):
    good, path, model, revision = weight_fixture
    if bad is None:
        bad = b"x" * len(good)  # same size must still fail the SHA-256 check
    path.write_bytes(bad)
    sentinel = tmp_path / "full" / "data" / "questions.jsonl"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("keep this research data", encoding="utf-8")
    calls = []

    def download(repo_id, filename, *, revision, cache_dir, force_download):
        calls.append(force_download)
        assert repo_id == model and revision == weight_fixture[3]
        assert filename == "model.safetensors" and Path(cache_dir) == tmp_path
        assert not path.exists(), "stale regular snapshot file must be moved before HF download"
        path.write_bytes(good)
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert embeddings.ensure_embedding_weights(model, revision, tmp_path) == path
    assert calls == [True] and path.read_bytes() == good
    assert next(path.parent.glob("*.invalid-*")).read_bytes() == bad
    assert sentinel.read_text(encoding="utf-8") == "keep this research data"


def test_bad_redownload_fails_without_loading_or_looping(tmp_path, monkeypatch, weight_fixture):
    good, path, model, revision = weight_fixture
    path.write_bytes(b"bad")
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs["force_download"])
        path.write_bytes(b"still bad")
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(RuntimeError, match="still fails integrity validation"):
        embeddings.ensure_embedding_weights(model, revision, tmp_path)
    assert calls == [True]


def test_minilm_is_checked_before_sentence_transformer_loading(monkeypatch, tmp_path):
    import asyncio
    import sentence_transformers
    events = []

    def validate(model, revision, cache):
        assert (model, revision) == (embeddings.MINILM_MODEL, embeddings.MINILM_REVISION)
        events.append("verified")
        return tmp_path / "persistent" / "model.safetensors"

    def stage(model, revision, weight, cache):
        assert events == ["verified"]
        assert weight == tmp_path / "persistent" / "model.safetensors"
        events.append("staged")
        return tmp_path / "local"

    def load(*args, **kwargs):
        assert events == ["verified", "staged"]
        assert args[0] == str(tmp_path / "local")
        assert kwargs["local_files_only"] is True
        assert "revision" not in kwargs and "cache_folder" not in kwargs
        raise RuntimeError("stop after verified weight reaches loader")

    monkeypatch.setattr(embeddings, "ensure_embedding_weights", validate)
    monkeypatch.setattr(embeddings, "stage_embedding_snapshot", stage)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", load)
    with pytest.raises(RuntimeError, match="stop after verified"):
        asyncio.run(embeddings.make_embedding_func(model=embeddings.MINILM_MODEL,
                    revision=embeddings.MINILM_REVISION, device="cpu", max_length=256))


def test_local_snapshot_materializes_verified_bytes_and_reuses_weight(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file, load_file
    from huggingface_hub.errors import LocalEntryNotFoundError

    source = tmp_path / "persistent" / "model.safetensors"
    source.parent.mkdir()
    save_file({"weight": torch.arange(6).reshape(2, 3)}, source)
    original = source.read_bytes()
    monkeypatch.setattr(embeddings, "MINILM_WEIGHT_BYTES", len(original))
    monkeypatch.setattr(embeddings, "MINILM_WEIGHT_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(embeddings.tempfile, "gettempdir", lambda: str(tmp_path / "runtime"))
    metadata = source.parent / "metadata.txt"
    metadata.write_text('{}', encoding="utf-8")
    calls = []

    def download(**kwargs):
        assert kwargs["filename"] != "model.safetensors", "staging must reuse verified weights"
        calls.append((kwargs["filename"], kwargs.get("local_files_only", False)))
        if kwargs["filename"] == "vocab.txt" and kwargs.get("local_files_only") and len(calls) < 10:
            raise LocalEntryNotFoundError("simulate uncached small tokenizer metadata")
        return str(metadata)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    model, revision = embeddings.MINILM_MODEL, embeddings.MINILM_REVISION
    local = embeddings.stage_embedding_snapshot(model, revision, source, str(source.parent))
    target = local / "model.safetensors"
    assert not target.is_symlink()
    assert target.read_bytes() == original == source.read_bytes()
    assert torch.equal(load_file(target)["weight"], torch.arange(6).reshape(2, 3))
    assert (local / "1_Pooling/config.json").is_file()
    assert ("vocab.txt", False) in calls
    timestamp = target.stat().st_mtime_ns
    embeddings.stage_embedding_snapshot(model, revision, source, str(source.parent))
    assert target.stat().st_mtime_ns == timestamp, "valid local weight should not be recopied"
    assert source.read_bytes() == original


def test_bad_runtime_copy_stops_before_loading_or_downloading_metadata(tmp_path, monkeypatch):
    source = tmp_path / "changed_source.safetensors"
    source.write_bytes(b"79-byte-pointer-or-changed-source")
    monkeypatch.setattr(embeddings.tempfile, "gettempdir", lambda: str(tmp_path / "runtime"))

    def forbidden(**kwargs):
        raise AssertionError("must stop before metadata download or model loading")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    with pytest.raises(RuntimeError, match="Local embedding copy failed checksum"):
        embeddings.stage_embedding_snapshot(embeddings.MINILM_MODEL, embeddings.MINILM_REVISION, source)
    assert source.read_bytes() == b"79-byte-pointer-or-changed-source"
