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


def test_minilm_is_checked_before_sentence_transformer_loading(monkeypatch):
    import asyncio
    import sentence_transformers
    events = []

    def validate(model, revision, cache):
        assert (model, revision) == (embeddings.MINILM_MODEL, embeddings.MINILM_REVISION)
        events.append("verified")

    def load(*args, **kwargs):
        assert events == ["verified"]
        raise RuntimeError("stop after verified weight reaches loader")

    monkeypatch.setattr(embeddings, "ensure_embedding_weights", validate)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", load)
    with pytest.raises(RuntimeError, match="stop after verified"):
        asyncio.run(embeddings.make_embedding_func(model=embeddings.MINILM_MODEL,
                    revision=embeddings.MINILM_REVISION, device="cpu", max_length=256))
