import asyncio
import threading
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from sa_jev.embeddings import make_embedding_func
from sa_jev.io import write_json
from sa_jev.retrieval import make_lightrag


def test_local_embeddings_need_no_key_and_keep_context_and_order(monkeypatch):
    st = pytest.importorskip("sentence_transformers")
    remote = pytest.importorskip("lightrag.llm.openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    forbidden = AsyncMock(side_effect=AssertionError("embedding API must not be called"))
    monkeypatch.setattr(remote, "openai_embed", forbidden)
    calls = []
    busy = threading.Lock()

    class Encoder:
        max_seq_length = 512
        def __init__(self, *args, **kwargs): pass
        def get_embedding_dimension(self): return 3
        def eval(self): return self
        def requires_grad_(self, enabled): assert not enabled
        def encode(self, texts, *, prompt, batch_size, normalize_embeddings, **kwargs):
            assert busy.acquire(blocking=False), "concurrent inference on one encoder"
            try:
                time.sleep(.01)
                calls.append((texts, prompt))
                assert batch_size == 2 and normalize_embeddings
                # Distinct vectors reveal any reordering or query/document prompt mix-up.
                rows = np.array([[len(t), len(prompt)+1, 1] for t in texts], dtype=np.float32)
                return rows / np.linalg.norm(rows, axis=1, keepdims=True)
            finally:
                busy.release()

    monkeypatch.setattr(st, "SentenceTransformer", Encoder)

    async def run():
        fn = await make_embedding_func(model="fixture", revision="fixed", dim=3,
                                       device="cpu", batch_size=2, query_prefix="query: ")
        docs, query = await asyncio.gather(fn(["a", "long"], context="document"),
                                          fn(["a"], context="query"))
        assert (await fn([], context="document")).shape == (0, 3)
        assert docs.shape == (2, 3) and query.shape == (1, 3)
        assert docs[0, 0] < docs[1, 0] and not np.allclose(query[0], docs[0])
        assert calls == [(["a", "long"], ""), (["a"], "query: ")]
        with pytest.raises(ValueError, match="unknown embedding context"):
            await fn(["a"], context="wrong")
        with pytest.raises(ValueError, match="does not match encoder dimension"):
            await make_embedding_func(model="fixture", dim=384)
    asyncio.run(run())
    forbidden.assert_not_called()


def test_changed_or_legacy_index_rejected_before_loading(monkeypatch, tmp_path):
    pytest.importorskip("lightrag")
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    forbidden = AsyncMock(side_effect=AssertionError("must reject before loading model"))
    monkeypatch.setattr("sa_jev.retrieval.make_embedding_func", forbidden)
    write_json(tmp_path / "index_config.json", {"embedding_model": "old-model"})
    with pytest.raises(ValueError, match="index model/config changed"):
        asyncio.run(make_lightrag(tmp_path))
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "old_vectors.json").write_text("{}")
    with pytest.raises(ValueError, match="existing index has no model binding"):
        asyncio.run(make_lightrag(legacy))
    forbidden.assert_not_called()


def test_deepseek_google_endpoint_mismatch_fails_before_model_or_api(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://user:password@generativelanguage.googleapis.com/v1beta/openai/?key=private-test-token")
    forbidden = AsyncMock(side_effect=AssertionError("do not load encoder for a known bad service pairing"))
    monkeypatch.setattr("sa_jev.retrieval.make_embedding_func", forbidden)
    with pytest.raises(ValueError, match="endpoint/model mismatch"):
        asyncio.run(make_lightrag(tmp_path, llm_model="DeepSeek-V4.1-Flash"))
    forbidden.assert_not_called()
    assert not tmp_path.joinpath("index_config.json").exists()
    text = capsys.readouterr().out
    assert "generativelanguage.googleapis.com" in text and "DeepSeek-V4.1-Flash" in text
    assert "password" not in text and "private-test-token" not in text
