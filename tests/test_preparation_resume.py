import asyncio
from types import SimpleNamespace

import pytest

from sa_jev.io import read_json, read_jsonl, write_json, write_jsonl
from sa_jev.retrieval import ingest_corpus
from sa_jev.schema import Evidence
from sa_jev.io import digest
from sa_jev.preparation import finish_preparation


def test_interrupted_writer_keeps_complete_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"complete": True}])

    def broken_download():
        yield {"partial": True}
        raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError): write_jsonl(path, broken_download())
    assert list(read_jsonl(path)) == [{"complete": True}]
    target = tmp_path / "manifest.json"
    write_json(target, {"complete": True})
    with pytest.raises(TypeError): write_json(target, {"unfinished": object()})
    assert read_json(target) == {"complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_ingestion_resumes_without_false_completion(tmp_path):
    pytest.importorskip("lightrag")
    docs = [Evidence.make("a", "First paragraph."), Evidence.make("b", "Second paragraph.")]
    statuses = {}; calls = []

    async def get_by_ids(ids): return [statuses.get(i) for i in ids]

    async def insert(texts, ids, file_paths):
        calls.append(ids)
        for i in ids: statuses[i] = {"status": "processed"}
        if len(calls) == 1: statuses[docs[1].id] = {"status": "failed"}

    rag = SimpleNamespace(doc_status=SimpleNamespace(get_by_ids=get_by_ids), ainsert=insert)
    path = tmp_path / "corpus_manifest.json"
    with pytest.raises(RuntimeError, match="NOT marked complete"):
        asyncio.run(ingest_corpus(rag, docs, path, {}))
    assert not path.exists()
    binding = asyncio.run(ingest_corpus(rag, docs, path, {}))
    assert calls == [[d.id for d in docs], [docs[1].id]]
    assert read_json(path)["fingerprint"] == binding
    assert asyncio.run(ingest_corpus(rag, docs, path, {})) == binding
    assert len(calls) == 2


def test_completion_marker_requires_finished_matching_stages(tmp_path):
    cfg = {"retrieval": {"llm_model": "fixture"}, "data": {"main_k": 8}}
    data = tmp_path / "data"
    write_json(data / "data_manifest.json", {"question_fingerprint": "questions", "download": {"formal": True}})
    write_json(data / "task_manifest.json", {"training_questions": 2, "task_states": 6,
                                            "validation_states": 2, "preferences": 1})
    write_json(data / "overlap_audit.json", {"question_fingerprint": "questions", "complete_scan": True})
    pools = {"complete": False, "questions": 1, "question_fingerprint": "questions",
             "config_fingerprint": digest({"retrieval": cfg["retrieval"], "main_k": 8})}
    write_json(data / "pool_cache_manifest.json", pools)
    write_json(tmp_path / "index/corpus_manifest.json", {})
    with pytest.raises(ValueError, match="candidate preparation is incomplete"):
        finish_preparation(tmp_path, cfg, mode="full")
    assert not (tmp_path / "00_complete.json").exists()
    pools["complete"] = True
    write_json(data / "pool_cache_manifest.json", pools)
    for name in ("train_states.jsonl", "validation_states.jsonl", "preferences.jsonl", "perturbations_draft.jsonl"):
        write_jsonl(data / name, [{"fixture": True}])
    result = finish_preparation(tmp_path, cfg, mode="full")
    assert result["status"] == "data_preparation_complete"
    assert read_json(tmp_path / "00_complete.json") == result
