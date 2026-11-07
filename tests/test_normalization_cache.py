import pytest

from sa_jev import preparation
from sa_jev.io import read_json, write_json, write_jsonl


def row(i):
    return {"_id": str(i), "question": f"Who founded company {i}?", "answer": "Ada",
            "context": [[f"Company {i}", [f"Ada founded company {i}."]]],
            "supporting_facts": [[f"Company {i}", 0]]}


@pytest.fixture
def prepared(tmp_path):
    raw, data = tmp_path/"raw", tmp_path/"data"
    write_jsonl(raw/"hotpot_train.jsonl", [row(0), row(1)])
    questions, manifest = preparation.normalize_sources(raw, data)
    return raw, data, questions, manifest


def test_cache_reuse_skips_processing_and_writes_and_can_skip_loading(prepared, monkeypatch):
    raw, data, original_questions, original_manifest = prepared
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("cache reuse must not process or rewrite the source")

    monkeypatch.setattr(preparation, "load_raw", forbidden)
    monkeypatch.setattr(preparation, "split_questions", forbidden)
    monkeypatch.setattr(preparation, "deduplicate", forbidden)
    monkeypatch.setattr(preparation, "write_jsonl", forbidden)
    questions, manifest = preparation.normalize_sources(raw, data)
    assert questions == original_questions and manifest == original_manifest
    monkeypatch.setattr(preparation, "read_jsonl", forbidden)
    assert preparation.normalize_sources(raw, data, load_questions=False) == (None, manifest)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in data.iterdir()}


@pytest.mark.parametrize("change", ["source", "seed", "family_links", "missing_output", "truncated_output", "legacy", "force"])
def test_changed_or_incomplete_cache_rebuilds(prepared, monkeypatch, change):
    raw, data, _, old_manifest = prepared
    kwargs = {}
    if change == "source":
        write_jsonl(raw/"hotpot_train.jsonl", [row(0), row(1), row(2)])
    elif change == "seed":
        kwargs["seed"] = 18
    elif change == "family_links":
        kwargs["family_links"] = [("hotpotqa:0", "hotpotqa:1")]
    elif change == "missing_output":
        (data/"corpus.jsonl").unlink()
    elif change == "truncated_output":
        (data/"questions.jsonl").write_text("", encoding="utf-8")
    elif change == "legacy":
        old_manifest.pop("normalization_cache")
        write_json(data/"data_manifest.json", old_manifest)
    elif change == "force":
        kwargs["reuse"] = False
    calls = []
    actual = preparation.load_raw

    def tracked(path):
        calls.append(path)
        return actual(path)

    monkeypatch.setattr(preparation, "load_raw", tracked)
    questions, manifest = preparation.normalize_sources(raw, data, **kwargs)
    assert calls and len(questions) == (3 if change == "source" else 2)
    assert manifest["seed"] == kwargs.get("seed", 17)
    assert manifest["family_links"] == len(kwargs.get("family_links", ()))
    if change == "family_links":
        assert len({q.group_id for q in questions}) == 1
    if change in {"missing_output", "truncated_output", "legacy", "force"}:
        assert manifest["question_fingerprint"] == old_manifest["question_fingerprint"]
        assert manifest["corpus_fingerprint"] == old_manifest["corpus_fingerprint"]


def test_interrupted_rebuild_is_not_treated_as_complete(prepared, monkeypatch):
    raw, data, _, _ = prepared
    previous_marker = (data/"data_manifest.json").read_bytes()
    write_jsonl(raw/"hotpot_train.jsonl", [row(0), row(1), row(2)])
    original_write = preparation.write_jsonl

    def interrupted(path, values):
        if path.name == "corpus.jsonl":
            raise OSError("interrupted before corpus publication")
        return original_write(path, values)

    monkeypatch.setattr(preparation, "write_jsonl", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        preparation.normalize_sources(raw, data)
    assert (data/"data_manifest.json").read_bytes() == previous_marker
    monkeypatch.setattr(preparation, "write_jsonl", original_write)
    questions, manifest = preparation.normalize_sources(raw, data)
    assert len(questions) == manifest["questions"] == 3
    assert read_json(data/"data_manifest.json") == manifest
