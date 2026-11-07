import io
import json
import sys
from types import SimpleNamespace
import zipfile

import pytest

from sa_jev import preparation
from sa_jev.io import read_jsonl


def test_streamed_sources_keep_limits_reuse_and_atomic_failure(tmp_path, monkeypatch, capsys):
    records = [{"id": i} for i in range(3)]
    body = b"".join(json.dumps(row).encode() + b"\n" for row in records)
    info = {"raw_data/musique_ans_train.jsonl": "train", "raw_data/musique_ans_dev.jsonl": "dev"}
    calls = []
    fail = True

    class BrokenStream(io.BytesIO):
        def __next__(self):
            if self.tell():
                raise OSError("connection interrupted")
            return super().__next__()

    def urlopen(url, timeout):
        assert timeout == 120
        calls.append(url)
        if "githubusercontent" in url:
            return io.BytesIO(json.dumps(info).encode())
        return BrokenStream(body) if fail else io.BytesIO(body)

    def dataset(*args, **kwargs):
        assert kwargs["streaming"]
        return iter(records)

    monkeypatch.setattr(preparation.urllib.request, "urlopen", urlopen)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=dataset))
    with pytest.raises(OSError, match="connection interrupted"):
        preparation.download_sources(tmp_path, limit=2)
    assert not (tmp_path / "musique_ans_train.jsonl").exists()
    assert not (tmp_path / "download_manifest.json").exists()
    assert not list(tmp_path.glob("*.tmp"))

    fail = False
    result = preparation.download_sources(tmp_path, limit=2)
    assert not result["formal"] and len(result["sources"]) == 4
    for entry in result["sources"]:
        assert list(read_jsonl(tmp_path / entry["path"])) == records[:2]
    snapshot = {p.name: p.read_bytes() for p in tmp_path.glob("*.jsonl")}
    calls.clear()
    preparation.download_sources(tmp_path, limit=2)
    assert calls == []  # Completed downloads are reusable without any network access.
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.jsonl")} == snapshot
    text = capsys.readouterr().out
    assert "Requesting musique_ans_train" in text and "2 records" in text
    assert "Reusing" in text and "ETA=unknown" in text


def test_2wiki_transfer_and_extract_preserve_bytes_and_reuse(tmp_path, monkeypatch, capsys):
    dev = b'[{"id":"fixture"}]'
    aliases = b'{"fixture":[]}'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("official/dev.json", dev)
        z.writestr("official/id_aliases.json", aliases)
    payload = stream.getvalue()
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        response = io.BytesIO(payload)
        response.headers = {"Content-Length": str(len(payload))}
        return response

    monkeypatch.setattr(preparation.urllib.request, "urlopen", urlopen)
    preparation.download_2wiki(tmp_path)
    assert (tmp_path / "2wiki_dev.json").read_bytes() == dev
    assert (tmp_path / "2wiki_id_aliases.json").read_bytes() == aliases
    assert not list(tmp_path.glob("*.part"))
    preparation.download_2wiki(tmp_path)
    assert len(calls) == 1
    text = capsys.readouterr().out
    assert "download 2Wiki ZIP" in text and "extract 2wiki_dev.json" in text
    assert "ETA=" in text and "Reusing" in text
