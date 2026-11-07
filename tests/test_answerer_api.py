"""Replay the Google seed rejection without API credentials or paid requests."""
import json
from types import SimpleNamespace

import pytest

from sa_jev.evaluation import Answerer
from sa_jev.schema import Evidence


def test_google_seed_compatibility_and_provider_cache_isolation(monkeypatch, tmp_path):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    real_client = openai.OpenAI
    state = SimpleNamespace(question="Where was Ada born?", conditions="",
                            selected=[Evidence.make("Ada", "Ada was born in London.", "fixture://ada")],
                            selected_ids=("test-evidence",))
    seen = []

    def handler(request):
        body = json.loads(request.content)
        google = request.url.host == "generativelanguage.googleapis.com"
        if google and "seed" in body:
            return httpx.Response(400, json={"error": {"message": 'Unknown name "seed"',
                                                       "type": "INVALID_ARGUMENT"}})
        if not google:
            assert body["seed"] == 17
        seen.append(request.url.host)
        return httpx.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "London"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        })

    results = []
    for endpoint in ("https://generativelanguage.googleapis.com/v1beta/openai/", "https://relay.example/v1"):
        with real_client(api_key="unused-fixture-key", base_url=endpoint, max_retries=0,
                         http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
            monkeypatch.setattr(openai, "OpenAI", lambda: client)
            answerer = Answerer("same-model-alias", tmp_path)
            result = answerer(state)
            assert result["answer"] == "London"
            assert answerer(state) == result  # Same endpoint reuses its own cache.
            results.append(result)
    assert seen == ["generativelanguage.googleapis.com", "relay.example"]
    assert "seed" not in results[0]["request_options"]
    assert results[1]["request_options"]["seed"] == 17
    assert results[0]["endpoint_hash"] != results[1]["endpoint_hash"]
