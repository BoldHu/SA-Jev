"""Small native LightRAG check with real local embeddings and optional live LLM.

python scripts/check_local_embeddings.py --live-llm
No training weights or benchmark results are produced. Uses two synthetic documents.
"""
import argparse
import asyncio
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from unittest.mock import patch

from sa_jev.embeddings import check_embeddings
from sa_jev.environment import load_environment
from sa_jev.experiments import load_config
from sa_jev.io import write_json
from sa_jev.retrieval import make_lightrag, ingest_corpus, retrieve_once, PoolCache
from sa_jev.schema import Evidence


async def run(root, config, live_llm):
    encoder = await check_embeddings(config["retrieval"], root / "encoder.json")
    args = {k: v for k, v in config["retrieval"].items() if k not in {"mode", "max_k"}}
    started = time.perf_counter()
    rag = await make_lightrag(root / "index", **args)
    try:
        docs = [Evidence.make("Ada", "Ada was born in London. London is a city in England.", "fixture://ada"),
                Evidence.make("Museum", "A museum in London has paintings.", "fixture://museum")]
        binding = await ingest_corpus(rag, docs, root / "index/corpus_manifest.json", args)
        cache = PoolCache(root / "pools")
        pool = await retrieve_once(rag, "fixture-query", "Where was Ada born?", 8, cache,
                                   corpus_fingerprint=binding, mode="mix", source_registry={e.id: e for e in docs})
        again = await retrieve_once(rag, "fixture-query", "Where was Ada born?", 8, cache,
                                    corpus_fingerprint=binding, mode="mix", source_registry={e.id: e for e in docs})
        assert pool == again and pool.candidates
        assert {e.source for e in pool.candidates} <= {e.source for e in docs}
        assert len({e.id for e in pool.candidates}) == len(pool.candidates)
        result = {"encoder": encoder, "live_llm": live_llm, "llm_model": args["llm_model"],
                  "native_ingestion": True, "native_mix_retrieval": True,
                  "fixed_cache_roundtrip": True, "source_traceability": True,
                  "returned_candidates": len(pool.candidates),
                  "pipeline_seconds": time.perf_counter()-started, "test_only": True}
        write_json(root / "verification.json", result)
        print(f"Native local-embedding check passed: {len(pool.candidates)} candidates. Results: {root}", flush=True)
    finally:
        await rag.finalize_storages()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-llm", action="store_true", help="Use the configured chat API on two toy documents")
    parser.add_argument("--root", type=Path, default=Path("outputs/local_embedding_check"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="run_", dir=args.root))
    os.environ.setdefault("HF_HOME", str(Path("outputs/huggingface").resolve()))
    config = load_config()
    if args.live_llm:
        load_environment(Path(os.environ.get("SAJEV_ENV_FILE", ".env")))
        asyncio.run(run(root, config, True))
    else:
        from check_lightrag_local import completion

        async def fixture_completion(model, prompt, **kwargs):
            return await completion(prompt, **kwargs)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "unused-fixture-key"}), \
             patch("lightrag.llm.openai.openai_complete_if_cache", new=fixture_completion):
            asyncio.run(run(root, config, False))


if __name__ == "__main__":
    main()
