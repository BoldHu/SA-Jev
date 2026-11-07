"""Reproduce native FAILED resubmission, then verify retry and mixed retrieval offline."""
import asyncio
from pathlib import Path
from tempfile import mkdtemp

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from check_lightrag_local import completion, embedding
from sa_jev.io import write_json
from sa_jev.retrieval import ingest_corpus, retrieve_once, PoolCache
from sa_jev.schema import Evidence


async def main():
    base = Path("outputs/lightrag_failed_resume"); base.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="run_", dir=base))
    failing = True
    llm_calls = 0

    async def controlled_completion(prompt, **kwargs):
        nonlocal llm_calls
        llm_calls += 1
        if failing:
            raise RuntimeError("fixture HTTP 400: unexpected model name format")
        return await completion(prompt, **kwargs)

    rag = LightRAG(working_dir=str(root/"index"), llm_model_func=controlled_completion,
                   embedding_func=EmbeddingFunc(embedding_dim=32, max_token_size=2048, func=embedding),
                   entity_extract_max_gleaning=0, cosine_better_than_threshold=-1.0,
                   chunk_token_size=128, chunk_overlap_token_size=16)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    docs = [Evidence.make("Ada", "Ada was born in London. London is a city in England.", "fixture://ada"),
            Evidence.make("Museum", "A museum in London has paintings.", "fixture://museum")]
    ids = [doc.id for doc in docs]
    marker = root/"index/corpus_manifest.json"
    try:
        try:
            await ingest_corpus(rag, docs, marker, {"fixture": True})
        except RuntimeError as error:
            assert "unexpected model name format" in str(error)
        else:
            raise AssertionError("injected extraction failure must be surfaced")
        assert not marker.exists()
        assert all(s["status"] == "failed" for s in await rag.doc_status.get_by_ids(ids))
        failing = False
        before = llm_calls
        # Reproduce the previous SA-Jev retry behavior against actual LightRAG.
        await rag.ainsert([d.title+"\n"+d.text for d in docs], ids=ids, file_paths=ids)
        assert llm_calls == before
        assert all(s["status"] == "failed" for s in await rag.doc_status.get_by_ids(ids))
        binding = await ingest_corpus(rag, docs, marker, {"fixture": True})
        assert marker.is_file() and llm_calls > before
        assert all(s["status"] == "processed" for s in await rag.doc_status.get_by_ids(ids))
        after = llm_calls
        await ingest_corpus(rag, docs, marker, {"fixture": True})
        assert llm_calls == after
        pool = await retrieve_once(rag, "fixture-question", "Where was Ada born?", 8, PoolCache(root/"pools"),
                                   corpus_fingerprint=binding, source_registry={d.id: d for d in docs})
        assert pool.candidates and {e.source for e in pool.candidates} <= {d.source for d in docs}
        write_json(root/"verification.json", {
            "naive_ainsert_does_not_retry_failed": True, "native_retry_restores_processed": True,
            "complete_marker_guarded": True, "processed_reuse_no_extra_llm_call": True,
            "native_mix_retrieval": True, "source_traceability": True,
            "returned_candidates": len(pool.candidates), "external_api_calls": 0,
            "scope": "real LightRAG; deterministic fixture LLM and embeddings, not paper results",
        })
        print("Native failure/resume check passed:", root/"verification.json", flush=True)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
