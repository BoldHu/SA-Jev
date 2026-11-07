"""One-time native LightRAG retrieval and immutable candidate caches."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import os
import re
import time
from .io import digest, read_json, write_json
from .schema import Evidence, Pool, deduplicate
from .embeddings import BGE_MODEL, BGE_REVISION, BGE_QUERY_PREFIX, make_embedding_func


class PoolCache:
    def __init__(self, directory):
        self.directory = Path(directory); self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, question_id): return self.directory / (digest(question_id) + ".json")

    def write(self, pool):
        path = self.path(pool.question_id)
        payload = {"fingerprint": pool.fingerprint, "pool": asdict(pool)}
        if path.exists() and read_json(path) != payload:
            raise ValueError("fixed candidate cache already exists with different contents")
        write_json(path, payload)

    def read(self, question_id):
        payload = read_json(self.path(question_id)); pool = Pool.from_dict(payload["pool"])
        if pool.fingerprint != payload["fingerprint"]: raise ValueError("candidate cache hash mismatch")
        return pool


def require_retrieved(pool):
    if pool.retriever != "lightrag" or pool.provenance.get("gold_injected") is not False:
        raise ValueError("formal evaluation requires actual LightRAG retrieval without gold injection")
    if not pool.provenance.get("corpus_fingerprint") or not pool.provenance.get("retrieval_config"):
        raise ValueError("formal evaluation pool is missing corpus/retrieval provenance")


async def retrieve_once(rag, question_id, question_text, k, cache, *, corpus_fingerprint, mode="mix", extra=None, source_registry=None):
    """mix preserves graph + vector retrieval; hybrid preserves local + global graph retrieval."""
    if mode not in {"mix", "hybrid"}: raise ValueError("keep native mixed/hybrid retrieval")
    config = {"mode": mode, "top_k": k, "chunk_top_k": k, "enable_rerank": False, **(extra or {})}
    provenance = {"corpus_fingerprint": corpus_fingerprint, "query_hash": digest(question_text),
                  "retrieval_config": config, "gold_injected": False,
                  "lightrag_version": importlib.metadata.version("lightrag-hku")}
    path = cache.path(question_id)
    if path.exists():
        pool = cache.read(question_id)
        if pool.provenance != {**provenance, "returned_count": len(pool.candidates)}:
            raise ValueError("cached pool belongs to a different query/corpus/configuration")
        return pool
    from lightrag import QueryParam
    response = await rag.aquery_data(question_text, param=QueryParam(**config))
    if response.get("status") != "success":
        raise RuntimeError(f"LightRAG retrieval failed: {response.get('message')}")
    evidence = []
    for chunk in response["data"]["chunks"]:
        if not chunk.get("chunk_id"): raise ValueError("LightRAG chunk missing traceable ID")
        source_id = chunk.get("file_path", "")
        original = (source_registry or {}).get(source_id)
        if source_registry is not None and original is None:
            raise ValueError(f"retrieved source ID is not in the fixed corpus registry: {source_id}")
        evidence.append(Evidence(str(chunk["chunk_id"]), original.title if original else "", chunk["content"],
                                 original.source if original else source_id))
    candidates = deduplicate(evidence)[:k]
    pool = Pool(question_id, candidates, "lightrag", {**provenance, "returned_count": len(candidates)})
    cache.write(pool)
    return pool


async def make_lightrag(directory, *, llm_model="DeepSeek-V4.1-Flash", embedding_model=BGE_MODEL,
                        embedding_dim=384, embedding_backend="local", embedding_revision=BGE_REVISION,
                        embedding_device="auto", embedding_batch_size=16, embedding_max_length=512,
                        embedding_query_prefix=BGE_QUERY_PREFIX, chunk_tokens=512, chunk_overlap=64):
    """Native mixed retrieval with a frozen local encoder and an API extraction LLM."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Fill it in MyDrive/SA-Jev/.env (Colab) or the project "
            ".env file (local), then rerun the third notebook code cell before cache_pools. "
            "Use the extraction LLM provider's key; local embeddings need no API key."
        )
    from lightrag import LightRAG
    from lightrag.llm.openai import openai_complete_if_cache
    from lightrag.kg.shared_storage import initialize_pipeline_status
    base_url = os.environ.get("OPENAI_BASE_URL")

    async def completion(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(llm_model, prompt, system_prompt=system_prompt,
                    history_messages=history_messages or [], api_key=api_key, base_url=base_url, **kwargs)

    identity = {"llm_model": llm_model, "embedding_backend": embedding_backend,
                "embedding_model": embedding_model, "embedding_revision": embedding_revision,
                "embedding_dim": embedding_dim, "embedding_max_length": embedding_max_length,
                "embedding_query_prefix": embedding_query_prefix,
                "chunk_tokens": chunk_tokens, "chunk_overlap": chunk_overlap}
    directory = Path(directory)
    binding_path = directory / "index_config.json"
    if binding_path.exists():
        if read_json(binding_path) != identity:
            raise ValueError("index model/config changed; use a fresh experiment ROOT and rebuild candidate pools")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError("existing index has no model binding (possibly an old API embedding index); "
                         "preserve it and use a fresh experiment ROOT to rebuild index and candidate pools")
    embedding = await make_embedding_func(
        backend=embedding_backend, model=embedding_model, dim=embedding_dim, revision=embedding_revision,
        device=embedding_device, batch_size=embedding_batch_size, max_length=embedding_max_length,
        query_prefix=embedding_query_prefix,
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_json(binding_path, identity)  # Bind even partially built indexes before the first insert.
    rag = LightRAG(working_dir=str(directory), llm_model_func=completion, llm_model_name=llm_model,
                   embedding_func=embedding, embedding_func_max_async=1 if embedding_backend == "local" else 8,
                   chunk_token_size=chunk_tokens, chunk_overlap_token_size=chunk_overlap)
    await rag.initialize_storages(); await initialize_pipeline_status()
    return rag


async def ingest_corpus(rag, evidence, manifest_path, config, batch_size=32):
    evidence = deduplicate(evidence)
    binding = digest({"documents": evidence, "config": config, "ingestion_format": "source_id_v2"})
    path = Path(manifest_path)
    if path.exists():
        if read_json(path)["fingerprint"] != binding: raise ValueError("index corpus/config changed")
        print(f"Reusing complete index: {len(evidence)} documents", flush=True)
        return binding
    from lightrag.base import DocStatus
    started = time.perf_counter()
    newly_processed = skipped = 0
    print(f"Indexing {len(evidence)} unique documents; completed documents will be reused", flush=True)
    for start in range(0, len(evidence), batch_size):
        batch = evidence[start:start+batch_size]
        statuses = await rag.doc_status.get_by_ids([e.id for e in batch])
        pending = [e for e, status in zip(batch, statuses)
                   if not status or status.get("status") != DocStatus.PROCESSED]
        skipped += len(batch) - len(pending)
        # Metadata contains stable source IDs, never question/gold/answer fields.
        # LightRAG normalizes file paths, which can strip URLs. Store stable IDs
        # here and restore full source URLs/titles from the fixed registry at retrieval.
        if pending:
            await rag.ainsert([e.title + "\n" + e.text for e in pending], ids=[e.id for e in pending],
                              file_paths=[e.id for e in pending])
        statuses = await rag.doc_status.get_by_ids([e.id for e in batch])
        failed = [e.id for e, status in zip(batch, statuses)
                  if not status or status.get("status") != DocStatus.PROCESSED]
        if len(statuses) != len(batch) or failed:
            raise RuntimeError(f"LightRAG has incomplete documents (example IDs: {failed[:5]}). "
                               "Index is NOT marked complete; resolve the API error and rerun this cell.")
        newly_processed += len(pending)
        done = min(start+batch_size, len(evidence))
        elapsed = time.perf_counter() - started
        rate = newly_processed / max(elapsed, 1e-9)
        eta = (len(evidence)-done) / rate if rate else None
        progress = {"documents": len(evidence), "completed_prefix": done, "reused": skipped,
                    "newly_processed": newly_processed, "elapsed_seconds": elapsed,
                    "new_documents_per_minute": rate*60, "eta_seconds": eta,
                    "eta_scope": "remaining documents assuming uncached; excludes candidate queries and task preparation"}
        write_json(path.parent / "ingestion_progress.json", progress)
        eta_text = f"{eta/3600:.2f} h" if eta is not None else "unknown (cache only)"
        print(f"indexed {done}/{len(evidence)}; reused={skipped}; "
              f"new={rate*60:.1f} docs/min; estimated remaining={eta_text}", flush=True)
    write_json(path, {"fingerprint": binding, "documents": len(evidence), "config": config})
    return binding


def hard_candidates(questions, corpus, k=64, dense=False, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Training-only mining. No negative label is inferred merely from retrieval."""
    from rank_bm25 import BM25Okapi
    import numpy as np
    corpus = deduplicate(corpus)
    tokenize = lambda s: re.findall(r"\w+", s.lower())
    bm25 = BM25Okapi([tokenize(e.title + " " + e.text) for e in corpus])
    encoder = vectors = None
    if dense:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(embedding_model)
        vectors = encoder.encode([e.title + " " + e.text for e in corpus], normalize_embeddings=True)
    result = {}
    for q in questions:
        lexical = np.argsort(-bm25.get_scores(tokenize(q.question)))[:k]
        indices = list(lexical)
        if dense:
            query = encoder.encode(q.question, normalize_embeddings=True)
            semantic = np.argsort(-(vectors @ query))[:k]
            indices = [i for pair in zip(lexical, semantic) for i in pair]
        result[q.id] = deduplicate([corpus[i] for i in indices])[:k]
    return result
