"""Frozen local text embeddings for native LightRAG; no embedding API required."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time

import numpy as np

from .io import digest, write_json

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


async def make_embedding_func(*, backend="local", model=BGE_MODEL, dim=384,
                              revision=BGE_REVISION, device="auto", batch_size=16,
                              max_length=512, query_prefix=BGE_QUERY_PREFIX):
    from lightrag.utils import EmbeddingFunc

    if backend == "openai":
        from lightrag.llm.openai import openai_embed
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise ValueError("OPENAI_API_KEY is required for embedding_backend=openai")

        async def remote(texts):
            fn = getattr(openai_embed, "func", openai_embed)
            return await fn(texts, model=model, api_key=key,
                            base_url=os.environ.get("OPENAI_BASE_URL"), embedding_dim=dim)

        return EmbeddingFunc(embedding_dim=dim, max_token_size=max_length,
                             model_name=model, func=remote)
    if backend != "local":
        raise ValueError("embedding_backend must be local or openai")
    if batch_size < 1 or max_length < 1:
        raise ValueError("embedding batch size and max length must be positive")

    import torch
    from sentence_transformers import SentenceTransformer

    actual_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    # Explicit cache location also works if HF_HOME changed in a live notebook.
    cache = os.environ.get("HF_HUB_CACHE")
    if not cache and os.environ.get("HF_HOME"):
        cache = str(Path(os.environ["HF_HOME"]) / "hub")
    print(f"Loading local embedding: {model}@{revision}, device={actual_device}", flush=True)
    started = time.perf_counter()
    encoder = await asyncio.to_thread(
        SentenceTransformer, model, revision=revision, device=actual_device,
        cache_folder=cache, trust_remote_code=False, model_kwargs={"use_safetensors": True},
    )
    if encoder.get_embedding_dimension() != dim:
        raise ValueError(f"embedding_dim={dim} does not match encoder dimension "
                         f"{encoder.get_embedding_dimension()}")
    if max_length > encoder.max_seq_length:
        raise ValueError(f"embedding_max_length={max_length} exceeds model limit {encoder.max_seq_length}")
    encoder.max_seq_length = max_length
    encoder.eval()
    encoder.requires_grad_(False)
    lock = asyncio.Lock()
    print(f"Local embedding ready in {time.perf_counter()-started:.1f}s: dim={dim}, "
          f"batch={batch_size}, max_length={max_length}, normalized=True", flush=True)

    def encode(texts, context):
        # Only queries receive the model's retrieval instruction; corpus text is unchanged.
        with torch.inference_mode():
            vectors = encoder.encode(
                texts, batch_size=batch_size, prompt=query_prefix if context == "query" else "",
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
            )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(texts), dim) or not np.isfinite(vectors).all():
            raise ValueError("local encoder returned invalid embeddings")
        return vectors

    async def local(texts, context="document"):
        if context not in {"query", "document"}:
            raise ValueError(f"unknown embedding context: {context}")
        if not texts:
            return np.empty((0, dim), dtype=np.float32)
        # LightRAG schedules several embedding calls concurrently. Serialize this one
        # GPU model to bound memory, and keep inference off the asyncio event loop.
        async with lock:
            return await asyncio.to_thread(encode, list(texts), context)

    identity = {"model": model, "revision": revision, "max_length": max_length,
                "query_prefix": query_prefix, "normalize": True}
    return EmbeddingFunc(embedding_dim=dim, max_token_size=max_length, func=local,
                         model_name=f"{model}-{digest(identity)[:12]}", supports_asymmetric=True)


async def check_embeddings(retrieval_config, output=None):
    """Download/cache and validate the configured encoder before corpus ingestion."""
    import torch

    options = {k.removeprefix("embedding_"): v for k, v in retrieval_config.items()
               if k.startswith("embedding_")}
    started = time.perf_counter()
    embedding = await make_embedding_func(**options)
    ready = time.perf_counter()
    documents = await embedding(["Paris is the capital of France.", "Penguins live in Antarctica."],
                                context="document")
    query = await embedding(["What is the capital of France?"], context="query")
    assert documents.shape == (2, embedding.embedding_dim) and query.shape == (1, embedding.embedding_dim)
    assert np.isfinite(documents).all() and np.isfinite(query).all()
    if options.get("backend", "local") == "local":
        assert np.allclose(np.linalg.norm(documents, axis=1), 1, atol=1e-4)
        assert np.allclose(np.linalg.norm(query, axis=1), 1, atol=1e-4)
    elapsed = time.perf_counter() - ready
    report = {"backend": options.get("backend", "local"), "model": options.get("model", BGE_MODEL),
              "revision": options.get("revision", BGE_REVISION), "dimension": embedding.embedding_dim,
              "device": (("cuda" if torch.cuda.is_available() else "cpu")
                         if options.get("device", "auto") == "auto" else options["device"]),
              "load_seconds": ready-started,
              "encode_seconds": elapsed, "texts_per_second": 3/max(elapsed, 1e-9),
              "document_norms": np.linalg.norm(documents, axis=1).tolist(),
              "toy_top1_correct": int((documents @ query[0]).argmax()) == 0,
              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved()/1024**3 if torch.cuda.is_available() else 0,
              "finite_vectors": True, "test_only": True}
    if output is not None:
        write_json(output, report)
    print(f"Embedding check: {documents.shape}, finite=True, {elapsed:.2f}s, "
          f"GPU peak reserved={report['peak_gpu_reserved_gib']:.3f} GiB", flush=True)
    return report
