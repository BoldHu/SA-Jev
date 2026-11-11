"""Frozen local text embeddings for native LightRAG; no embedding API required."""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

import numpy as np

from .io import digest, write_json

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Official Hub LFS metadata for model.safetensors at BGE_REVISION.
BGE_WEIGHT_BYTES = 133466304
BGE_WEIGHT_SHA256 = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_WEIGHT_BYTES = 90868376
MINILM_WEIGHT_SHA256 = "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"


def ensure_embedding_weights(model, revision, cache_folder=None):
    """Validate the pinned small encoder; repair only its corrupt cached weight once."""
    from huggingface_hub import hf_hub_download, constants

    specs = {(BGE_MODEL, BGE_REVISION): (BGE_WEIGHT_BYTES, BGE_WEIGHT_SHA256),
             (MINILM_MODEL, MINILM_REVISION): (MINILM_WEIGHT_BYTES, MINILM_WEIGHT_SHA256)}
    if (model, revision) not in specs:
        raise ValueError("No pinned weight checksum for this embedding model/revision")
    weight_bytes, weight_sha256 = specs[model, revision]
    cache = Path(cache_folder or os.environ.get("HF_HUB_CACHE") or
                 (str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME")
                  else constants.HF_HUB_CACHE)).expanduser().absolute()
    snapshot = cache / ("models--" + model.replace("/", "--")) / "snapshots" / revision
    expected_path = snapshot / "model.safetensors"

    def validate(path):
        size = path.stat().st_size
        if size != weight_bytes:
            raise ValueError(f"expected {weight_bytes:,} bytes, got {size:,}")
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != weight_sha256:
            raise ValueError("SHA-256 differs from the official pinned weight")

    def download(force=False):
        return Path(hf_hub_download(model, "model.safetensors", revision=revision,
                                   cache_dir=str(cache), force_download=force))

    print(f"Checking pinned {model} weight: {expected_path}", flush=True)
    # Inspect even a dangling snapshot link instead of treating it as a new download.
    path = expected_path if expected_path.exists() or expected_path.is_symlink() else download()
    try:
        validate(path)
    except (OSError, ValueError) as error:
        print(f"Embedding weight cache is invalid: {error}. Re-downloading the same revision once.", flush=True)
        # HF force_download replaces the blob, but may leave an existing regular
        # snapshot file untouched (e.g. Drive/non-symlink caches). Preserve that
        # single entry under another name before asking HF to rebuild the pointer.
        if path.absolute() != expected_path:
            raise RuntimeError("Unexpected embedding cache path; refusing to move it") from error
        if path.exists() or path.is_symlink():
            quarantine = path.with_name(path.name + ".invalid-" + uuid4().hex[:8])
            path.rename(quarantine)
            print(f"Preserved invalid cache entry: {quarantine}", flush=True)
        path = download(force=True)
        try:
            validate(path)
        except (OSError, ValueError) as retry_error:
            raise RuntimeError("Downloaded embedding weight still fails integrity validation. "
                               "Use a fresh HF_HUB_CACHE and check the download/Drive connection; "
                               "do not delete raw data or graph indexes.") from retry_error
    print(f"{model} weight verified: {weight_bytes:,} bytes, SHA-256 matches.", flush=True)
    return path


def ensure_bge_weights(cache_folder=None):
    """Backward-compatible entry point for the legacy BGE protocol."""
    return ensure_embedding_weights(BGE_MODEL, BGE_REVISION, cache_folder)


def stage_embedding_snapshot(model, revision, weight_path, cache_folder=None):
    """Load the verified encoder from regular local files, not Drive-backed mmap.

    Only these two pinned encoder layouts are supported. Persistent HF downloads
    stay in the original cache; the runtime copy is disposable and never changes
    the model identity used to bind graphs/candidate pools.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from safetensors import safe_open

    specs = {(BGE_MODEL, BGE_REVISION): (BGE_WEIGHT_BYTES, BGE_WEIGHT_SHA256),
             (MINILM_MODEL, MINILM_REVISION): (MINILM_WEIGHT_BYTES, MINILM_WEIGHT_SHA256)}
    size, checksum = specs[(model, revision)]
    local = Path(tempfile.gettempdir()) / "sajev_embedding_models" / digest([model, revision])[:24]
    local.mkdir(parents=True, exist_ok=True)

    def weight_ok(path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == checksum

    def copy_regular(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp-" + uuid4().hex)
        try:
            # Stream bytes through Python IO, avoiding FUSE mmap/copy-file shortcuts
            # and materializing symlinks rather than copying the link itself.
            with Path(source).open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    target = local / "model.safetensors"
    if not weight_ok(target):
        print(f"Copying verified embedding weight to local runtime: {target}", flush=True)
        copy_regular(weight_path, target)
    if not weight_ok(target):
        raise RuntimeError(f"Local embedding copy failed checksum validation: {target}. "
                           "The source may have changed during copying; graph/pool files were not modified.")
    # Check the same local file the transformer will open, including mmap parsing.
    with safe_open(str(target), framework="pt", device="cpu") as tensors:
        if not tensors.keys():
            raise ValueError(f"Embedding weight contains no tensors: {target}")

    # Both pinned models use a root Transformer and 1_Pooling; Normalize needs no
    # file. Explicit files avoid downloading duplicate PyTorch/ONNX/OpenVINO weights.
    metadata = ("config.json", "config_sentence_transformers.json", "modules.json",
                "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json")
    for filename in metadata:
        args = dict(repo_id=model, filename=filename, revision=revision, cache_dir=cache_folder)
        try:
            source = hf_hub_download(**args, local_files_only=True)
        except LocalEntryNotFoundError:
            source = hf_hub_download(**args)
        copy_regular(source, local / filename)
    print(f"Local embedding snapshot verified; loading ONLY from {local}", flush=True)
    return local


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
    weight_path = None
    if model == BGE_MODEL and revision == BGE_REVISION:
        weight_path = await asyncio.to_thread(ensure_bge_weights, cache)
    elif model == MINILM_MODEL and revision == MINILM_REVISION:
        weight_path = await asyncio.to_thread(ensure_embedding_weights, model, revision, cache)
    load_path = model
    load_options = {"revision": revision, "cache_folder": cache}
    if weight_path is not None:
        local = await asyncio.to_thread(stage_embedding_snapshot, model, revision, weight_path, cache)
        load_path = str(local)
        load_options = {"local_files_only": True}
    encoder = await asyncio.to_thread(
        SentenceTransformer, load_path, device=actual_device, **load_options,
        trust_remote_code=False, model_kwargs={"use_safetensors": True},
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
