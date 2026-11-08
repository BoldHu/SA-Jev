"""Pinned public LightRAG graphs: import, audit, and query; never re-extract documents."""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
from zipfile import ZipFile

from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .schema import Evidence, Question, Pool, deduplicate, normalize

REPO = "thomoal/graph-rag-qa-indexes"
REVISION = "b187c2460ab021a8d0a947f24ea5743043253ad0"
ARCHIVE = "lightrag_indexes_N500.zip"
SHA256 = "919a6a156e2274b3495d5fdbccdee32914e6e6be5e679b71f3f5a04145b42ba3"
QUERY_REPO = "osunlp/HippoRAG_2"
QUERY_REVISION = "5ec05b38deecc3318bb432c69865959c56058990"
DATASETS = ("hotpotqa", "musique", "2wiki")
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
FILES = ("graph_chunk_entity_relation.graphml", "kv_store_full_docs.json",
         "kv_store_text_chunks.json", "kv_store_full_entities.json", "kv_store_full_relations.json",
         "kv_store_entity_chunks.json", "kv_store_relation_chunks.json", "kv_store_doc_status.json",
         "vdb_chunks.json", "vdb_entities.json", "vdb_relationships.json",
         "kv_store_llm_response_cache.json", "lightrag_contexts.json")
REPAIRS = {"graph_chunk_entity_relation.graphml": "graph_chunk_entity_relation-DESKTOP-0KMHV60.graphml",
           "kv_store_full_relations.json": "kv_store_full_relations-DESKTOP-0KMHV60.json",
           "vdb_chunks.json": "vdb_chunks-DESKTOP-0KMHV60.json"}


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download_file(url, path, sha256=None):
    """Resume HTTP downloads; publish only a complete, optionally checksum-verified file."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256 and file_hash(path) != sha256:
            raise ValueError(f"Checksum mismatch: {path}; preserve and move the invalid file before retrying")
        print(f"Reusing {path}", flush=True)
        return path
    part = path.with_suffix(path.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    started = last = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server returned an unexpected byte range")
        if not append: offset = 0
        expected = int(response.headers.get("Content-Length", 0))
        received = 0
        with part.open("ab" if append else "wb") as f:
            while block := response.read(4 * 1024**2):
                f.write(block); received += len(block)
                now = time.perf_counter()
                if now - last > 5:
                    rate = received / max(now-started, 1e-9)
                    eta = (expected-received)/rate if expected else None
                    print(f"{path.name}: {(offset+received)/1024**2:.1f} MiB; ETA={eta}s", flush=True)
                    last = now
        if expected and received != expected: raise IOError("Incomplete download; rerun to resume")
    if sha256 and file_hash(part) != sha256: raise ValueError("Downloaded archive checksum mismatch")
    part.replace(path)
    return path


def download_public_assets(directory):
    directory = Path(directory)
    archive = download_file(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{ARCHIVE}",
                            directory/ARCHIVE, SHA256)
    for dataset in DATASETS:
        name = "2wikimultihopqa" if dataset == "2wiki" else dataset
        download_file(f"https://huggingface.co/datasets/{QUERY_REPO}/resolve/{QUERY_REVISION}/{name}.json",
                      directory/f"{dataset}_queries.json")
    return archive


def audit_index(directory):
    """Validate graph/vector/source alignment without changing the upstream graph."""
    import networkx as nx
    directory = Path(directory)
    docs = read_json(directory/"kv_store_full_docs.json")
    chunks = read_json(directory/"kv_store_text_chunks.json")
    graph = nx.read_graphml(directory/"graph_chunk_entity_relation.graphml")
    vectors = {key: read_json(directory/f"vdb_{key}.json") for key in ("chunks", "entities", "relationships")}
    if any(v["embedding_dim"] != 384 for v in vectors.values()): raise ValueError("Unexpected vector dimension")
    chunk_ids = {x["__id__"] for x in vectors["chunks"]["data"]}
    if chunk_ids != set(chunks): raise ValueError("Chunk vector index is incomplete or mismatched")
    entity_names = {x["entity_name"] for x in vectors["entities"]["data"]}
    if entity_names != set(graph): raise ValueError("Graph/entity vector index is incomplete or mismatched")
    if any(c["full_doc_id"] not in docs for c in chunks.values()): raise ValueError("Missing chunk source document")
    for _, attrs in graph.nodes(data=True):
        if set(attrs.get("source_id", "").split("<SEP>")) - set(chunks) - {""}:
            raise ValueError("Graph node refers to missing source chunks")
    for row in vectors["relationships"]["data"]:
        if not graph.has_edge(row["src_id"], row["tgt_id"]): raise ValueError("Relationship vector has no graph edge")
    # The public MuSiQue graph has one more edge than its vector store; preserve and disclose it.
    return {"documents": len(docs), "chunks": len(chunks), "nodes": len(graph),
            "edges": graph.number_of_edges(), "relation_vectors": len(vectors["relationships"]["data"]),
            "published_questions": len(read_json(directory/"lightrag_contexts.json"))}


def import_public_indexes(archive, root):
    """ZIP has three occurrences of identical paths. Never call extractall on it."""
    root = Path(root); marker = root/"index/public_manifest.json"
    if marker.exists():
        manifest = read_json(marker)
        if manifest["archive_sha256"] != SHA256: raise ValueError("Public release changed")
        print("Reusing imported public indexes; no document extraction", flush=True)
        return manifest
    if file_hash(archive) != SHA256: raise ValueError("Not the pinned public archive")
    stats, entries = {}, {}
    with ZipFile(archive) as z:
        for index, dataset in enumerate(DATASETS):
            target = root/"index"/dataset; target.mkdir(parents=True, exist_ok=True)
            entries[dataset] = {}
            for name in FILES:
                actual = REPAIRS.get(name, name) if dataset == "2wiki" else name
                members = [m for m in z.infolist() if m.filename == "large_scale_lightrag/" + actual]
                if actual != name:
                    if len(members) != 1: raise ValueError("Ambiguous public repair artifact")
                    member = members[0]
                else:
                    if len(members) != 3: raise ValueError("Expected three distinct dataset occurrences")
                    member = members[index]
                # Read the ZipInfo, NOT its repeated filename (which resolves to the last dataset).
                with z.open(member) as src, (target/(name+".tmp")).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                (target/(name+".tmp")).replace(target/name)
                entries[dataset][name] = {"zip_name": member.filename, "crc": member.CRC,
                                          "sha256": file_hash(target/name)}
            stats[dataset] = audit_index(target)
            docs = read_json(target/"kv_store_full_docs.json")
            chunks = read_json(target/"kv_store_text_chunks.json")
            registry = {}
            for cid, chunk in chunks.items():
                title = docs[chunk["full_doc_id"]]["content"].split("\n", 1)[0].strip()
                registry[cid] = {"id": cid, "title": title, "text": chunk["content"],
                                 "source": f"{dataset}/{title}"}
            write_json(target/"source_registry.json", registry)
            print(f"Imported {dataset}: {stats[dataset]}", flush=True)
    manifest = {"repo": REPO, "revision": REVISION, "archive_sha256": SHA256,
                "stats": stats, "files": entries, "embedding_model": ENCODER,
                "embedding_revision": ENCODER_REVISION,
                "upstream_embedding_revision": "not recorded by upstream; local compatibility probe required",
                "upstream_extraction_model": "llama-3.1-8b-instant",
                "repairs": {"2wiki": REPAIRS}, "document_extraction_api_calls": 0}
    write_json(marker, manifest)
    return manifest


def prepare_public_data(root, assets, raw, config, *, mode="pilot"):
    """Public 500 questions stay test. Validation/heldout come from unused question families."""
    from .data import hotpot, musique, twowiki, split_questions, sample_training, audit_splits
    from .preparation import load_raw, _progress_rows
    root, assets, raw = Path(root), Path(assets), Path(raw)
    counts = config["data"]["train_counts"] if mode == "full" else config["data"].get("pilot_train_counts", {"hotpotqa": 32, "musique": 32})
    binding = {"protocol": "public_lightrag_n500_v1", "mode": mode, "counts": counts,
               "seed": config["seed"], "validation_per_dataset": config["data"]["validation_per_dataset"],
               "heldout_per_dataset": config["data"]["heldout_per_dataset"],
               "queries": {ds: file_hash(assets/f"{ds}_queries.json") for ds in DATASETS},
               "raw": {ds: file_hash(raw/name) for ds, name in
                       (("hotpotqa", "hotpot_train.jsonl"), ("musique", "musique_ans_train.jsonl"))}}
    marker = root/"data/data_manifest.json"
    if marker.exists():
        previous = read_json(marker)
        if previous.get("preparation_binding") != binding: raise ValueError("Prepared protocol changed; use a fresh ROOT")
        if all((root/"data"/name).is_file() for name in ("questions.jsonl", "corpus.jsonl")):
            return previous
    public, test_ids = [], set()
    for ds, adapter in zip(DATASETS, (hotpot, musique, twowiki)):
        rows = read_json(assets/f"{ds}_queries.json")
        published = set(read_json(root/"index"/ds/"lightrag_contexts.json"))
        public_rows = [adapter(row, split="dev") for row in rows]
        matches = {q.id.split(":", 1)[1] for q in public_rows} & published
        if matches != published or len(published) != 500: raise ValueError(f"Public query IDs mismatch for {ds}")
        public.extend(public_rows)
        test_ids.update(q.id for q in public_rows if q.id.split(":", 1)[1] in published)
    # Split all source train questions jointly with the public questions BEFORE sampling.
    training_raw = [hotpot(r) for r in _progress_rows(load_raw(raw/"hotpot_train.jsonl"), "read HotpotQA training source")]
    training_raw += [musique(r) for r in _progress_rows(load_raw(raw/"musique_ans_train.jsonl"), "read MuSiQue training source") if r.get("answerable", True)]
    unique = {q.id: q for q in training_raw}
    unique.update({q.id: q for q in public})  # public/dev overrides any same-ID training row
    all_questions = split_questions(list(unique.values()), seed=config["seed"])
    public_ids = {q.id for q in public}
    test_groups = {q.group_id for q in all_questions if q.id in test_ids}
    selected = sample_training(all_questions, counts, config["seed"])
    chosen = Counter(q.dataset for q in selected)
    if any(chosen[ds] < n for ds, n in counts.items()): raise ValueError(f"Insufficient disjoint training data: {chosen}, need {counts}")
    family_roles = {g: "test" for g in test_groups}
    for ds in DATASETS:
        candidates = sorted((q for q in all_questions if q.dataset == ds and q.id in public_ids and q.id not in test_ids
                             and q.group_id not in test_groups), key=lambda q: digest([config["seed"], q.group_id, q.id]))
        # 2Wiki remains cross-dataset test only: no validation or stability tuning on it.
        if ds != "2wiki":
            for role, n in (("validation", config["data"]["validation_per_dataset"] if mode == "full" else 2),
                            ("heldout", config["data"]["heldout_per_dataset"] if mode == "full" else 4)):
                available = [q for q in candidates if q.group_id not in family_roles]
                taken = available[:n]
                if len(taken) < n: raise ValueError(f"Insufficient independent {role} questions for {ds}")
                selected.extend(replace(q, split=role) for q in taken)
                family_roles.update({q.group_id: role for q in taken})
        tests = sorted((q for q in all_questions if q.dataset == ds and q.id in test_ids), key=lambda q: q.id)
        selected.extend(replace(q, split="test") for q in (tests if mode == "full" else tests[:2]))
    split_counts = audit_splits(selected)
    # Only sampled training context paragraphs enter BM25 hard-negative mining. No graph insertion.
    corpus = deduplicate(e for q in selected if q.split == "train" for e in q.evidence)
    write_jsonl(root/"data/questions.jsonl", selected)
    write_jsonl(root/"data/corpus.jsonl", corpus)
    manifest = {"preparation_binding": binding, "protocol": binding["protocol"],
                "question_fingerprint": digest(selected), "corpus_fingerprint": digest(corpus),
                "questions": len(selected), "corpus_documents": len(corpus), "splits": split_counts,
                "dataset_splits": {ds: dict(Counter(q.split for q in selected if q.dataset == ds)) for ds in DATASETS},
                "download": {"formal": mode == "full"},
                "scope": "500 published test questions/dataset; training corpus is NOT graph-indexed",
                "validation_scope": "unused HippoRAG question families on the fixed public graph; support may be absent",
                "seed": config["seed"]}
    write_json(marker, manifest)
    print(manifest["dataset_splits"], flush=True)
    return manifest


async def open_public_index(root, dataset, config, embedding=None, llm_func=None):
    from lightrag import LightRAG
    from .embeddings import make_embedding_func
    cfg = config["retrieval"]
    if cfg["embedding_model"] != ENCODER or cfg["embedding_revision"] != ENCODER_REVISION:
        raise ValueError("Public index requires its MiniLM encoder; BGE vectors are incompatible")
    if (cfg["embedding_dim"] != 384 or cfg["embedding_max_length"] != 256
            or cfg["embedding_query_prefix"] or cfg["embedding_backend"] != "local"):
        raise ValueError("Public MiniLM requires local 384-dimensional embeddings, max_length=256 and no query prefix")
    if embedding is None:
        embedding = await make_embedding_func(backend="local", model=ENCODER, revision=ENCODER_REVISION,
                     dim=384, device=cfg["embedding_device"], batch_size=cfg["embedding_batch_size"],
                     max_length=256, query_prefix="")
    await verify_encoder(Path(root)/"index"/dataset, embedding)
    if llm_func is None:
        from lightrag.llm.openai import openai_complete_if_cache
        async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            if not os.environ.get("OPENAI_API_KEY"): raise ValueError("Fill Drive .env for query keyword extraction")
            return await openai_complete_if_cache(cfg["llm_model"], prompt, system_prompt=system_prompt,
                        history_messages=history_messages or [], api_key=os.environ["OPENAI_API_KEY"],
                        base_url=os.environ.get("OPENAI_BASE_URL"), **kwargs)
    rag = LightRAG(working_dir=str(Path(root)/"index"), workspace=dataset,
                   llm_model_func=llm_func, llm_model_name=cfg["llm_model"], embedding_func=embedding,
                   chunk_token_size=1200, chunk_overlap_token_size=100, embedding_func_max_async=1)
    await rag.initialize_storages()
    # Explicitly prevent accidental paid extraction through this read/query adapter.
    async def no_insert(*args, **kwargs): raise RuntimeError("Public indexes are query-only; rebuilding is disabled")
    rag.ainsert = no_insert
    return rag


async def verify_encoder(directory, embedding):
    """Compare real stored vectors with pinned local MiniLM; dimension equality is insufficient."""
    import base64
    import numpy as np
    vectors = read_json(Path(directory)/"vdb_chunks.json")
    matrix = np.frombuffer(base64.b64decode(vectors["matrix"]), dtype=np.float32).reshape(-1, 384)
    rows = vectors["data"]
    if len(matrix) != len(rows): raise ValueError("Invalid public vector matrix")
    indices = sorted({0, len(rows)//2, len(rows)-1})
    actual = np.asarray(await embedding([rows[i]["content"] for i in indices]))
    reference = matrix[indices]
    cosines = np.sum(actual*reference, axis=1)/(np.linalg.norm(actual, axis=1)*np.linalg.norm(reference, axis=1))
    if not np.isfinite(cosines).all() or float(cosines.min()) < .995:
        raise ValueError(f"Public embedding compatibility check failed: cosine={cosines.tolist()}")
    write_json(Path(directory)/"encoder_check.json", {"model": ENCODER, "revision": ENCODER_REVISION,
               "sample_ids": [rows[i]["__id__"] for i in indices], "cosines": cosines.tolist(),
               "scope": "sampled vector compatibility, not proof of exact upstream revision"})
    print(f"Stored MiniLM compatibility: min cosine={cosines.min():.6f}", flush=True)


async def cache_public_pools(root, config, limit=None, *, llm_func=None):
    from .retrieval import PoolCache, retrieve_once
    from .experiments import read_questions, slice_pool
    root = Path(root)
    if not (root/"index/public_manifest.json").exists():
        raise ValueError("Public indexes are not imported in this ROOT. Run the updated notebook 00 import cell; "
                         "do not run ingestion on the old full directory.")
    imported = read_json(root/"index/public_manifest.json")
    source = read_json(root/"data/data_manifest.json")
    questions = [q for q in read_questions(root) if q.split != "train"]
    cache, main = PoolCache(root/"data/pools_max"), PoolCache(root/"data/pools")
    complete_path = root/"data/pool_cache_manifest.json"
    expected_config = digest({"retrieval": config["retrieval"], "main_k": config["data"]["main_k"]})
    if limit is None and complete_path.exists():
        previous = read_json(complete_path)
        if (previous.get("complete") and previous.get("question_fingerprint") == source["question_fingerprint"]
                and previous.get("config_fingerprint") == expected_config
                and previous.get("corpus_binding") == digest(imported)
                and previous.get("questions") == len(questions)
                and all(cache.path(q.id).is_file() and main.path(q.id).is_file() for q in questions)):
            print(f"Reusing {len(questions)} completed public candidate pools; no model loading or API calls", flush=True)
            return previous
    total = 0; started = time.perf_counter(); coverage_rows = []
    from .embeddings import make_embedding_func
    cfg = config["retrieval"]
    embedding = await make_embedding_func(backend="local", model=cfg["embedding_model"], revision=cfg["embedding_revision"],
                 dim=384, device=cfg["embedding_device"], batch_size=cfg["embedding_batch_size"], max_length=256, query_prefix="")
    for ds in DATASETS:
        selected = [q for q in questions if q.dataset == ds]
        if limit: selected = selected[:limit]
        rag = await open_public_index(root, ds, config, embedding, llm_func=llm_func)
        registry = {cid: Evidence(**row) for cid, row in read_json(root/"index"/ds/"source_registry.json").items()}
        binding = digest({"release": REVISION, "dataset": ds, "files": imported["files"][ds],
                          "query_llm": cfg["llm_model"], "embedding_revision": cfg["embedding_revision"]})
        try:
            for q in selected:
                pool = await retrieve_once(rag, q.id, q.question, config["retrieval"]["max_k"], cache,
                           corpus_fingerprint=binding, mode=config["retrieval"]["mode"], chunk_registry=registry,
                           extra={"max_total_tokens": 180000, "max_entity_tokens": 20000, "max_relation_tokens": 20000})
                main.write(slice_pool(pool, config["data"]["main_k"]))
                total += 1
                from .schema import State, labels_for
                label = labels_for(q, State(q.id, q.group_id, q.question, pool, budget=config["data"]["budget"]))
                coverage_rows.append({"question_id": q.id, "dataset": ds, "split": q.split, "actual_k": len(pool.candidates),
                                      "initial_rank_valid": label.rank_valid, "initial_suf_valid": label.suf_valid,
                                      "positive_candidates": sum(g > 0 for g in label.gains)})
                elapsed = time.perf_counter()-started
                print(f"Public pool {total}/{len(questions)} {ds}; K={len(pool.candidates)}; "
                      f"ETA={(len(questions)-total)*elapsed/total:.0f}s; extraction calls=0", flush=True)
        finally:
            await rag.finalize_storages()
    from .reporting import csv_rows
    csv_rows(root/"data/public_pool_coverage.csv", coverage_rows)
    write_json(root/"data/pool_cache_manifest.json", {"complete": limit is None, "questions": total,
               "corpus_binding": digest(imported), "question_fingerprint": source["question_fingerprint"],
               "config_fingerprint": expected_config,
               "protocol": "public_lightrag_n500_v1", "document_extraction_api_calls": 0})
