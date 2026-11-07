"""ViDoRe V3 Industrial: official OCR/page retrieval case study only."""
from __future__ import annotations

from pathlib import Path
import time
import torch
from .io import write_jsonl, write_json
from .schema import Evidence, Pool, State
from .metrics import ndcg, answer_metrics
from .evaluation import rollout
from .retrieval import require_retrieved

REPO = "vidore/vidore_v3_industrial"
REVISION = "233d20721f4deb392a09a86cca01761adbc91157"


def download_ocr(output, revision=REVISION):
    from datasets import load_dataset
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # Column projection on parquet avoids decoding/downloading image columns.
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=revision)
    urls = [f"hf://datasets/{REPO}@{revision}/{p}" for p in files if p.startswith("corpus/") and p.endswith(".parquet")]
    corpus = load_dataset("parquet", data_files={"test": urls}, split="test", streaming=True,
                          columns=["corpus_id", "doc_id", "markdown", "page_number_in_doc"])
    metadata = list(load_dataset(REPO, "documents_metadata", split="test", revision=revision))
    by_doc = {r["doc_id"]: r for r in metadata}
    pages = []
    for row in corpus:
        doc = by_doc[row["doc_id"]]
        pages.append({"id": "vidore:" + str(row["corpus_id"]), "title": row["doc_id"],
                      "text": row["markdown"], "source": doc["url"] + "#page=" + str(row["page_number_in_doc"] + 1),
                      "page_number_in_doc": row["page_number_in_doc"], "corpus_id": row["corpus_id"]})
    queries = [r for r in load_dataset(REPO, "queries", split="test", revision=revision) if r["language"] in {"en", "english"}]
    qrels = list(load_dataset(REPO, "qrels", split="test", revision=revision))
    for name, values in (("pages", pages), ("queries", queries), ("qrels", qrels), ("documents_metadata", metadata)):
        write_jsonl(output / (name + ".jsonl"), values)
    write_json(output / "provenance.json", {"repository": REPO, "revision": revision, "input": "official markdown OCR", "visual_understanding": False})
    return pages, queries, qrels


def page_pool(question_id, retrieved_pool, pages, k):
    """Collapse retrieved chunks to their source pages, preserving retrieval order.

    The input retrieval must identify each chunk's page source. No qrel input is
    accepted here, so gold pages cannot be inserted into the candidate set.
    """
    require_retrieved(retrieved_pool)
    by_source = {p["source"]: p for p in pages}; seen = set(); result = []
    for chunk in retrieved_pool.candidates:
        if chunk.source not in by_source: raise ValueError("retrieved chunk has no traceable official page")
        p = by_source[chunk.source]
        if p["id"] not in seen:
            result.append(Evidence(p["id"], p["title"], p["text"], p["source"])); seen.add(p["id"])
    return Pool(question_id, tuple(result[:k]), "lightrag", {**retrieved_pool.provenance,
                "parent_chunk_pool": retrieved_pool.fingerprint, "page_aggregation": "first_chunk_occurrence"})


@torch.no_grad()
def evaluate_industrial(model, queries, qrels, pools, *, budget=4, answerer=None, name="sa_jev"):
    relevance = {}
    for r in qrels:
        relevance.setdefault(str(r["query_id"]), {})["vidore:" + str(r["corpus_id"])] = int(r["score"])
    rows = []; model.eval()
    for q in queries:
        qid = "vidore:" + str(q["query_id"]); pool = pools[qid]; require_retrieved(pool)
        state = State(qid, qid, q["query"], pool, budget=budget)
        begin = time.perf_counter(); p = model(state, include_rel=False)
        gold = relevance.get(str(q["query_id"]), {})
        ranking = sorted(range(len(p.p)), key=lambda i: -float(p.p[i]))
        top = [pool.candidates[i] for i in ranking[:5]]
        row = {"question_id": qid, "group_id": qid, "method": name, "dataset": "vidore_industrial_ocr",
               "pool_hash": pool.fingerprint, "page_ndcg5": ndcg([gold.get(e.id, 0) for e in pool.candidates], p.p.cpu().tolist(), 5),
               "page_recall5": len({e.id for e in top} & set(gold)) / len(gold) if gold else None,
               "query": q["query"], "top_pages": [{"id": e.id, "source": e.source, "ocr_excerpt": e.text[:500],
                            "qrel": gold.get(e.id)} for e in top], "seconds": time.perf_counter()-begin,
               "sufficiency_gold": None, "input_modality": "official OCR text", "success_definition": "at least one annotated page in top 5"}
        row["retrieval_success"] = bool(any(e.id in gold for e in top))
        if answerer:
            result = rollout(model, state, answerer=answerer)
            row["answer"] = result["qa"]["answer"]
            row.update(answer_metrics(row["answer"], q["answer"]))
        rows.append(row)
    return rows
