"""Read-only support-text coverage audit; no retrieval, model loads, or API calls."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv

from sa_jev.data import hotpot, musique, twowiki
from sa_jev.io import read_json, write_json
from sa_jev.schema import Evidence, Pool, State, deduplicate, labels_for, normalize


def audit(root, assets, output):
    root, assets, output = map(Path, (root, assets, output))
    manifest = read_json(root / "index/public_manifest.json")
    rows = []
    for dataset, adapter in (("hotpotqa", hotpot), ("musique", musique), ("2wiki", twowiki)):
        directory = root / "index" / dataset
        published = set(read_json(directory / "lightrag_contexts.json"))
        by_title = {"documents": defaultdict(list), "chunks": defaultdict(list)}
        for key, doc in read_json(directory / "kv_store_full_docs.json").items():
            title = doc["content"].split("\n", 1)[0].strip()
            by_title["documents"][normalize(title)].append(
                Evidence(key, title, doc["content"], f"{dataset}/{title}"))
        for value in read_json(directory / "source_registry.json").values():
            e = Evidence(**value)
            by_title["chunks"][normalize(e.title)].append(e)
        for raw in read_json(assets / f"{dataset}_queries.json"):
            q = adapter(raw, split="test")
            titles = {normalize(e.title) for e in q.evidence if q.coverage.get(e.id)}
            row = {"dataset": dataset, "question_id": q.id,
                   "subset": "published_500" if q.id.split(":", 1)[1] in published else "unused_500",
                   "gold_complete": q.gold_complete, "support_units": len(set(q.units)),
                   "support_titles": len(titles),
                   "support_titles_found": sum(bool(by_title["documents"].get(t)) for t in titles)}
            for kind in ("documents", "chunks"):
                # Match via the actual annotation-transfer logic. Selecting all matching
                # source material is an availability probe, NOT a legal policy rollout.
                evidence = deduplicate(e for t in sorted(titles) for e in by_title[kind].get(t, ()))
                pool = Pool(q.id, evidence, "coverage_audit_only")
                state = State(q.id, q.group_id, q.question, pool,
                              selected_ids=tuple(e.id for e in evidence))
                labels = labels_for(q, state)
                row[f"covered_{kind}"] = len(labels.covered)
                row[f"all_support_{kind}"] = bool(labels.sufficient and q.gold_complete)
                row[f"missing_{kind}"] = sorted(set(q.units) - set(labels.covered))
            rows.append(row)
        print(f"Audited {dataset}: 1,000 questions, no API calls", flush=True)
    summary = []
    for dataset in ("hotpotqa", "musique", "2wiki"):
        for subset in ("published_500", "unused_500"):
            group = [r for r in rows if r["dataset"] == dataset and r["subset"] == subset]
            summary.append({"dataset": dataset, "subset": subset, "questions": len(group),
                            "gold_complete": sum(r["gold_complete"] for r in group),
                            "all_support_documents": sum(r["all_support_documents"] for r in group),
                            "all_support_chunks": sum(r["all_support_chunks"] for r in group),
                            "zero_support_chunks": sum(r["covered_chunks"] == 0 for r in group),
                            "all_support_titles_found": sum(r["support_titles"] > 0 and r["support_titles_found"] == r["support_titles"] for r in group),
                            "support_units": sum(r["support_units"] for r in group),
                            "covered_units_chunks": sum(r["covered_chunks"] for r in group)})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "coverage_audit.json", {
        "scope": "Exact source/text annotation transfer using current labels_for; not semantic verification or top-K retrieval recall",
        "archive_sha256": manifest["archive_sha256"], "api_calls": 0,
        "summary": summary, "questions": rows})
    with (output / "coverage_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0])
        writer.writeheader(); writer.writerows(summary)
    for row in summary:
        print(row, flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.root, args.assets, args.output)
