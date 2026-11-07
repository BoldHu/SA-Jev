"""Reproducible raw-data preparation and manifests, separate from experiment runs."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import itertools
import json
import urllib.request
from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .schema import Question, TaskExample, Evidence, Pool, deduplicate
from .data import hotpot, musique, twowiki, split_questions, audit_splits, sample_training, build_training_states, task_states
from .trajectories import build_preferences
from .retrieval import PoolCache, hard_candidates


def load_raw(path):
    path = Path(path)
    return read_jsonl(path) if path.suffix == ".jsonl" else iter(read_json(path))


def download_sources(directory, limit=None):
    """The official MuSiQue distribution includes Hotpot/2Wiki support-format data.

    Only answerable MuSiQue is fetched. A limit produces explicitly marked smoke
    subsets, not formal training/test sets. Google Drive quota errors are surfaced.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    if (directory/"download_manifest.json").exists() and read_json(directory/"download_manifest.json")["limit"] != limit:
        raise ValueError("download limit changed; use a separate pilot/full directory to avoid relabeling partial files as complete")
    source = "https://raw.githubusercontent.com/StonyBrookNLP/musique/main/.all_data_information.json"
    info = json.load(urllib.request.urlopen(source))
    wanted = ["musique_ans_train.jsonl", "musique_ans_dev.jsonl"]
    manifest = {"limit": limit, "formal": limit is None, "sources": []}
    for name in wanted:
        file_id = info["raw_data/" + name]
        url = "https://drive.usercontent.google.com/download?id=" + file_id + "&export=download&confirm=t"
        target = directory / name
        if not target.exists():
            response = urllib.request.urlopen(url, timeout=120)
            rows = []
            for line in response:
                if line.strip(): rows.append(json.loads(line))
                if limit and len(rows) >= limit: break
            write_jsonl(target, rows)
        manifest["sources"].append({"path": name, "url": url})
    from datasets import load_dataset
    for split in ("train", "validation"):
        target = directory / f"hotpot_{split}.jsonl"
        if not target.exists():
            data = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
            write_jsonl(target, itertools.islice(data, limit) if limit else data)
        manifest["sources"].append({"path": target.name, "repository": "hotpotqa/hotpot_qa", "split": split})
    write_json(directory / "download_manifest.json", manifest)
    return manifest


def normalize_sources(raw, output, *, seed=17, family_links=()):
    raw, output = Path(raw), Path(output)
    sources = [("hotpot_train.jsonl", hotpot, "train"), ("hotpot_validation.jsonl", hotpot, "dev"),
               ("musique_ans_train.jsonl", musique, "train"), ("musique_ans_dev.jsonl", musique, "dev"),
               ("2wiki_dev.json", twowiki, "test"), ("2wiki_dev.jsonl", twowiki, "test")]
    questions, provenance = [], []
    for name, adapter, split in sources:
        path = raw / name
        if not path.exists(): continue
        rows = list(load_raw(path))
        questions.extend(adapter(row, split=split) for row in rows if row.get("answerable", True))
        provenance.append({"path": name, "records": len(rows), "sha256_content": digest(rows)})
    if not questions: raise ValueError("no dataset files found; run download_sources or supply official raw files")
    questions = split_questions(questions, family_links, seed)
    counts = audit_splits(questions)
    write_jsonl(output / "questions.jsonl", questions)
    corpus = deduplicate(e for q in questions for e in q.evidence)
    write_jsonl(output / "corpus.jsonl", corpus)
    manifest = {"sources": provenance, "question_fingerprint": digest(questions), "corpus_fingerprint": digest(corpus),
                "questions": len(questions), "corpus_documents": len(corpus),
                "splits": counts, "seed": seed, "family_links": len(family_links),
                "corpus_protocol": "union of raw source paragraphs; no answers, decompositions or support labels indexed",
                "scope": "bounded benchmark corpus, not full Wikipedia; report this corpus scope explicitly"}
    if (raw/"download_manifest.json").exists():
        manifest["download"] = read_json(raw/"download_manifest.json")
    write_json(output / "data_manifest.json", manifest)
    return questions, manifest


def download_2wiki(directory):
    """Official April 2021 archive, extracted by explicit filename."""
    import zipfile
    import shutil
    import os
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    url = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
    archive = directory/"2wiki_official.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        partial = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f)
        if not zipfile.is_zipfile(partial):
            raise ValueError("2Wiki download is not a complete ZIP; rerun to retry the download")
        os.replace(partial, archive)
    with zipfile.ZipFile(archive) as z:
        for target, basename in (("2wiki_dev.json", "dev.json"), ("2wiki_id_aliases.json", "id_aliases.json")):
            names = [n for n in z.namelist() if Path(n).name == basename]
            if len(names) != 1: raise ValueError(f"expected one {basename} in the official archive")
            if (directory/target).exists(): continue
            partial = directory / (target + ".part")
            with z.open(names[0]) as source, partial.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            os.replace(partial, directory/target)
    write_json(directory/"2wiki_source.json", {"url": url, "format": "official April 2021", "split": "dev used as cross-dataset test"})


def prepare_tasks(data_directory, *, seed=17, budget=4, counts=None, mine=False, dense=False, small_k=False):
    directory = Path(data_directory)
    questions = [Question.from_dict(r) for r in read_jsonl(directory / "questions.jsonl")]
    training = sample_training(questions, counts, seed)
    retrieved = None
    if mine:
        corpus = [Evidence(**r) for r in read_jsonl(directory / "corpus.jsonl")]
        retrieved = hard_candidates(training, corpus, dense=dense)
    states = build_training_states(training, retrieved, seed, budget, small_k=small_k)
    initials = [x.state for x in states if not x.state.selected_ids]
    pairs = build_preferences(training, initials, seed=seed)
    audit_splits(questions, states, pairs, [q.group_id for q in questions if q.split == "heldout"])
    suffix = "_small_k" if small_k else ""
    write_jsonl(directory / f"train_states{suffix}.jsonl", states)
    write_jsonl(directory / f"preferences{suffix}.jsonl", pairs)
    # Validation MUST come from retrieved pools, no training-pool gold injection.
    cache = PoolCache(directory / "pools")
    validation = []
    for q in questions:
        if q.split == "validation" and cache.path(q.id).exists():
            validation.extend(task_states(q, cache.read(q.id), budget, seed))
    write_jsonl(directory / "validation_states.jsonl", validation)
    manifest = {"training_questions": len(training), "task_states": len(states), "preferences": len(pairs),
                "validation_states": len(validation), "rank_valid": sum(x.labels.rank_valid for x in states),
                "suf_valid": sum(x.labels.suf_valid for x in states),
                "rel_valid": sum(sum(x.labels.rel_valid) for x in states),
                "requested_k_counts": {k: sum(x.state.pool.provenance["requested_k"] == k for x in states if not x.state.selected_ids) for k in (8, 16, 32, 64)},
                "actual_k_counts": {k: sum(len(x.state.pool.candidates) == k for x in states if not x.state.selected_ids) for k in sorted({len(x.state.pool.candidates) for x in states})},
                "state_fingerprint": digest(states), "preference_fingerprint": digest(pairs),
                "unknown_labels": "masked; short pools are never padded"}
    write_json(directory / f"task_manifest{suffix}.json", manifest)
    return manifest


def finish_preparation(root, config, *, mode=None, run_id=None):
    """Write the notebook completion marker only after all preparation stages succeeded."""
    from datetime import datetime, timezone
    root = Path(root); data = root / "data"
    source = read_json(data / "data_manifest.json")
    if mode == "full" and source.get("download", {}).get("formal") is not True:
        raise ValueError("full preparation requires complete source downloads, not pilot files")
    tasks = read_json(data / "task_manifest.json")
    audit = read_json(data / "overlap_audit.json")
    pools = read_json(data / "pool_cache_manifest.json")
    read_json(root / "index/corpus_manifest.json")
    expected = digest({"retrieval": config["retrieval"], "main_k": config["data"]["main_k"]})
    if not pools["complete"] or pools["config_fingerprint"] != expected:
        raise ValueError("candidate preparation is incomplete or uses a different retrieval configuration")
    if any(x["question_fingerprint"] != source["question_fingerprint"] for x in (audit, pools)):
        raise ValueError("preparation manifests refer to different question sets")
    if not audit.get("complete_scan"):
        raise ValueError("complete OpenJevData scan is required before marking 00 complete")
    if not tasks["task_states"] or not tasks["validation_states"]:
        raise ValueError("training/validation states are missing; do not mark 00 complete")
    for name in ("train_states.jsonl", "validation_states.jsonl", "preferences.jsonl", "perturbations_draft.jsonl"):
        if not (data / name).is_file(): raise FileNotFoundError(data / name)
    result = {"status": "data_preparation_complete", "mode": mode, "run_id": run_id,
              "completed_at": datetime.now(timezone.utc).isoformat(),
              "config_fingerprint": digest(config), "question_fingerprint": source["question_fingerprint"],
              "training_questions": tasks["training_questions"], "task_states": tasks["task_states"],
              "validation_states": tasks["validation_states"], "preferences": tasks["preferences"],
              "cached_questions": pools["questions"], "perturbation_review": "draft_only; manual review still required",
              "openjev_4b_training": "not run by notebook 00"}
    write_json(root / "00_complete.json", result)
    print(f"00 COMPLETE: {root / '00_complete.json'}", flush=True)
    return result
