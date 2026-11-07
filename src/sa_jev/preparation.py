"""Reproducible raw-data preparation and manifests, separate from experiment runs."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import itertools
import json
import time
import urllib.request
from .io import digest, read_json, read_jsonl, write_json, write_jsonl
from .schema import Question, TaskExample, Evidence, Pool, deduplicate
from .data import hotpot, musique, twowiki, split_questions, audit_splits, sample_training, build_training_states, task_states
from .trajectories import build_preferences
from .retrieval import PoolCache, hard_candidates


def load_raw(path):
    path = Path(path)
    return read_jsonl(path) if path.suffix == ".jsonl" else iter(read_json(path))


def _progress_rows(rows, label, *, total=None):
    """Report real streaming progress; an unknown source length has no ETA."""
    started = last = time.perf_counter()
    count = 0
    print(f"[data] {label}: starting; waiting for records...", flush=True)
    for count, row in enumerate(rows, 1):
        yield row
        now = time.perf_counter()
        if count == 1 or now - last >= 5:
            rate = count / max(now - started, 1e-9)
            eta = f"{max(0, total-count)/rate:.0f}s" if total else "unknown (streaming)"
            print(f"[data] {label}: {count:,} records, {rate:.1f}/s, ETA={eta}", flush=True)
            last = now
    print(f"[data] {label}: finished {count:,} records in {time.perf_counter()-started:.1f}s", flush=True)


def _copy_with_progress(source, dest, label, *, total=None):
    started = last = time.perf_counter()
    done = 0
    print(f"[data] {label}: transferring...", flush=True)
    while block := source.read(1024 * 1024):
        dest.write(block)
        done += len(block)
        now = time.perf_counter()
        if done == len(block) or now - last >= 5:
            rate = done / max(now-started, 1e-9)
            target = f" / {total/1024**2:.1f} MiB" if total else ""
            eta = f"{max(0, total-done)/rate:.0f}s" if total else "unknown (no Content-Length)"
            print(f"[data] {label}: {done/1024**2:.1f} MiB{target}, "
                  f"{rate/1024**2:.2f} MiB/s, ETA={eta}", flush=True)
            last = now
    print(f"[data] {label}: transferred {done/1024**2:.1f} MiB in {time.perf_counter()-started:.1f}s", flush=True)


def download_sources(directory, limit=None):
    """The official MuSiQue distribution includes Hotpot/2Wiki support-format data.

    Only answerable MuSiQue is fetched. A limit produces explicitly marked smoke
    subsets, not formal training/test sets. Google Drive quota errors are surfaced.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "download_manifest.json"
    previous = read_json(manifest_path) if manifest_path.exists() else None
    if previous is not None and previous["limit"] != limit:
        raise ValueError("download limit changed; use a separate pilot/full directory to avoid relabeling partial files as complete")
    expected = ("musique_ans_train.jsonl", "musique_ans_dev.jsonl", "hotpot_train.jsonl", "hotpot_validation.jsonl")
    if previous and {row["path"] for row in previous.get("sources", [])} == set(expected) and all(
            (directory / name).is_file() and (directory / name).stat().st_size for name in expected):
        for name in expected:
            print(f"[data] Reusing {directory/name}", flush=True)
        print("[data] All source downloads already complete; no network request needed.", flush=True)
        return previous
    print(f"[data] Download sources: limit={limit}, destination={directory}", flush=True)
    source = "https://raw.githubusercontent.com/StonyBrookNLP/musique/main/.all_data_information.json"
    print("[data] Fetching MuSiQue source list from GitHub...", flush=True)
    with urllib.request.urlopen(source, timeout=120) as response:
        info = json.load(response)
    wanted = ["musique_ans_train.jsonl", "musique_ans_dev.jsonl"]
    manifest = {"limit": limit, "formal": limit is None, "sources": []}
    for name in wanted:
        file_id = info["raw_data/" + name]
        url = "https://drive.usercontent.google.com/download?id=" + file_id + "&export=download&confirm=t"
        target = directory / name
        if not target.exists():
            print(f"[data] Requesting {name} from Google Drive...", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                rows = (json.loads(line) for line in response if line.strip())
                rows = itertools.islice(rows, limit) if limit else rows
                write_jsonl(target, _progress_rows(rows, name))
            print(f"[data] Saved {target} ({target.stat().st_size/1024**2:.1f} MiB)", flush=True)
        else:
            print(f"[data] Reusing {target} ({target.stat().st_size/1024**2:.1f} MiB)", flush=True)
        manifest["sources"].append({"path": name, "url": url})
    print("[data] Loading Hugging Face dataset client...", flush=True)
    from datasets import load_dataset
    for split in ("train", "validation"):
        target = directory / f"hotpot_{split}.jsonl"
        if not target.exists():
            print(f"[data] Opening HotpotQA {split} stream from Hugging Face...", flush=True)
            data = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
            rows = itertools.islice(data, limit) if limit else data
            write_jsonl(target, _progress_rows(rows, target.name))
            print(f"[data] Saved {target} ({target.stat().st_size/1024**2:.1f} MiB)", flush=True)
        else:
            print(f"[data] Reusing {target} ({target.stat().st_size/1024**2:.1f} MiB)", flush=True)
        manifest["sources"].append({"path": target.name, "repository": "hotpotqa/hotpot_qa", "split": split})
    write_json(directory / "download_manifest.json", manifest)
    print("[data] HotpotQA and MuSiQue downloads complete.", flush=True)
    return manifest


def normalize_sources(raw, output, *, seed=17, family_links=(), reuse=True, load_questions=True):
    """Reuse completed unchanged artifacts; optionally return None instead of loading questions.

    File size/mtime stamps provide a fast local cache check, not a cryptographic
    re-audit. Set reuse=False after external edits that preserve those attributes.
    """
    raw, output = Path(raw), Path(output)
    family_links = tuple(tuple(pair) for pair in family_links)
    sources = [("hotpot_train.jsonl", hotpot, "train"), ("hotpot_validation.jsonl", hotpot, "dev"),
               ("musique_ans_train.jsonl", musique, "train"), ("musique_ans_dev.jsonl", musique, "dev"),
               ("2wiki_dev.json", twowiki, "test"), ("2wiki_dev.jsonl", twowiki, "test")]

    def stamps(directory, names):
        result = {}
        for name in names:
            path = directory / name
            if path.is_file():
                stat = path.stat()
                result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return result

    download = read_json(raw/"download_manifest.json") if (raw/"download_manifest.json").exists() else None
    source_names = [name for name, _, _ in sources]
    inputs = {"version": 1, "seed": seed, "family_links": digest(family_links), "download": download,
              "source_files": stamps(raw, source_names),
              "adapter_code": digest([(Path(__file__).parent/name).read_text(encoding="utf-8")
                                      for name in ("data.py", "schema.py")])}
    artifacts = ("questions.jsonl", "corpus.jsonl")
    manifest_path = output / "data_manifest.json"
    if reuse and manifest_path.is_file():
        previous = read_json(manifest_path)
        cached = previous.get("normalization_cache", {})
        current_outputs = stamps(output, artifacts)
        if (cached.get("inputs") == inputs and len(current_outputs) == len(artifacts)
                and cached.get("outputs") == current_outputs):
            print("[data] Reusing completed normalization: source/config/output stamps match. "
                  "Skipping raw parsing, splitting, deduplication and rewriting.", flush=True)
            questions = None
            if load_questions:
                questions = [Question.from_dict(row) for row in _progress_rows(
                    read_jsonl(output/"questions.jsonl"), "load cached questions", total=previous["questions"])]
            return questions, previous
        print("[data] Normalization cache missing or changed; rebuilding (legacy manifests require one rebuild).", flush=True)
    questions, provenance = [], []
    for name, adapter, split in sources:
        path = raw / name
        if not path.exists(): continue
        print(f"[data] Reading {path}...", flush=True)
        rows = list(_progress_rows(load_raw(path), f"read {name}"))
        questions.extend(adapter(row, split=split) for row in
                         _progress_rows(rows, f"normalize {name}", total=len(rows)) if row.get("answerable", True))
        print(f"[data] Fingerprinting {name}...", flush=True)
        provenance.append({"path": name, "records": len(rows), "sha256_content": digest(rows)})
    if not questions: raise ValueError("no dataset files found; run download_sources or supply official raw files")
    print(f"[data] Splitting {len(questions):,} questions by family and auditing leakage...", flush=True)
    questions = split_questions(questions, family_links, seed)
    counts = audit_splits(questions)
    write_jsonl(output / "questions.jsonl", _progress_rows(questions, "write questions", total=len(questions)))
    evidence_count = sum(len(q.evidence) for q in questions)
    corpus = deduplicate(_progress_rows((e for q in questions for e in q.evidence),
                                       "deduplicate evidence", total=evidence_count))
    write_jsonl(output / "corpus.jsonl", _progress_rows(corpus, "write corpus", total=len(corpus)))
    print("[data] Fingerprinting questions/corpus and writing data manifest...", flush=True)
    manifest = {"sources": provenance, "question_fingerprint": digest(questions), "corpus_fingerprint": digest(corpus),
                "questions": len(questions), "corpus_documents": len(corpus),
                "splits": counts, "seed": seed, "family_links": len(family_links),
                "corpus_protocol": "union of raw source paragraphs; no answers, decompositions or support labels indexed",
                "scope": "bounded benchmark corpus, not full Wikipedia; report this corpus scope explicitly"}
    if download is not None:
        manifest["download"] = download
    if stamps(raw, source_names) != inputs["source_files"]:
        raise RuntimeError("Raw source files changed during normalization; rerun before using these artifacts")
    manifest["normalization_cache"] = {"inputs": inputs, "outputs": stamps(output, artifacts)}
    write_json(manifest_path, manifest)
    print(f"[data] Normalization complete: {len(questions):,} questions, "
          f"{len(corpus):,} unique evidence paragraphs; saved to {output}", flush=True)
    return questions if load_questions else None, manifest


def download_2wiki(directory):
    """Official April 2021 archive, extracted by explicit filename."""
    import zipfile
    import os
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    url = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
    provenance = {"url": url, "format": "official April 2021", "split": "dev used as cross-dataset test"}
    source_path = directory / "2wiki_source.json"
    if source_path.is_file() and read_json(source_path) == provenance and all(
            (directory/name).is_file() and (directory/name).stat().st_size
            for name in ("2wiki_dev.json", "2wiki_id_aliases.json")):
        print("[data] Reusing completed 2Wiki extraction; no download or unzip needed.", flush=True)
        return
    archive = directory/"2wiki_official.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        print("[data] Requesting official 2Wiki archive from Dropbox...", flush=True)
        partial = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as f:
            total = int(response.headers.get("Content-Length", "0")) or None
            _copy_with_progress(response, f, "download 2Wiki ZIP", total=total)
        print("[data] Checking 2Wiki ZIP...", flush=True)
        if not zipfile.is_zipfile(partial):
            raise ValueError("2Wiki download is not a complete ZIP; rerun to retry the download")
        os.replace(partial, archive)
    else:
        print(f"[data] Reusing {archive}", flush=True)
    with zipfile.ZipFile(archive) as z:
        for target, basename in (("2wiki_dev.json", "dev.json"), ("2wiki_id_aliases.json", "id_aliases.json")):
            names = [n for n in z.namelist() if Path(n).name == basename]
            if len(names) != 1: raise ValueError(f"expected one {basename} in the official archive")
            if (directory/target).exists():
                print(f"[data] Reusing {directory/target}", flush=True)
                continue
            partial = directory / (target + ".part")
            with z.open(names[0]) as source, partial.open("wb") as dest:
                _copy_with_progress(source, dest, f"extract {target}", total=z.getinfo(names[0]).file_size)
            os.replace(partial, directory/target)
    write_json(source_path, provenance)
    print("[data] 2Wiki download/extraction complete.", flush=True)


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
