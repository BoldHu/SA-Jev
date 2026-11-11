"""Verify/repair pinned OpenJev artifacts without changing weights or manifests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from uuid import uuid4

from .io import digest


def _valid(path, info):
    if not path.is_file() or path.stat().st_size != info["bytes"]:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == info["sha256"]


def _copy_verified(source, target, info):
    """Stream from Drive instead of mmap; publish only complete verified bytes."""
    if not source.is_file() or source.stat().st_size != info["bytes"]:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp-" + uuid4().hex)
    started = last = time.perf_counter()
    size = 0; checksum = hashlib.sha256()
    try:
        with source.open("rb") as src, temp.open("xb") as dst:
            while block := src.read(8 * 1024 * 1024):
                dst.write(block); checksum.update(block); size += len(block)
                now = time.perf_counter()
                if now - last >= 5:
                    eta = (now-started) / max(size, 1) * max(0, info["bytes"]-size)
                    print(f"[OpenJev] {target.name}: {size/1024**3:.2f}/{info['bytes']/1024**3:.2f} GiB "
                          f"copied; ETA={eta:.0f}s", flush=True)
                    last = now
        if size != info["bytes"] or checksum.hexdigest() != info["sha256"]:
            return False
        os.replace(temp, target)
        return True
    finally:
        temp.unlink(missing_ok=True)


def prepare_openjev_artifact(artifact, revision=None, *, local_directory=None):
    """Resolve pinned Hub files, repair only failed entries, stage Drive bytes locally.

    Local user-supplied artifacts are checked but never repaired from an assumed
    repository. Hub repair preserves the invalid snapshot entry and downloads the
    same resolved commit exactly once. No data/index/checkpoint directory is touched.
    """
    from openjev.runtime.hub import resolve_artifact
    from openjev.artifacts import load_manifest
    local_input = Path(artifact).expanduser().is_dir()
    snapshot = Path(resolve_artifact(artifact, revision=revision)).absolute()
    print(f"[OpenJev] Checking artifact: {snapshot}", flush=True)
    if local_input:
        load_manifest(snapshot, verify_hashes=True)
        return str(snapshot)
    # snapshot_download resolves a ref to a commit directory. Never repair from main.
    commit = snapshot.name
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("OpenJev repair requires a resolved immutable Hub commit")
    raw = (snapshot / "manifest.json").read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    if (snapshot / "COMPLETE").read_text(encoding="utf-8").strip() != manifest_sha:
        raise ValueError("OpenJev manifest/COMPLETE mismatch; refusing weight repair from an unverified manifest")
    manifest = json.loads(raw)
    if manifest.get("artifact_format_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("invalid OpenJev artifact manifest")
    files = manifest["files"]
    for name, info in files.items():
        rel = PurePosixPath(name)
        if (rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name
                or not rel.parts or name in {"manifest.json", "COMPLETE"}
                or not isinstance(info.get("bytes"), int) or info["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", info.get("sha256", ""))):
            raise ValueError(f"invalid artifact manifest entry: {name}")
    if "decision_head.safetensors" not in files:
        raise ValueError("complete pretrained decision head required")
    stage = local_directory is not None or snapshot.as_posix().startswith("/content/drive/")
    target = (Path(local_directory) if local_directory is not None else
              Path(tempfile.gettempdir()) / "sajev_openjev_models" / digest([artifact, commit, manifest_sha])[:24]) if stage else snapshot
    if stage:
        target = target.absolute()
        if target == snapshot or target.is_relative_to(snapshot):
            raise ValueError("local runtime copy must be outside the Hub snapshot")
        print(f"[OpenJev] Loading verified ordinary files from local runtime: {target}", flush=True)
    repaired = 0
    for i, (name, info) in enumerate(files.items(), 1):
        source, dest = snapshot / name, target / name
        print(f"[OpenJev] {i}/{len(files)} {name}: expected {info['bytes']:,} bytes; checking SHA-256...", flush=True)
        # Reuse a valid local runtime copy without rereading multi-GB Drive files.
        if (not stage or not dest.is_symlink()) and _valid(dest, info):
            continue
        if stage and _copy_verified(source, dest, info):
            continue
        actual_size = source.stat().st_size if source.is_file() else None
        print(f"[OpenJev] Invalid source: {name}; actual bytes={actual_size}; expected={info['bytes']}. "
              "Re-downloading this file at the same commit once.", flush=True)
        if source.exists() or source.is_symlink():
            # Rename only the snapshot entry, never move a symlink's target/blob.
            preserved = source.with_name(source.name + ".invalid-" + uuid4().hex[:8])
            source.rename(preserved)
            print(f"[OpenJev] Preserved invalid snapshot entry: {preserved}", flush=True)
        from huggingface_hub import hf_hub_download
        downloaded = Path(hf_hub_download(str(artifact), name, revision=commit,
                            cache_dir=str(snapshot.parent.parent.parent), force_download=True)).absolute()
        if downloaded != source:
            raise RuntimeError("HF download returned an unexpected snapshot path; refusing to load it")
        ok = _copy_verified(source, dest, info) if stage else _valid(source, info)
        if not ok:
            raise RuntimeError(f"OpenJev file still fails size/SHA-256 validation after one repair: {name}. "
                               "Check disk space, download and Drive connection; keep raw data and graph caches.")
        repaired += 1
    if stage:
        # Small metadata copies also use checked, atomic publication; no HF symlinks.
        for name in ("manifest.json", "COMPLETE"):
            source = snapshot / name
            content = source.read_bytes()
            info = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            if not _copy_verified(source, target / name, info):
                raise RuntimeError(f"Unable to stage artifact metadata: {name}")
    # All contents were hashed above. Upstream still enforces manifest/head/calibration.
    load_manifest(target, verify_hashes=False)
    print(f"[OpenJev] Artifact verified; repaired files={repaired}; loading path={target}", flush=True)
    return str(target)
