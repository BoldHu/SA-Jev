from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from pathlib import Path
from dataclasses import asdict, is_dataclass


def progress_rows(rows, label, *, total=None):
    """Plain flushed stdout, including in notebooks without working widgets.

    Reports completed items after the caller processes them. A slow individual
    item can delay the next line; this is progress reporting, not a heartbeat.
    """
    if total is None and hasattr(rows, "__len__"):
        total = len(rows)
    started = last = time.perf_counter()
    count = 0
    target = f"/{total:,}" if total is not None else ""
    print(f"[data] {label}: starting 0{target}; waiting for records...", flush=True)
    for count, row in enumerate(rows, 1):
        yield row
        now = time.perf_counter()
        if count == 1 or now - last >= 5:
            rate = count / max(now - started, 1e-9)
            eta = f"{max(0, total-count)/rate:.0f}s" if total is not None else "unknown (streaming)"
            print(f"[data] {label}: {count:,}{target} records, {rate:.1f}/s, ETA={eta}", flush=True)
            last = now
    print(f"[data] {label}: finished {count:,} records in {time.perf_counter()-started:.1f}s", flush=True)


def canonical(value):
    return json.dumps(asdict(value) if is_dataclass(value) else value,
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda x: asdict(x) if is_dataclass(x) else str(x) if isinstance(x, Path) else _unsupported(x))


def _unsupported(value):
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with atomic_text(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def atomic_text(path):
    """Publish a complete file, so a disconnected writer is never mistaken for a finished download."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            yield stream
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    with atomic_text(path) as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
