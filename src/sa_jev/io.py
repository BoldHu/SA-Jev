from __future__ import annotations

import hashlib
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
