"""Measured-only CSV and publication-exportable PNG/PDF figures."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
from .io import write_json


def csv_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def export_report(rows, directory, metrics=("support_recall", "ndcg", "suf_brier", "qa_f1", "seconds"), group_keys=("method", "dataset")):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    csv_rows(directory / "results.csv", rows)
    groups = {}
    for row in rows:
        key = tuple(row.get(k, "all") for k in group_keys)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, group in groups.items():
        result = {**dict(zip(group_keys, key)), "n_rows": len(group), "errors": sum(r.get("status") == "error" for r in group)}
        for metric in metrics:
            # Mean over question groups within each seed, then across seeds.
            seeds = {}
            for r in group:
                if r.get(metric) is not None:
                    seeds.setdefault(r.get("seed", 17), {}).setdefault(r.get("group_id", r.get("question_id", "aggregate")), []).append(r[metric])
            means = [np.mean([np.mean(v) for v in groups.values()]) for groups in seeds.values()]
            result[metric + "_mean"] = float(np.mean(means)) if means else None
            result[metric + "_std_across_seeds"] = float(np.std(means, ddof=1)) if len(means) > 1 else None
            result[metric + "_n"] = sum(len(g) for g in seeds.values())
        summary.append(result)
    csv_rows(directory / "summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    available = [m for m in metrics if any(x.get(m+"_mean") is not None for x in summary)]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(4.2*len(available), 4.5), squeeze=False)
        for ax, metric in zip(axes[0], available):
            valid = [x for x in summary if x.get(metric+"_mean") is not None]
            ax.bar(range(len(valid)), [x[metric+"_mean"] for x in valid],
                   yerr=[x[metric+"_std_across_seeds"] or 0 for x in valid], color="#356c9b")
            ax.set_xticks(range(len(valid)), [" / ".join(str(x[k]) for k in group_keys) for x in valid], rotation=35, ha="right")
            ax.set_ylabel(metric); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(directory / "figure.png", dpi=180); fig.savefig(directory / "figure.pdf"); plt.close(fig)
    write_json(directory / "report_metadata.json", {"rows": len(rows), "metrics": list(metrics),
               "note": "Only measured values; blank means unavailable. Single-seed variance is not estimated."})
    return summary


def training_report(history, directory):
    csv_rows(Path(directory) / "training.csv", history)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for field in ("task", "rank", "suf", "rel", "dpo", "stability"):
        axes[0].plot([r["step"] for r in history], [r[field] for r in history], label=field)
    valid = [r for r in history if "validation" in r]
    axes[0].plot([r["step"] for r in valid], [r["validation"]["task"] for r in valid], "ko", label="validation task")
    axes[0].legend(fontsize=7); axes[0].set_ylabel("loss")
    for field in ("peak_allocated_gib", "peak_reserved_gib"):
        axes[1].plot([r["step"] for r in history], [r[field] for r in history], label=field)
    axes[1].legend(fontsize=7); axes[1].set_ylabel("GiB")
    axes[2].plot([r["step"] for r in history], [r["eta_seconds"] for r in history]); axes[2].set_ylabel("ETA (s)")
    for ax in axes: ax.set_xlabel("optimizer step")
    fig.tight_layout(); fig.savefig(Path(directory)/"training.png", dpi=180); fig.savefig(Path(directory)/"training.pdf"); plt.close(fig)
