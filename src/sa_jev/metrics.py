from __future__ import annotations

from collections import Counter
import re
import string
import numpy as np


def ndcg(gains, scores, k=None):
    gains, scores = np.asarray(gains, dtype=float), np.asarray(scores, dtype=float)
    if len(gains) != len(scores): raise ValueError("ranking alignment mismatch")
    k = len(gains) if k is None else min(k, len(gains))
    if not k: return None
    discount = np.log2(np.arange(k) + 2)
    actual = ((2**gains[np.argsort(-scores, kind="stable")[:k]] - 1) / discount).sum()
    ideal = ((2**np.sort(gains)[::-1][:k] - 1) / discount).sum()
    return float(actual / ideal) if ideal > 0 else None


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or not len(p): return None
    if not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1): raise ValueError("invalid probability distribution")
    m = (p + q) / 2
    def kl(x):
        mask = x > 0
        return np.sum(x[mask] * np.log(x[mask] / m[mask]))
    return float((kl(p) + kl(q)) / 2)


def normalize_answer(s):
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def answer_metrics(prediction, gold, aliases=()):
    def score(target):
        p, g = normalize_answer(prediction), normalize_answer(target)
        exact = float(p == g)
        if (p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}) and p != g:
            return exact, 0.
        pt, gt = p.split(), g.split()
        common = sum((Counter(pt) & Counter(gt)).values())
        f1 = 2 * common / (len(pt) + len(gt)) if pt or gt else 1.
        return exact, f1
    values = [score(g) for g in (gold, *aliases)]
    return {"qa_em": max(x[0] for x in values), "qa_f1": max(x[1] for x in values)}


def binary_metrics(probabilities, targets, threshold=.5):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if not len(p): return {"suf_accuracy": None, "suf_brier": None, "suf_ece": None, "suf_auroc": None}
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == i) * abs(p[bins == i].mean() - y[bins == i].mean()) for i in range(10) if np.any(bins == i))
    pos, neg = p[y == 1], p[y == 0]
    auc = float(np.mean([(a > b) + .5*(a == b) for a in pos for b in neg])) if len(pos) and len(neg) else None
    return {"suf_accuracy": float(((p >= threshold) == y).mean()), "suf_brier": float(((p-y)**2).mean()),
            "suf_ece": float(ece), "suf_auroc": auc}


def distribution_change(ids_a, p_a, h_a, ids_b, p_b, h_b, threshold=.5):
    if set(ids_a) != set(ids_b): raise ValueError("distribution comparison requires the same effective candidate IDs")
    p_a = np.asarray(p_a); p_b = np.asarray([p_b[ids_b.index(i)] for i in ids_a])
    k = min(5, len(ids_a))
    ra, rb = np.argsort(-p_a, kind="stable"), np.argsort(-p_b, kind="stable")
    return {"js": js_divergence(p_a, p_b), "top1_flip": float(ra[0] != rb[0]) if k else None,
            "top5_change": 1 - len(set(ra[:k]) & set(rb[:k])) / k if k else None,
            "suf_flip": float((h_a >= threshold) != (h_b >= threshold)), "suf_delta": abs(h_a - h_b)}


def paired_bootstrap(rows_a, rows_b, metric, repetitions=2000, seed=17):
    """Pair on question + seed, then resample question families (not individual states)."""
    key = lambda r: (r["question_id"], r.get("seed", 17))
    a = {key(r): r for r in rows_a}; b = {key(r): r for r in rows_b}
    if set(a) != set(b): raise ValueError("paired methods must cover identical question/seed sets, including failures")
    grouped = {}
    for k in a:
        if a[k]["pool_hash"] != b[k]["pool_hash"]: raise ValueError("paired methods used different candidate pools")
        for field in ("budget", "answer_model", "threshold"):
            if a[k].get(field) != b[k].get(field): raise ValueError(f"paired methods differ in {field}")
        if a[k].get(metric) is None or b[k].get(metric) is None: continue
        grouped.setdefault(a[k]["group_id"], []).append(a[k][metric] - b[k][metric])
    values = np.asarray([np.mean(v) for v in grouped.values()])
    if not len(values): return {"n_groups": 0, "difference": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)]
    low, high = np.quantile(boot, [.025, .975])
    return {"n_groups": len(values), "difference": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}
