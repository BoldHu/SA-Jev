"""Independent hard perturbation evaluation; never a stability-training dataset."""
from __future__ import annotations

from dataclasses import asdict, replace
import torch
from .schema import State, Labels, labels_for
from .io import digest, write_jsonl
from .metrics import distribution_change, ndcg


def review_packets(questions, states, output, limit=600):
    lookup = {q.id: q for q in questions}; rows = []
    for state in states:
        q = lookup[state.question_id]
        if q.split != "heldout": raise ValueError("robustness packets must use reserved question families")
        y = labels_for(q, state)
        if not y.gold_available: continue
        # Draft related distractors; annotation must establish that they add no
        # support, alter no effective condition and preserve the intended labels.
        distractors = [e for e in q.evidence if e.id in q.known_negative_ids]
        proposals = sorted(distractors, key=lambda e: -len(set(e.text.lower().split()) & set(q.question.lower().split())))[:3]
        if len(proposals) < 3: continue
        variants = [{"state": asdict(replace(state, context=e.title + "\n" + e.text)),
                     "source": e.source, "verified": False, "reviewer": "", "rationale": ""} for e in proposals]
        changed = None
        if state.budget:
            for eid, gain in zip(y.candidate_ids, y.gains):
                if gain > 0:
                    s = state.read(eid); new_y = labels_for(q, s)
                    changed = {"state": asdict(s), "labels": asdict(new_y), "verified": False,
                               "reviewer": "", "rationale": "", "change_type": "new_selected_evidence"}
                    break
        if changed is None: continue
        rows.append({"packet_id": digest(state), "split": "heldout", "clean": asdict(state),
                     "clean_labels": asdict(y), "perturbations": variants, "effective": changed})
        if len(rows) >= limit: break
    write_jsonl(output, rows)
    return rows


def validate_packet(packet, reserved_groups):
    clean = State.from_dict(packet["clean"])
    labels = Labels.from_dict(packet["clean_labels"])
    if packet["split"] != "heldout" or clean.group_id not in set(reserved_groups):
        raise ValueError("test family was not reserved independently")
    if len(packet["perturbations"]) != 3: raise ValueError("each packet needs three reviewed irrelevant perturbations")
    views = []
    for v in packet["perturbations"]:
        if not v["verified"] or not v["reviewer"] or not v["rationale"]:
            raise ValueError("unverified perturbation; do not treat a generated draft as verified")
        s = State.from_dict(v["state"])
        if replace(s, context=clean.context) != clean or s.context == clean.context:
            raise ValueError("irrelevant perturbation must only change contextual text")
        views.append(s)
    e = packet["effective"]
    if not e["verified"] or not e["reviewer"] or not e["rationale"]:
        raise ValueError("effective-state change needs verified labels and rationale")
    changed = State.from_dict(e["state"]); changed_labels = Labels.from_dict(e["labels"])
    if changed == clean or changed.pool != clean.pool or changed.group_id != clean.group_id:
        raise ValueError("effective change must retain the fixed candidate pool and question family")
    if changed.selected_ids == clean.selected_ids and changed.conditions == clean.conditions and changed.question == clean.question:
        raise ValueError("context-only or budget-only changes do not establish changed evidence needs")
    if changed_labels.candidate_ids != tuple(x.id for x in changed.candidates):
        raise ValueError("changed-state labels misaligned")
    return clean, labels, views, changed, changed_labels


@torch.no_grad()
def evaluate_robustness(model, packets, reserved_groups, name="model", threshold=.5):
    model.eval(); rows = []
    for packet in packets:
        clean, y, views, changed, yc = validate_packet(packet, reserved_groups)
        p0 = model(clean, include_rel=False)
        for i, view in enumerate(views):
            p = model(view, include_rel=False)
            rows.append({"question_id": clean.question_id, "group_id": clean.group_id, "method": name,
                         "kind": "irrelevant", "view": i, "pool_hash": clean.pool.fingerprint,
                         **distribution_change(p0.candidate_ids, p0.p.cpu().tolist(), float(p0.h), p.candidate_ids, p.p.cpu().tolist(), float(p.h), threshold),
                         "ndcg": ndcg(y.gains, p.p.cpu().tolist()) if y.rank_valid else None,
                         "false_stop": float(p.h >= threshold and not y.sufficient) if y.suf_valid else None})
        p = model(changed, include_rel=False)
        # Changed candidate sets are deliberately NOT forced into an invariant JS.
        top = int(torch.argmax(p.p)) if len(p.p) else None
        rows.append({"question_id": clean.question_id, "group_id": clean.group_id, "method": name,
                     "kind": "effective", "view": 0, "pool_hash": clean.pool.fingerprint,
                     "suf_correct": float((p.h >= threshold) == bool(yc.sufficient)) if yc.suf_valid else None,
                     "next_support_correct": float(yc.gains[top] > 0) if top is not None and yc.rank_valid else None,
                     "ndcg": ndcg(yc.gains, p.p.cpu().tolist()) if yc.rank_valid else None,
                     "false_stop": float(p.h >= threshold and not yc.sufficient) if yc.suf_valid else None})
    return rows
