from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_valid(values, reference):
    return torch.stack(values).mean() if values else reference.sum() * 0.


def task_loss(predictions, labels):
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("nonempty aligned predictions and labels required")
    rank, suf, rel = [], [], []
    for p, y in zip(predictions, labels):
        if p.candidate_ids != y.candidate_ids:
            raise ValueError("candidate IDs not aligned with task labels")
        if y.rank_valid:
            target = p.log_p.new_tensor(y.rank)
            positive = target > 0
            rank.append(-(target[positive] * p.log_p[positive]).sum())
        if y.suf_valid:
            suf.append(-p.log_h[int(y.sufficient)])
        if p.log_rel.shape[0] != len(y.rel_valid):
            raise ValueError("auxiliary Noul predictions required for task training")
        rel.extend(-p.log_rel[i, int(y.relevant[i])] for i, valid in enumerate(y.rel_valid) if valid)
    anchor = predictions[0].log_h
    components = {"rank": mean_valid(rank, anchor), "suf": mean_valid(suf, anchor),
                  "rel": mean_valid(rel, anchor)}
    components["task"] = components["rank"] + components["suf"] + .2 * components["rel"]
    return components, {"rank": len(rank), "suf": len(suf), "rel": len(rel)}


def stability_loss(a, b, labels, states_a, states_b, tolerance=.05):
    rank, suf = [], []
    for pa, pb, y, sa, sb in zip(a, b, labels, states_a, states_b):
        # A semantic State equality check forbids matching real transitions,
        # including budget/conditions/context/content changes. Views live outside State.
        if sa != sb or not y.gold_available:
            continue
        if y.suf_valid:
            suf.append(F.relu(torch.abs(pa.h - pb.h) - tolerance).square())
        if y.suf_valid and not y.sufficient and pa.candidate_ids and set(pa.candidate_ids) == set(pb.candidate_ids):
            index = [pb.candidate_ids.index(e) for e in pa.candidate_ids]
            rank.append(F.relu(torch.linalg.vector_norm(pa.p - pb.p[index]) - tolerance).square())
    return mean_valid(rank, a[0].log_h) + mean_valid(suf, a[0].log_h), {"stab_rank": len(rank), "stab_suf": len(suf)}


def dpo_loss(chosen_logp, rejected_logp, reference_chosen, reference_rejected, beta=.1):
    advantage = (chosen_logp - rejected_logp) - (reference_chosen - reference_rejected).detach()
    return -F.logsigmoid(beta * advantage).mean()
