"""Real transitions in a fixed pool, full-trajectory probability and conservative pairs."""
from __future__ import annotations

from dataclasses import dataclass
import random
import torch
from .io import digest
from .schema import State, labels_for
from .prompts import View


@dataclass(frozen=True)
class Trajectory:
    initial: State
    actions: tuple[str, ...]  # evidence ID for READ, or the literal STOP
    termination: str  # stop, budget, exhausted

    def replay(self):
        state = self.initial
        steps = []
        for index, action in enumerate(self.actions):
            if state.budget <= 0:
                raise ValueError("budget exhaustion is not an autonomous STOP")
            steps.append((state, action))
            if action == "STOP":
                if index != len(self.actions) - 1 or self.termination != "stop":
                    raise ValueError("STOP must be the last action")
            else:
                state = state.read(action)
        if self.termination == "stop":
            if not self.actions or self.actions[-1] != "STOP":
                raise ValueError("missing autonomous STOP")
        elif self.termination == "budget":
            if state.budget != 0 or "STOP" in self.actions:
                raise ValueError("invalid budget termination")
        elif self.termination == "exhausted":
            if state.candidates or "STOP" in self.actions:
                raise ValueError("invalid candidate exhaustion")
        else:
            raise ValueError("unknown trajectory termination")
        return steps, state

    @property
    def fingerprint(self): return digest(self)

    @classmethod
    def from_dict(cls, d):
        return cls(State.from_dict(d["initial"]), tuple(d["actions"]), d["termination"])


@dataclass(frozen=True)
class PreferencePair:
    chosen: Trajectory
    rejected: Trajectory
    reason: str

    def __post_init__(self):
        if self.chosen.initial != self.rejected.initial:
            raise ValueError("preferences must share an identical starting state and fixed pool")
        self.chosen.replay(); self.rejected.replay()

    @property
    def fingerprint(self): return digest(self)

    @classmethod
    def from_dict(cls, d):
        return cls(Trajectory.from_dict(d["chosen"]), Trajectory.from_dict(d["rejected"]), d["reason"])


def trajectory_log_probability(model, trajectory):
    steps, _ = trajectory.replay()
    values = []
    for state, action in steps:
        prediction = model(state, View(0), include_rel=False)
        if action == "STOP":
            values.append(prediction.log_h[1])
        else:
            index = prediction.candidate_ids.index(action)
            values.append(prediction.log_h[0] + prediction.log_p[index])
    if not values:
        return next(model.parameters()).sum() * 0.
    return torch.stack(values).sum()  # no length normalization


def _rollout(q, initial, mode, rng):
    state, actions = initial, []
    while state.budget and state.candidates:
        y = labels_for(q, state)
        if mode == "early" or (y.suf_valid and y.sufficient and mode != "redundant"):
            return Trajectory(initial, tuple(actions + ["STOP"]), "stop")
        if mode == "random":
            e = rng.choice(state.candidates)
        else:
            gain = dict(zip(y.candidate_ids, y.gains))
            ordered = sorted(state.candidates, key=lambda e: (-gain[e.id], e.id))
            e = ordered[-1] if mode == "redundant" and not actions else ordered[0]
        actions.append(e.id)
        state = state.read(e.id)
    return Trajectory(initial, tuple(actions), "budget" if not state.budget else "exhausted")


def quality(q, trajectory):
    _, final = trajectory.replay()
    y = labels_for(q, final)
    if not y.gold_available or not y.suf_valid:
        return None
    return bool(y.sufficient), len(final.selected_ids) - len(trajectory.initial.selected_ids)


def preference(q, a, b):
    qa, qb = quality(q, a), quality(q, b)
    if qa is None or qb is None or a.actions == b.actions:
        return None
    if qa[0] != qb[0]:
        return PreferencePair(a, b, "sufficient_vs_missing_support") if qa[0] else PreferencePair(b, a, "sufficient_vs_missing_support")
    if qa[0] and abs(qa[1] - qb[1]) >= 1:
        return PreferencePair(a, b, "sufficient_with_fewer_reads") if qa[1] < qb[1] else PreferencePair(b, a, "sufficient_with_fewer_reads")
    return None  # uncertain failure-vs-failure and ties are not forced preferences


def build_preferences(questions, initial_states, limit=3000, seed=17):
    lookup = {q.id: q for q in questions}
    rng = random.Random(seed)
    states = list(initial_states); rng.shuffle(states)
    result, seen = [], set()
    for state in states:
        q = lookup[state.question_id]
        if q.split != "train": raise ValueError("preference data is training-only")
        candidates = [_rollout(q, state, mode, rng) for mode in ("greedy", "redundant", "early", "random")]
        for other in candidates[1:]:
            pair = preference(q, candidates[0], other)
            if pair and pair.fingerprint not in seen:
                result.append(pair); seen.add(pair.fingerprint)
                if len(result) >= limit: return result
    return result
