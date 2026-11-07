from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
import unicodedata
from .io import digest


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


@dataclass(frozen=True)
class Evidence:
    id: str
    title: str
    text: str
    source: str = ""

    @classmethod
    def make(cls, title, text, source=""):
        return cls("e_" + digest([normalize(title), normalize(text)])[:20], title, text, source)


def deduplicate(evidence):
    result, seen = [], set()
    for e in evidence:
        # Text equality deduplicates aliases/titles; never pad a short pool with copies.
        key = normalize(e.text)
        if key and key not in seen:
            seen.add(key)
            result.append(e)
    return tuple(result)


@dataclass(frozen=True)
class Question:
    id: str
    group_id: str
    dataset: str
    split: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...]
    # Gold stays outside State/Decision/model inputs.
    units: tuple[str, ...]
    coverage: dict[str, tuple[str, ...]]
    unit_text: dict[str, str]
    known_negative_ids: tuple[str, ...] = ()
    gold_complete: bool = True
    answer_aliases: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["evidence"] = tuple(Evidence(**x) for x in d["evidence"])
        for k in ("units", "known_negative_ids", "answer_aliases"):
            d[k] = tuple(d.get(k, ()))
        d["coverage"] = {k: tuple(v) for k, v in d["coverage"].items()}
        return cls(**d)


@dataclass(frozen=True)
class Pool:
    question_id: str
    candidates: tuple[Evidence, ...]
    retriever: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if len({e.id for e in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate candidate IDs")
        if len(deduplicate(self.candidates)) != len(self.candidates):
            raise ValueError("duplicate candidate content")

    @property
    def fingerprint(self):
        return digest(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{**d, "candidates": tuple(Evidence(**x) for x in d["candidates"])})


@dataclass(frozen=True)
class State:
    question_id: str
    group_id: str
    question: str
    pool: Pool
    selected_ids: tuple[str, ...] = ()
    conditions: str = ""
    context: str = ""
    budget: int = 4

    def __post_init__(self):
        if self.pool.question_id != self.question_id:
            raise ValueError("state and pool question mismatch")
        ids = {e.id for e in self.pool.candidates}
        if len(set(self.selected_ids)) != len(self.selected_ids) or not set(self.selected_ids) <= ids:
            raise ValueError("selected IDs must be unique members of the fixed initial pool")
        if self.budget < 0:
            raise ValueError("negative budget")

    @property
    def candidates(self):
        return tuple(e for e in self.pool.candidates if e.id not in self.selected_ids)

    @property
    def selected(self):
        by_id = {e.id: e for e in self.pool.candidates}
        return tuple(by_id[x] for x in self.selected_ids)

    def read(self, evidence_id):
        if self.budget <= 0 or evidence_id not in {e.id for e in self.candidates}:
            raise ValueError("illegal READ action")
        return replace(self, selected_ids=self.selected_ids + (evidence_id,), budget=self.budget - 1)

    @classmethod
    def from_dict(cls, d):
        return cls(**{**d, "pool": Pool.from_dict(d["pool"]), "selected_ids": tuple(d["selected_ids"])})


@dataclass(frozen=True)
class Labels:
    candidate_ids: tuple[str, ...]
    gains: tuple[float, ...]
    rank: tuple[float, ...]
    rank_valid: bool
    sufficient: float
    suf_valid: bool
    relevant: tuple[float, ...]
    rel_valid: tuple[bool, ...]
    gold_available: bool
    covered: tuple[str, ...]

    @classmethod
    def from_dict(cls, d):
        fields = ("candidate_ids", "gains", "rank", "relevant", "rel_valid", "covered")
        return cls(**{k: tuple(v) if k in fields else v for k, v in d.items()})


@dataclass(frozen=True)
class TaskExample:
    state: State
    labels: Labels
    split: str

    @classmethod
    def from_dict(cls, d):
        return cls(State.from_dict(d["state"]), Labels.from_dict(d["labels"]), d["split"])


def labels_for(q: Question, state: State):
    if q.id != state.question_id or q.question != state.question:
        raise ValueError("labels must match the actual question; changed questions need new gold")
    if state.conditions:
        raise ValueError("condition changes need explicitly reviewed task labels; coverage-only labeling cannot infer their effect")
    units = set(q.units)
    known_negative = set(q.known_negative_ids)

    def annotation(e):
        if e.id in q.coverage:
            return set(q.coverage[e.id]), True
        if e.id in known_negative:
            return set(), True
        # LightRAG assigns chunk IDs. Exact source-paragraph matches (with optional
        # title prefix) preserve its existing annotations despite the new ID.
        for original in q.evidence:
            same_source = (bool(e.title) and normalize(e.title) == normalize(original.title)) or (bool(e.source) and e.source == original.source)
            if same_source and normalize(e.text) in {normalize(original.text), normalize(original.title + "\n" + original.text)}:
                if original.id in q.coverage:
                    return set(q.coverage[original.id]), True
        # Exact gold sentence/paragraph containment permits positive transfer only.
        # Bind support to its source: identical pronoun-bearing sentences from
        # unrelated pages are not evidence for the same support fact.
        source_units = set().union(*(set(q.coverage.get(original.id, ())) for original in q.evidence
                         if (bool(e.title) and normalize(e.title) == normalize(original.title))
                         or (bool(e.source) and e.source == original.source)))
        covered = {u for u in source_units if q.unit_text.get(u) and normalize(q.unit_text[u]) in normalize(e.text)}
        # Even a positive chunk can include unannotated alternative evidence: only
        # its known support coverage is used; unknown zero-gain items stay masked.
        return covered, bool(covered)

    selected = [annotation(e) for e in state.selected]
    covered = set().union(*(c for c, _ in selected)) if selected else set()
    sufficient = bool(units) and units <= covered
    gold = bool(units) and q.gold_complete
    suf_valid = gold and (sufficient or all(known for _, known in selected))
    annotations = [annotation(e) for e in state.candidates]
    gains = tuple(len(c & (units - covered)) / len(units) if units else 0. for c, _ in annotations)
    total = sum(gains)
    # A zero target in softmax is negative supervision. Mask the entire ranking
    # when any candidate's coverage is unknown; do not silently assign it zero.
    rank_valid = gold and suf_valid and not sufficient and total > 0 and all(k for _, k in annotations)
    return Labels(tuple(e.id for e in state.candidates), gains,
                  tuple(g / total if total else 0. for g in gains), rank_valid,
                  float(sufficient), suf_valid,
                  tuple(float(bool(c & units)) for c, _ in annotations),
                  tuple(known and gold for _, known in annotations), gold,
                  tuple(sorted(covered & units)))
