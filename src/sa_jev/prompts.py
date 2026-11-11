"""Whitelist-only model inputs. Gold, answers and decomposition never enter here."""
from __future__ import annotations

from dataclasses import dataclass
import json
from .schema import State, Evidence


@dataclass(frozen=True)
class View:
    style: int = 0
    context_suffix: str = ""  # Only supplied by the reviewed training-context bank.


def render_state(state: State, view=View()):
    payload = {"question": state.question, "conditions": state.conditions,
               "selected_evidence": [{"id": e.id, "title": e.title, "text": e.text} for e in state.selected],
               "context": "\n\n".join(x for x in (state.context, view.context_suffix) if x),
               "remaining_budget": state.budget}
    # Review metadata never enters the prompt. Original state/trajectory stays immutable.
    if view.style % 3 == 0:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if view.style % 3 == 1:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in reversed(list(payload.items())))


def requests(state: State, view=View(), include_rel=True):
    body = render_state(state, view)
    suf = {"state": body, "questions": {"sufficient": {
        "type": "noul", "instructions": "Can the question be answered completely from the selected evidence under the conditions? Use only selected evidence as support.",
        "criteria": {"false": "Required support is missing.", "true": "Selected evidence provides all required support."}}}}
    choice = None
    if state.candidates:
        choice = {"state": body, "questions": {"next": {"type": "choice",
                  "instructions": "Which remaining evidence adds the most missing support for answering the question, given the selected evidence and conditions?",
                  "criteria": {e.id: json.dumps({"title": e.title, "text": e.text}, ensure_ascii=False) for e in state.candidates}}}}
    rel = []
    if include_rel:
        for e in state.candidates:
            rel.append({"state": json.dumps({"question": state.question, "evidence": {"id": e.id, "title": e.title, "text": e.text}}, ensure_ascii=False),
                        "questions": {"relevant": {"type": "noul", "instructions": "Does this evidence support answering the question? Relevance is independent of what is already selected."}}})
    return choice, suf, rel
