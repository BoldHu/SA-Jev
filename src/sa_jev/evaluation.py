from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit
import os
import time
import torch
import numpy as np
from .io import digest, read_json, write_json, write_jsonl
from .model import Prediction
from .prompts import View, requests
from .schema import State, labels_for
from .retrieval import require_retrieved
from .metrics import ndcg, answer_metrics, distribution_change, binary_metrics
from .training import memory_stats, synchronize


class OfficialJev:
    """Verified System One wire contract; secrets only in process environment."""
    device = torch.device("cpu")

    def __init__(self, model="jev-1.13.0", endpoint="https://api.typesafe.ai/v1/systemone", timeout=120):
        import requests as http
        self.session = http.Session(); self.model = model; self.endpoint = endpoint; self.timeout = timeout
        key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
        if not key: raise ValueError("Fill TYPESAFE_API_KEY in MyDrive/SA-Jev/.env (Colab) or the project .env file, then reload it")
        self.session.headers["Authorization"] = "Bearer " + key
        self.actual_models = set()
        self.output_tokens = 0

    def eval(self): return self

    def decision(self, body):
        response = self.session.post(self.endpoint, json={**body, "model": self.model}, timeout=self.timeout)
        response.raise_for_status()
        data = response.json(); self.actual_models.add(data["model"])
        self.output_tokens += data.get("usage", {}).get("output_tokens", 0)
        if len(self.actual_models) > 1: raise RuntimeError("Jev version changed during the experiment")
        key, spec = next(iter(body["questions"].items())); value = data["answers"][key]
        if spec["type"] == "noul":
            probs = [1-value["noul"], value["noul"]]
        else:
            expected = tuple(spec["criteria"])
            if set(expected) != set(value["probabilities"]): raise ValueError("API omitted candidate probabilities")
            probs = [value["probabilities"][i] for i in expected]
        p = torch.tensor(probs, dtype=torch.float32)
        if not torch.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not torch.isclose(p.sum(), torch.tensor(1.), atol=2e-5):
            raise ValueError("invalid API probabilities")
        return p.log(), data.get("usage", {}).get("input_tokens", 0), 0

    def __call__(self, state, view=View(), include_rel=False):
        choice, suf, rel = requests(state, view, include_rel)
        h, n, m = self.decision(suf)
        p = h.new_empty(0)
        if choice:
            p, a, b = self.decision(choice); n += a; m += b
        r = []
        for body in rel:
            v, a, b = self.decision(body); r.append(v); n += a; m += b
        return Prediction(tuple(e.id for e in state.candidates), p, h,
                          torch.stack(r) if r else h.new_empty((0, 2)), n, m)


class Answerer:
    def __init__(self, model, cache_directory, *, seed=17):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model; self.seed = seed
        # Google's OpenAI-compatible endpoint rejects seed rather than ignoring it.
        self.request_options = {"temperature": 0}
        if urlsplit(str(self.client.base_url)).hostname != "generativelanguage.googleapis.com":
            self.request_options["seed"] = seed
        # Provider changes must not reuse an answer cached under the same model alias.
        self.endpoint_hash = digest(str(self.client.base_url).rstrip("/"))
        self.cache = Path(cache_directory); self.cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, state):
        payload = {"question": state.question, "conditions": state.conditions,
                   "evidence": [{"id": e.id, "title": e.title, "text": e.text} for e in state.selected]}
        key = digest({"payload": payload, "model": self.model, "seed": self.seed,
                      "request_options": self.request_options, "endpoint_hash": self.endpoint_hash,
                      "prompt_version": 1})
        path = self.cache / (key + ".json")
        if path.exists(): return read_json(path)
        import json
        response = self.client.chat.completions.create(model=self.model, **self.request_options,
            messages=[{"role": "system", "content": "Answer the question using only the supplied evidence. Return only the short answer. If evidence is insufficient, return noanswer."},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        result = {"answer": response.choices[0].message.content.strip(), "model": response.model,
                  "input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens,
                  "selected_ids": list(state.selected_ids), "request_options": self.request_options,
                  "endpoint_hash": self.endpoint_hash}
        write_json(path, result); return result


@torch.no_grad()
def rollout(model, initial, *, policy="dynamic", threshold=.5, answerer=None):
    if policy not in {"dynamic", "static", "retrieval", "state_filtered", "no_state_update"}:
        raise ValueError("unknown selection baseline")
    model.eval(); state = initial
    static = None; trace = []; logical = executed = 0
    while True:
        model_state = replace(state, context="") if policy == "state_filtered" else state
        # Independent sufficiency first, with no unselected evidence bodies.
        _, body, _ = requests(model_state, include_rel=False)
        log_h, n, m = model.decision(body); logical += n; executed += m
        h = float(log_h[1].exp())
        entry = {"selected_ids": list(state.selected_ids), "budget": state.budget, "h": h}
        if h >= threshold:
            terminal = "budget" if state.budget == 0 else "stop"
            entry["action"] = "BUDGET_END" if terminal == "budget" else "STOP"
            trace.append(entry); accepted = True; break
        if state.budget == 0 or not state.candidates:
            terminal = "budget" if state.budget == 0 else "exhausted"
            entry["action"] = terminal.upper(); trace.append(entry); accepted = False; break
        if policy == "retrieval":
            ids = [e.id for e in state.candidates]
            p = np.arange(len(ids), 0, -1, dtype=float); p /= p.sum()
        elif policy == "static" and static is not None:
            ids = [e.id for e in state.candidates]
            p = np.asarray([static[i] for i in ids]); p = p / p.sum() if p.sum() else np.ones(len(ids))/len(ids)
        else:
            choice_state = model_state
            if policy == "no_state_update":
                # Remove selected content only from Choice's state; current remaining
                # candidates and independent Noul continue to obey actual state.
                choice_state = replace(model_state, context=initial.context)
            body, _, _ = requests(choice_state, include_rel=False)
            if policy == "no_state_update":
                import json
                value = json.loads(body["state"]); value["selected_evidence"] = []
                body["state"] = json.dumps(value, sort_keys=True, ensure_ascii=False)
            log_p, n, m = model.decision(body); logical += n; executed += m
            ids = [e.id for e in state.candidates]; p = log_p.exp().cpu().numpy()
            if policy == "static": static = dict(zip(ids, p.tolist()))
        selected = ids[int(np.argmax(p))]
        entry.update(action=selected, candidate_ids=ids, p=p.tolist()); trace.append(entry)
        state = state.read(selected)
    qa = answerer(state) if accepted and answerer else {"answer": "noanswer" if not accepted else None,
                                                       "input_tokens": 0, "output_tokens": 0}
    return {"state": state, "trace": trace, "accepted": accepted, "termination": terminal,
            "logical_tokens": logical, "executed_tokens": executed, "qa": qa}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


@torch.no_grad()
def evaluate(model, questions, pools, output, *, name="openjev_frozen", policy="dynamic", budget=4,
             threshold=.5, answerer=None, seed=17, formal=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []; started = time.perf_counter(); model.eval()
    for index, q in enumerate(questions):
        pool = pools[q.id]
        if formal: require_retrieved(pool)
        initial = State(q.id, q.group_id, q.question, pool, budget=budget)
        row = {"question_id": q.id, "group_id": q.group_id, "dataset": q.dataset, "method": name,
               "seed": seed, "pool_hash": pool.fingerprint, "k": len(pool.candidates), "budget": budget,
               "formal": formal, "answer_model": getattr(answerer, "model", None), "threshold": threshold}
        if model.device.type == "cuda": torch.cuda.reset_peak_memory_stats(model.device)
        synchronize(model.device); begin = time.perf_counter()
        api_output_start = getattr(model, "output_tokens", 0)
        try:
            result = rollout(model, initial, policy=policy, threshold=threshold, answerer=answerer)
            synchronize(model.device)
            final = result.pop("state")
            labels = labels_for(q, final)
            rankings, h, z = [], [], []; state = initial
            for entry in result["trace"]:
                y = labels_for(q, state)
                if y.suf_valid: h.append(entry["h"]); z.append(y.sufficient)
                if "p" in entry:
                    if y.rank_valid: rankings.append(ndcg(y.gains, entry["p"]))
                    state = state.read(entry["action"])
            pool_state = replace(initial, selected_ids=tuple(e.id for e in pool.candidates))
            pool_labels = labels_for(q, pool_state)
            row.update(status="ok", support_recall=len(labels.covered)/len(q.units) if q.units else None,
                       pool_support_recall=len(pool_labels.covered)/len(q.units) if q.units else None,
                       complete_support=float(labels.sufficient) if labels.suf_valid else None,
                       false_stop=float(result["accepted"] and not labels.sufficient) if labels.suf_valid else None,
                       refused=float(not result["accepted"]), reads=len(final.selected_ids), ndcg=_mean(rankings),
                       rank_evaluable_states=len(rankings), suf_evaluable_states=len(z),
                       selected_ids=list(final.selected_ids), **binary_metrics(h, z, threshold),
                       **result)
            if answerer is not None and result["qa"]["answer"] is not None:
                row.update(answer_metrics(result["qa"]["answer"], q.answer, q.answer_aliases))
            else: row.update(qa_em=None, qa_f1=None)
        except Exception as exc:
            # Preserve failed questions in denominators; missing quality measurements
            # are explicit, never replace OOM/token-limit failures with a smaller K.
            row.update(status="error", error=type(exc).__name__ + ": " + str(exc),
                       qa_em=0. if answerer else None, qa_f1=0. if answerer else None,
                       support_recall=0., complete_support=0., ndcg=None)
            if model.device.type == "cuda": torch.cuda.empty_cache()
        row.update(seconds=time.perf_counter()-begin, **memory_stats(model.device))
        if isinstance(model, OfficialJev):
            row.update(cost_scope="client_wall_time_and_reported_API_usage", executed_tokens=None,
                       api_output_tokens=model.output_tokens-api_output_start,
                       allocated_gib=None, reserved_gib=None, peak_allocated_gib=None, peak_reserved_gib=None)
        rows.append(row); write_jsonl(output / "raw.jsonl", rows)
        elapsed = time.perf_counter()-started
        print(f"{name} {index+1}/{len(questions)} status={row['status']} ETA={elapsed/(index+1)*(len(questions)-index-1):.1f}s peak_reserved_gib={row['peak_reserved_gib']}", flush=True)
    return rows


@torch.no_grad()
def repeat_diagnostics(model, states, repeats=3, output=None):
    model.eval(); rows = []
    started = time.perf_counter()
    total = len(states) * repeats
    print(f"[diagnostics] {len(states)} questions x {repeats} repeats = {total} forwards; device={model.device}", flush=True)
    if output is not None:
        output = Path(output)
        write_jsonl(output / "raw.jsonl", [])
    for index, state in enumerate(states):
        first = None
        for repeat in range(repeats):
            print(f"[diagnostics] START {len(rows)+1}/{total}: question={index+1}/{len(states)} "
                  f"repeat={repeat+1}/{repeats} K={len(state.candidates)} id={state.question_id}", flush=True)
            synchronize(model.device); start = time.perf_counter()
            if model.device.type == "cuda": torch.cuda.reset_peak_memory_stats(model.device)
            row = {"question_id": state.question_id, "group_id": state.group_id, "repeat": repeat,
                         "pool_hash": state.pool.fingerprint, "k": len(state.candidates),
                         "status": "ok"}
            try:
                p = model(state, include_rel=False); synchronize(model.device)
                data = (p.candidate_ids, p.p.cpu().tolist(), float(p.h))
                if first is None: first = data
                row.update(logical_tokens=p.logical_tokens, executed_tokens=p.executed_tokens,
                           **distribution_change(*first, *data))
            except Exception as exc:
                row.update(status="error", error=type(exc).__name__ + ": " + str(exc))
            row.update(seconds=time.perf_counter()-start, **memory_stats(model.device))
            if isinstance(model, OfficialJev):
                row.update(executed_tokens=None, allocated_gib=None, reserved_gib=None, peak_allocated_gib=None, peak_reserved_gib=None,
                           cost_scope="client_wall_time_and_reported_API_usage")
            rows.append(row)
            if output is not None: write_jsonl(output / "raw.jsonl", rows)
            elapsed = time.perf_counter()-started
            print(f"[diagnostics] DONE {len(rows)}/{total}: status={row['status']} "
                  f"seconds={row['seconds']:.1f} ETA={elapsed/len(rows)*(total-len(rows)):.1f}s "
                  f"peak_reserved_gib={row['peak_reserved_gib']}" +
                  (f" error={row['error']}" if row['status'] == "error" else ""), flush=True)
    return rows
