"""Native OpenJev head/interactor and differentiable branch execution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import torch
from torch import nn
from .prompts import View, requests
from .io import digest, read_json, write_json


@dataclass
class Prediction:
    candidate_ids: tuple[str, ...]
    log_p: torch.Tensor
    log_h: torch.Tensor  # [false, true], a separate Noul decision
    log_rel: torch.Tensor  # [K, false/true]
    logical_tokens: int = 0
    executed_tokens: int = 0

    @property
    def p(self): return self.log_p.exp()

    @property
    def h(self): return self.log_h[1].exp()


class SAJev(nn.Module):
    def __init__(self, core, tokenizer, serializer, *, temperature=1., base_identity="", lora_config=None,
                 decision_batch_size=8, optimized_execution=False, branch_token_budget=0):
        super().__init__()
        if isinstance(decision_batch_size, bool) or not isinstance(decision_batch_size, int) or decision_batch_size < 1:
            raise ValueError("decision_batch_size must be a positive integer")
        self.core, self.tokenizer, self.serializer = core, tokenizer, serializer
        self.temperature = float(temperature)
        self.base_identity = base_identity
        self.lora_config = lora_config
        self.decision_batch_size = decision_batch_size
        if isinstance(branch_token_budget, bool) or not isinstance(branch_token_budget, int) or branch_token_budget < 0:
            raise ValueError('branch_token_budget must be a nonnegative integer')
        self.optimized_execution = optimized_execution
        self.branch_token_budget = branch_token_budget
        self.execution_stats = {}

    def reset_execution_stats(self):
        self.execution_stats = dict(requested_decisions=0, unique_decisions=0, encoded_branches=0,
                                    forward_padded_tokens=0, backbone_calls=0, max_branch_tokens=0)

    @property
    def device(self): return next(self.parameters()).device

    def view_within_budget(self, state, view):
        """Check the complete augmented decisions without truncating any evidence."""
        from openjev.api.typesafe import parse_request
        choice, suf, _ = requests(state, view, include_rel=False)
        for body in (suf, choice):
            if body is None: continue
            try:
                self.serializer(parse_request(body, model_ids=set())[0].decision)
            except ValueError as exc:
                if "no automatic truncation" in str(exc): return False
                raise
        return True

    def decision(self, body):
        return self.decisions([body])[0]

    def decisions(self, bodies):
        """Batch independent complete decisions, preserving order and separate softmaxes.

        Similar lengths share padding; each decision keeps its own option mask and
        Interactor row. No candidate is removed and no support input is merged.
        """
        from openjev.api.typesafe import parse_request
        from openjev.data.collator import collate_serialized
        from .execution import deterministic_forward, score_bucketed
        optimized = self.optimized_execution and deterministic_forward(self)
        bodies = list(bodies)
        if not self.execution_stats:
            self.reset_execution_stats()
        self.execution_stats['requested_decisions'] += len(bodies)
        unique, mapping, keys = [], [], {}
        for body in bodies:
            key = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(',', ':')) if optimized else len(unique)
            if key not in keys:
                keys[key] = len(unique); unique.append(body)
            mapping.append(keys[key])
        self.execution_stats['unique_decisions'] += len(unique)
        encoded = []
        for body in unique:
            parsed = parse_request(body, model_ids=set())
            if len(parsed) != 1:
                raise ValueError("one complete decision per request")
            encoded.append(self.serializer(parsed[0].decision))
        if encoded:
            self.execution_stats['max_branch_tokens'] = max(self.execution_stats['max_branch_tokens'],
                max(len(branch.input_ids) for item in encoded for branch in item.branches))
        pad = self.tokenizer.pad_token_id
        if pad is None: pad = self.tokenizer.eos_token_id
        order = sorted(range(len(encoded)), key=lambda i: max(len(b.input_ids) for b in encoded[i].branches))
        results = [None] * len(encoded)
        for start in range(0, len(order), self.decision_batch_size):
            indices = order[start:start+self.decision_batch_size]
            items = [encoded[i] for i in indices]
            if optimized:
                output, costs, calls = score_bucketed(self.core, items, pad, self.temperature, self.branch_token_budget)
            else:
                batch = collate_serialized(items, pad)
                length = batch["input_ids"].shape[1]
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output = self.core(batch, temperature=self.temperature)
                costs = [len(item.branches)*length for item in items]
                n = sum(len(item.branches) for item in items)
                chunk = self.core.branch_microbatch_size or n
                calls = (n+chunk-1)//chunk
            self.execution_stats['encoded_branches'] += sum(len(item.branches) for item in items)
            self.execution_stats['forward_padded_tokens'] += sum(costs)
            self.execution_stats['backbone_calls'] += calls
            for row, index in enumerate(indices):
                item = encoded[index]
                results[index] = (output.log_probs[row, :len(item.option_ids)],
                                  item.logical_token_count, costs[row])
        # Reused outputs retain the SAME differentiable graph, never detach or
        # cache across optimizer steps. Charge executed tokens only on first use.
        seen, expanded = set(), []
        for index in mapping:
            value, logical, executed = results[index]
            expanded.append((value, logical, 0 if index in seen else executed))
            seen.add(index)
        return expanded

    def forward(self, state, view=View(), include_rel=True):
        return self.forward_batch([state], [view], include_rel=include_rel)[0]

    def forward_batch(self, states, views=None, include_rel=True, *, choice_needed=None, relevance_masks=None):
        """Multiple states share execution, never inputs, labels or normalization.

        Group task types separately to avoid padding short relevance inputs to
        the length of selected-evidence contexts. ``decisions`` further groups
        by length and restores order, with a complete K-way mask for each row.
        """
        states = list(states)
        views = [View()] * len(states) if views is None else list(views)
        if len(states) != len(views):
            raise ValueError("one view is required per state")
        if not states:
            return []
        bodies = [requests(state, view, include_rel) for state, view in zip(states, views)]
        if choice_needed is not None:
            if len(choice_needed) != len(states): raise ValueError('choice mask length mismatch')
            bodies = [(c if needed else None, s, r) for (c, s, r), needed in zip(bodies, choice_needed)]
        if relevance_masks is None:
            relevance_masks = [[True]*len(r) for _, _, r in bodies]
        if len(relevance_masks) != len(states) or any(len(m) != len(r) for m, (_, _, r) in zip(relevance_masks, bodies)):
            raise ValueError('relevance mask length mismatch')
        sufficient = self.decisions([suf for _, suf, _ in bodies])
        choices = iter(self.decisions([choice for choice, _, _ in bodies if choice is not None]))
        relevance = iter(self.decisions([body for (_, _, rel), mask in zip(bodies, relevance_masks)
                                        for body, valid in zip(rel, mask) if valid]))
        predictions = []
        for state, (choice, _, rel), (log_h, logical, executed), mask in zip(states, bodies, sufficient, relevance_masks):
            # NaNs make accidental use of uncomputed predictions detectable.
            log_p = log_h.new_full((len(state.candidates),), float('nan'))
            if choice is not None:
                log_p, n, m = next(choices)
                logical += n; executed += m
            log_rel = []
            for valid in mask:
                value, n, m = next(relevance) if valid else (log_h.new_full((2,), float('nan')), 0, 0)
                log_rel.append(value); logical += n; executed += m
            predictions.append(Prediction(tuple(e.id for e in state.candidates), log_p, log_h,
                               torch.stack(log_rel) if log_rel else log_h.new_empty((0, 2)), logical, executed))
        return predictions

    @classmethod
    def from_pretrained(cls, artifact="shenjunhao/OpenJev-4B", *, revision=None, device="cuda",
                        branch_microbatch=1, trainable=True, lora_rank=16, lora_alpha=32,
                        max_branch_length=None, max_decision_tokens=None, decision_batch_size=8,
                        optimized_execution=False, branch_token_budget=0, checkpoint_mode='branch'):
        from openjev.artifacts import load_decision_artifact, sha256_file
        from .artifact_cache import prepare_openjev_artifact
        from .io import progress_phase
        from openjev.models.decision.serializer import DecisionSerializer
        path = prepare_openjev_artifact(artifact, revision=revision)
        with progress_phase(f"[OpenJev] Load backbone/head/tokenizer; target={device}, dtype=bfloat16"):
            core, tokenizer, config, calibration = load_decision_artifact(path, device=device, dtype="bfloat16")
        print(f"[OpenJev] Model loaded on {device}; configuring decision branches", flush=True)
        delta = next((m for m in core.modules() if hasattr(m, "chunk_gated_delta_rule")), None)
        if delta is not None:
            kernel = delta.chunk_gated_delta_rule
            print(f"[OpenJev] Delta kernel={kernel.__module__}.{kernel.__name__}; "
                  f"convolution={'causal_conv1d' if delta.causal_conv1d_fn is not None else 'torch'}", flush=True)
        core.branch_microbatch_size = branch_microbatch
        core.checkpoint_branches = trainable
        print(f"[OpenJev] branch_microbatch={branch_microbatch}, independent decision batch={decision_batch_size}", flush=True)
        core.requires_grad_(False)
        lora = None
        if trainable:
            from peft import LoraConfig, get_peft_model
            # Attention + gated-delta + MLP projections in the existing backbone.
            # Generic PEFT wrapper avoids causal-LM task wrappers and generation paths.
            lora = dict(r=lora_rank, lora_alpha=lora_alpha, target_modules="all-linear", lora_dropout=0., bias="none")
            core.adapter.backbone = get_peft_model(core.adapter.backbone, LoraConfig(**lora))
            core.head.requires_grad_(True)
            from .execution import configure_checkpointing
            configure_checkpointing(core, checkpoint_mode)
        args = {k: v for k, v in config.serializer.items() if k != "version"}
        if max_branch_length is not None: args["max_branch_length"] = max_branch_length
        if max_decision_tokens is not None: args["max_decision_tokens"] = max_decision_tokens
        result = cls(core, tokenizer, DecisionSerializer(tokenizer, **args),
                     temperature=calibration["temperature"],
                     base_identity=sha256_file(Path(path) / "manifest.json"), lora_config=lora,
                     decision_batch_size=decision_batch_size, optimized_execution=optimized_execution,
                     branch_token_budget=branch_token_budget)
        print(f'[OpenJev] optimized_execution={optimized_execution}; branch token budget={branch_token_budget}; '
              f'checkpoint_mode={checkpoint_mode}', flush=True)
        if trainable:
            total = sum(p.numel() for p in result.parameters())
            updated = sum(p.numel() for p in result.parameters() if p.requires_grad)
            print(f"[OpenJev] LoRA + decision head trainable parameters: {updated:,}/{total:,} "
                  f"({100*updated/total:.2f}%); original backbone weights frozen", flush=True)
        result.train(trainable)
        return result

    def save_trainable(self, directory, metadata=None):
        from safetensors.torch import save_file
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        values = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if k in names}
        if not values:
            raise ValueError("no trainable parameters to save")
        save_file(values, str(directory / "trainable.safetensors"))
        write_json(directory / "metadata.json", {"base_identity": self.base_identity,
                   "lora": self.lora_config, "temperature": self.temperature,
                   "parameter_names": sorted(values), "metadata": metadata or {}})

    def load_trainable(self, directory):
        from safetensors.torch import load_file
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        if meta["base_identity"] != self.base_identity or meta["lora"] != self.lora_config:
            raise ValueError("checkpoint base artifact or LoRA configuration mismatch")
        values = load_file(str(directory / "trainable.safetensors"), device=str(self.device))
        current = self.state_dict()
        if sorted(values) != sorted(meta["parameter_names"]):
            raise ValueError("incomplete checkpoint")
        for name, value in values.items():
            if name not in current or current[name].shape != value.shape:
                raise ValueError(f"checkpoint parameter mismatch: {name}")
        self.load_state_dict(values, strict=False)
        return meta


class TinyTokenizer:
    """Deterministic smoke tokenizer, never used as a scientific baseline."""
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def encode(self, text, add_special_tokens=False):
        return [2 + int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 2046
                for t in re.findall(r"\w+|[^\w\s]", text)]


class TinyAdapter(nn.Module):
    """Small causal encoder for plumbing tests; the real OpenJev head is reused."""
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.embedding = nn.Embedding(2048, hidden)
        self.backbone = nn.GRU(hidden, hidden, batch_first=True)

    def get_hidden_size(self): return self.hidden

    def forward_hidden(self, input_ids, attention_mask, position_ids=None):
        return self.backbone(self.embedding(input_ids))[0]


def tiny_model(device="cpu", seed=17, branch_microbatch=4):
    from openjev.models.decision.head import DecisionHead
    from openjev.models.decision.model import OpenJevDecisionModel
    from openjev.models.decision.serializer import DecisionSerializer
    torch.manual_seed(seed)
    core = OpenJevDecisionModel(TinyAdapter(), DecisionHead(32, set_dim=32, set_layers=1,
                                set_heads=4, ffn_dim=64), branch_microbatch_size=branch_microbatch,
                                checkpoint_branches=True).to(device)
    tokenizer = TinyTokenizer()
    return SAJev(core, tokenizer, DecisionSerializer(tokenizer, max_branch_length=8192,
                 max_decision_tokens=65536), base_identity=f"TEST_ONLY_tiny_seed_{seed}")
