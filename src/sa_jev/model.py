"""Native OpenJev head/interactor and differentiable branch execution."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
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
    def __init__(self, core, tokenizer, serializer, *, temperature=1., base_identity="", lora_config=None):
        super().__init__()
        self.core, self.tokenizer, self.serializer = core, tokenizer, serializer
        self.temperature = float(temperature)
        self.base_identity = base_identity
        self.lora_config = lora_config

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
        from openjev.api.typesafe import parse_request
        from openjev.data.collator import collate_serialized
        parsed = parse_request(body, model_ids=set())
        if len(parsed) != 1:
            raise ValueError("one complete decision per call")
        encoded = self.serializer(parsed[0].decision)
        pad = self.tokenizer.pad_token_id
        if pad is None: pad = self.tokenizer.eos_token_id
        batch = collate_serialized([encoded], pad)
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = self.core(batch, temperature=self.temperature)
        return output.log_probs[0], encoded.logical_token_count, batch["input_ids"].numel()

    def forward(self, state, view=View(), include_rel=True):
        choice, suf, rel = requests(state, view, include_rel)
        log_h, logical, executed = self.decision(suf)
        if choice:
            log_p, n, m = self.decision(choice)
            logical += n; executed += m
        else:
            log_p = log_h.new_empty((0,))
        log_rel = []
        for body in rel:
            value, n, m = self.decision(body)
            log_rel.append(value); logical += n; executed += m
        return Prediction(tuple(e.id for e in state.candidates), log_p, log_h,
                          torch.stack(log_rel) if log_rel else log_h.new_empty((0, 2)), logical, executed)

    @classmethod
    def from_pretrained(cls, artifact="shenjunhao/OpenJev-4B", *, revision=None, device="cuda",
                        branch_microbatch=1, trainable=True, lora_rank=16, lora_alpha=32,
                        max_branch_length=None, max_decision_tokens=None):
        from openjev.artifacts import load_decision_artifact, sha256_file
        from openjev.runtime.hub import resolve_artifact
        from openjev.models.decision.serializer import DecisionSerializer
        path = resolve_artifact(artifact, revision=revision)
        core, tokenizer, config, calibration = load_decision_artifact(path, device=device, dtype="bfloat16")
        core.branch_microbatch_size = branch_microbatch
        core.checkpoint_branches = trainable
        core.requires_grad_(False)
        lora = None
        if trainable:
            from peft import LoraConfig, get_peft_model
            # Attention + gated-delta + MLP projections in the existing backbone.
            # Generic PEFT wrapper avoids causal-LM task wrappers and generation paths.
            lora = dict(r=lora_rank, lora_alpha=lora_alpha, target_modules="all-linear", lora_dropout=0., bias="none")
            core.adapter.backbone = get_peft_model(core.adapter.backbone, LoraConfig(**lora))
            core.head.requires_grad_(True)
        args = {k: v for k, v in config.serializer.items() if k != "version"}
        if max_branch_length is not None: args["max_branch_length"] = max_branch_length
        if max_decision_tokens is not None: args["max_decision_tokens"] = max_decision_tokens
        result = cls(core, tokenizer, DecisionSerializer(tokenizer, **args),
                     temperature=calibration["temperature"],
                     base_identity=sha256_file(Path(path) / "manifest.json"), lora_config=lora)
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
