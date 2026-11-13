"""Exact expanded-branch execution: length buckets, complete native decision heads.

No prefix/KV reuse, detach, quantization, or evidence truncation is used here.
Only deterministic identical requests can share a graph within one forward call.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def deterministic_forward(model):
    """Conservatively disable sharing/reordering for supported stochastic layers."""
    if not model.training:
        return True
    for module in model.modules():
        if not module.training:
            continue
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.p:
            return False
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            return False
        for name in ('dropout', 'attention_dropout', 'hidden_dropout', 'drop_path_rate'):
            value = getattr(module, name, 0)
            if isinstance(value, (int, float)) and value > 0:
                return False
        config = getattr(module, 'config', None)
        if config is not None and any(getattr(config, name, 0) for name in
                                     ('attention_dropout', 'hidden_dropout', 'hidden_dropout_prob')):
            return False
    return True


def configure_checkpointing(core, mode):
    if mode not in {'branch', 'layer', 'branch_layer'}:
        raise ValueError('checkpoint_mode must be branch, layer or branch_layer')
    backbone = core.adapter.backbone
    if mode in {'layer', 'branch_layer'}:
        if not hasattr(backbone, 'gradient_checkpointing_enable'):
            raise ValueError('This backbone does not support layer checkpointing')
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    elif hasattr(backbone, 'gradient_checkpointing_disable'):
        backbone.gradient_checkpointing_disable()
    core.checkpoint_branches = mode != 'layer'


def score_bucketed(core, encoded, pad_token_id, temperature, token_budget=0):
    """Encode length-sorted branches, then restore ALL options before the head.

    token_budget caps padded tokens per backbone call, not input length. A single
    long branch is kept intact. All checkpoint inputs are explicit, so backward
    replay cannot capture the wrong loop indices. Cost excludes backward replay.
    """
    from openjev.compute import compute_context, model_compute_contract
    from openjev.models.decision.pooling import group_branch_features, pool_branch_endpoints

    device = next(core.parameters()).device
    flat = [(row, branch) for row, item in enumerate(encoded) for branch in item.branches]
    order = sorted(range(len(flat)), key=lambda i: len(flat[i][1].input_ids))
    groups, group = [], []
    limit = core.branch_microbatch_size or len(flat)
    for index in order:
        length = len(flat[index][1].input_ids)
        if group and (len(group) == limit or (token_budget and (len(group)+1)*length > token_budget)):
            groups.append(group); group = []
        group.append(index)
    if group:
        groups.append(group)
    question, option, seen = [], [], []
    costs = [0] * len(encoded)

    def encode(ids, mask, positions, q_end, o_end):
        with compute_context(model_compute_contract(core), 'backbone', ids.device):
            hidden = core.adapter.forward_hidden(ids, mask, positions)
        return pool_branch_endpoints(hidden, q_end, o_end, mask)

    for group in groups:
        length = max(len(flat[i][1].input_ids) for i in group)
        ids = torch.full((len(group), length), pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        positions = torch.zeros_like(ids)
        q_end, o_end = [], []
        for j, index in enumerate(group):
            row, branch = flat[index]
            size = len(branch.input_ids)
            if not 0 <= branch.question_end < branch.option_end < size:
                raise ValueError('Invalid branch endpoints')
            ids[j, :size] = torch.tensor(branch.input_ids)
            mask[j, :size] = True
            positions[j, :size] = torch.arange(size)
            q_end.append(branch.question_end); o_end.append(branch.option_end)
            costs[row] += length
        args = [t.to(device) for t in (ids, mask, positions, torch.tensor(q_end), torch.tensor(o_end))]
        if core.checkpoint_branches and core.training and torch.is_grad_enabled():
            q, e = checkpoint(encode, *args, use_reentrant=False)
        else:
            q, e = encode(*args)
        question.append(q); option.append(e); seen.extend(group)
    inverse = torch.tensor(seen, device=device).argsort()
    q = torch.cat(question).index_select(0, inverse)
    e = torch.cat(option).index_select(0, inverse)
    ptr = [0]
    mask = torch.zeros((len(encoded), max(len(x.branches) for x in encoded)), dtype=torch.bool, device=device)
    for row, item in enumerate(encoded):
        ptr.append(ptr[-1] + len(item.branches))
        mask[row, :len(item.branches)] = True
    features = group_branch_features(q, e, torch.tensor(ptr), mask)
    output = core.score_features(features.question_features, features.option_features,
                                 features.option_mask, temperature=temperature)
    return output, costs, len(groups)
