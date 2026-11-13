"""Execution optimizations must preserve supervised and full-trajectory gradients."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from sa_jev.data import task_states
from sa_jev.losses import task_loss, stability_loss, dpo_loss
from sa_jev.model import tiny_model
from sa_jev.prompts import View
from sa_jev.schema import Pool
from sa_jev.smoke import fixture_questions
from sa_jev.training import task_predictions, cache_reference, fit, TrainConfig
from sa_jev.trajectories import build_preferences, trajectory_log_probabilities
from test_context_views import FakeReviewer, prepared


def setup():
    torch.set_num_threads(2)
    qs = fixture_questions()
    q = qs[0]
    evidence = list(q.evidence)
    evidence[-1] = replace(evidence[-1], text=evidence[-1].text + ' ordinary background' * 70)
    train = task_states(q, Pool(q.id, tuple(evidence), 'fixture'))
    val = task_states(qs[2], Pool(qs[2].id, qs[2].evidence, 'fixture'))
    return qs, train, val


def assert_grads(a, b):
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-4, msg=name)


@pytest.mark.parametrize('stage', [1, 2])
def test_full_objective_and_gradients_preserved(stage):
    qs, train, _ = setup()
    # Mixed masks, full/sufficient states, long/short branches and actual transitions.
    train[1] = replace(train[1], labels=replace(train[1].labels, rank_valid=False,
                       rel_valid=(True, False, True, False, True)))
    old = tiny_model(branch_microbatch=2).train()
    new = deepcopy(old); new.optimized_execution = True; new.branch_token_budget = 700
    states = [x.state for x in train]
    labels = [x.labels for x in train]
    pairs = build_preferences(qs, [train[0].state], limit=2)
    losses = []
    for model in (old, new):
        examples = train if stage == 1 else [x for x in train for _ in range(2)]
        views = None if stage == 1 else [v for x in train for v in (View(), View(0, 'Background with no new support.'))]
        stable = [bool(x.labels.gold_available and x.labels.suf_valid and not x.labels.sufficient)
                  for x in examples] if stage == 2 else None
        p = task_predictions(model, examples, views, stable)
        loss = task_loss(p, [x.labels for x in examples])[0]['task']
        if stage == 2:
            a, b = p[::2], p[1::2]
            loss = loss + .1 * stability_loss(a, b, labels, states, states)[0]
            chosen, rejected = trajectory_log_probabilities(model, (pairs[0].chosen, pairs[0].rejected), 2)
            loss = loss + .2*dpo_loss(chosen, rejected, chosen.new_tensor(-3.), chosen.new_tensor(-4.))
        loss.backward(); losses.append(loss)
    torch.testing.assert_close(*losses, atol=3e-6, rtol=3e-5)
    assert_grads(old, new)
    assert new.execution_stats['encoded_branches'] < old.execution_stats['encoded_branches']
    assert new.execution_stats['forward_padded_tokens'] < old.execution_stats['forward_padded_tokens']
    # Comparing the actual AdamW update detects changes hidden by scalar loss agreement.
    for model in (old, new):
        torch.optim.AdamW(model.parameters(), lr=2e-5).step()
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)


def test_identical_views_reuse_graph_but_never_cache_updated_weights():
    _, train, _ = setup()
    old = tiny_model().train(); new = deepcopy(old); new.optimized_execution = True
    state = train[0].state
    a = old.forward_batch([state, state], [View(), View(0, 'Background.')])
    b = new.forward_batch([state, state], [View(), View(0, 'Background.')])
    for p, q in zip(a, b):
        for key in ('log_p', 'log_h', 'log_rel'):
            torch.testing.assert_close(getattr(p, key), getattr(q, key), atol=2e-6, rtol=2e-5)
    assert new.execution_stats['encoded_branches'] == 4*len(state.candidates)+4
    assert old.execution_stats['encoded_branches'] == 6*len(state.candidates)+4
    before = new.execution_stats['encoded_branches']
    new(state)
    assert new.execution_stats['encoded_branches'] > before  # No persistent graph/features cache.


def test_trajectory_common_states_and_stop_pruning_match(tmp_path):
    qs, train, _ = setup()
    pairs = build_preferences(qs, [train[0].state], limit=3)
    old = tiny_model().train(); new = deepcopy(old); new.optimized_execution = True
    trajectories = [t for p in pairs for t in (p.chosen, p.rejected)]
    a = trajectory_log_probabilities(old, trajectories, 2)
    b = trajectory_log_probabilities(new, trajectories, 2)
    torch.testing.assert_close(torch.stack(a), torch.stack(b), atol=3e-6, rtol=3e-5)
    sum(a).backward(); sum(b).backward(); assert_grads(old, new)
    assert new.execution_stats['encoded_branches'] < old.execution_stats['encoded_branches']
    reference = cache_reference(new, pairs, tmp_path/'reference.json', 2)
    assert new.training
    assert cache_reference(new, pairs, tmp_path/'reference.json', 2)['values'] == reference['values']


def test_stochastic_forward_falls_back_to_original_execution():
    _, train, _ = setup()
    model = tiny_model().train(); model.optimized_execution = True
    model.core.extra_dropout = torch.nn.Dropout(.1)
    state = train[0].state
    model.forward_batch([state, state])
    assert model.execution_stats['requested_decisions'] == model.execution_stats['unique_decisions']


def test_optimized_fit_matches_original_and_resume(tmp_path):
    _, train, val = setup()
    old, new = tiny_model(), tiny_model()
    new.optimized_execution = True; new.branch_token_budget = 512
    cfg = TrainConfig(accumulation=2, state_batch_size=2, max_steps=2, save_every=1, validate_every=2)
    a = fit(old, train, val, cfg, tmp_path/'old')
    b = fit(new, train, val, cfg, tmp_path/'new')
    for x, y in zip(a, b):
        for key in ('task', 'rank', 'suf', 'rel'):
            assert x[key] == pytest.approx(y[key], abs=3e-6, rel=3e-5)
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)
    resumed = tiny_model(); resumed.optimized_execution = True; resumed.branch_token_budget = 512
    fit(resumed, train, val, cfg, tmp_path/'resume', resume=tmp_path/'new/step_000001')
    for p, q in zip(new.parameters(), resumed.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_reviewed_stage2_fit_preserves_updates_with_partial_batch(prepared):
    from sa_jev.context_views import prepare_context_views, ReviewedContexts
    root, qs, train, val, config, _ = prepared
    torch.set_num_threads(2)
    prepare_context_views(root, config, reviewer=FakeReviewer())
    bank = ReviewedContexts(root/'data/training_contexts.json', train)
    pairs = build_preferences(qs, [x.state for x in train if not x.state.selected_ids], limit=3)
    old, new = tiny_model(), tiny_model()
    new.optimized_execution = True; new.branch_token_budget = 700
    ref = cache_reference(old, pairs, root/'ref.json')
    cfg = TrainConfig(stage=2, context_view_mode='reviewed_context', accumulation=2, state_batch_size=2,
                      max_steps=2, validate_every=2, save_every=2, stability_warmup_steps=1)
    a = fit(old, train, val, cfg, root/'old_execution', pairs=pairs, reference=ref, context_bank=bank)
    b = fit(new, train, val, cfg, root/'new_execution', pairs=pairs, reference=ref, context_bank=bank)
    for x, y in zip(a, b):
        for key in ('task', 'dpo', 'stability', 'total_loss'):
            assert x[key] == pytest.approx(y[key], abs=3e-6, rel=3e-5)
        assert x['context_pairs'] == y['context_pairs']
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        torch.testing.assert_close(p, q, atol=3e-6, rtol=3e-5, msg=name)


def test_interrupted_reference_reuses_completed_pairs_and_restores_mode(tmp_path, monkeypatch):
    import sa_jev.training as training
    from sa_jev.io import read_json
    qs, train, _ = setup()
    pairs = build_preferences(qs, [train[0].state], limit=3)
    assert len(pairs) > 1
    model = tiny_model().train(); model.optimized_execution = True
    original = training.trajectory_log_probabilities
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('simulated interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(training, 'trajectory_log_probabilities', interrupted)
    path = tmp_path/'ref.json'
    with pytest.raises(RuntimeError, match='simulated interruption'): cache_reference(model, pairs, path)
    assert model.training and len(read_json(path)['values']) == 1
    monkeypatch.setattr(training, 'trajectory_log_probabilities', original)
    resumed = cache_reference(model, pairs, path)
    assert resumed['new_pairs'] == len(pairs)-1
    assert resumed['values'] == cache_reference(model, pairs, tmp_path/'fresh.json')['values']
