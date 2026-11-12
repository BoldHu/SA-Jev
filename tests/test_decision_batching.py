from dataclasses import replace

import pytest
import torch

from sa_jev.losses import task_loss
from sa_jev.model import tiny_model
from sa_jev.prompts import requests
from sa_jev.schema import Pool, State, labels_for
from sa_jev.smoke import fixture_questions


@pytest.mark.parametrize("branches", [2, 4])
def test_batching_preserves_predictions_loss_and_all_gradients(branches):
    torch.set_num_threads(2)
    q = fixture_questions()[0]
    state = State(q.id, q.group_id, q.question, Pool(q.id, q.evidence, "fixture"), budget=3)
    model = tiny_model(branch_microbatch=1)
    model.train()
    model.decision_batch_size = 1
    before = model(state)
    loss, _ = task_loss([before], [labels_for(q, state)])
    loss["task"].backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.decision_batch_size = 4  # Exercises a partial last decision batch.
    model.core.branch_microbatch_size = branches
    after = model(state)
    updated, _ = task_loss([after], [labels_for(q, state)])
    updated["task"].backward()
    assert before.candidate_ids == after.candidate_ids
    for key in ("log_h", "log_p", "log_rel"):
        torch.testing.assert_close(getattr(before, key), getattr(after, key), atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(loss["task"], updated["task"], atol=2e-6, rtol=2e-5)
    assert grads.keys() == {n for n, p in model.named_parameters() if p.grad is not None}
    for n, p in model.named_parameters():
        if n in grads:
            torch.testing.assert_close(p.grad, grads[n], atol=2e-6, rtol=2e-4, msg=n)


def test_batched_decisions_keep_isolated_inputs_masks_order_and_costs():
    torch.set_num_threads(2)
    q = fixture_questions()[0]
    state = State(q.id, q.group_id, q.question, Pool(q.id, q.evidence, "fixture"), budget=3)
    model = tiny_model(branch_microbatch=4).eval()
    choice, suf, rel = requests(state)
    bodies = [rel[-1], choice, suf, rel[0]]  # Different K, lengths and task types.
    model.decision_batch_size = 1
    with torch.no_grad():
        expected = model.decisions(bodies)
        model.decision_batch_size = 8
        captured = []
        handle = model.core.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        actual = model.decisions(bodies)
        handle.remove()
    assert len(captured) == 1
    assert captured[0]["option_mask"].sum().item() == 12  # 2 + 6 + 2 + 2
    assert sum(row[2] for row in actual) == captured[0]["input_ids"].numel()
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a[0], b[0], atol=2e-6, rtol=2e-5)
        assert a[1] == b[1]
        torch.testing.assert_close(b[0].exp().sum(), torch.tensor(1.))
    # Changing candidates must never affect the sufficient-only decision.
    changed = replace(state, pool=replace(state.pool, candidates=state.pool.candidates[:2]))
    assert requests(changed)[1] == suf
    assert model.decisions([]) == []


def test_full_state_uses_fewer_core_calls_without_changing_tokens():
    torch.set_num_threads(2)
    q = fixture_questions()[0]
    state = State(q.id, q.group_id, q.question, Pool(q.id, q.evidence, "fixture"), budget=3)
    model = tiny_model().eval()
    calls = []
    handle = model.core.register_forward_pre_hook(lambda module, args: calls.append(1))
    with torch.no_grad():
        model.decision_batch_size = 1
        old = model(state)
        assert len(calls) == 8
        calls.clear()
        model.decision_batch_size = 8
        new = model(state)
        assert len(calls) == 3
    handle.remove()
    assert old.logical_tokens == new.logical_tokens


@pytest.mark.parametrize("include_rel", [True, False])
def test_multiple_states_keep_outputs_and_gradients_isolated(include_rel):
    torch.set_num_threads(2)
    qs = fixture_questions()
    states = [State(q.id, q.group_id, q.question, Pool(q.id, q.evidence[:k], "fixture"), budget=3)
              for q, k in zip(qs, [6, 3, 4])]
    states[1] = states[1].read(states[1].candidates[0].id)
    states[2] = replace(states[2], selected_ids=tuple(e.id for e in states[2].candidates), budget=0)
    model = tiny_model(branch_microbatch=2).train()
    before = [model(s, include_rel=include_rel) for s in states]

    def objective(predictions):
        # Include all output branches even for fully selected states.
        return sum(p.log_h.square().sum() + p.log_p.square().sum() + p.log_rel.square().sum()
                   for p in predictions)

    objective(before).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    after = model.forward_batch(states, include_rel=include_rel)
    objective(after).backward()
    for a, b in zip(before, after):
        assert a.candidate_ids == b.candidate_ids and a.logical_tokens == b.logical_tokens
        for name in ("log_h", "log_p", "log_rel"):
            torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=2e-6, rtol=2e-5)
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n], atol=1e-5, rtol=2e-4, msg=n)
    with torch.no_grad():
        changed = model.forward_batch([states[0], replace(states[1], context="Completely different input")], include_rel=include_rel)[0]
        torch.testing.assert_close(changed.log_h, after[0].log_h)
        torch.testing.assert_close(changed.log_p, after[0].log_p)


@pytest.mark.parametrize("state_batch_size", [2, 3, 4])
def test_effective_batch_update_matches_serial_with_partial_microbatch(tmp_path, state_batch_size):
    from sa_jev.data import task_states
    from sa_jev.training import TrainConfig, fit

    torch.set_num_threads(2)
    qs = fixture_questions()
    train = [x for q in qs[:2] for x in task_states(q, Pool(q.id, q.evidence, "fixture"))]
    # Differing valid-label denominators, not simply one mean per state.
    train[0] = replace(train[0], labels=replace(train[0].labels, rank_valid=False,
                       rel_valid=(True, False, False, False, False, False)))
    q = qs[2]
    val = task_states(q, Pool(q.id, q.evidence, "fixture"))
    serial, batched = tiny_model(), tiny_model()
    cfg = TrainConfig(accumulation=5, max_steps=2, validate_every=2, save_every=2)
    a = fit(serial, train, val, cfg, tmp_path/'serial')
    b = fit(batched, train, val, replace(cfg, state_batch_size=state_batch_size), tmp_path/'batched')
    for x, y in zip(a, b):
        for name in ('rank', 'suf', 'rel', 'task'):
            assert x[name] == pytest.approx(y[name], abs=2e-6, rel=2e-5)
        for name in ('valid_rank', 'valid_suf', 'valid_rel'):
            assert x[name] == y[name]
    for (name, x), (_, y) in zip(serial.named_parameters(), batched.named_parameters()):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5, msg=name)
