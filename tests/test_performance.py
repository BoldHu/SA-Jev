from copy import deepcopy

import pytest
import torch

from sa_jev.performance import execution_config, throughput_report, combined_time_report, stress_probe
from sa_jev.training import TrainConfig
from sa_jev.model import tiny_model
from test_training_progress import examples


def test_presets_change_only_execution_and_refuse_wrong_gpu():
    config = {'model': {'max_branch_length': 8192}, 'training': {'accumulation': 8, 'epochs': 1,
              'learning_rate': 2e-5, 'stability_weight': .1}, 'data': {'main_k': 32}}
    before = deepcopy(config)
    result = execution_config(config, stage=2, gpu_gib=39.49)
    assert config == before
    assert result['training']['accumulation'] == 8 and result['training']['state_batch_size'] == 1
    assert result['model']['max_branch_length'] == 8192 and result['data'] == config['data']
    with pytest.raises(ValueError, match='larger GPU'):
        execution_config(config, 'a100_80gb', gpu_gib=39.49)


def test_time_report_excludes_partial_and_warmup_and_does_not_predict_other_gpu(tmp_path):
    def row(n, seconds):
        return dict(batch_states=n, step_seconds=seconds, task_seconds=seconds-1, dpo_seconds=1,
                    peak_reserved_gib=12., min_k=4, max_k=10, context_pairs=0)
    rows = [row(8, 100), row(8, 10), row(1, 1), row(8, 20)]
    result = throughput_report(rows, tmp_path/'s1', training_states=36000, effective_batch=8)
    assert result['seconds_per_update_median'] == 15
    assert result['formal_updates'] == 4500
    assert result['training_hours'] == 18.75
    assert result['required_seconds_per_update_for_stage_alone_24h'] == 19.2
    assert result['reference_hours_estimate'] is None
    throughput_report(rows, tmp_path/'s2', stage=2)
    joined = combined_time_report(tmp_path/'s1', tmp_path/'s2', tmp_path)
    assert joined['same_hardware_two_stage_hours_estimate'] == 37.5
    assert not throughput_report(rows[:1], tmp_path/'short')['estimate_available']


def test_stress_does_not_change_weights_or_rng(tmp_path):
    torch.set_num_threads(2)
    train, _ = examples()
    model = tiny_model(); model.optimized_execution = True
    weights = deepcopy(model.state_dict()); rng = torch.get_rng_state().clone()
    reports = stress_probe(model, train, TrainConfig(state_batch_size=2), tmp_path)
    assert len(reports) == 2 and torch.equal(rng, torch.get_rng_state())
    for key, value in weights.items(): torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    assert all(p.grad is None for p in model.parameters())


def test_training_log_appends_and_resume_repairs_incomplete_tail(tmp_path, monkeypatch):
    import sa_jev.training as training
    from sa_jev.io import read_jsonl
    torch.set_num_threads(2)
    train, val = examples()
    cfg = TrainConfig(accumulation=2, max_steps=2, save_every=1, validate_every=2)
    calls = []
    original = training.write_jsonl
    def counted(path, values):
        if path.name == 'training.jsonl': calls.append(path)
        return original(path, values)
    monkeypatch.setattr(training, 'write_jsonl', counted)
    training.fit(tiny_model(), train, val, cfg, tmp_path/'run')
    assert len(calls) == 1  # No O(steps^2) history rewrite during training.
    log = tmp_path/'run/training.jsonl'
    with log.open('a') as stream: stream.write('{"interrupted":')
    training.fit(tiny_model(), train, val, cfg, tmp_path/'run', resume=tmp_path/'run/step_000001')
    assert [r['step'] for r in read_jsonl(log)] == [1, 2]
