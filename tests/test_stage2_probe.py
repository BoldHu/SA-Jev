import pytest
import torch

from test_context_views import FakeReviewer, prepared
from sa_jev.io import read_json, write_json, write_jsonl
from sa_jev.model import tiny_model
from sa_jev.probes import latest_stage1_probe, prepare_stage2_probe, run_stage2_probe
from sa_jev.trajectories import build_preferences


def setup_probe(prepared):
    root, qs, train, val, cfg, extra = prepared
    torch.set_num_threads(2)
    write_json(root/'data/data_manifest.json', {'question_fingerprint': 'TEST_ONLY'})
    write_json(root/'data/overlap_audit.json', {'question_fingerprint': 'TEST_ONLY', 'complete_scan': True})
    write_jsonl(root/'data/validation_states.jsonl', val)
    pairs = build_preferences(qs, [x.state for x in train if not x.state.selected_ids], limit=3)
    write_jsonl(root/'data/preferences.jsonl', pairs)
    checkpoint = root/'runs/stage1_probe/20261112_test/seed_17/step_000006'
    tiny_model().save_trainable(checkpoint, {'stage': 1, 'step': 6})
    write_json(checkpoint.parent/'latest.json', {'checkpoint': str(checkpoint), 'step': 6})
    cfg['model'] = {}
    cfg['training'] = {'accumulation': 2, 'state_batch_size': 1, 'context_view_mode': 'reviewed_context',
                       'validate_every': 1, 'save_every': 1}
    output = root/'runs/stage2_probe/test/seed_17'
    return root, cfg, checkpoint, output, pairs


def test_stage2_probe_bounds_reference_and_preserves_formal_artifacts(prepared, monkeypatch):
    import sa_jev.probes as probes
    from sa_jev import experiments
    root, cfg, checkpoint, output, pairs = setup_probe(prepared)
    assert latest_stage1_probe(root) == checkpoint
    protected = [root/'data/training_contexts.json', root/'reference/seed_17.json',
                 root/'runs/full/seed_17/latest.json', root/'runs/stage1/seed_17/latest.json']
    for p in protected: write_json(p, {'sentinel': 'preserve formal data'})
    originals = {p: p.read_bytes() for p in protected}
    reviewer = FakeReviewer()
    result = prepare_stage2_probe(root, cfg, output, checkpoint, question_limit=1, reviewer=reviewer)
    assert result['questions'] == 1 and reviewer.calls <= 2
    calls = reviewer.calls
    prepare_stage2_probe(root, cfg, output, checkpoint, question_limit=1, reviewer=reviewer)
    assert reviewer.calls == calls
    monkeypatch.setattr(probes.SAJev, 'from_pretrained', lambda **kwargs: tiny_model())
    monkeypatch.setattr(experiments, 'environment_manifest', lambda *args: None)
    history = run_stage2_probe(root, cfg, output, checkpoint, max_steps=2, pair_limit=1, validation_limit=2)
    assert len(history) == 2
    reference = read_json(output/'reference.json')
    assert len(reference['values']) == 1
    assert set(reference['values']) <= {p.fingerprint for p in pairs}
    checks = read_json(output/'probe_checks.json')
    assert checks['paths_exercised']
    assert checks['context_pairs'] > 0 and checks['valid_stab_rank'] > 0
    assert (output/'training.csv').exists()
    for p, before in originals.items(): assert p.read_bytes() == before
    with pytest.raises(FileExistsError):
        run_stage2_probe(root, cfg, output, checkpoint)


def test_probe_rejects_formal_output_and_missing_audit_before_reviews(prepared):
    root, cfg, checkpoint, output, _ = setup_probe(prepared)
    reviewer = FakeReviewer()
    with pytest.raises(ValueError, match='separate run'):
        prepare_stage2_probe(root, cfg, root/'runs/full/seed_17', checkpoint, reviewer=reviewer)
    write_json(root/'data/overlap_audit.json', {'question_fingerprint': 'stale', 'complete_scan': True})
    with pytest.raises(ValueError, match='overlap audit'):
        prepare_stage2_probe(root, cfg, output, checkpoint, reviewer=reviewer)
    assert reviewer.calls == 0


def test_probe_does_not_force_rejected_context(prepared):
    root, cfg, checkpoint, output, _ = setup_probe(prepared)
    with pytest.raises(ValueError, match='No approved eligible context'):
        prepare_stage2_probe(root, cfg, output, checkpoint, question_limit=1, reviewer=FakeReviewer(reject=True))
    assert not (output/'latest.json').exists()
