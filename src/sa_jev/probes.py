"""Bounded Stage-2 plumbing checks using real data and a Stage-1 probe checkpoint."""
from __future__ import annotations

import gc
import random
from pathlib import Path

import numpy as np
import torch

from .context_views import ReviewedContexts, prepare_context_views
from .io import digest, read_json, read_jsonl, write_json
from .model import SAJev
from .schema import TaskExample
from .trajectories import PreferencePair
from .training import TrainConfig, cache_reference, fit


def latest_stage1_probe(root, seed=17):
    """Find a completed probe, never manufacture a formal Stage-1 run."""
    markers = sorted((Path(root)/'runs/stage1_probe').glob(f'*/seed_{seed}/latest.json'), reverse=True)
    for marker in markers:
        path = marker.parent / Path(read_json(marker)['checkpoint']).name
        if (path/'trainable.safetensors').is_file() and (path/'metadata.json').is_file():
            if read_json(path/'metadata.json').get('metadata', {}).get('stage') == 1:
                return path
    raise FileNotFoundError('No completed Stage-1 probe found; set STAGE1_CHECKPOINT to the successful 02 checkpoint.')


def _check_sources(root, config, output, checkpoint):
    from .public_indexes import require_public_question_protocol
    root, output, checkpoint = Path(root), Path(output), Path(checkpoint)
    base = (root/'runs/stage2_probe').resolve()
    if output.resolve() == base or not output.resolve().is_relative_to(base):
        raise ValueError('Stage-2 probe output must be a separate run under ROOT/runs/stage2_probe/')
    require_public_question_protocol(root, config)
    audit = read_json(root/'data/overlap_audit.json')
    manifest = read_json(root/'data/data_manifest.json')
    if not audit.get('complete_scan') or audit.get('question_fingerprint') != manifest['question_fingerprint']:
        raise ValueError('Current data require a matching completed OpenJevData overlap audit')
    if not (checkpoint/'trainable.safetensors').is_file():
        raise FileNotFoundError(f'Stage-1 weights not found: {checkpoint}')
    if read_json(checkpoint/'metadata.json').get('metadata', {}).get('stage') != 1:
        raise ValueError('Probe initialization requires a Stage-1 checkpoint')
    return root, output, checkpoint


def prepare_stage2_probe(root, config, output, stage1_checkpoint, *, question_limit=4, reviewer=None):
    root, output, checkpoint = _check_sources(root, config, output, stage1_checkpoint)
    if question_limit < 1:
        raise ValueError('question_limit must be positive')
    if (output/'latest.json').exists():
        raise FileExistsError('This probe already trained; rerun the settings cell for a new isolated directory')
    # Review only questions for which a real, already-created preference pair exists.
    ids = []
    for raw in read_jsonl(root/'data/preferences.jsonl'):
        pair = PreferencePair.from_dict(raw)
        qid = pair.chosen.initial.question_id
        if qid not in ids:
            ids.append(qid)
        if len(ids) == question_limit:
            break
    if not ids:
        raise ValueError('No real trajectory preferences available; do not synthesize a preference to pass the probe')
    print(f'[stage2 probe] Reviewing at most {len(ids)} training questions; no retrieval or trajectory regeneration.', flush=True)
    summary = prepare_context_views(root, config, question_ids=ids, reviewer=reviewer,
                                    output_path=output/'training_contexts.json')
    write_json(output/'probe_plan.json', {'purpose': 'pipeline_smoke_only_not_paper_results',
               'source_root': str(root), 'stage1_checkpoint': str(checkpoint), 'question_ids': ids,
               'question_limit': question_limit, 'review_summary': summary})
    if not summary['eligible_states']:
        raise ValueError('No approved eligible context views. Inspect review results; stability must not be forced.')
    return summary


def run_stage2_probe(root, config, output, stage1_checkpoint, *, max_steps=2, pair_limit=2,
                     validation_limit=2, seed=17):
    from .experiments import environment_manifest
    from .reporting import training_report
    root, output, checkpoint = _check_sources(root, config, output, stage1_checkpoint)
    if max_steps is None or max_steps < 1 or pair_limit < 1 or validation_limit < 1:
        raise ValueError('Probe steps, preference count and validation count must be explicitly bounded')
    if (output/'latest.json').exists():
        raise FileExistsError('Completed probe exists; use a new probe directory, not the formal run directory')
    all_train = [TaskExample.from_dict(x) for x in read_jsonl(root/'data/train_states.jsonl')]
    bank = ReviewedContexts(output/'training_contexts.json', all_train, require_complete=False)
    train = [x for x in all_train if digest(x) in bank.rows]
    del all_train
    if not train:
        raise ValueError('No audited task states available for the probe')
    groups = {x.state.group_id for x in train}
    ids = {x.state.question_id for x in train}
    pairs = []
    for raw in read_jsonl(root/'data/preferences.jsonl'):
        pair = PreferencePair.from_dict(raw)
        if pair.chosen.initial.question_id in ids and pair.chosen.initial.group_id in groups:
            pairs.append(pair)
        if len(pairs) == pair_limit:
            break
    if not pairs:
        raise ValueError('No existing preference pair matches the reviewed training questions')
    val = []
    for raw in read_jsonl(root/'data/validation_states.jsonl'):
        val.append(TaskExample.from_dict(raw))
        if len(val) == validation_limit:
            break
    cfg = TrainConfig(**config['training'], stage=2, seed=seed, max_steps=max_steps)
    if cfg.context_view_mode != 'reviewed_context' or cfg.dpo_weight <= 0 or cfg.stability_weight <= 0:
        raise ValueError('Full Stage-2 probe must exercise reviewed-context views, DPO and stability')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    print(f'[stage2 probe] {len(train)} task states, {len(pairs)} complete preference pairs, '
          f'{len(val)} validation states, at most {max_steps} updates; TEST ONLY.', flush=True)
    environment_manifest(output/'environment.json', config)
    write_json(output/'probe_inputs.json', {'purpose': 'pipeline_smoke_only_not_paper_results',
               'stage1_checkpoint': str(checkpoint), 'question_ids': sorted(ids),
               'states': len(train), 'pairs': len(pairs), 'max_steps': max_steps,
               'reference_cache': str(output/'reference.json')})
    model = SAJev.from_pretrained(**config['model'], trainable=True)
    try:
        model.load_trainable(checkpoint)
        reference = cache_reference(model, pairs, output/'reference.json')
        history = fit(model, train, val, cfg, output, pairs=pairs, reference=reference, context_bank=bank)
        training_report(history, output)
        checks = {'purpose': 'pipeline_smoke_only_not_paper_results',
                  'completed_updates': len(history), 'dpo_finite': all(np.isfinite(h['dpo']) for h in history),
                  'context_pairs': sum(h['context_pairs'] for h in history),
                  'valid_stab_rank': sum(h['valid_stab_rank'] for h in history),
                  'valid_stab_suf': sum(h['valid_stab_suf'] for h in history),
                  'checkpoint_saved': (output/'latest.json').is_file()}
        checks['paths_exercised'] = bool(checks['completed_updates'] and checks['dpo_finite']
                                        and checks['context_pairs'] and checks['valid_stab_rank']
                                        and checks['valid_stab_suf'] and checks['checkpoint_saved'])
        write_json(output/'probe_checks.json', checks)
        print('[stage2 probe]', checks, flush=True)
        if not checks['paths_exercised']:
            print('Some paths were not exercised (possibly views exceed the token budget); do not call this a complete Stage-2 check.', flush=True)
        return history
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
