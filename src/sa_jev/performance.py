"""Conservative execution presets and measured-only training-time projections."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
from statistics import median

from .io import read_json, write_json


def execution_config(config, profile='a100_40gb', *, stage=1, gpu_gib=None,
                     checkpoint_mode='branch'):
    """Starting points to benchmark, NOT assertions of speed or memory safety.

    Effective batch, data, optimizer, precision, losses and token limits stay intact.
    branch_token_budget controls only execution grouping; long inputs are not cut.
    """
    if stage not in (1, 2): raise ValueError('stage must be 1 or 2')
    recipes = {'a100_40gb': (4, 2048 if stage == 1 else 1536, 2 if stage == 1 else 1),
               'a100_80gb': (8, 4096, 4 if stage == 1 else 2),
               'h100_80gb': (8, 4096, 4 if stage == 1 else 2)}
    if profile not in recipes: raise ValueError(f'Unknown execution profile: {profile}')
    if gpu_gib is None:
        import torch
        if not torch.cuda.is_available(): raise ValueError('Select a CUDA GPU before setting the execution profile')
        gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_gib < (35 if profile == 'a100_40gb' else 70):
        raise ValueError(f'{profile} requires a larger GPU than {gpu_gib:.2f} GiB')
    if checkpoint_mode not in {'branch', 'layer', 'branch_layer'}:
        raise ValueError('Unknown checkpoint mode')
    result = deepcopy(config)
    branches, tokens, states = recipes[profile]
    result['model'].update(optimized_execution=True, branch_microbatch=branches,
                           branch_token_budget=tokens, decision_batch_size=8, checkpoint_mode=checkpoint_mode)
    result['training']['state_batch_size'] = min(states, result['training'].get('accumulation', 8))
    result['training']['memory_limit_gib'] = min(76., gpu_gib-3.)
    print(f'[execution] {profile}: branches <= {branches}; padded tokens/call <= {tokens} '
          f'(one longer branch remains intact); states={result["training"]["state_batch_size"]}; '
          f'effective batch={result["training"].get("accumulation", 8)}; checkpoint={checkpoint_mode}. '
          'Unmeasured starting configuration: inspect GPU peak and throughput.', flush=True)
    return result


def throughput_report(history, output, *, training_states=36000, effective_batch=8,
                      epochs=1, validation_states=600, validate_every=500, reference=None,
                      reference_pairs=3000, stage=1):
    """Only project the same hardware/config; no fabricated 80GB/H100 speedup."""
    full = [r for r in history if r.get('batch_states') == effective_batch]
    timed = full[1:]  # Exclude first full update; retain all later complete updates.
    if not timed:
        result = {'stage': stage, 'estimate_available': False, 'reason': 'Need at least two complete effective batches'}
    else:
        steps = math.ceil(training_states/effective_batch)*epochs
        step_seconds = median(r['step_seconds'] for r in timed)
        validation = [r for r in history if r.get('validation_states') and 'validation_seconds' in r]
        val_pass = median(r['validation_seconds']/r['validation_states'] for r in validation)*validation_states if validation else None
        val_total = val_pass*math.ceil(steps/validate_every) if val_pass is not None else None
        ref_total = None
        if stage == 2 and reference and reference.get('new_pairs', 0) > 0:
            ref_total = reference['compute_seconds']/reference['new_pairs']*reference_pairs
        measured_component_total = steps*step_seconds+(val_total or 0)+(ref_total or 0)
        result = dict(stage=stage, estimate_available=True, full_updates_measured=len(timed),
                      effective_batch=effective_batch, formal_updates=steps, seconds_per_update_median=step_seconds,
                      task_seconds_median=median(r['task_seconds'] for r in timed),
                      dpo_seconds_median=median(r['dpo_seconds'] for r in timed),
                      training_hours=steps*step_seconds/3600,
                      validation_hours_estimate=None if val_total is None else val_total/3600,
                      reference_hours_estimate=None if ref_total is None else ref_total/3600,
                      measured_components_hours_estimate=measured_component_total/3600,
                      required_seconds_per_update_for_stage_alone_24h=86400/steps,
                      peak_reserved_gib=max(r['peak_reserved_gib'] for r in history),
                      observed_remaining_k=[min(r['min_k'] for r in history), max(r['max_k'] for r in history)],
                      sampled_context_pair_fraction=sum(r.get('context_pairs',0) for r in history)/sum(r['batch_states'] for r in history),
                      limitation='Small probe, possibly repeated/selected questions. Same GPU/config only; '
                                 'not a full-data timing guarantee. Excludes checkpoint I/O, initialization, '
                                 'API review and interruptions. Reference/validation extrapolate sampled items.')
    import torch
    result['hardware'] = ({'gpu': torch.cuda.get_device_name(0),
                           'memory_gib': torch.cuda.get_device_properties(0).total_memory/1024**3}
                          if torch.cuda.is_available() else {'gpu': 'cpu'})
    write_json(Path(output)/'throughput.json', result)
    print('[throughput]', result, flush=True)
    return result


def combined_time_report(stage1_directory, stage2_directory, output):
    a, b = [read_json(Path(p)/'throughput.json') for p in (stage1_directory, stage2_directory)]
    if not (a.get('estimate_available') and b.get('estimate_available')):
        raise ValueError('Both probes need measured complete effective batches')
    if a.get('hardware') != b.get('hardware'):
        raise ValueError('Cannot add timings from different GPUs as a single-device estimate')
    hours = a['measured_components_hours_estimate'] + b['measured_components_hours_estimate']
    result = {'same_hardware_two_stage_hours_estimate': hours,
              'speedup_needed_to_fit_24h_before_unmeasured_overheads': max(1., hours/24),
              'scope': 'One seed, one configured epoch per stage; includes sampled validation/reference estimates. '
                       'Not a forecast for another GPU. API/checkpoint/loading overhead remains unmeasured.'}
    write_json(Path(output)/'combined_time.json', result)
    print('[24h target]', result, flush=True)
    return result


def stress_probe(model, examples, config, output, context_bank=None, pairs=()):
    """Read-only-weight forward/backward checks on large K and long inputs.

    No optimizer update, API call, truncation or task change. This samples extremes
    from the supplied data, not an assertion that every possible input fits.
    """
    import random
    import torch
    from .training import task_predictions, memory_stats, synchronize
    from .losses import task_loss, stability_loss, dpo_loss
    from .prompts import View
    from .trajectories import trajectory_log_probabilities
    from time import perf_counter

    if not examples: raise ValueError('Stress probe requires task examples')
    largest = max(examples, key=lambda x: len(x.state.candidates))
    longest = max(examples, key=lambda x: len(x.state.question)+len(x.state.context)
                  +sum(len(e.text) for e in x.state.selected)
                  +max((len(e.text) for e in x.state.candidates), default=0))
    # Keep a full microbatch, including duplicated examples if it is very small.
    rows = ([longest, largest]*config.state_batch_size)[:config.state_batch_size]
    rows = [rows, [largest]*config.state_batch_size]
    device = model.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    was_training = model.training
    reports = []
    try:
        with torch.random.fork_rng(devices=devices):
            model.train()
            for index, batch in enumerate(rows):
                model.zero_grad(set_to_none=True)
                if devices: torch.cuda.reset_peak_memory_stats(device)
                expanded, views, stable, original, augmented = [], [], [], [], []
                for x in batch:
                    second = context_bank.view(x, random.Random(17)) if context_bank is not None else None
                    if second is not None and not model.view_within_budget(x.state, second): second = None
                    needs = bool(second is not None and x.labels.suf_valid and x.labels.gold_available and not x.labels.sufficient)
                    original.append(len(expanded)); expanded.append(x); views.append(View()); stable.append(needs)
                    if second is not None:
                        augmented.append(len(expanded)); expanded.append(x); views.append(second); stable.append(needs)
                    else: augmented.append(None)
                synchronize(device); started = perf_counter()
                print(f'[stress] {index+1}/2: K={[len(x.state.candidates) for x in batch]}, '
                      f'states={len(batch)}, views={len(expanded)}; backward, NO optimizer update', flush=True)
                predictions = task_predictions(model, expanded, views, stable)
                loss = task_loss(predictions, [x.labels for x in expanded])[0]['task']
                for x, a, b in zip(batch, original, augmented):
                    if b is not None:
                        loss = loss + .1*stability_loss([predictions[a]], [predictions[b]], [x.labels], [x.state], [x.state])[0]
                loss.backward()
                del loss, predictions
                model.zero_grad(set_to_none=True)
                if pairs and index == 0:
                    pair = max(pairs, key=lambda p: sum(len(s.context)+sum(len(e.text) for e in s.selected)
                               +sum(len(e.text) for e in s.candidates) for t in (p.chosen,p.rejected) for s,_ in t.replay()[0]))
                    a, b = trajectory_log_probabilities(model, (pair.chosen, pair.rejected), config.state_batch_size)
                    (.2*dpo_loss(a, b, a.detach(), b.detach())).backward()
                    del a, b
                    model.zero_grad(set_to_none=True)
                synchronize(device)
                reports.append({'seconds': perf_counter()-started, 'remaining_k': [len(x.state.candidates) for x in batch],
                                'views': len(expanded), **memory_stats(device)})
                print('[stress]', reports[-1], flush=True)
                if reports[-1]['peak_reserved_gib'] >= config.memory_limit_gib:
                    raise RuntimeError('Stress probe exceeds memory threshold; reduce branch concurrency/token budget')
    finally:
        model.zero_grad(set_to_none=True); model.train(was_training)
    write_json(Path(output)/'stress_probe.json', {'weight_updates': 0, 'measurements': reports,
               'scope': 'Selected high-K/long states within supplied data; not a guarantee for all inputs'})
    return reports
