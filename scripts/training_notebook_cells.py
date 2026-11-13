"""Editable training notebook cells; generated notebooks keep research code in src/."""

CHECK = '''# Small Qwen3.5 + LoRA check: no downloads, no API, no 4B training.
import torch
assert torch.cuda.is_available(), 'Select an A100 GPU runtime'
assert torch.cuda.is_bf16_supported(including_emulation=False), 'Native BF16 GPU required'
for check_dtype in ('float32', 'bfloat16'):
    log_path = ROOT / f'qwen_execution_{check_dtype}.log'
    cmd = [sys.executable, '-u', 'scripts/check_qwen_lora.py', '--device', 'cuda',
           '--compare-execution', '--dtype', check_dtype,
           '--output', str(ROOT/f'qwen_execution_{check_dtype}.json')]
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                status = process.wait()
            except KeyboardInterrupt:
                process.terminate(); process.wait(); raise
    if status:
        raise RuntimeError(f'Execution equivalence check failed; inspect {log_path}. Do not start 4B training.')
'''

STAGE1 = [
('md', '''## 40GB A100：优化执行路径测试
保留 MODE='full' 读取已准备的数据，不重跑 00。默认仅 8 步；不会覆盖旧的 probe 或正式模型。
先运行下面的真实 Qwen3.5 小结构 FP32/BF16 损失、梯度检查，再做完整数据中大 K／长样本的无参数更新压力测试。
优化只改变执行：相同请求在同次前向中共享梯度图、分支按长度组批、跳过未参与损失的输出。候选归一化、精度、数据和损失权重不变。
新版单元格的旧输出已清空；旧 notebook 原件备份在本地 outputs/notebook_backups。'''),
('code', CHECK),
('md', '## 设置（在这里改）\n当前用 a100_40gb；换到 80GB 后改 PROFILE，而不是改变数据 MODE。CHECKPOINT_MODE 默认 branch；若压力测试显存不足，可单独测试 branch_layer。所有配置都是待实测起点，不能保证最坏输入的显存。'),
('code', '''import copy, importlib, torch
from sa_jev import execution, model, trajectories, training, performance, experiments
for module in (execution, model, trajectories, training, performance, experiments):
    importlib.reload(module)
RUN_FULL_STAGE1 = False
PROBE_STEPS = 8
TRAIN_SEEDS = [17]
PROFILE = 'a100_40gb'  # Later: 'a100_80gb' or 'h100_80gb'.
CHECKPOINT_MODE = 'branch'  # Optional memory tradeoff: 'branch_layer'; compare measured speed.
STRESS_CHECK = True
config = performance.execution_config(config, PROFILE, stage=1, checkpoint_mode=CHECKPOINT_MODE)
config['training'].update(accumulation=8, validate_every=500, save_every=25)
print('ROOT:', ROOT, 'GPU:', torch.cuda.get_device_name(0), flush=True)
print('Formal training' if RUN_FULL_STAGE1 else f'Isolated {PROBE_STEPS}-step test; same effective batch of 8', flush=True)
'''),
('code', '''from datetime import datetime, timezone
from uuid import uuid4
from sa_jev.io import read_json
stage1_config = copy.deepcopy(config)
if not RUN_FULL_STAGE1:
    stage1_config['training'].update(validate_every=PROBE_STEPS, save_every=PROBE_STEPS)
probe_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
for seed in (TRAIN_SEEDS if RUN_FULL_STAGE1 else [17]):
    probe_output = None if RUN_FULL_STAGE1 else ROOT/'runs/stage1_probe'/probe_id/f'seed_{seed}'
    history = experiments.train_variant(
        ROOT, stage1_config, 'stage1', seed,
        max_steps=None if RUN_FULL_STAGE1 else PROBE_STEPS,
        validation_limit=None if RUN_FULL_STAGE1 else 6,
        output_directory=probe_output, stress_check=STRESS_CHECK and not RUN_FULL_STAGE1,
    )
    output = probe_output or ROOT/'runs/stage1'/f'seed_{seed}'
    manifest = read_json(ROOT/'data/task_manifest.json')
    timing = performance.throughput_report(history, output,
        training_states=manifest['task_states'], effective_batch=8,
        epochs=config['training']['epochs'], validation_states=manifest['validation_states'],
        validate_every=config['training']['validate_every'], stage=1)
    print('Results/checkpoint:', output, flush=True)
'''),
('md', '''## 看哪些结果
检查小模型 execution_comparison.passed、stress_probe.json、training.jsonl、throughput.json。
每步记录 task_seconds、dpo_seconds、实际分支数与补齐 token 数，loss、梯度范数、显存及 ETA。
8 步只是起点；要估算正式时间，使用同一 PROFILE 再测 20～30 步并核对长样本。不要把小模型耗时当作 4B 吞吐。
OOM 后重启会话，降低 branch_microbatch 或 branch_token_budget；不要截断证据或降低有效批量。branch_token_budget 只限制并行分组，超过它的单条长分支完整保留。
一天目标：两阶段各 36,000 状态、有效批量 8，共 9,000 次更新；两阶段平均单步之和须小于 19.2 秒，且还要留参考、验证、保存时间。未测得该吞吐前，不承诺一天内完成。''')]

STAGE2 = [
('md', '''## 40GB A100：Stage 2 优化路径测试
MODE='full'、RUN_FULL_STAGE2=False。使用 02 最近成功的 probe 检查点，最多审核 16 道训练题、每题 2 段（最多 32 次未缓存审核，网络重试另计），最多 8 对真实完整轨迹。
选题保留此前最先审核的 4 题，并加入大 K／长证据样本及固定随机样本。不是用于报告模型效果的测试集。
默认 8 步、有效批量 8；若小样本不足会明确重复若干 epoch。审核失败的题仍做干净任务，不强制稳定性。
输出写入独立 stage2_probe；不覆盖正式上下文包、参考概率和模型。先完成 02 的两种精度检查；新运行时也可运行下面的小模型检查。'''),
('code', CHECK),
('md', '## 设置（在这里改）\n40GB 先保持 PROFILE=a100_40gb。RUN_FULL_STAGE2=True 会启动全量审核及正式训练，并要求正式 Stage 1 检查点；现在保持 False。'),
('code', '''import copy, importlib, torch
from datetime import datetime, timezone
from uuid import uuid4
from sa_jev import execution, model, trajectories, training, performance, experiments, context_views, probes
for module in (execution, model, trajectories, training, performance, experiments, context_views, probes):
    importlib.reload(module)
RUN_FULL_STAGE2 = False
PROBE_STEPS = 8
PROBE_QUESTIONS = 16
PROBE_PAIRS = 8
TRAIN_SEEDS = [17]
PROFILE = 'a100_40gb'  # Later: 'a100_80gb' or 'h100_80gb'.
CHECKPOINT_MODE = 'branch'
STRESS_CHECK = True
STAGE1_CHECKPOINT = None  # None: most recent successful 02 probe; pin a path for A/B comparisons.
config = performance.execution_config(config, PROFILE, stage=2, checkpoint_mode=CHECKPOINT_MODE)
config['training'].update(accumulation=8, validate_every=500, save_every=25)
probe_config = copy.deepcopy(config)
probe_config['training'].update(validate_every=PROBE_STEPS, save_every=PROBE_STEPS)
probe_config.setdefault('context_views', {})['max_candidates'] = 2
if not RUN_FULL_STAGE2:
    STAGE1_CHECKPOINT = Path(STAGE1_CHECKPOINT) if STAGE1_CHECKPOINT else probes.latest_stage1_probe(ROOT)
    probe_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6]
    PROBE_OUTPUT = ROOT/'runs/stage2_probe'/probe_id/'seed_17'
    print('Stage 1 checkpoint:', STAGE1_CHECKPOINT, flush=True)
    print('Stage 2 output:', PROBE_OUTPUT, flush=True)
    print('Limits:', PROBE_QUESTIONS, 'questions;', PROBE_PAIRS, 'pairs;', PROBE_STEPS, 'updates', flush=True)
else:
    print('FORMAL Stage 2: complete reviews, all reference pairs, formal Stage 1 required.', flush=True)
'''),
('code', '''if RUN_FULL_STAGE2:
    context_summary = context_views.prepare_context_views(ROOT, config)
else:
    context_summary = probes.prepare_stage2_probe(ROOT, probe_config, PROBE_OUTPUT, STAGE1_CHECKPOINT,
                                                  question_limit=PROBE_QUESTIONS)
context_summary
'''),
('code', '''assert torch.cuda.is_available(), 'GPU required for training'
if RUN_FULL_STAGE2:
    for seed in TRAIN_SEEDS:
        experiments.train_variant(ROOT, config, 'full', seed)
else:
    history = probes.run_stage2_probe(ROOT, probe_config, PROBE_OUTPUT, STAGE1_CHECKPOINT,
        max_steps=PROBE_STEPS, pair_limit=PROBE_PAIRS, validation_limit=6, stress_check=STRESS_CHECK)
    print('Inspect probe_checks.json, stress_probe.json and throughput.json:', PROBE_OUTPUT, flush=True)
    if (STAGE1_CHECKPOINT.parent/'throughput.json').exists():
        performance.combined_time_report(STAGE1_CHECKPOINT.parent, PROBE_OUTPUT, PROBE_OUTPUT)
    else:
        print('Stage 1 checkpoint has no new throughput report. Run updated 02 for a combined estimate.', flush=True)
'''),
('md', '''## 验收与后续
probe_checks.json 必须有 completed_updates=8、dpo_finite=True、checkpoint_saved=True、paths_exercised=True。
比较 task_seconds 与 dpo_seconds；throughput.json 只外推同卡同配置，不凭显存容量推算 H100/A100 80GB 速度。
稳定性按原 100 步升温保留，8 步不能证明最终稳定性或 QA 效果；验证 rank=None 仍表示没有可用金标，不能算排序通过。
一次 OOM 不会自动跳过样本或缩小候选；重启后减少并发／token 分组上限，或测试 branch_layer 重计算模式。
冻结参考与 DPO 仍使用干净完整轨迹，不截断、不改变偏好、不将预算耗尽当作 STOP。新增性能优化不需要重跑 00 或重新生成全部轨迹。
正式实验保留 36,000 状态、所有原有监督、学习率、有效批量和完整轨迹；24 小时是需要实测达成的目标，不以减少数据或省略目标换取。''')]
