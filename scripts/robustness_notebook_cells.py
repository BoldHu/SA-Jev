"""05 notebook cells, shared with the notebook generator."""
TITLE = '# 独立干扰与有效变化\n先对 heldout 草稿做可缓存的机器审核，再评测；人工抽查另行记录。禁止用于训练或调参。'
ROBUSTNESS = [
    ('md', '先审核已有 heldout 扰动草稿，再评测。审核通过表示机器审核，不代表人工核验或绝对语义保证。默认只预检查 3 个包；确认 API 返回正常后设置 REVIEW_LIMIT=None。预检查不写正式测试文件，不运行模型评测。'),
    ('code', '''from pathlib import Path
import importlib
from sa_jev import context_views, perturbation_review
importlib.reload(context_views)
importlib.reload(perturbation_review)

# 保持 MODE='full' 使用已经准备好的数据；不要为了少量检查切换 ROOT。
REVIEW_LIMIT = 3  # 首次预检查；随后改为 None，审核全部草稿并导出正式文件。
config.setdefault('perturbation_review', {}).update(max_tokens=4096, disable_thinking=True)
data_dir = ROOT / 'data'
draft_path = data_dir / 'perturbations_draft.jsonl'
if not draft_path.exists():
    cli('review-packets')
print('草稿:', draft_path)
print('审核范围:', '全部草稿' if REVIEW_LIMIT is None else f'前 {REVIEW_LIMIT} 个包（预检查）')'''),
    ('code', '''review_summary = perturbation_review.prepare_perturbation_reviews(
    ROOT, config, limit=REVIEW_LIMIT)
print({k: v for k, v in review_summary.items() if k != 'decisions'})'''),
    ('md', '审核中断后重跑审核格，会复用逐包缓存。API 输出不完整时停止并保留原始响应；不伪造审核通过。全部审核后，data/perturbations_reviewed.jsonl 保存合格包，配套 manifest 记录实际数量、通过率和拒绝原因；perturbation_reviews/ 保存原文、Gold 和 API 审核记录（仅审核使用，不输入评测模型）。perturbations_reviewed.human_spotcheck.json 是待填写的人工抽查清单，不代表人工审核已完成。论文需报告机器审核及真实人工抽查结果；若发现错误，应修正或移除对应包并更新评测集记录，不能用于选择模型或调参。'),
    ('code', '''from sa_jev.experiments import robustness_experiment
if REVIEW_LIMIT is not None:
    print('预检查完成，尚未运行正式评测。将 REVIEW_LIMIT 改为 None，重跑设置格、审核格及本格。')
else:
    from sa_jev.io import read_jsonl, digest
    reviewed_path = ROOT / 'data/perturbations_reviewed.jsonl'
    if (review_summary['scope'] != 'full_draft' or review_summary['status'] != 'complete'
            or digest(list(read_jsonl(reviewed_path))) != review_summary['output_hash']):
        raise ValueError('正式审核导出不完整或已变更，请检查审核记录。')
    rows = robustness_experiment(ROOT, config)'''),
    ('md', '同时检查 JS、Top-1/Top-5、充分性翻转、错误提前停止、nDCG 和有效变化后正确补证。新增已选证据会改变剩余候选集合，不强行比较为不变分布。独立扰动测试禁止用于训练或调参。'),
]
