"""Generate self-contained Colab entrypoints; research logic lives in src/sa_jev."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

BOOT = '''from pathlib import Path
import os, sys, subprocess, zipfile
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE = Path('/content/drive/MyDrive/SA-Jev')
    PROJECT = Path('/content/graphrag-jev')
except ImportError:
    DRIVE = Path(os.environ.get('SAJEV_WORKDIR', '../outputs/colab_local')).resolve()
    PROJECT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DRIVE.mkdir(parents=True, exist_ok=True)
REPO_URL = os.environ.get('SAJEV_REPO_URL', 'https://github.com/BoldHu/SA-Jev.git')  # Override with SAJEV_REPO_URL if needed.
SYNC_GIT = True
if not PROJECT.exists():
    if REPO_URL:
        subprocess.run(['git', 'clone', REPO_URL, str(PROJECT)], check=True)
    elif (DRIVE / 'sa-jev-source.zip').exists():
        PROJECT.mkdir(parents=True)
        with zipfile.ZipFile(DRIVE / 'sa-jev-source.zip') as archive:
            archive.extractall(PROJECT)
    else:
        raise RuntimeError('Set SAJEV_REPO_URL or copy sa-jev-source.zip to MyDrive/SA-Jev/')
if SYNC_GIT and (PROJECT / '.git').exists():
    status = subprocess.check_output(['git', '-C', str(PROJECT), 'status', '--porcelain'], text=True)
    if status.strip():
        print('Local edits present; preserving them. Git synchronization skipped.')
    else:
        subprocess.run(['git', '-C', str(PROJECT), 'pull', '--ff-only'], check=True)
os.chdir(PROJECT)
MODE = 'pilot'  # Change to 'full' only after the pilot completes and memory is measured.
ROOT = DRIVE / MODE
ROOT.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(DRIVE / 'huggingface')
print('Code:', PROJECT, 'Artifacts:', ROOT)
'''

INSTALL = '''# Run this in each fresh Colab runtime. No model weights are downloaded here.
subprocess.run([sys.executable, 'scripts/bootstrap_upstream.py'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.[train,retrieval,test]'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', 'vendor/OpenJev'], check=True)
if 'google.colab' in sys.modules:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'fla-core==0.5.2'], check=True)
# Activate editable-install .pth files in this already-running notebook kernel.
import importlib, site
site_dirs = site.getsitepackages()
if site.ENABLE_USER_SITE:
    site_dirs.append(site.getusersitepackages())
for site_dir in dict.fromkeys(site_dirs):
    site.addsitedir(site_dir)
importlib.invalidate_caches()
import sa_jev
print('SA-Jev source:', sa_jev.__file__)
# If package installation asks for a runtime restart, restart before importing torch.
from sa_jev.experiments import load_config, environment_manifest
config = load_config()
environment_manifest(ROOT / 'environment.json', config)
def cli(command, *args):
    from collections import deque
    tail = deque(maxlen=40)
    cmd = [sys.executable, '-u', '-m', 'sa_jev.cli', command, '--root', str(ROOT), *map(str,args)]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                tail.append(line)
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if returncode:
        raise RuntimeError(f'sajev {command} failed (exit {returncode}). Last output: ' + ''.join(tail))
'''

ENVIRONMENT = '''# Read the same credential file in every notebook. Fill it in once.
from sa_jev.environment import load_environment
ENV_FILE = (DRIVE if 'google.colab' in sys.modules else PROJECT) / '.env'
credential_status = load_environment(ENV_FILE, template=PROJECT / '.env.example')
print('Credential file:', ENV_FILE)
for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'TYPESAFE_API_KEY', 'HF_TOKEN'):
    print(name + ':', 'loaded' if credential_status[name] else 'empty')
if not credential_status['OPENAI_API_KEY']:
    print('Fill OPENAI_API_KEY in the file above, save it, then rerun this cell before building the index or answering questions.')
'''


def create(name, title, content):
    notebook = nbf.v4.new_notebook()
    boot = BOOT
    if name == '00_environment_data.ipynb':
        boot = BOOT.replace("MODE = 'pilot'  # Change to 'full' only after the pilot completes and memory is measured.",
                            "MODE = 'pilot'  # Use 'full' after the pilot; separate Drive directories.\n"
                            "AUTO_DISCONNECT = True  # Release Colab after successful preparation and Drive flush.\n"
                            "from uuid import uuid4\nPREPARATION_RUN_ID = uuid4().hex")
    notebook.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(boot),
                      nbf.v4.new_code_cell(INSTALL), nbf.v4.new_code_cell(ENVIRONMENT)]
    for kind, text in content:
        notebook.cells.append(nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text))
    notebook.metadata = {"kernelspec": {"display_name":"Python 3", "language":"python", "name":"python3"},
                         "language_info":{"name":"python"}, "colab":{"name":name}, "accelerator":"GPU"}
    if name == '00_environment_data.ipynb':
        notebook.metadata.pop('accelerator', None)  # CPU is sufficient for data preparation.
    target=ROOT/'notebooks'/name; target.parent.mkdir(exist_ok=True)
    nbf.write(notebook,target)


create('00_environment_data.ipynb', '# SA-Jev · 环境与数据准备\n先运行 MODE=pilot，产物保存到 MyDrive/SA-Jev/pilot；改为 full 后另存 full 目录。CPU 可运行，不训练 OpenJev-4B。pilot 每个源文件 64 条，不是完整论文数据，也不下载 2Wiki。流程成功生成本次 00_complete.json 后，默认刷新 Drive 写入并释放 Colab 运行时。首个代码单元的 AUTO_DISCONNECT=False 可关闭自动断开。', [
('code', "import torch\nfrom sa_jev.smoke import run_fresh\n# Execute in the notebook so any failure shows its original traceback.\nsmoke_result = run_fresh(ROOT, device='cuda' if torch.cuda.is_available() else 'cpu')\nsmoke_result"),
('md', '本地向量编码器：首次运行下载固定版本的 BAAI/bge-small-en-v1.5（384 维），缓存到 Drive/huggingface；之后复用。加载前校验官方权重大小和 SHA-256；损坏时隔离该缓存文件，并自动重下同一版本一次，校验仍失败则停止。不会删除原始数据或图谱。此检查不调用聊天/向量 API，自动使用 GPU，无 GPU 时用 CPU。文本抽取与最终回答继续使用 LLM API。'),
('code', "from sa_jev.embeddings import check_embeddings\nembedding_check = await check_embeddings(config['retrieval'], ROOT/'embedding_check.json')\nembedding_check"),
('md', '下载原始数据：pilot 每个文件 64 条；full 读取完整源。2Wiki 的官方 dev.json 另置于 ROOT/raw/2wiki_dev.json。MuSiQue 两种官方格式均受支持。'),
('code', "from time import perf_counter\nfrom sa_jev.preparation import download_sources, download_2wiki, normalize_sources\nprepare_started = perf_counter()\nprint(f'[1/3] Download HotpotQA and MuSiQue: MODE={MODE}, directory={ROOT / \"raw\"}', flush=True)\ndownload_sources(ROOT/'raw', limit=64 if MODE=='pilot' else None)\nif MODE=='full':\n    print('[2/3] Download and extract official 2Wiki data...', flush=True)\n    download_2wiki(ROOT/'raw')\nelse:\n    print('[2/3] Pilot: skip 2Wiki.', flush=True)\nprint('[3/3] Reuse completed normalization, or prepare changed/missing data...', flush=True)\n# Subsequent cells load their own data; avoid loading all cached questions here.\nquestions, manifest = normalize_sources(ROOT/'raw', ROOT/'data', load_questions=False)\nprint(f'Download/normalization complete in {perf_counter()-prepare_started:.1f}s. Graph indexing is the next cell.', flush=True)\nmanifest"),
('md', '固定 LightRAG 混合召回。原始段落统一入库，不索引答案或标签。向量使用本地 BGE 编码器，文本抽取调用 LLM API。先检查 configs/research.yaml。旧 API 向量索引不能复用；切换建图模型或编码器后使用新的实验 ROOT，重建索引和候选池。首次建图通常比训练 pilot 更耗时。'),
('code', "from sa_jev.experiments import cache_pools\nawait cache_pools(ROOT, config)"),
('code', "from sa_jev.preparation import prepare_tasks\nprepare_tasks(ROOT/'data', counts=config['data']['train_counts'], mine=True, dense=config['data']['dense_mining'])"),
('md', '训练前完整扫描公开 OpenJevData。检出的重叠保留记录并在论文报告；未检出不代表证明完全未见。此步骤不使用独立扰动测试集调参。'),
('code', "cli('overlap')\ncli('review-packets')"),
('code', "from sa_jev.preparation import finish_preparation\ncompletion = finish_preparation(ROOT, config, mode=MODE, run_id=PREPARATION_RUN_ID)\ncompletion"),
('md', '核对实际 K 分布和有效标签数量。缺候选不复制补齐，未知相关性不硬判负例。若排序有效状态太少，核查候选注释映射/补充核验，不能把掩码简单删掉。把独立扰动草稿人工核验后保存为 perturbations_reviewed.jsonl。'),
('md', '结束后自动断开：只有本次运行已成功完成才执行。先刷新并卸载 Drive，再释放 Colab 运行时。Drive 刷新失败或流程失败时不主动释放；重新连接需重装环境并挂载 Drive，已保存数据保留。本地运行会跳过此步骤。'),
('code', "from sa_jev.environment import disconnect_after_preparation\ndisconnect_after_preparation(ROOT, run_id=PREPARATION_RUN_ID, enabled=AUTO_DISCONNECT)")])

create('01_original_diagnostics.ipynb', '# 原模型诊断\n先测重复调用噪声，再测无关上下文与大 K；不预设原模型不稳定。', [
('code', "from sa_jev.experiments import diagnostics_experiment, efficiency_experiment\ndiagnostics_experiment(ROOT, config, limit=8 if MODE=='pilot' else 100)\nefficiency_experiment(ROOT, config, limit=8 if MODE=='pilot' else 100)"),
('code', "if os.environ.get('TYPESAFE_API_KEY') or os.environ.get('JEV_API_KEY'):\n    diagnostics_experiment(ROOT, config, method='dynamic_jev', limit=8 if MODE=='pilot' else 100)\n    efficiency_experiment(ROOT, config, method='dynamic_jev', limit=8 if MODE=='pilot' else 100)\nelse:\n    print('Official Jev API unavailable; no official-Jev result is reported.')"),
('md', '无关状态干扰使用独立且经过核验的测试包，保持不调参。未核验草稿不会被自动当成测试结果。'),
('code', "from sa_jev.experiments import robustness_experiment\nif (ROOT/'data/perturbations_reviewed.jsonl').exists():\n    robustness_experiment(ROOT, config, tag='original_perturbations', methods=['openjev_frozen'] + (['dynamic_jev'] if os.environ.get('TYPESAFE_API_KEY') else []))\nelse:\n    print('Awaiting reviewed independent perturbation packets; repeat/K diagnostics above are separate measurements.')")])

create('02_stage1.ipynb', '# Stage 1 · 任务学习\n复用已训练 OpenJev-4B，BF16 骨干 + LoRA，FP32 原决策头与交互层。先实测单卡 pilot，再开始正式训练。', [
('code', "import torch\nassert torch.cuda.is_available(), 'Select a GPU runtime'\nprint(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3, 'GiB')\nsubprocess.run([sys.executable, 'scripts/check_qwen_lora.py', '--device', 'cuda', '--output', str(ROOT/'qwen_lora_check.json')], check=True)"),
('code', "from sa_jev.experiments import train_variant\nseeds = [17] if MODE=='pilot' else config['seeds']\nfor seed in seeds:\n    train_variant(ROOT, config, 'stage1', seed, max_steps=2 if MODE=='pilot' else None, validation_limit=6 if MODE=='pilot' else None)"),
('md', '检查 training.csv/PNG/PDF、验证 loss、实际显存和 ETA。4GB 本地 tiny 测试不能代替 4B A100 测试；OOM 时降低 branch_microbatch，不切分 softmax，不截断证据，不改变标签。恢复训练使用 train_variant(..., resume=checkpoint_path)。')])

create('03_stage2.ipynb', '# Stage 2 · 整轨迹 DPO + 等义双视图\nL_task(two views) + 0.2 DPO + 0.1 stability；β=0.1，容忍区间 0.05。预算耗尽不算自主 STOP。', [
('code', "from sa_jev.experiments import train_variant\nfor seed in ([17] if MODE=='pilot' else config['seeds']):\n    train_variant(ROOT, config, 'full', seed, max_steps=2 if MODE=='pilot' else None, validation_limit=6 if MODE=='pilot' else None)"),
('md', '参考缓存与冻结 Stage 1 权重、完整轨迹和输入视图绑定。双视图仅改变呈现，不使用 8,000 对人工无关样本；真实相邻证据状态继续分别计算任务损失。')])

create('04_main_comparison.ipynb', '# 主性能 · 共享召回池与回答模型\nHotpotQA / MuSiQue / 2Wiki 跨数据集；保存逐题原始轨迹和配对统计。', [
('code', "from sa_jev.experiments import main_experiment\nrows = main_experiment(ROOT, config, seeds=[17] if MODE=='pilot' else config['seeds'], limit=8 if MODE=='pilot' else None)"),
('md', '候选缓存相同、预算相同、回答器相同。原始召回排序使用冻结 OpenJev 的独立充分性头。Jev API 缺失会写 unavailable.json，不伪造可比较结果。未知标签的指标留空并报告有效样本数。')])

create('05_robustness_response.ipynb', '# 独立干扰与有效变化\n约 600 保留状态 × 3 个核验无关扰动 + 1 个有效变化；禁止用于训练或调参。', [
('code', "from sa_jev.experiments import robustness_experiment\nrows = robustness_experiment(ROOT, config)"),
('md', '同时检查 JS、Top-1/Top-5、充分性翻转、错误提前停止、nDCG 和有效变化后正确补证。新增已选证据会改变剩余候选集合，不强行比较为不变分布。')])

create('06_ablations.ipynb', '# 消融\nStage 1；DPO；稳定性；完整模型；无状态更新；小 K；少量稳定性权重。', [
('code', "from sa_jev.experiments import train_variant, main_experiment\nfrom sa_jev.preparation import prepare_tasks\nprepare_tasks(ROOT/'data', counts=config['data']['train_counts'], mine=True, small_k=True)\nfor seed in ([17] if MODE=='pilot' else config['seeds']):\n    for variant in ['dpo','stability','small_stage1','small_k']:\n        train_variant(ROOT, config, variant, seed, max_steps=2 if MODE=='pilot' else None, validation_limit=6 if MODE=='pilot' else None)\nmain_experiment(ROOT, config, table_name='ablations', methods=['openjev_frozen','stage1','dpo','stability','full','no_state_update','small_k'], seeds=[17] if MODE=='pilot' else config['seeds'], limit=8 if MODE=='pilot' else None)"),
('md', '权重选择只用 validation task loss/验证指标，不查看独立扰动测试来选权重。以下小扫掠先保留单种子。'),
('code', "for weight in [0.05,0.2]:\n    train_variant(ROOT, config, f'stab_weight_{weight}', 17, max_steps=2 if MODE=='pilot' else None, validation_limit=6 if MODE=='pilot' else None)")])

create('07_efficiency_k.ipynb', '# K 扩展与效率\nK=16/32/64/128/256；K>64 明确标记为外推。报告实际 K、token、耗时与显存。', [
('code', "from sa_jev.experiments import efficiency_experiment\nfor method in ['openjev_frozen','stage1','full']:\n    efficiency_experiment(ROOT, config, method=method, limit=8 if MODE=='pilot' else 100)"),
('md', '所有 K 使用同一次最大候选池缓存的前缀。实际候选不足时 exact_k=False，不能当作完整 K 测量；token 上限/OOM 记录失败，不复制候选或静默截断。')])

create('08_industrial_case_study.ipynb', '# ViDoRe V3 Industrial · 独立案例\n只使用官方 OCR 文本与页面来源，不是端到端视觉理解；页级相关性不能自动转成多页充分性。', [
('code', "from sa_jev.industrial import download_ocr\npages, queries, qrels = download_ocr(ROOT/'industrial')"),
('code', "from sa_jev.industrial import page_pool, evaluate_industrial\nfrom sa_jev.retrieval import make_lightrag, ingest_corpus, retrieve_once, PoolCache\nfrom sa_jev.schema import Evidence\nfrom sa_jev.experiments import load_method\nfrom sa_jev.reporting import export_report\nfrom sa_jev.io import write_jsonl\nretrieval_args = {k: v for k, v in config['retrieval'].items() if k not in {'mode', 'max_k'}}\nrag = await make_lightrag(ROOT/'industrial/index', **retrieval_args)\ntry:\n    docs = [Evidence(p['id'],p['title'],p['text'],p['source']) for p in pages if p['text'].strip()]\n    binding = await ingest_corpus(rag, docs, ROOT/'industrial/index/manifest.json', config['retrieval'])\n    selected_queries = queries[:8] if MODE=='pilot' else queries\n    cache = PoolCache(ROOT/'industrial/chunk_pools')\n    pools = {}\n    for q in selected_queries:\n        qid = 'vidore:'+str(q['query_id'])\n        chunks = await retrieve_once(rag, qid, q['query'], 64, cache, corpus_fingerprint=binding, source_registry={e.id:e for e in docs})\n        pools[qid] = page_pool(qid, chunks, pages, 32)\nfinally:\n    await rag.finalize_storages()\nmodel, _ = load_method(ROOT, config, 'full', 17)\nrows = evaluate_industrial(model, selected_queries, qrels, pools)\nwrite_jsonl(ROOT/'results/industrial/raw.jsonl', rows)\nexport_report(rows, ROOT/'results/industrial', ('page_ndcg5','page_recall5','seconds'))"),
('code', "# Export sourced successes AND failures; never fabricate a case if one category is absent.\nfor success in (True, False):\n    subset = [r for r in rows if r['retrieval_success'] is success]\n    print('success' if success else 'failure', subset[:2])")])

print('Created', len(list((ROOT/'notebooks').glob('*.ipynb'))), 'Colab notebooks')
