# SA-Jev

按照根目录 `AGENTS.md` 实现的研究框架：固定 GraphRAG 初始候选池，复用训练好的 OpenJev-4B，进行动态边际证据选择、独立充分性判断、两阶段微调与稳定性/有效变化联合评测。尚未运行 4B 正式训练，也没有论文性能结论。

## 入口与首次运行

主入口是 `notebooks/`，代码复用 `src/sa_jev/`。每个 notebook 都挂载 Drive、支持 Git 快进同步、初始化相同依赖；模型、候选缓存、原始结果、CSV 与 PNG/PDF 保存到 `MyDrive/SA-Jev/{pilot,full}`。

1. 每个 notebook 第一个代码单元的 `REPO_URL` 默认指向 `https://github.com/BoldHu/SA-Jev.git`，运行时自动克隆并同步代码，可通过环境变量 `SAJEV_REPO_URL` 覆盖。如需使用 ZIP，将 `REPO_URL` 改为 `''`，并将 `artifacts/sa-jev-source.zip` 放到 `MyDrive/SA-Jev/`。
2. Colab 打开 `00_environment_data.ipynb`。保持 `MODE='pilot'`，先检查 tiny 流程，再下载小样本、构建固定 LightRAG 缓存、生成状态/轨迹、扫描 OpenJevData。配置 Colab Secrets：抽取/嵌入/回答服务使用 `OPENAI_API_KEY`，可选 `OPENAI_BASE_URL`；官方 Jev 使用 `TYPESAFE_API_KEY`；可选 `HF_TOKEN`。密钥不会进入结果文件。
3. 运行 `01_original_diagnostics.ipynb`，先看重复调用与 K 扩展。独立无关扰动部分需要完成核验的测试包，不会用生成草稿冒充结果。
4. 在单卡 A100 上依次运行 `02_stage1.ipynb`、`03_stage2.ipynb`。pilot 默认各两个 optimizer step。检查 loss、梯度、实测显存、checkpoint 重载和 ETA，再决定 full。
5. 运行 `04_main_comparison.ipynb`，确认相同池、预算、回答器以及逐题结果可比较。pilot 可以只取少量题。
6. 将各 notebook 的 MODE 改成 `full`，从 00 重新准备完整数据和独立缓存，再执行 01→02→03→04；随后执行 05（独立稳定性/有效变化）、06（消融）、07（效率）、08（工业案例）。正式主实验默认三个种子。

full 模式会下载官方 2Wiki April-2021 修订数据；若其托管链接不可用，将官方 `dev.json` 和 `id_aliases.json` 分别放到 `ROOT/raw/2wiki_dev.json` 与 `2wiki_id_aliases.json`。官方 dev 被用作跨数据集测试，不参与训练。数据下载与建图可能耗时，应先估计 pilot 的实际吞吐。

旧版 notebook 若在安装后报 `ModuleNotFoundError: No module named 'sa_jev'`，是当前内核可能尚未读取 editable 安装生成的 `.pth` 路径。新版第二个代码单元会在安装后调用 `site.addsitedir()` 刷新路径，并打印 `SA-Jev source:` 以核对实际加载位置。更新 notebook 后重新运行安装单元即可；若 pip 提示已经加载的依赖需要重启，则重启会话后按顺序重跑开头的初始化单元。

## 本地 Conda

已在此机器创建 `SA-Jev`：Python 3.11，PyTorch 2.6.0+cu118（兼容本机旧 NVIDIA 驱动）。RTX 3050 仅用于小模型流程与接口验证。

```powershell
conda activate SA-Jev
python scripts/bootstrap_upstream.py
python -m pip install --no-deps -e vendor/OpenJev
python -m pip install -e ".[train,test]"
python -m pytest -q
python -m sa_jev.cli smoke --root outputs/smoke --device cuda
python scripts/check_qwen_lora.py --device cuda
python scripts/check_native_artifact.py
python scripts/check_lightrag_local.py
```

若 PowerShell 未初始化 Conda，可使用 `D:\anaconda\envs\SA-Jev\python.exe`。新环境可用 `conda env create -f environment.yml`，再按驱动安装对应 PyTorch CUDA wheel。Colab 的 CUDA/FLA 路径必须单独验证；不要把本地 cu118 测试当成 FLA 测试。

## 实现对应关系

| 研究内容 | 实现 |
|---|---|
| 原生 OpenJev、LoRA、全候选分布、checkpoint | `model.py` |
| 数据适配、问题族划分、三类标签、未知掩码 | `data.py`、`schema.py`、`preparation.py` |
| 原生 LightRAG mix/hybrid 与固定候选缓存 | `retrieval.py` |
| 白名单模型输入、在线等义视图 | `prompts.py` |
| 任务损失、容忍区间稳定性、DPO | `losses.py`、`training.py` |
| READ/STOP 轨迹和保守偏好构造 | `trajectories.py` |
| 动态/静态/召回/过滤基线、QA、成本 | `evaluation.py`、`experiments.py` |
| 核验独立扰动与有效变化 | `robustness.py` |
| ViDoRe 官方 OCR、页面来源与页级指标 | `industrial.py` |
| 配对统计、CSV、PNG/PDF | `metrics.py`、`reporting.py` |

Stage 1：`L_rank + L_suf + 0.2 L_rel`。Stage 2：两视图任务损失的**平均值** + `0.2 DPO + 0.1 stability`。DPO 使用完整轨迹概率，β=0.1；两个稳定性容忍阈值均为 0.05。有效状态变化只按各自标签训练。各损失分母按有效样本分别计算，并跨梯度累积的有效 batch 归一化。

原始 OpenJev 第二阶段是 REINFORCE-Analysis；这里的轨迹 DPO 是研究扩展。原生头与 Option Set Interactor 从发布 artifact 一起加载，未新建替代决策头。编码分支可分 microbatch，但全部候选特征汇合后只通过一次原生 interactor 与 softmax，梯度不断开。

## 研究边界

- 训练候选允许支持事实监督构池；正式验证/测试只接受原生 LightRAG 召回缓存。召回完成后不重新检索。默认语料是原始数据段落的去重并集，**不是完整 Wikipedia 开放域检索**，论文必须说明。
- 外部候选若没有可确认支持覆盖/负例注释，则保留未知。它不会成为软目标的隐式负例；该状态排序监督被屏蔽。`task_manifest.json` 报告有效数量和实际 K 分布。低有效覆盖需要补充可核验注释，不能删掩码凑数据量。相关重复证据可 `rel=1, gain=0`。
- 充分性只看问题、条件、已选证据和状态信息，不看剩余候选正文。独立 Noul 是支持事实覆盖的监督代理，不是天然具有真实语义完备性的 oracle。
- 训练的等义视图只改变 JSON/字段呈现。独立约 600 个状态的困难自然语言扰动测试需要人工核验，测试文件不会进入训练或调参。
- 无状态更新消融只屏蔽 Choice 的已选内容；剩余候选与独立充分性仍使用真实状态。静态基线只重排一次，以后按剩余候选重新归一化原分数。状态过滤基线去除普通 context，保留有效条件和已选证据。
- OpenJev 原接口上限 255，bootstrap 只将两个选项数量校验扩到 256。K>64 为外推；效率实验显式把逻辑总 token 预算放宽到 262144，单分支仍为 8192。超限/OOM 记录失败，不改 K、不截断或复制证据。
- ViDoRe 输入仅为官方 `markdown` OCR；页级 qrels 用于检索指标，不自动构造多页充分性标签。返回页面 URL 和 OCR 摘录供核查成功/失败案例。
- 缺失 Jev API、人工核验、回答器调用或模型运行时，相关结果保持缺失；不会补入预测数值。已实际验证和仍待运行项见 [验证记录](docs/VALIDATION.md)。

## 上游与来源

- [OpenJev 源码](https://github.com/ejhshen/OpenJev)，固定 commit `e279c44829370cbdc5c48784a912f38ffd73df60`；其 [训练说明](https://github.com/ejhshen/OpenJev/blob/e279c44829370cbdc5c48784a912f38ffd73df60/docs/training.md) 记录上游测试环境。
- [OpenJev-4B](https://huggingface.co/shenjunhao/OpenJev-4B)，固定 revision `064bd435c6713486450194e29d526266b28941e2`。
- [LightRAG](https://github.com/HKUDS/LightRAG)、[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)、[MuSiQue](https://github.com/StonyBrookNLP/musique)、[2Wiki](https://github.com/Alab-NII/2wikimultihop)。
- [官方 Jev Choice](https://docs.typesafe.ai/primitives/choice)、[Noul](https://docs.typesafe.ai/primitives/noul)、[System One API](https://docs.typesafe.ai/api/system-one)。
- [ViDoRe V3 Industrial](https://huggingface.co/datasets/vidore/vidore_v3_industrial)，按官方卡片端到端评测 revision 固定 `233d20721f4deb392a09a86cca01761adbc91157`。
