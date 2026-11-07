# SA-Jev

按照根目录 `AGENTS.md` 实现的研究框架：固定 GraphRAG 初始候选池，复用训练好的 OpenJev-4B，进行动态边际证据选择、独立充分性判断、两阶段微调与稳定性/有效变化联合评测。尚未运行 4B 正式训练，也没有论文性能结论。

## 入口与首次运行

主入口是 `notebooks/`，代码复用 `src/sa_jev/`。每个 notebook 都挂载 Drive、支持 Git 快进同步、初始化相同依赖；模型、候选缓存、原始结果、CSV 与 PNG/PDF 保存到 `MyDrive/SA-Jev/{pilot,full}`。

1. 每个 notebook 第一个代码单元的 `REPO_URL` 默认指向 `https://github.com/BoldHu/SA-Jev.git`，运行时自动克隆并同步代码，可通过环境变量 `SAJEV_REPO_URL` 覆盖。如需使用 ZIP，将 `REPO_URL` 改为 `''`，并将 `artifacts/sa-jev-source.zip` 放到 `MyDrive/SA-Jev/`。
2. 填写项目根目录 `.env`（没有时从 `.env.example` 复制），上传到 `MyDrive/SA-Jev/.env`。所有 notebook 的第三个代码单元自动读取同一份文件；无需配置 Colab Secrets。`00_environment_data.ipynb` 默认恢复为 `MODE='pilot'`：先检查 tiny 流程和本地编码器，再下载小规模数据、构建固定 LightRAG 缓存、生成状态/轨迹、扫描 OpenJevData。结果保存在 `MyDrive/SA-Jev/pilot`，不下载或训练 OpenJev-4B 主模型。首个代码单元的 `AUTO_DISCONNECT=True` 会在准备成功且 Drive 写入刷新后释放 Colab 运行时。
3. 运行 `01_original_diagnostics.ipynb`，先看重复调用与 K 扩展。独立无关扰动部分需要完成核验的测试包，不会用生成草稿冒充结果。
4. 在单卡 A100 上依次运行 `02_stage1.ipynb`、`03_stage2.ipynb`。保持 `MODE='pilot'` 和 `ROOT = DRIVE / MODE`，直接读取 00 准备的 pilot 数据；训练 pilot 默认各两个 optimizer step。检查 loss、梯度、实测显存、checkpoint 重载和 ETA，再决定全量训练。确认实际分配显存，主实验上限是 A100 80GB。
5. 运行 `04_main_comparison.ipynb`，确认相同池、预算、回答器以及逐题结果可比较。pilot 可以只取少量题。
6. 模型 pilot 通过后，先将 **00 首个代码单元**的 `MODE` 改为 `'full'`，完整运行一次，另存 `MyDrive/SA-Jev/full`。pilot 数据不能代替这次全量准备。然后将实验 notebook 的 `MODE` 改成 `'full'`，执行 01→02→03→04；随后执行 05（独立稳定性/有效变化）、06（消融）、07（效率）、08（工业案例）。正式主实验默认三个种子。已完成的 full 主实验数据无需重复准备。若此前已完成 full 数据却尚未测试模型，可先保持训练 notebook 的 `MODE='pilot'`，仅把 `ROOT` 改成 `DRIVE / 'full'` 做少步验证，之后再恢复 `ROOT = DRIVE / MODE`。

full 模式会下载官方 2Wiki April-2021 修订数据；若其托管链接不可用，将官方 `dev.json` 和 `id_aliases.json` 分别放到 `ROOT/raw/2wiki_dev.json` 与 `2wiki_id_aliases.json`。官方 dev 被用作跨数据集测试，不参与训练。数据下载与建图可能耗时，应先估计 pilot 的实际吞吐。

00 的原始数据单元分三段打印进度：HotpotQA/MuSiQue 下载、2Wiki 下载解压、问题规范化与证据去重。流式下载显示记录数和速度；2Wiki 显示 MiB、速度及有总大小时的 ETA。源记录总量未知时明确显示 `ETA=unknown`；网络等待时最后一条阶段消息用于定位，不能把没有新计数当作已经完成。MuSiQue 改为边读取边写临时文件，完整成功后才发布正式文件。full 在此读取完整来源，8,000/4,000 的训练抽样发生在后续 `prepare_tasks`，此单元尚不调用 DeepSeek 建图。旧版只有末尾 `manifest` 输出，执行期间可能一直空白；更新代码后重载模块或重新连接才会使用新日志，已经执行中的调用不会自动更新。

重复运行时，`Reusing` 表示复用原始文件；旧版的 `[3/3] Normalize...` 则每次都会重新解析、划分、去重、计算内容指纹并写文件，并不等于重复下载。新版下载清单与文件齐备时不再请求来源列表，2Wiki 解压文件与来源记录齐备时也直接复用。`normalize_sources` 默认复用已完成的加工结果：检查来源及产物的大小/修改时间、种子、问题族关联、下载协议与适配代码；匹配时不重做加工。00 使用 `load_questions=False`，连缓存问题列表也不重新载入内存，后续各阶段自行读取所需数据。大小/时间戳是快速缓存检查，并不代替内容哈希审计；如果外部修改刻意保留这些属性，调用 `normalize_sources(..., reuse=False)`。旧 manifest 没有复用标记，自动路径首次需要重建一次；若刚才已经成功完成加工且原始数据、MODE/ROOT 都未变化，只为修复后续编码器/建图错误时，可以直接跳过该单元继续建图。

旧版 notebook 若在安装后报 `ModuleNotFoundError: No module named 'sa_jev'`，是当前内核可能尚未读取 editable 安装生成的 `.pth` 路径。新版第二个代码单元会在安装后调用 `site.addsitedir()` 刷新路径，并打印 `SA-Jev source:` 以核对实际加载位置。更新 notebook 后重新运行安装单元即可；若 pip 提示已经加载的依赖需要重启，则重启会话后按顺序重跑开头的初始化单元。

`00` 的 smoke 现在直接在 notebook 中执行，失败时显示原始 Python 异常。每次执行都在 `ROOT/smoke/run_*` 创建独立目录，可重复运行并保留旧结果；本地 `sajev smoke` 采用相同方式。其他 notebook CLI 命令会实时显示子进程输出，并在失败异常中附上最后 40 行，避免只看到没有内部原因的 `CalledProcessError`。smoke 通过后再继续数据准备；它使用 tiny 测试模型，不能代替 OpenJev-4B 的显存与性能验证。

所有密钥集中在一个文件中，填写等号右侧，不用的项目留空：

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://www.autodl.art/api/v1
TYPESAFE_API_KEY=
HF_TOKEN=
```

`OPENAI_API_KEY` 填服务商签发的密钥，用于建图时的文本抽取和最终回答；默认向量编码器在本地运行，不使用 API 密钥。`TYPESAFE_API_KEY` 用于官方 Jev 基线；`HF_TOKEN` 用于需要认证的 Hugging Face 下载。最终选择的服务地址为 `https://www.autodl.art/api/v1`，保留 `/api/v1`，不要追加 `/chat/completions`。更换聊天服务商时修改此地址，并在 `configs/research.yaml` 匹配聊天模型名称；若改用默认 OpenAI 服务，可将 `OPENAI_BASE_URL` 留空。更新本地文件后也要更新 Drive 中的 `.env`，再运行第三个单元，已有的 Drive 文件不会被模板覆盖。

`configs/research.yaml` 中的建图模型 `retrieval.llm_model` 和最终回答模型 `evaluation.answer_model` 最终统一使用 `DeepSeek-V4.1-Flash`；ViDoRe 建图也读取相同配置。证据排序/充分性训练仍使用 `shenjunhao/OpenJev-4B`。2026-11-07 最后一次聊天检查返回 200、正文 `OK`，耗时 3.687 秒，服务返回模型名 `DeepSeek-Flash`；本地 BGE + 该模型的两文档建图、混合检索和缓存检查也通过。实际请求选项会记入回答缓存；训练随机种子仍按配置执行，不把聊天接口响应视为严格确定性的结果。切换建图模型后应重建索引与固定候选池，不能混用旧模型的缓存。

向量已改为本地运行的 [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)，固定 revision，384 维、L2 归一化；查询按官方用法加检索前缀，文档不加。`configs/research.yaml` 的 `retrieval.embedding_backend: local` 是默认值，不再请求 `/embeddings`。`embedding_device: auto` 有 GPU 时使用 GPU，否则使用 CPU；batch size 默认 16，显存不足可调小。模型冻结，仅用于一次建库和固定候选池检索，不参与 SA-Jev 训练。

`00_environment_data.ipynb` 在 smoke 后新增“本地向量编码器”检查单元：自动下载、检查向量形状/归一化、打印速度和 GPU 显存，记录到 `ROOT/embedding_check.json`。Colab 权重缓存位于 `MyDrive/SA-Jev/huggingface/hub`，以后复用；已有运行时更新代码和安装单元后需重启，再按顺序初始化。主英文任务可使用该编码器；工业案例若包含非英文 OCR，应单独配置合适的多语言模型并注明检索设置，不把英文编码器结果当作多语言能力。编码器输入上限为 512 个自身 tokenizer 的 token，长输入按模型规则截断，完整候选正文保留，需在检索覆盖率中评估影响。

更改编码器后旧向量索引及候选缓存必须重建，程序会拒绝混用。可在 Drive 中把旧的 `pilot/index`、`pilot/data/pools`、`pilot/data/pools_max` 重命名备份后，从 `00` 的建图单元继续，再运行后面的任务状态准备单元；原始数据无需重新下载。所有方法共享这次重建后的固定候选池，已用旧池得到的评测结果不能混入新对比。

如果出现 `SafetensorError: ... header too large`，先处理编码器权重缓存，不能据此判定 CPU 不支持或显存不足。当前固定 BGE 的官方 `model.safetensors` 是 **133,466,304 bytes**，SHA-256 为 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`。加载前会检查这两项；不匹配时仅将该 snapshot 权重条目改名为 `.invalid-*`，强制下载同一 revision 一次并重新校验，避免反复下载仍继续运行。这样也处理了 Drive/无符号链接缓存中旧 snapshot 文件不随 blob 强制下载而替换的情况。修复的是相同权重文件，不改变编码器与索引配置，因此原始数据、已加工状态和原有图谱不需要因此删除或重建。重试仍失败时停止并报告；不能保证修复外部 Drive 或下载链路的故障。

本地验证入口：`python scripts/check_local_embeddings.py` 使用真实编码器和测试 LLM；加 `--live-llm` 则读取 `.env`，用真实聊天 API 对两篇合成文档跑原生 LightRAG 建图、混合检索和缓存检查。每次创建独立测试目录，不修改正式索引。

Colab 固定读取 `/content/drive/MyDrive/SA-Jev/.env`，本地 notebook 和 CLI 读取项目根目录 `.env`。首次运行第三个单元时，若文件不存在会创建空白模板并打印位置。填写或修改后重新运行第三个单元，再继续实验；该文件中的设置会覆盖会话旧值，留空会清除对应旧值。第三个单元只显示 `loaded`/`empty`，不打印密钥；`loaded` 只表示变量已读取，不代表服务权限或额度已验证。实际 `.env` 已被 Git 忽略，源码 ZIP 只包含空白 `.env.example` 模板。

## 00 保存、恢复与自动断开

当前先跑 pilot：在 Colab 的“运行时 → 更改运行时类型”选择 **Python 3、硬件加速器 None（CPU）、标准 RAM**。00 不加载 4B 主模型；本地 BGE 支持 CPU，图谱文本抽取在 API 服务端执行。T4 可加速本地向量编码，但不能直接加速服务端抽取，不必为 00 占用 A100。全量语料及图谱会使用更多系统内存，有条件时选高 RAM；00 尚未完成全量实测，不能保证标准 RAM 足够。模型训练 pilot 在 02/03 中再切换 A100，并核对实际显存。

睡前先完成 Drive 授权、依赖安装/必要重启、tiny smoke、本地向量检查，再顺序执行其余单元。准备期间不要改变聊天模型、编码器、语料范围或分块配置。API 服务余额/限额和 Drive 可用空间需要能覆盖本次运行。2Wiki 官方归档已做小范围下载检查（HTTP 200、ZIP 头正确，约 259 MB）；完整全量数据和 Colab 长时间运行尚未在本机验证。

| Drive 位置（均在 `MyDrive/SA-Jev` 下） | 内容 |
|---|---|
| `<MODE>/raw/` | 下载的原始数据与来源记录；2Wiki 仅 full 模式下载 |
| `<MODE>/data/` | 去重语料、问题划分、固定候选池、任务状态、轨迹偏好、重叠审计、扰动核验草稿 |
| `<MODE>/index/` | 图谱、向量库、抽取响应缓存、文档处理状态与 `ingestion_progress.json` |
| `huggingface/` | 本地编码器与 Hugging Face 下载缓存 |
| `<MODE>/00_complete.json` | 00 全部准备步骤成功后才生成的完成标记；包含本次运行 ID、完成时间和状态数量 |

`<MODE>` 当前为 `pilot`，全量时为 `full`。成功结束后，最后一个代码单元先调用 `drive.flush_and_unmount()`，再调用 `runtime.unassign()`；旧运行的完成标记不会触发本次自动断开。Drive 刷新失败时不会主动释放运行时。需要保留会话查看结果时，将首个代码单元的 `AUTO_DISCONNECT = False`。这只是完成后的主动释放，不能阻止 Colab 提前回收会话。下一次连接仍需挂载、安装依赖及加载 `.env`；Drive 数据保留。

建图按已处理文档恢复；完成索引及已写入的问题候选池可以复用。失败文档不会再被当作“全部建库完成”。原始 JSON/JSONL 和准备产物先写临时文件，成功后替换目标，2Wiki 下载/解压亦如此。硬中断可能留下 `.tmp`/`.part` 文件或使正在执行的批次重做，不能承诺零重复或外部 Drive 永不损坏。重新连接时先运行每个 notebook 开头的挂载/安装/环境读取单元，再从失败阶段继续；中断的状态构造或重叠扫描目前会重跑该阶段。

若建图报 `LightRAG has incomplete documents`，这是未完成状态检查，不能单凭它认定 API 失败。新版异常会附上最多三条失败文档的原生 `error_msg`，并把当前批次的文档 ID、状态及错误类型/阶段保存到 `ROOT/index/ingestion_failure.json`，对当前环境的服务令牌和常见 Bearer/sk-令牌做脱敏。旧版需读取 `ROOT/index/kv_store_doc_status.json` 中失败文档的 `error_msg`（如使用 workspace，文件可能在 index 子目录）。保留原始数据和已完成索引；根据具体错误修复后再重跑建图，不跳过失败文档，不把未完成索引用于正式候选缓存。尚无具体错误记录时，不盲目改模型、强行标记 processed 或反复调用服务。

**失败重跑修正**：当前 LightRAG 的 `ainsert` 不会重试已有 FAILED 文档，重复提交还可能新增重复文件的失败记录。修正后的 `ingest_corpus` 发现已有失败文档时，使用与原生 `/documents/reprocess_failed` 相同的 manual-retry 提交机制，触发一次工作区失败重处理；仅对没有状态记录的新文档调用 `ainsert`。失败重试仍失败则报告具体原因并停止，不无限重试，不重新插入原 ID，也不直接改状态 JSON。历史重复失败记录如果没有正文，会被 LightRAG 保留；完成条件始终是本语料全部原始文档已 processed，而非历史状态文件中所有附带记录都成功。本地原生集成测试已复现旧行为，并验证正确重试、processed 状态、完成标记和来源可追溯的 mix 检索。

建图初始化现在会显示实际 API 地址、模型名和 `.env` 路径（不显示密钥），并提前阻止 DeepSeek 配 Google 官方 Gemini 地址这一已知错误组合。若实际地址是 AutoDL 且模型正确，而状态文件仍显示旧 `GenerateContentRequest.model` 错误，先用当前 Colab 密钥做一条小请求确认服务可用，再进行失败重处理；历史 `error_msg` 本身不能证明更正配置后已发起新请求。

以前用旧代码中断留下的数据不保证完整；以前更换过 Gemini/DeepSeek、编码器或分块设置的索引不能直接混用。若启动时提示旧索引配置不匹配，备份旧 `pilot` 目录后使用新目录重新准备，不必删除备份。当前代码的恢复保护不能修复旧版本已写坏的文件；原始文件解析失败时需要重新下载该文件。

最后应看到 `00 COMPLETE`，并检查 Drive 的 `pilot/00_complete.json`（全量则是 `full/00_complete.json`）为本次运行时间。仅有 `raw/`、索引目录或进度输出不表示全部结束。数据与检索设置保持一致时，后续主实验直接读取这些文件，无需再次跑 00。**00 不完成所有后续实验的数据工作**：05 仍需人工核验独立扰动及有效变化草稿；06 会从已有数据生成小 K 消融状态/轨迹；08 独立下载 ViDoRe OCR/页面数据并构建工业案例索引与候选池，产物保存在当前 `ROOT/industrial`。Stage 2 的等义视图在线生成，冻结参考模型的轨迹概率缓存也在训练阶段产生，无需重建主图谱。

pilot 每个源文件取 64 条，四个源文件合计最多 256 条原始记录，每题又有多个段落；实际问题/去重段落数会打印在 `data_manifest` 的 `questions`/`corpus_documents`。即使 pilot 也会完整扫描 OpenJevData，所以不要按“两文档联调 33 秒”推算整个流程几分钟结束。首次 00 pilot 建议预留数小时，并以首批 32 个文档之后的 `estimated remaining` 和后续候选查询 ETA 为准；建图 ETA 不含下载、候选查询和重叠审计。尚未实测完整 00 pilot 或 A100 4B pilot，无法承诺固定时长。02/03 pilot 各两步，但首次仍有 4B 下载、加载、验证和 DPO 参考计算开销；全量训练和消融也可能很耗时。

全量数据准备可能很久，不能保证一夜完成；[Colab 官方说明](https://research.google.com/colaboratory/faq.html)明确运行时有最大寿命和动态资源限制，Drive 中已落盘的文件不等于运行时会一直持续。

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
