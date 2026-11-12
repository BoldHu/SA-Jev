# 实际验证记录

日期：2026-11-07 至 2026-11-12（Asia/Shanghai）。所有数值来自本机实际运行，均为工程/方法契约测试，不是 SA-Jev-4B 论文结果。

## 已运行

| 检查 | 实际结果 | 可追溯输出 |
|---|---|---|
| A100 40GB 起步与训练进度 | 相关测试 **22 passed**（28.89s），含真实 tiny 两阶段反向传播/断点续训一致性、step/验证/保存进度输出、独立试跑目录不覆盖正式检查点、已有正式检查点时在分配模型前拒绝误重跑。9 个 notebook 格式与代码语法通过。02 默认复用 full 数据，20 step 试跑独立保存，显存阈值按实际容量减 2 GiB 并受配置上限约束；原始有效候选全集归一化、损失、梯度累积语义未改。01 保存日志显示 300 次本地重复诊断及 500 条效率轨迹均成功，推理峰值约 8.84 GiB；不能据此宣称真实 4B 训练可在 40GB 内完成。未实际运行 A100 40GB 训练；官方 API 全量输出未见完成，独立扰动仍缺核验文件，追加两题实验是否覆盖全量结果需只读行数审计 | `tests/test_training_progress.py`；`notebooks/02_stage1.ipynb`；`notebooks/01_original_diagnostics.ipynb` |
| 01 加载与重复诊断进度 | 相关测试 **11 passed**（5.71s）；新增测试确认中断后已完成诊断行保存、重复推理分布指标不变、异常即时输出且落盘、效率外推以实际训练 K 判断。真实本地 tiny Qwen3.5/OpenJev 原生加载、原 head 保留、LoRA 梯度与检查点重载通过；9 个 notebook 格式/代码语法通过。01 保留旧输出，新增两题检查设置并拆分两个实验；加载等待提示不是完成百分比。未实际运行 T4/A100 上的 4B 诊断，无法从用户显存占用证明远程进程正常或估算全量耗时 | `tests/test_diagnostic_progress.py`；`outputs/diagnostic_loader_bybo_smv/verification.json` |
| 训练未知候选与排序覆盖修复 | 完整套件 **89 passed**（103.23s），9 个 notebook 格式与代码语法通过。新增测试覆盖未知项剔除但掩码不取消、保留已标注干扰段、完整候选归一化与排序反向梯度、修复备份、验证状态/问题/缓存不改动、问题抽样不一致时拒绝发布、偏好轨迹合法重放。真实本地 HotpotQA/MuSiQue 各 4 题共 24 状态：排序有效 **6 → 16**，剩余 8 状态已充分；实际 K 为 8/10/16/20。无 API、无 BM25、无 4B 训练。云盘连接仍为另一账号，未直接读取或修改用户云端全量文件；正式验证排序覆盖仍未修复 | `tests/test_task_repair.py`；`outputs/rank_repair_real_4i0trmx9/verification.json`；`docs/RANK_LABEL_REPAIR_ZH.md` |
| OpenJev 分片完整性修复与 Drive 本地物化 | 完整套件 **86 passed**（37.10s）；新增 8 个测试覆盖第三分片缺失/大小错误、同大小内容损坏、单文件修复与正常文件复用、普通本地副本及复用、重下一次仍失败时停止、manifest/路径检查、用户本地产物不自动修复、模型使用已验证路径。9 个 notebook 格式/语法通过，01 保留原用户设置和历史输出并新增校验格。仅联网读取固定 revision 的小 manifest：第三分片官方大小 **440,425,856 bytes**、SHA-256 `03f15fc4f5773f2ca6bcd863d968ae2db0c002b206dd357d2edfa2e68a4e8289`。使用真实本地 tiny Qwen3.5/OpenJev 产物验证原决策头完全一致、BF16 骨干/FP32 head、LoRA 梯度与检查点重载。未下载完整 4B、未连接用户 Drive、未确认其坏分片的具体成因 | `tests/test_artifact_cache.py`；`outputs/artifact_repair_native_mzf73fb1/verification.json` |
| 核验上下文双视图 | 完整套件 **78 passed**（29.93s），包含 context 来源排除、审核缓存复用、截断/格式错误中止、修改状态/标签拒绝、混合 batch 分母、超 token 预算跳过、真实反向传播/DPO 和逐参数一致的断点续训。9 个 notebook 格式/代码语法通过，03/06 新入口已接入。真实 API 最终三个探针均返回完整 JSON，一条主题背景通过、两条不贴题拒绝；请求别名 DeepSeek-V4.1-Flash，响应名 DeepSeek-Flash，不据此确认后端权重身份。开发中实际共 7 次探针请求（包含前两次推理预算耗尽）；已加 DeepSeek 关闭 thinking 与不完整审核立即停止。用通过审核的一条真实 MuSiQue 问题、3 个真实状态、3 对既有偏好运行 tiny Stage 1 一步/Stage 2 两步，3 个 context 视图实际参与训练，轨迹指纹不变。该检查的稳定性值为 0（落在容忍区间），不证明鲁棒性提升；无 4B/A100/full 训练 | `tests/test_context_views.py`；`outputs/context_screen_kupxnv6y/live_reviews_final.json`；`outputs/context_real_smoke_fu58oquf/verification.json` |
| 编码器校验与实际加载路径绑定 | **14 passed**（38.85s），覆盖持久缓存修复、本地普通文件物化、仅补缺失的小元数据、本地权重复用、复制失败拒绝加载、SentenceTransformer 显式离线路径。真实 MiniLM 全程离线 CPU 加载/编码通过，直接加载与本地副本向量最大差 **0.0**；三套公开图各三个已存向量兼容检查最小 cosine 日志均为 1.000000。持久权重 SHA-256 不变，无权重下载、无 API。00 编码器格重载模块，格式语法通过，历史输出/执行次数保留。未复现用户 Drive 具体故障，不将路径/映射推断称为已确认根因 | `outputs/embedding_local_snapshot_gh9tj7vq/verification.json`；`pytest -q tests/test_embedding_weight_repair.py tests/test_local_embeddings.py` |
| 任务加工进度输出 | 为 prepare_tasks 添加 8 个阶段提示、读写计数；BM25 分词/查询、任务状态与轨迹偏好添加进度及吞吐/ETA。用真实教学题派生的 3 训练＋1 验证夹具对比 Git HEAD 修改前实现，任务状态、偏好、验证状态及 manifest 四个文件逐字节一致；全部阶段日志出现。相关现有测试 **20 passed**（8.57s）。无 API、无正式训练，未测 full 耗时；索引构建和整体指纹计算只提供开始提示，不承诺持续心跳 | `outputs/task_progress_m9h8_1gw/verification.json`、`after.log`；`pytest -q tests/test_research_contract.py tests/test_download_progress.py tests/test_preparation_resume.py` |
| 00 从头运行前最终检查 | **34 passed**（8.77s）：公开题划分/迁移与中断恢复、缓存复用、MiniLM 校验修复、环境文件、子进程日志及完成后释放。另校验 00 的格式/代码语法与 full 配置。发现用户 notebook 残留三段一次性修复（ZIP 路径补丁、无条件重下 MiniLM、从 Drive ZIP 覆盖源码），已将活动单元改为提示跳过，原代码保存在单元 metadata，历史输出/执行次数与用户 MODE=full、AUTO_DISCONNECT=True 均保留。远端 HEAD 与本地源码提交一致 `329e1ea`；本次 notebook 清理尚为本地更改。无收费 API、无正式训练，未实际执行 Colab 全量流程 | `outputs/final_00_review/checks.json`；`outputs/final_00_review/00_before_retiring_repairs.ipynb` |
| 公开配套题内重新划分与旧数据迁移 | 用户选定每库 300 test / 100 validation / 100 heldout（HotpotQA、MuSiQue），2Wiki 500 test。实际公开问题文件按固定种子划分：HotpotQA 支持完整数 validation 99/100、heldout 100/100、test 297/300；MuSiQue 100/100、100/100、300/300；2Wiki 498/500。覆盖审计在与完整训练源合并前执行；Colab 最终清单另行核查。真实小样本旧派生文件归档、旧完成标记移除、图与缓存保留通过。完整回归 **67 passed**（92.13s）；其后新增缓存幂等测试另见下一项 | `outputs/public_coverage_audit/v2_split_support.json`；`outputs/split_v2_si1ah8ew/verification.json` |
| 新划分下原生检索缓存续跑 | 固定真实公开图＋MiniLM，以关键词夹具生成 6 个原生 mix 候选池；禁用编码器加载与图打开后再次调用，6 个池全部复用，主池/最大池字节不变，无收费 API。发现并修复 PoolCache.write 把 JSON list 与 Python tuple 直接比较、导致相同池被误拒绝的问题。最终相关 **19 passed**（3.37s），包含归档中断后恢复；9 个 notebook 格式与代码语法通过，用户输出/执行次数保留。非常早期缺少完整查询模型绑定的历史缓存继续拒绝，不伪造来源升级 | `outputs/split_v2_cache__af1743p/verification.json`；`pytest -q tests/test_public_indexes.py tests/test_protocol_integration.py tests/test_notebook_cli.py` |
| 全部公开问题的支持覆盖审计 | **实际发现当前额外验证/heldout 配方不适用，不能宣布正式实验就绪。** 3,000 题逐题检查源文档和可检索分块，两者完整覆盖计数相同：发布的 500 题为 HotpotQA 496 / MuSiQue 500 / 2Wiki 498（2Wiki 另两题原始支持标注不完整）；未使用的 500 题为 0 / 128 / 5。额外 HotpotQA 中 488 题无任何可识别支持。复用当前 labels_for 的来源/文本匹配，不等同语义验证、图边正确性或 top-K 召回；无模型/API 调用 | `scripts/audit_public_coverage.py`；`outputs/public_coverage_audit/coverage_audit.json`、`coverage_summary.csv` |
| MiniLM 权重缓存修复 | **12 passed**（8.35s），覆盖 BGE/MiniLM 的正常缓存离线复用、大小或哈希错误后的单次重下、失败后停止、MiniLM 加载前校验。另在独立缓存放入 LFS 文本指针，实际复现 `SafetensorError: header too large`；重新下载固定 revision 的 90,868,376-byte MiniLM 权重，SHA-256 与 Hub 元数据一致，CPU 编码 384 维且归一化通过，错误文件保留。没有访问用户 Colab 的实际缓存，未确认其损坏的具体来源 | `pytest -q tests/test_embedding_weight_repair.py tests/test_local_embeddings.py`；`outputs/minilm_repair_yjzpds2t/verification.json` |
| Colab ZIP 路径兼容修复 | **8 passed**（13.93s）。真实发布包使用反斜杠路径，Windows 的 ZipInfo 自动转换掩盖了原导入器在 Linux 的匹配错误；现在仅在成员匹配时统一分隔符。以保留原始反斜杠的 ZipInfo 模拟 Linux 行为，从真实固定 SHA-256 压缩包全新导入三套图并通过完整性检查，文档数为 4,927 / 7,674 / 3,258，二次运行复用成功。无下载、无模型 API 调用；未在真实 Colab 执行 | `pytest -q tests/test_public_indexes.py`；`outputs/public_index_inspection/colab_zip_uzi99i5u/verification.json` |
| 公开索引适配完整回归 | **57 passed**（154.77s）。包含重复 ZIP 路径分离、2Wiki 完整副本选择、来源追踪、编码器不匹配拒绝、完整 500×3 测试题划分和问题族隔离、9 个 notebook 格式与语法，以及原两阶段训练契约。旧测试要求 notebook 无输出的约束改为允许保留用户真实日志；CLI 测试加入新的配置路径 | Conda SA-Jev：`torch.set_num_threads(2); pytest.main(['-q'])` |
| 公开候选缓存复用最终检查 | 后续新增完整缓存快速复用后，相关 **12 passed**（5.80s）。清单、问题、配置、图绑定和所有池文件齐备时不再加载编码器/图、不再调用 API；原准备失败/完成条件检查仍通过 | `pytest -q tests/test_public_indexes.py tests/test_preparation_resume.py` |
| 公开 LightRAG 索引真实导入 | 下载并验证固定 ZIP SHA-256，按重复 ZipInfo 分开导入三套图；2Wiki 使用包内完整冲突副本。实际文档数 HotpotQA 4,927、MuSiQue 7,674、2Wiki 3,258。真实 MiniLM 与每图三个分块的已存向量余弦匹配（接近 1）。查询前后图、向量、原文、分块文件哈希不变 | `outputs/public_index_inspection/public_oqp5aaw8/index/public_manifest.json`、`unchanged_public_artifacts.json` |
| 新协议小规模端到端 | 真实原始数据采样 8 道训练题，24 个任务状态、19 对偏好；18 个公开题原生 mix 候选缓存、12 个验证状态。tiny 模型 Stage 1/2 各 2 步，冻结参考与 DPO 路径运行，6 题评测均 status=ok，生成 CSV 与 PNG/PDF。查询关键词使用确定性夹具，外部 API 调用为 0。验证 rank 全掩蔽，不能作为排序学习或鲁棒性证据；不是 4B 论文结果 | `scripts/check_public_indexes.py`；`outputs/public_index_inspection/public_oqp5aaw8/verification.json` |
| 新协议真实 DeepSeek 查询 | 本地当前 .env，DeepSeek-V4.1-Flash 查询 HotpotQA 公开图，原生 mix 实际返回 K=256，全部候选有原文来源；查询耗时 7.87 秒（不含加载）。文档抽取调用为 0，未生成最终答案。仅一题，不推算正式吞吐、费用或论文性能 | `scripts/check_public_live.py`；`outputs/public_index_inspection/public_oqp5aaw8/live_verification.json` |
| Conda 环境 | 新建 `SA-Jev`；Python 3.11.18、PyTorch 2.6.0+cu118；RTX 3050 Laptop 4GB 可用 | `docs/validation_environment.json` |
| 此前完整测试套件 | **23 passed，1 failed**（约 81.89s）；唯一失败是本地 `00` notebook 保留用户运行输出，不满足源码 notebook 应无输出的检查；未删除用户日志。smoke 重入、子进程错误显示与环境文件检查均通过 | `python -m pytest -q` |
| 全量准备前回归检查 | 本次 **29 passed，1 deselected**（49.30s）；包括失败写入不覆盖完整文件、失败建图不写完成标记、重跑仅补未完成文档、00 完成标记的前置条件。跳过要求用户 notebook 输出为空的旧检查；9 个 notebook 的格式及代码语法另行通过。未跑真实全量语料 | `pytest -q -k "not notebook_cells_parse_and_outputs_empty"` |
| pilot 默认值与结束自动释放 | 本次 **9 passed**（6.94s）：环境文件、准备完成/恢复及自动断开相关检查通过；模拟 Colab 验证本次完成标记校验、先刷新 Drive 后释放、刷新失败不释放、本地跳过。9 个 notebook 格式及代码语法通过，保留 00 原有用户输出。默认恢复 pilot，00 不默认请求 GPU。未在真实 Colab 会话执行释放 | `pytest -q tests/test_colab_disconnect.py tests/test_preparation_resume.py tests/test_environment.py` |
| 下载/加工进度与流式保存 | 本次 **5 passed**（3.82s）：模拟下载中断不发布残缺文件、限制条数保持一致、完整文件复用、2Wiki 下载解压字节正确及进度输出。另用 7 题合成数据对比修改前后，问题、语料与 manifest 文件逐字节一致；9 个 notebook 格式和语法通过，用户输出及 MODE 保留。未运行完整下载 | `pytest -q tests/test_download_progress.py tests/test_preparation_resume.py`；`outputs/download_progress_check/verification.json` |
| 重复运行复用已加工数据 | 本次 **14 passed**（3.63s）：复用时不读原始语料、不划分/去重、不重写产物，并可不载入问题列表；来源、种子、问题族关联改变，产物缺失/截断、旧清单或强制重建时不会误用缓存；中断不发布新的完成标记。完整原始下载二次调用不请求网络，2Wiki 解压复用通过。9 个 notebook 格式及语法通过，用户 MODE 和输出保留。未测全量 Drive 的耗时 | `pytest -q tests/test_normalization_cache.py tests/test_download_progress.py tests/test_preparation_resume.py` |
| LightRAG 未完成文档诊断 | 本次 **4 passed**（3.34s）：原生错误信息进入异常及 `ingestion_failure.json`，配置令牌/Bearer 令牌脱敏，缺失状态行不漏掉文档，失败不写完成标记，重跑仅提交未完成文档。使用模拟状态存储；用户 Colab 此次实际失败根因仍待其 `error_msg` 输出，未声称已修复服务或真实语料建图 | `pytest -q tests/test_preparation_resume.py` |
| 原生 FAILED 恢复修正 | **8 passed**（12.66s）：检查失败诊断、脱敏、原生重试调用、重试仍失败时停止且不插入新工作、完成文档复用及已知地址/模型错配提前拦截。原生 LightRAG 集成另实际复现：失败后仅再 `ainsert` 不产生新 LLM 调用，旧错误保留；显式 manual retry 后原文档 processed、完成标记生成、mix 检索与来源追踪通过。集成使用确定性模拟 LLM/向量，外部 API 调用为 0；真实 Colab 96 条失败的恢复仍需用户执行 | `pytest -q tests/test_preparation_resume.py tests/test_local_embeddings.py`；`scripts/check_lightrag_resume.py` |
| 本次 AutoDL 服务复查 | 使用本地环境密钥请求 `DeepSeek-V4.1-Flash`：HTTP 200，正文 `OK`，响应模型 `DeepSeek-Flash`，5.802 秒，94 tokens。用户反馈的 Colab 地址/模型相同，但该检查不验证 Colab 中的密钥或批量抽取 | `outputs/ingestion_service_recheck/verification.json` |
| BGE 权重缓存损坏修复 | 本次 **6 passed**（11.60s）：正确缓存无需网络，错误大小/哈希触发隔离并重下，重试仍坏则停止，原数据保留。另在独立缓存放置错误的 LFS 文本指针，实际重现 `header too large`，从 Hub 重下真实 133,466,304-byte BGE 权重，SHA-256 与官方固定 revision 一致，CPU 编码 384 维及归一化检查通过。没有读取用户 Colab 的坏文件，未验证真实 Drive 文件系统的恢复 | `pytest -q tests/test_embedding_weight_repair.py tests/test_local_embeddings.py`；`outputs/embedding_weight_repair_check/run_o0_ns57q/verification.json` |
| 本地向量相关测试 | 本次 **6 passed，1 deselected**（13.35s）；验证本地编码不调用 embedding API、查询/文档前缀、候选向量顺序、并发序列化及旧索引拒绝混用。跳过要求用户 notebook 输出为空的检查；另对全部 9 个 notebook 实际通过 nbformat 与代码语法验证 | `pytest -q tests/test_local_embeddings.py tests/test_protocol_integration.py -k "not notebook_cells_parse_and_outputs_empty"` |
| smoke 重复运行 | RTX 3050 上连续两次完整 tiny smoke：每次 Stage 1/2 各 2 步、检查点重载通过、评测错误 0；各自独立目录，第一次检查点字节保持不变 | `outputs/notebook_smoke_reentry/verification.json` |
| 中转站模型检查 | 使用用户配置的服务令牌查询模型列表返回 200，包含 `gemini-3.1-flash-lite`，不包含 `text-embedding-3-small`；Gemini 小请求返回 401；embedding 小请求返回 503（当前分组无可用通道）。仅确认模型名称与当前失败状态，未宣称接口调用成功 | `outputs/relay_model_check/verification.json` |
| Gemini 3 Flash Preview 试调用 | 同一中转站与令牌：模型列表返回 200，包含 `gemini-3-flash-preview`；聊天请求仍返回 401 `Invalid token`，未获得文本。配置已切换，不能据此声称生成可用；本次未重复调用 embedding | `outputs/relay_gemini3_flash_check/verification.json` |
| AutoDL DeepSeek 试调用 | 使用更新后的 `.env`：模型列表 200，包含 `DeepSeek-V4.1-Flash`；聊天请求 200，正文 `OK`，耗时 4.187 秒，服务返回模型名 `DeepSeek-Flash`，usage 为 48 tokens。向量最小请求 `/api/v1/embeddings` 返回 404；聊天连通，不代表 LightRAG 全链路通过。配置已改用请求模型名，未独立验证服务后端权重身份 | `outputs/api_deepseek_v41_flash_check/verification.json` |
| Google Gemini 2.5 Flash-Lite 试调用 | 更新后的 Google 官方兼容接口模型列表返回 200，含 `models/gemini-2.5-flash-lite`。带 `seed` 的聊天请求返回 400（不接受该参数）；去掉后返回 404，明确该模型不再向新用户开放。未获得生成文本；服务错误信息建议 `gemini-3.5-flash-lite`，其后续独立测试见下一行。回答器已对 Google 接口省略 `seed`，并按服务地址隔离缓存 | `outputs/api_google_gemini25_flash_lite_check/verification.json` |
| Google Gemini 3.5 Flash-Lite 试调用 | 请求返回 200，正文 `OK.`，响应模型名 `gemini-3.5-flash-lite`，耗时 1.343 秒，usage 共 7 tokens；项目实际 Answerer 根据合成证据返回 `London`，83 tokens，缓存重读一致。已将建图/回答默认模型及环境模板切换到 Google 官方兼容接口 | `outputs/api_google_gemini35_flash_lite_check/verification.json` |
| 本地 BGE + 真实 Gemini + 原生 LightRAG | 在独立目录用两篇合成文档实际完成建图、`mix` 查询、来源还原与固定缓存重读，返回 2 个候选；未使用旧 DeepSeek 索引或候选池。全部向量由固定版本本地 BGE 生成。仅是微型联调，不是论文性能或正式训练验证 | `outputs/google_gemini35_lightrag_check/run_j4geu2_9/verification.json` |
| 本地 BGE 编码器 | 实际下载 `BAAI/bge-small-en-v1.5` 固定 revision 的 safetensors（133,466,304 bytes）；RTX 3050 上生成 384 维归一化向量，2 文档 + 1 查询的小测试峰值 reserved 0.158203125 GiB。另在 `HF_HUB_OFFLINE=1` 且无 API key 的 CPU 进程中验证缓存加载与编码成功 | `outputs/local_embedding_check/run_et0opc_q/encoder.json`、`outputs/local_embedding_check/cpu_offline.json` |
| 本地 BGE + 真实 DeepSeek + 原生 LightRAG | 两篇合成文档实际建图、`mix` 查询、来源还原与固定缓存重读通过，返回 2 个候选；流程耗时 28.94 秒（含一次 API 超时后重试恢复）。全部向量由本地编码器生成。使用测试 LLM 的独立运行亦通过。仅是微型联调，未估计正式数据建图成本或论文性能 | `outputs/local_embedding_check/run_et0opc_q/verification.json`、`outputs/local_embedding_check/run_ernui78a/verification.json` |
| 最终 DeepSeek 复测与配置 | 用户最终选择 `DeepSeek-V4.1-Flash`。最小聊天请求 200、`OK`、3.687 秒，服务返回名 `DeepSeek-Flash`；更新后的持久化/状态检查代码实际跑通两文档建图、混合查询和候选缓存，流程 33.45 秒，返回 2 候选。默认模型与模板恢复为 AutoDL DeepSeek，00 默认 full 数据准备；此微型耗时不能外推全量 | `outputs/deepseek_final_check/verification.json`、`outputs/deepseek_final_lightrag_check/run_b2fhpk6a/verification.json` |
| 2Wiki 全量入口预检 | 官方 ZIP 链接实际返回 HTTP 200、ZIP 文件头正确，响应长度 258,968,175 bytes；仅读取文件头，未声称完整下载、解压或正式全量处理通过 | 官方下载地址在 `src/sa_jev/preparation.py` |
| 真实小数据完整路径 | 下载 HotpotQA train/dev、作者 MuSiQue answerable train/dev 各 8 条；选其中小部分作流程测试，12 个训练状态、6 个验证状态、6 对轨迹；Stage 1 与 Stage 2 各实际更新 2 步；2 题动态评测无运行错误 | `outputs/real_smoke_final/verification.json` |
| 检查点重载 | tiny 编码器 + **实际上游** DecisionHead/OptionSetInteractor 的保存、重载预测一致 | 同上及 `stage1/step_000002/` |
| 随机小 Qwen3.5 | 103,936 参数，包含 linear/full attention；CPU 和 CUDA 均完成原生 adapter + PEFT LoRA + 原生 interactor/head 的反传及 optimizer 更新 | `outputs/qwen_lora_check.json`、`outputs/qwen_lora_cuda.json` |
| 原生 artifact 加载 | 本地随机小 Qwen3.5 artifact，BF16 backbone/FP32 head；确认加载原 head 参数、LoRA 梯度、LoRA+head checkpoint 重载一致 | `outputs/native_artifact_check/verification.json` |
| 原生 LightRAG 联调 | 使用确定性测试 LLM/embedding，实际完成原生 `ainsert`、图/向量存储、`mix` 查询、固定缓存；返回 2 条候选，完整来源还原正确 | `outputs/lightrag_local_v2/verification.json` |
| 原始数据预处理 | 32 条真实源记录归一化；问题级划分得到 train=14、validation=0、test=16、heldout=2；BM25 候选/状态生成得到 42 状态、31 对偏好 | `outputs/data_check/data/` |
| OpenJevData 接口 | 实际流式扫描前 100 条，验证数据读取与匹配流程；**不是完整重叠审计** | `outputs/data_check/data/overlap_audit.json` 的 `complete_scan=false` |
| Notebooks | 9 个 notebook 通过 nbformat 和全部代码单元语法检查；每个含 Drive 挂载与 Git 同步 | `tests/test_protocol_integration.py` |
| 结果产物 | 实际生成训练/评测 CSV、PNG、PDF；已查看真实 smoke 的训练曲线与评测图 | `outputs/real_smoke_final/` |

完整小数据路径用的是**测试用 tiny causal encoder**，没有下载/运行训练好的 OpenJev-4B。其 PyTorch peak reserved 为 **0.0546875 GiB**，仅说明这个小测试的显存占用，不能估算 4B 训练显存。其 stability loss 为 0 是两视图差异落在容忍区间内；另有构造分布的测试验证超阈值时稳定性损失及梯度确实非零，不能由该小测试推导鲁棒性结论。

32 条数据的正式拆分恰好没有 validation 题，准备器如实保留这个事实，训练入口会拒绝把 test 借作 validation。完整路径 smoke 另外使用互不重复的原始问题作明确的代码测试划分。这不是论文数据划分。42 状态中只有 6 个 rank-valid 状态；外部未核验候选导致掩码，已通过 manifest 暴露，不伪装为 42 个有效排序监督状态。

## 已覆盖的方法检查

- Choice 在全部有效候选上归一化；K=256 的第 256 个候选仍参与 interactor 和梯度。
- 分支 checkpointing 反向传播到编码器与 interactor；候选重排后概率按 ID 等变。
- 充分性输入不含未选证据正文；任务输入不含答案、支持标签或问题分解。
- 边际增益、支持相关性、充分性分离；相关重复证据可以零增益；未知标签被屏蔽。
- 支持文本匹配绑定原始来源，不能用其他页面中相同的代词句冒充支持；有效条件变化必须提供重新核验的标签。
- 问题族隔离、正式评测 gold 注入阻断、候选去重、缓存内容不可覆盖和 ID 追踪。
- 不同有效样本数的任务损失分母；稳定性 ID 对齐、0.05 容忍区间、真实状态变化屏蔽。
- 完整 READ/STOP 对数概率、预算耗尽不计 STOP、DPO 参考 detach 和梯度方向。
- 先 Noul 后 Choice、预算不足拒答；未配置回答器时 QA 指标不填值。
- 核验过的测试包才能评分；有效变化按新标签评分；对不同缓存的配对比较拒绝计算。
- 恢复 optimizer 和 RNG 后，剩余训练参数与未中断运行逐项完全一致。

## 仍需在 Colab/真实服务运行

2026-11-12：针对 Colab 预装 `torchao==0.10.0` 导致 PEFT 0.19.1 在创建 LoRA 时抛出 ImportError，所有 notebook 的 Colab 安装单元移除本项目不使用的 torchao 量化包。02 的小模型检查实时显示 stdout/stderr，并保存 `ROOT/qwen_lora_check.log`。本地 Conda SA-Jev（PEFT 0.19.1、Transformers 5.12.1、无 torchao）在 CPU 和 RTX3050 CUDA 上均通过原小模型的 LoRA/Interactor 反向传播及优化器更新检查，结果在 `outputs/torchao_fix_check/`；9 个 notebook 的格式和代码语法校验通过。这不代表 Colab Python 3.13、FLA 或真实 4B A100 训练已经通过，仍须在用户运行时复测。已有会话卸载 torchao 后应重启内核再从 02 的环境准备开始运行，避免已导入模块的可用性缓存残留。

1. **OpenJev-4B 发布权重 + FLA 0.5.2 + A100**：代表性长状态/长轨迹的前向、反向、显存与吞吐；确认单卡峰值满足约束后才正式训练。
2. 完整 8k/4k 数据准备、全部 OpenJevData 扫描、实际 K/有效监督覆盖核查；必要时对未知候选补可核验注释。本地 BGE 分别搭配真实 DeepSeek、Gemini 的微型 LightRAG 全链路已通过；仍需在真实 pilot 语料上测量召回覆盖、建图耗时和 API 成本。
3. 官方 Jev API、真实回答模型的正式数据测量；最终采用的 AutoDL DeepSeek 已完成最后一次聊天及小规模建图复测，向量使用本地编码器。此前 Google Gemini 的接口验证保留为历史记录。未读取或使用 `JEV_API_KEY.txt`，也未调用官方 Jev API。
4. 原模型上下文干扰与 K 诊断、三个种子的两阶段训练、共享缓存主对比及 2Wiki 跨数据集评测。
5. 人工核验约 600 个独立状态的 3 个无关扰动与 1 个有效变化；本次没有生成或声称已有 600 份核验结果。
6. DPO/稳定性/完整模型/无状态更新/小 K/权重消融；K=16…256 的实测质量、延迟、token、显存及失败比例。
7. ViDoRe 官方 OCR 流式列读取、真实检索及有来源的成功/失败案例。本次只核对了官方字段与指定 revision，没有运行工业案例结论。

没有正式训练权重、论文指标、显著性或“优于基线”的结论。Colab 顺序见根 README：00 pilot → 01–04 用少量问题/步骤验证 4B 路径 → 00 full 准备一次 → 正式 01–04 → 05–08。已经完成的 full 主实验数据与固定候选缓存不需要再建一次；独立扰动人工核验、小 K 消融派生数据和 ViDoRe 工业案例仍在后续步骤完成。00 pilot 与 4B pilot 均尚未完整实测，不承诺固定耗时。
