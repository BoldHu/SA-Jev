# 实际验证记录

日期：2026-11-07 至 2026-11-08（Asia/Shanghai）。所有数值来自本机实际运行，均为工程/方法契约测试，不是 SA-Jev-4B 论文结果。

## 已运行

| 检查 | 实际结果 | 可追溯输出 |
|---|---|---|
| Conda 环境 | 新建 `SA-Jev`；Python 3.11.18、PyTorch 2.6.0+cu118；RTX 3050 Laptop 4GB 可用 | `docs/validation_environment.json` |
| 此前完整测试套件 | **23 passed，1 failed**（约 81.89s）；唯一失败是本地 `00` notebook 保留用户运行输出，不满足源码 notebook 应无输出的检查；未删除用户日志。smoke 重入、子进程错误显示与环境文件检查均通过 | `python -m pytest -q` |
| 全量准备前回归检查 | 本次 **29 passed，1 deselected**（49.30s）；包括失败写入不覆盖完整文件、失败建图不写完成标记、重跑仅补未完成文档、00 完成标记的前置条件。跳过要求用户 notebook 输出为空的旧检查；9 个 notebook 的格式及代码语法另行通过。未跑真实全量语料 | `pytest -q -k "not notebook_cells_parse_and_outputs_empty"` |
| pilot 默认值与结束自动释放 | 本次 **9 passed**（6.94s）：环境文件、准备完成/恢复及自动断开相关检查通过；模拟 Colab 验证本次完成标记校验、先刷新 Drive 后释放、刷新失败不释放、本地跳过。9 个 notebook 格式及代码语法通过，保留 00 原有用户输出。默认恢复 pilot，00 不默认请求 GPU。未在真实 Colab 会话执行释放 | `pytest -q tests/test_colab_disconnect.py tests/test_preparation_resume.py tests/test_environment.py` |
| 下载/加工进度与流式保存 | 本次 **5 passed**（3.82s）：模拟下载中断不发布残缺文件、限制条数保持一致、完整文件复用、2Wiki 下载解压字节正确及进度输出。另用 7 题合成数据对比修改前后，问题、语料与 manifest 文件逐字节一致；9 个 notebook 格式和语法通过，用户输出及 MODE 保留。未运行完整下载 | `pytest -q tests/test_download_progress.py tests/test_preparation_resume.py`；`outputs/download_progress_check/verification.json` |
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

1. **OpenJev-4B 发布权重 + FLA 0.5.2 + A100**：代表性长状态/长轨迹的前向、反向、显存与吞吐；确认单卡峰值满足约束后才正式训练。
2. 完整 8k/4k 数据准备、全部 OpenJevData 扫描、实际 K/有效监督覆盖核查；必要时对未知候选补可核验注释。本地 BGE 分别搭配真实 DeepSeek、Gemini 的微型 LightRAG 全链路已通过；仍需在真实 pilot 语料上测量召回覆盖、建图耗时和 API 成本。
3. 官方 Jev API、真实回答模型的正式数据测量；最终采用的 AutoDL DeepSeek 已完成最后一次聊天及小规模建图复测，向量使用本地编码器。此前 Google Gemini 的接口验证保留为历史记录。未读取或使用 `JEV_API_KEY.txt`，也未调用官方 Jev API。
4. 原模型上下文干扰与 K 诊断、三个种子的两阶段训练、共享缓存主对比及 2Wiki 跨数据集评测。
5. 人工核验约 600 个独立状态的 3 个无关扰动与 1 个有效变化；本次没有生成或声称已有 600 份核验结果。
6. DPO/稳定性/完整模型/无状态更新/小 K/权重消融；K=16…256 的实测质量、延迟、token、显存及失败比例。
7. ViDoRe 官方 OCR 流式列读取、真实检索及有来源的成功/失败案例。本次只核对了官方字段与指定 revision，没有运行工业案例结论。

没有正式训练权重、论文指标、显著性或“优于基线”的结论。Colab 顺序见根 README：00 pilot → 01–04 用少量问题/步骤验证 4B 路径 → 00 full 准备一次 → 正式 01–04 → 05–08。已经完成的 full 主实验数据与固定候选缓存不需要再建一次；独立扰动人工核验、小 K 消融派生数据和 ViDoRe 工业案例仍在后续步骤完成。00 pilot 与 4B pilot 均尚未完整实测，不承诺固定耗时。
