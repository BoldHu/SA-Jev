# 实际验证记录

日期：2026-11-07（Asia/Shanghai）。所有数值来自本机实际运行，均为工程/方法契约测试，不是 SA-Jev-4B 论文结果。

## 已运行

| 检查 | 实际结果 | 可追溯输出 |
|---|---|---|
| Conda 环境 | 新建 `SA-Jev`；Python 3.11.18、PyTorch 2.6.0+cu118；RTX 3050 Laptop 4GB 可用 | `docs/validation_environment.json` |
| 测试套件 | **20 passed**，最后一次约 13.62s | `python -m pytest -q` |
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
2. 完整 8k/4k 数据准备、全部 OpenJevData 扫描、实际 K/有效监督覆盖核查；必要时对未知候选补可核验注释。真实 LightRAG 的抽取/嵌入服务尚未调用。
3. 官方 Jev API、真实回答模型的接入与测量；本次没有读取或使用项目中的 API key 文件。
4. 原模型上下文干扰与 K 诊断、三个种子的两阶段训练、共享缓存主对比及 2Wiki 跨数据集评测。
5. 人工核验约 600 个独立状态的 3 个无关扰动与 1 个有效变化；本次没有生成或声称已有 600 份核验结果。
6. DPO/稳定性/完整模型/无状态更新/小 K/权重消融；K=16…256 的实测质量、延迟、token、显存及失败比例。
7. ViDoRe 官方 OCR 流式列读取、真实检索及有来源的成功/失败案例。本次只核对了官方字段与指定 revision，没有运行工业案例结论。

没有正式训练权重、论文指标、显著性或“优于基线”的结论。Colab 首次顺序见根 README：00 → 01 → 02 → 03 → 04；pilot 确认后在 full 根目录重做，最后 05–08。
