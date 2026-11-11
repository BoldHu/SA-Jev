# 公开 LightRAG：固定图 v1，问题划分 v2

默认配置为 `configs/public_lightrag.yaml`；所有 notebook 使用 `MyDrive/SA-Jev/public_lightrag_v1/{pilot,full}`。
**2026-11-08：用户确认采用优先主测试的配方，已修正选题来源。** 新增问题协议标识 `published_family_split_v2`，只在每库公开的 500 题内按问题族划分；不改变图、检索器、损失或训练样本来源。HotpotQA/MuSiQue 每库 300 test、100 validation、100 heldout；2Wiki 500 test。固定种子与问题族决定划分，不依据覆盖率或模型成绩筛题；较大问题族可能使题数略超目标，实际清单为准。

历史问题：旧版从额外 500 题抽验证/heldout。实际完整支持可识别数为 HotpotQA 0、MuSiQue 128，不能用于原定充分支持任务。发布的 500 题中为 HotpotQA 496、MuSiQue 500、2Wiki 498（另 2 题原始支持标注不完整）。新划分对实际公开题文件的来源覆盖检查：HotpotQA validation 99/100、heldout 100/100、test 297/300；MuSiQue 为 100/100、100/100、300/300；2Wiki test 498/500。此检查在与完整训练源的问题族合并之前执行；Colab 会对最终划分再次报告覆盖。它不证明图关系正确，也不保证 top-K 召回；缺失题保留并报告，不补入金标。

复核命令：`python scripts/audit_public_coverage.py --root <ROOT> --assets <PUBLIC_ASSETS> --output <OUTPUT>`。只读本地文件，不调用模型 API。实际报告：`outputs/public_coverage_audit/coverage_audit.json`、`coverage_summary.csv`。训练仍使用原始训练上下文，并未验证它们能由公开图召回；训练未知候选导致的排序标签掩蔽是另一个需核查的问题。

旧的 `MyDrive/SA-Jev/full` 及其图保留，不继续自动建图。`full` 现在表示完整的本研究采样协议，不表示 benchmark 全测试集。
SA-Jev 的 OpenJev-4B 初始化、固定候选决策、标签掩码、两阶段损失和单卡训练限制不变。

## 公开产物与实际检查

- 图索引：[thomoal/graph-rag-qa-indexes](https://huggingface.co/datasets/thomoal/graph-rag-qa-indexes)，revision `b187c2460ab021a8d0a947f24ea5743043253ad0`。
- ZIP：`lightrag_indexes_N500.zip`，560,874,369 字节，SHA-256 `919a6a156e2274b3495d5fdbccdee32914e6e6be5e679b71f3f5a04145b42ba3`。
- 原始带支持标签的问题：[osunlp/HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2)，revision `5ec05b38deecc3318bb432c69865959c56058990`。每个数据集 1,000 题；公开索引的 `lightrag_contexts.json` 仅用于确定作者选中的 500 个测试 ID，**不复用其压缩上下文、答案或结果作为我们的候选或性能**。
- 上游代码：[graph-rag-qa-pub](https://github.com/thomouvic/graph-rag-qa-pub)，论文 *The Reasoning Bottleneck in Graph-RAG*。其原实验为 hybrid 检索；我们保留既定原生 **mix（图＋向量）**，新生成最大 K=256 候选，不能称为逐项复现原论文分数。

| 数据集 | 文档 | 分块 | 图节点 | 图边 | 关系向量 | 公开测试题 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 4,927 | 4,929 | 28,858 | 26,330 | 26,330 | 500 |
| MuSiQue | 7,674 | 7,674 | 44,172 | 45,613 | 45,612 | 500 |
| 2Wiki | 3,258 | 3,260 | 22,330 | 20,543 | 20,543 | 500 |

这些是实际下载后统计的产物规模，不是论文性能。三个库共 15,859 段（各库内部去重，跨库不合并）。

**ZIP 导入注意**：三套文件在 ZIP 内使用相同路径，直接 `extractall` 会覆盖。导入器按经过校验的 ZIP 条目顺序分别写入数据集目录，再用公开问题 ID 检查归属。
2Wiki 的默认图仅 1,625 节点、默认分块向量仅 206 条；采用同包中的三个 `-DESKTOP-0KMHV60` 完整副本（图、分块向量、full_relations）。每个文件的原名、CRC、SHA-256 和选择记录保存在 `index/public_manifest.json`。
MuSiQue 的一条图边没有对应关系向量，保持原样并报告，不宣称发布产物完美一致。

上游图抽取模型是 `llama-3.1-8b-instant`。我们的 DeepSeek 只用于新查询关键词和最终回答；不能声称这些图由 DeepSeek 构建。
上游编码器为 `all-MiniLM-L6-v2`，未记录模型 revision；本地固定 `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，256 token、384 维、无 BGE 查询前缀。每次打开索引比较三个已有分块的向量余弦，相似度低于 0.995 则停止。该检查验证抽样兼容性，不证明上游版本身份。

编码器加载兼容修复（2026-11-11）：Drive 权重即使顺序读取的 SHA-256 正确，也不能由此确认模型实际解析的缓存路径或内存映射读取正常。固定 MiniLM/BGE 现在将已校验权重及必要的小型配置/分词文件复制到运行时临时目录 `sajev_embedding_models/`，不复制软链接；本地副本再次核对大小、SHA-256 和 safetensors 解析，再由 SentenceTransformer 使用该显式目录和 `local_files_only=True` 加载。Drive 下载缓存仍保留，新连接只需重新复制；此修复不改变模型身份、向量设置、公开图或候选池。日志打印实际加载目录。已在本地用真实 MiniLM 离线验证，尚未在用户 Colab 上验证其具体 Drive 故障。

## 数据与泄漏边界

1. 完整训练仍从官方 train 中选 HotpotQA 8,000、MuSiQue 4,000 题。先将官方训练题与所有公开评测题联合按问题族及规范化问题文本隔离，再采样。训练原始上下文与 BM25 难例构造方式保持原协议；这些段落不进入公开图。
2. 所有非训练角色仅从作者发布的每库 500 题选择。HotpotQA/MuSiQue 各 100 validation、100 heldout，其余约 300 test。同题族整体分配；触及 2Wiki 的跨数据集同题族全部保留 test。2Wiki 不用于训练、验证调参或稳定性训练。
3. 检索前 `data/public_source_coverage.json` 报告全部源分块能覆盖的支持；检索后 `data/public_pool_coverage.csv` 报告实际候选中的初始标签有效性和正增益候选数。二者不能混同。绝不往图或缓存注入 gold，不因结果差而筛题。缺失证据与未知标签继续掩蔽。200 道保留题可产生多个状态，但不保证达到 600 个合格扰动包；`perturbations_draft.manifest.json` 记录实际包数与独立题数，草稿仍须人工核验。
4. full 共 1,500 个非训练问题（1,100 test、200 validation、200 heldout）。pilot 为 64 道训练题及 18 个检索问题（每库 2 test、Hotpot/MuSiQue 各 2 validation 和 4 heldout）。图不变；原来的 ROOT 路径保留以复用下载和候选缓存。
5. 支持标签仍由原始问题的支持事实/分解生成；索引的 `unknown_source` 用 `chunk_id → full_doc_id → 标题/正文` 恢复来源，不把所有 unknown_source 当作同一文档。
6. 仅用固定图上的实际混合召回构建 C0。后续状态更新不调用召回器。所有比较方法共享同一问题和缓存，K 外推从同一个 K=256 池切片；短池如实报告。

## 首次运行与费用

旧划分迁移：停止当前检索单元，等待停止完成后更新代码与配置。重跑 00 中 `download_public_assets / prepare_public_data` 数据单元（新版会重新加载配置及相关模块），再运行 `cache_pools`。旧的 data 顶层派生文件、扰动核验文件和 `00_complete.json` 移入 `ROOT/protocol_archive/`，不会删除；raw、index、逐题 pools/pools_max 保留。同题、同图、同模型/检索配置且缓存校验一致的池直接复用，旧图外问题缓存留存但不再使用。随后重跑任务状态、偏好、OpenJevData 核查和扰动草稿单元。不要同时运行旧检索进程和迁移。旧问题清单会在调用 API 或训练前被拒绝。特别早期的本地缓存缺少完整查询模型绑定时仍会被拒绝，不能伪造来源使其通过。

1. 停止旧 full 的建图单元，保留 Drive 数据。将代码同步到 GitHub 后，在 Colab 打开更新后的 00。新的代码不会自动传到已有 Colab 内存。
2. 首个代码单元选 `MODE='pilot'`，环境安装及 `.env` 仍按顺序运行。相同 Drive `.env`，无需额外 Groq 密钥。
3. 00 下载/复用 ZIP、小型编码器和原始数据。full 若发现旧 `full/raw` 的完整下载清单则复用；不重跑旧的 65 万段标准化或建图。
4. 导入、来源覆盖审计、编码兼容性检查、标签加工不产生抽取 API 费用。候选检索仍需要查询关键词调用；缓存命中可减少调用，不能承诺恰好 1,500 次或严格零费用。后续最终回答、官方 Jev 和独立工业案例另计。
5. 先完成 00 pilot，再运行 02/03 pilot 验证真实 4B 的单卡显存和训练，之后才执行 00 full 与正式实验。全部权重/原始结果/表格/图仍在 Drive。00 成功后可按原设置自动断开。
6. 01–07 自动使用新配置和目录。08 工业案例仍是独立 OCR 图，**没有**包含在这个公开 ZIP 中，仍需单独建图及费用评估。

公开图已加载并完成离线关键词夹具下的原生 mix 测试。另实际验证一条 DeepSeek 查询：HotpotQA 原生 mix 返回 256 个可追溯候选，约 7.87 秒，未调用文档抽取或最终回答。新划分全量 1,500 问题缓存、OpenJev-4B 正式训练和 A100 显存仍需在 Colab 验证。不得将夹具联调结果作为论文性能。历史小规模真实检索验证集的排序标签全部因未知候选而掩蔽，验证报告中的 rank 为 null；来源覆盖修复不自动解决排序监督掩蔽，应检查新 full 的有效标签数量，并在需要时补充可核验注释。
