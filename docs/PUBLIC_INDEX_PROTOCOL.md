# 公开 LightRAG 协议 v1

默认配置为 `configs/public_lightrag.yaml`；所有 notebook 使用 `MyDrive/SA-Jev/public_lightrag_v1/{pilot,full}`。
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

## 数据与泄漏边界

1. 完整训练仍从官方 train 中选 HotpotQA 8,000、MuSiQue 4,000 题。先将官方训练题与所有公开评测题联合按问题族及规范化问题文本隔离，再采样。训练原始上下文与 BM25 难例构造方式保持原协议；这些段落不进入公开图。
2. 每个数据集作者发布的 500 题全部保留为 test。另从 HippoRAG 剩余问题中，按固定种子从 HotpotQA 和 MuSiQue 各选 100 题 validation、300 题 heldout。问题族不能跨角色。2Wiki 不用于训练、验证调参或稳定性训练。
3. 这些额外验证/保留题仍查询固定的公开图；它们的支持证据可能不在图中。绝不往图或缓存注入 gold，不因结果差而筛题；`data/public_pool_coverage.csv` 报告初始标签有效性和正增益候选数量。缺失证据与未知标签继续掩蔽。600 个独立扰动状态仍需人工核验，草稿不能当真实鲁棒性结果。
4. full 共 2,300 个非训练问题（1,500 test、200 validation、600 heldout）。pilot 为 64 道训练题及 18 个检索问题（每库 2 test、Hotpot/MuSiQue 各 2 validation 和 4 heldout）。公共图大小相同，避免 pilot 重建另一套图。
5. 支持标签仍由原始问题的支持事实/分解生成；索引的 `unknown_source` 用 `chunk_id → full_doc_id → 标题/正文` 恢复来源，不把所有 unknown_source 当作同一文档。
6. 仅用固定图上的实际混合召回构建 C0。后续状态更新不调用召回器。所有比较方法共享同一问题和缓存，K 外推从同一个 K=256 池切片；短池如实报告。

## 首次运行与费用

1. 停止旧 full 的建图单元，保留 Drive 数据。将代码同步到 GitHub 后，在 Colab 打开更新后的 00。新的代码不会自动传到已有 Colab 内存。
2. 首个代码单元选 `MODE='pilot'`，环境安装及 `.env` 仍按顺序运行。相同 Drive `.env`，无需额外 Groq 密钥。
3. 00 下载/复用 ZIP、小型编码器和原始数据。full 若发现旧 `full/raw` 的完整下载清单则复用；不重跑旧的 65 万段标准化或建图。
4. 导入、编码兼容性检查、标签加工不产生抽取 API 费用。候选检索仍需要查询关键词调用；缓存命中可减少调用，不能承诺恰好 2,300 次或严格零费用。后续最终回答、官方 Jev 和独立工业案例另计。
5. 先完成 00 pilot，再运行 02/03 pilot 验证真实 4B 的单卡显存和训练，之后才执行 00 full 与正式实验。全部权重/原始结果/表格/图仍在 Drive。00 成功后可按原设置自动断开。
6. 01–07 自动使用新配置和目录。08 工业案例仍是独立 OCR 图，**没有**包含在这个公开 ZIP 中，仍需单独建图及费用评估。

公开图已加载并完成离线关键词夹具下的原生 mix 测试。另实际验证一条 DeepSeek 查询：HotpotQA 原生 mix 返回 256 个可追溯候选，约 7.87 秒，未调用文档抽取或最终回答。全量 2,300 问题缓存、OpenJev-4B 正式训练和 A100 显存仍需在 Colab 验证。不得将夹具联调结果作为论文性能。小规模真实检索验证集的排序标签全部因未知候选而掩蔽，验证报告中的 rank 为 null；这不是排序质量已验证，应检查 full 的有效标签数量，并在需要时补充可核验注释。
