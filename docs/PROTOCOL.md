# 实现与研究协议

根目录 AGENTS.md 是方法定义。本文件解释实现中的可检查选择，不改动研究主体。

## 原生模型复用

`load_decision_artifact` 一起加载发布的 Qwen3.5 文本骨干、`decision_head.safetensors`、配置、tokenizer 与校准温度。决策头包括 projection、两层原生 Option Set Interactor 和双线性评分。Choice 和 Noul 都使用相同头；Noul 是 False/True 两个选项的完整决策。

`SAJev.decision` 直接走 serializer → `collate_serialized` → `OpenJevDecisionModel.forward`，不使用被 `inference_mode` 包装的 runtime.predict。上游 branch checkpointing 保存可微分计算，分支 microbatch 仅减少骨干激活驻留。所有候选汇合到一个 interactor/softmax；不会把 K 个候选切成多个独立归一化组，也不对池化特征 detach。

发布配置为 BF16 backbone / FP32 head、FLA 0.5.2。LoRA 仅作用于原骨干线性层，原 head/interactor 全部可训练。没有添加额外奖励模型、planner、检索器或充分性头。上游的 full-parameter SFT→REINFORCE-A 训练器没有被冒充为本研究的离线 DPO。

仅一处语义无关的兼容扩展：`schema.py` 和 `api/typesafe.py` 中的数量校验从 255 放宽到 256，方便定义中的 K=256。源码版本被固定，bootstrap 检查被修改的字串。该扩展没有改变头维数、参数、排序概率或梯度。

## 数据、状态与标签

Question 保存 gold/答案/分解得到的支持映射；State 只保存问题、有效条件、普通上下文、固定候选池、已选 ID 与预算。prompts 以白名单构造模型文本，不序列化 Question/Labels。

HotpotQA/2Wiki 的 unit 为 `(title, sent_id)`。MuSiQue 的 unit 为分解步骤支持段；支持新发布的 paragraphs 格式和作者实验代码中的 contexts/decomposed_instances 格式。分解文本和答案从不进入输入。完整性检查不把缺失支持段、失效 sentence index 或不可回答上下文扩展视为负例。

支持事实覆盖是监督代理，不表示穷尽所有自然语言替代证明。原始明确干扰段可作为已知负例；检索得到的外部候选只按可追溯注释或完整支持文本匹配转移已知正覆盖，未知候选保持 mask。未标注的新候选若被赋零软目标会产生隐式负梯度，因此未知候选存在时默认屏蔽整个排序项。保留候选全集，不把未知候选移除后做子集 softmax。

每题最多 3 个任务状态：初始、中间、轨迹末端；相邻 READ 真正更新已选证据并扣减预算。充分状态和无可判断正增益的状态不算 rank loss。相关而已覆盖的另一条证据可以保留 relevance=1、gain=0。

K 采样比例为 8/16/32/64 的 40/30/20/10。实际池不足时不补重复；实际和请求 K 都写入 manifest。训练时可保留 gold 支持段，优先使用 BM25、可选向量候选和原始干扰段；正式验证/测试只能用无 gold 注入的 LightRAG 缓存。

先对 question group 和规范化题面建立连通分量，再划分 train/validation/heldout；官方 dev 作为 test。可通过 `normalize_sources(..., family_links=[...])` 加入原始数据已知同题变体的关联。自动精确题面识别无法保证发现第三方未标注的释义题；发布额外题目变体前应提供 family_links。训练生成的状态、轨迹、双视图继承相同 group，不再独立划分。

完整源文件必须与 pilot 子集分目录。正式训练检查完整 OpenJevData 扫描与数据 fingerprint；扫描只检查规范化包含关系，不把未命中包装成完全未见泛化。正式论文应报告已知来源接触、命中清单和这个检测方法的局限。

## 两阶段目标

Stage 1：`rank + suf + 0.2 rel`。rank 是正边际增益归一化后的软交叉熵；suf/rel 使用原生 Noul 的 log probabilities，数学上等价二元交叉熵且避免 log(sigmoid) 数值不稳。

Stage 2 的两视图任务项取平均。每个状态在线选择另一种等义序列化；问题、条件、context 内容、证据内容/ID、预算、标签都不变。没有 `irrelevant_noise` 输入字段。呈现变化不意味着获得了自然语言干扰鲁棒性。

稳定性两项分别除以自己的有效数，L2 和充分性绝对差均先减 0.05，再 ReLU 平方。排序稳定性可用于 Gold 可用且确认未充分、有非空同一候选集的等义状态，即使当前没有正增益而 rank task loss 被屏蔽；充分性稳定性要求充分性标签可用。State 发生真实变化时不得调用等义配对损失。

参考概率在训练 Stage 2 前从 Stage 1 以 eval/no_grad 计算，绑定权重、温度、完整轨迹与输入视图哈希。缓存后只驻留一份策略模型。轨迹 log probability 累加每个 READ 的 `log(1-h)+log(p_i)` 与自主 STOP 的 `log(h)`；不做长度平均，不给预算耗尽补一个 STOP。

偏好构造实际重放 READ 动作，比较确定的完整覆盖与缺失支持，或都充分时至少相差一次 READ。失败对失败、平局、未知终局不强制排序。短轨迹优势来自偏好定义，不是错误的对数概率长度归一化。

## 评测与统计

主表共用相同问题、pool fingerprint、预算、阈值与回答模型；配对函数检查这些绑定。默认阈值 0.5，若调阈值只能用 validation 并固定到全部对应比较。原始召回基线用冻结 OpenJev 的独立 Noul；官方 Jev 的静态/动态/状态过滤基线使用官方 Noul。无状态更新消融仅移除 Choice 已选内容。

记录支持单元召回、完整覆盖、有效状态 nDCG、独立充分性 accuracy/Brier/ECE/AUROC、错误接受/提前停止、拒答、READ 次数、最终答案 EM/F1、token/延迟/显存。QA 使用同一回答器且只输入已选证据。当前答案分数是统一规范化字符串 EM/F1，不等价于 2Wiki 的实体别名/关系三元组官方完整指标；对 2Wiki 官方实体别名评测应另跑官方评测器。

错误调用保留逐题行与失败计数；无法观测的质量字段留空。题目级失败的 QA（配置回答器时）及覆盖计为 0，避免通过删掉失败题提高分数。均值先在每种子内按 question family 汇总，再跨种子汇总；单种子不报假标准差。配对 bootstrap 以问题族为重采样单位。

重复调用诊断不缓存模型输出。K 扩展采用同一次最大候选池的前缀，记录 exact_k。效率运行显式提高逻辑总 token guard，但保持每分支上下文上限并记录修改；超限和 OOM 都是实验观测，不以改变候选集合来隐藏。

独立测试包的 3 个无关 context 必须逐条填写 verified、reviewer、rationale；有效变化必须更改已选证据/有效条件/问题，且有核验标签。不得为前后候选集不同的真实变化强算 JS 不变。测试题族从 train/validation 中隔离。

ViDoRe 单列 OCR case study。页面 qrels 只评价页面排序，充分性 gold 留空。source_registry 将 LightRAG 归一化文件名还原成官方页面 URL，保留页面来源及 OCR 摘录。成功与失败均从真实返回结果选择，不预编案例。

## 显存与运行边界

默认分支 microbatch=1、有效 batch 用梯度累积、FP32 可训练头、BF16 发布骨干、LoRA、checkpointing 和离线参考缓存。实测 allocated/reserved 及峰值写入日志；79GiB 为检查阈值。A100 80GB 是否满足还需在实际 4B/FLA 路径跑代表性最长状态和轨迹确认；本地 tiny 的显存数不能外推。

超长输入不静默截断。发生 token guard 或显存错误时先检查长度、候选分布与预算；只能使用不改变全候选归一化、梯度路径或状态语义的执行设置。当前没有为性能而引入 detach、候选 top-k 截断、基模替换或多卡方案。
