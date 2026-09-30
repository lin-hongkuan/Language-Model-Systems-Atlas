---
title: 30 天学习路线：从 Python 到语言模型系统
description: 每天 2 小时，以中文讲义、手算和编程练习串起 Python/PyTorch、CS224N L01–L14、CS336 A1–A5 与 Hands-on LLM。
tags:
  - roadmap
  - curriculum
  - CS224N
  - CS336
---

本路线按 **30 天、每天 2 小时，共 60 小时**安排，从 Python/PyTorch 前置开始，衔接 CS224N L01–L14、CS336 A1–A5 与 Hands-on LLM。全程以中文讲义、官方文字 handout、手算、小程序和自测为主，不需要看网课视频。

这里的“学完”指讲义主题全部覆盖、关键机制能解释、各作业主题完成缩小版练习并留下可检查产物。官方完整作业、长时间 GPU 训练或大数据实验可能超过 60 小时；计划会读懂五项作业目标并做小规模练习，不把阅读导读说成已完成官方作业。开始官方作业前，先核对当期 handout 的规则和接口。

## 依赖路径

![知识从 Python、张量和梯度，逐步连接到词向量、序列模型、Transformer、训练与评估的主题依赖图](assets/learning-dependency.svg)

*图 1. Python、shape 和梯度是两门课的共同前置；CS224N 建立概念主干，CS336 将概念扩展为实现、系统、数据和对齐。*

推荐顺序：[[../01-Foundations/index|Python、NumPy、PyTorch]] → [[../03-CS224N/index|CS224N L01–L14]] → [[../04-CS336/course-map|CS336 讲次与 A1–A5 地图]] → [[../06-Hands-on-LLM/index|Hands-on LLM 五模块]]。Hands-on 理论、训练、PEFT、Agent、RAG 笔记分别在课程对应日穿插，用作工程案例和实验记录阅读材料。遇到术语可查 [[../05-Reference/Glossary|中英术语表]]、[[../05-Reference/Formula-Sheet|公式速查]]、[[../05-Reference/Tensor-Shape-Cheat-Sheet|shape 速查]] 和 [[../05-Reference/Debugging-Checklist|调试清单]]。

## 每日 120 分钟安排

每天固定为 **10 分钟闭卷回忆 + 45 分钟读中文讲义/文字 handout + 55 分钟手算或编程 + 10 分钟整理产物**。每日主任务总计 2 小时。建议把产物记在同一本笔记或 day-01 至 day-30 文件中。

- **验收产物**应可展示、可复查；“读完”不算通过。
- **遇阻回退**先完成更小的闭环，把未掌握点记下并放入次日回忆；必要时压缩加练。
- **可选加练**仅在通过当日验收后进行，不挤占主任务；官方作业依当期课程规则独立完成。

## 第一阶段｜Python、张量和训练一步（第 1–4 天）

| 天 | 2 小时主任务与材料 | 验收产物 | 遇阻回退 | 可选加练 |
|---|---|---|---|---|
| 1 Python | [[../01-Foundations/Python-Prerequisites|Python 前置能力]]；读列表/字典、循环、函数、切片和报错，写文本切词到 token id、输入/目标的小程序。 | 可运行程序、3 个自测；手算 [2,5,8,3] 的错位输入和目标。 | 两个词和小词表逐行追踪；推导式改普通循环。 | 为未知词实现报错与映射到预留 UNK 的策略。 |
| 2 NumPy/shape | [[../01-Foundations/NumPy-and-Tensor-Shapes|NumPy 与张量形状]]；练索引、广播、矩阵乘法、embedding 查表，标注 B/T/V/D 轴。 | 手算 2×3 乘 3×2；画 [B,T] 查 [V,D] 得 [B,T,D] 的 shape 流。 | 只跟讲义的小矩阵例，说明共享维度为何消失。 | 造小 batch 查表例，用 NumPy 验证 shape。 |
| 3 PyTorch | [[../01-Foundations/PyTorch-Train-Step|PyTorch 训练一步]]；CPU 跑 next-token 小 batch，追踪 logits、loss、backward、optimizer.step。 | 代码或运行记录标注 logits/labels shape；证明梯度有限且参数改变。 | 缩为 Linear→CrossEntropyLoss，分清求梯度与更新参数。 | 同一小 batch 训练数步、记录 loss 和排查过程。 |
| 4 前置检查 | 复习 [[../01-Foundations/index|基础讲义]]、[[../05-Reference/Tensor-Shape-Cheat-Sheet|shape 速查]]、[[../05-Reference/Debugging-Checklist|调试清单]]；闭卷画文本到参数更新全流程。 | 数据流图和 5 题自测（索引、shape、错位目标、loss、更新），至少 4 题能说明白。 | 缩到单条两 token；shape 回第 2 天，梯度回第 3 天。 | 做异常、shape/dtype/device 与修复原因速查卡。 |

## 第二阶段｜CS224N L01–L14（第 5–18 天）

| 天 | 2 小时主任务与材料 | 验收产物 | 遇阻回退 | 可选加练 |
|---|---|---|---|---|
| 5 L01 NLP 全景 | [[../03-CS224N/L01-课程导论与NLP演进|L01]]；追踪句子从 token、模型预测到目标和评估。 | NLP 路线图；任务、监督、输出、指标表，含分类和生成。 | 只解释三 token 例子里看到什么、预测什么、如何评价。 | 提出可测问题、基线和潜在评测泄漏。 |
| 6 L02 Word2Vec | [[../03-CS224N/L02-词向量与 Word2Vec|L02]]；学中心词/上下文、嵌入表、Skip-gram、负采样。 | 数据流图标 id、[B,D]、[B,V]、正负样本和目标；手算小词表概率。 | 只留两个中心词和两个上下文，暂略负采样。 | 用 PyTorch Embedding 验证玩具例的 shape。 |
| 7 L03 反传 | [[../03-CS224N/L03-神经网络与反向传播|L03]]；手算损失/梯度，连接链式法则、logits、softmax、交叉熵。 | 标注局部梯度含义的计算图；核对一次梯度方向，解释 batch 求和/均值。 | 退回一维参数和平方损失，判断参数和 loss 变化方向。 | 用有限差分与自动微分互核。 |
| 8 L04 RNN/LM | [[../03-CS224N/L04-语言模型与 RNN|L04]]；学概率分解、RNN 状态、teacher forcing、困惑度。 | RNN 展开图；4 token 错位输入/目标；负对数似然与 perplexity 小例。 | 先解释“历史状态预测下个 token”和条件概率，不算矩阵。 | 比较玩具语料 unigram 与 bigram 分布。 |
| 9 L05 Transformer | [[../03-CS224N/L05-Attention 与 Transformer|L05]]；学 Q/K/V、多头、causal mask、位置、残差、FFN，手算双位置注意力。 | 标注主要 shape 的 block 图；测试位置 0 不关注未来。 | 单头两个 token，依次算 QKᵀ、softmax 和 mask。 | 对照 RNN/Attention 的上下文顺序、计算和长依赖。 |
| 10 L06 研究方法 | [[../03-CS224N/L06-期末项目与研究方法|L06]]；定义问题、假设、基线、变量控制、切分、指标。 | 一页实验方案，含混杂因素和失败报告方式。 | 用“LoRA rank 是否影响效果”示范如何控制变量。 | 加入消融，记版本、种子和数据来源。 |
| 11 L07 预训练 | [[../03-CS224N/L07-预训练|L07]]；Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/01_L1-L3_分词_资源核算_Transformer|理论 L1–L3]]。比较 causal LM、MLM、encoder-decoder。 | 三种目标输入/预测位置/用途表；一组 next-token 标签。 | 只比较 causal 与 MLM 预测位置及是否看未来。 | 粗估参数、token、计算预算和忽略项。 |
| 12 L08 后训练 | [[../03-CS224N/L08-后训练|L08]]；Hands-on [[../06-Hands-on-LLM/source/02-train-from-scratch/notes/alignment|对齐推导]]。比较 SFT、偏好、RLHF、DPO。 | 训练流程图；方法、数据、优化对象表；一个目标漏洞。 | 先分清示范监督和成对偏好，再补奖励/策略更新。 | 用偏好对解释 DPO 方向，不训练模型。 |
| 13 L09 PEFT | [[../03-CS224N/L09-高效适配与 PEFT|L09]]；Hands-on [[../06-Hands-on-LLM/source/03-peft-framework/README|PEFT 模块]]。比较 prompt、LoRA、adapter。 | 小矩阵全量/低秩参数量及显存、适用条件对照。 | 手算 2×3 权重与 rank-1 更新。 | 比较 Full、LoRA、QLoRA 资源取舍。 |
| 14 L10 RAG/Agent | [[../03-CS224N/L10-RAG 与语言 Agent|L10]]；Hands-on [[../06-Hands-on-LLM/source/05-rag-application/docs/RAG入门导读|RAG 入门]]、[[../06-Hands-on-LLM/source/04-agent/docs/Agent架构自测问答|Agent 自测]]。 | 三段文档带来源问答轨迹；RAG/Agent 比较与评估项。 | 手工排序三段材料；证据不足就明确写不足。 | 分别设计召回与答案忠实度指标。 |
| 15 L11 评估 | [[../03-CS224N/L11-评估与基准|L11]]；手算混淆矩阵，学生成评估、污染、切片、不确定性。 | 最小评估协议：目标、基线、指标、切片、污染检查和不确定性。 | 十个样本标 TP/FP/FN/TN，再算 precision/recall。 | 为第 14 天设计检索和生成两阶段指标。 |
| 16 L12 解码/RL | [[../03-CS224N/L12-推理与解码（一）|L12]]；比较 greedy、temperature、top-k/top-p、多采样并手算温度影响。 | 两种温度概率表；区分解码采样与策略参数更新。 | 只算两个候选 softmax，不混淆 prompt、采样和训练。 | 四个回答奖励手算组内优势，预热 A5。 |
| 17 L13 推测解码 | [[../03-CS224N/L13-推测解码与长上下文|L13]]；Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/03_L9-L11_缩放定律_推理优化|缩放/推理]]。 | 首个拒绝位置的推测解码轨迹；注明假设的 KV cache 估算表。 | 只算一个草稿 token 接受率；解释朴素注意力随长度平方增长。 | 举加速/不加速场景并分析瓶颈。 |
| 18 L14 分词/多语 | [[../03-CS224N/L14-分词与多语言|L14]]；复习 Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/01_L1-L3_分词_资源核算_Transformer|L1–L3 分词]]；手做 BPE 合并。 | BPE 合并记录；中英文 token 数与上下文预算对比。 | 三个短字符串统计符号对并做一次合并。 | 说明分词如何影响预算、推理和多语公平。 |

## 第三阶段｜CS336 A1–A5（第 19–28 天）

作业入口见 [[../04-CS336/course-map|CS336 课程地图]]。每项两天：概念/最小实践，再到实验/系统判断。讲义不代替当期官方规范。

| 天 | 2 小时主任务与材料 | 验收产物 | 遇阻回退 | 可选加练 |
|---|---|---|---|---|
| 19 A1 BPE | [[../04-CS336/a1-basics|A1 Basics]] 与官方 handout；追踪 UTF-8、pair merge、encode/decode，手推或写极小 tokenizer。 | 原文→byte→token id→decode 表；三种字符串 round-trip。 | 先验证 byte 序列还原原文，再看 BPE 小例。 | 解释词表大小、merge 数和速度；按当前规则做小测试。 |
| 20 A1 Transformer LM | [[../04-CS336/a1-basics|A1]]；Hands-on [[../06-Hands-on-LLM/source/02-train-from-scratch/from_scratch/README|手写组件]]、[[../06-Hands-on-LLM/source/02-train-from-scratch/notes/training-pipeline-explained-zh|训练流程]]。 | 前向/更新图；标 tokenizer、attention、loss、优化器和 checkpoint I/O；测 mask 或错位断言。 | 退回第 3 天训练步，只追 shape 并画组件接口。 | 按课程 AI 政策实现并测一组件，记设备/batch。 |
| 21 A2 测量/资源 | [[../04-CS336/a2-systems|A2]]；Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/01_L1-L3_分词_资源核算_Transformer|资源核算]]；定义计时边界、估显存。 | 带单位/假设的账本；实验模板含设备、dtype、shape、warm-up、测量次数。 | 无 GPU 就纸面估算并标理论值。 | 同设备测两种序列长度或精度，记录波动。 |
| 22 A2 并行 | [[../04-CS336/a2-systems|A2]]；Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/02_L4-L8_注意力替代_MoE_GPU系统_Triton|系统并行]]、[[../06-Hands-on-LLM/source/03-peft-framework/notes/peft-distributed-quant|分布式选型]]。比较精度、checkpointing、FlashAttention、DDP/FSDP。 | 表格列瓶颈、节省项、额外计算/通信、适用条件、失败边界。 | 先比 checkpoint 重算换内存与 FlashAttention 减少搬运。 | 有双卡跑最小 DDP，否则估算且明确非实测。 |
| 23 A3 预算 | [[../04-CS336/a3-scaling|A3 Scaling Laws]]；解释参数、训练 token、计算预算与验证 loss，构造同预算配置。 | 预算近似、固定/变化变量、观测量和公平边界表。 | 只比较两种假设配置，数字标成估算。 | 设计短训大模型与长训小模型的选择指标。 |
| 24 A3 拟合 | [[../04-CS336/a3-scaling|A3]]；Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/03_L9-L11_缩放定律_推理优化|缩放笔记]]；分析拟合、残差、噪声、外推。 | 批注实验图或标明玩具数据的拟合图；说明外推风险。 | 逐点记 loss、圈异常并讨论影响。 | 做 log-log 小拟合，报告内插/外推与残差。 |
| 25 A4 抽取/过滤 | [[../04-CS336/a4-data|A4 Data]]；画网页记录到抽取、识别、过滤、规范化流程。 | schema 和八条自编样例过滤表，写决策与覆盖损失；不下载受限样本。 | 五条手写文本应用两条规则并说清误伤。 | 抽检阈值两侧样本，分析 precision/recall。 |
| 26 A4 去重/评估 | [[../04-CS336/a4-data|A4]]、CS224N [[../03-CS224N/L11-评估与基准|L11]]、Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/04_L12-L14_模型评估_训练数据|评估与数据]]。 | exact/near duplicate 判定；过滤前后评估方案，分清数量和质量。 | 六条样例含完全及近似重复，先定规则再分类。 | 抽检误删/漏删，解释删除率为何不够。 |
| 27 A5 GRPO | [[../04-CS336/a5-alignment-rl|A5]]；Hands-on [[../06-Hands-on-LLM/source/01-foundations/notes/05_L15-L17_对齐_GRPO_多模态|对齐笔记]]；手算组内奖励与优势。 | 玩具数值例和伪代码，标采样、奖励、优势、概率比、裁剪。 | 仅算三项奖励的中心化优势，复习 L12。 | 读实验曲线并追踪数据、步数、基线。 |
| 28 A5 实验 | [[../04-CS336/a5-alignment-rl|A5]]；Hands-on [[../06-Hands-on-LLM/source/02-train-from-scratch/notes/alignment|对齐推导]]、[[../06-Hands-on-LLM/source/02-train-from-scratch/experiments/README|实验记录]]。设计推理任务与防奖励投机。 | 协议含基线、奖励、独立评估、停止条件、资源和失败标准。 | 算术小例写奖励和作弊答案，再添检查，不训练模型。 | 实现组内优势 toy 函数并逐项手算核对。 |

## 第四阶段｜Hands-on LLM 与小项目（第 29–30 天）

| 天 | 2 小时主任务与材料 | 验收产物 | 遇阻回退 | 可选加练 |
|---|---|---|---|---|
| 29 Hands-on 实战 | [[../06-Hands-on-LLM/index|总览]]、[[../06-Hands-on-LLM/source/LEARNING_PATH|学习路径]]、[[../06-Hands-on-LLM/source/02-train-from-scratch/README|从零训练]]、[[../06-Hands-on-LLM/source/03-peft-framework/README|PEFT 框架]]。按 10/45/55/10 分钟节奏：回忆、映射五模块、拆一个实验、整理。 | 覆盖基础、从零训练、PEFT/框架、Agent、RAG 的映射表；详细拆解一个训练阶段的前置概念和指标。 | 只读总览/路线，五模块各写问题、先修、产出；不让安装占满时间。 | 做 LoRA 矩阵或小样本 toy demo，遵守许可与数据边界。 |
| 30 小项目/总复盘 | 复习 [[../03-CS224N/现代主题地图-L07-L14|现代主题图]]、[[../04-CS336/course-map|CS336 地图]]、Hands-on [[../06-Hands-on-LLM/source/04-agent/README|Agent]]、[[../06-Hands-on-LLM/source/05-rag-application/docs/RAG入门导读|RAG]]、[[../06-Hands-on-LLM/source/appendix/core-self-check|核心自测]]。用 3 段自写短文、3 个问题和 Python 关键词重合度，做一个无需外部 API 的极小检索问答程序。 | 程序为每个问题列出 top-2 文档 id/片段并附来源；手工核对 3 题的 Recall@2；另交系统图和“会解释/需复习/需实现”自测表。 | 先人工给 3 段短文排序并引用证据；至少说明 token、检索证据如何进入回答及何时应答“资料不足”。 | 将关键词重合替换为 BM25 或比较两种排序，记录误召回案例；重做一个最薄弱练习。 |

## 结束检查

- **Python/PyTorch：**读懂函数和 batch，解释关键 shape，完成含 loss、梯度、更新的一步训练。
- **CS224N L01–L14：**串起 Word2Vec、反传、RNN、Transformer、预训练、后训练、PEFT、RAG、评估、解码和分词；至少两个主题有手算或代码例。
- **CS336 A1–A5：**解释 BPE/LM 实现、系统测量、缩放实验、数据过滤去重和 reasoning RL 的决策与评估边界。
- **Hands-on LLM：**定位理论、训练、PEFT、Agent、RAG 笔记和实验记录，并从证据总结结论与局限。
- **学习记录：**每天至少一件可复查产物；未通过项有回退任务。

60 小时能覆盖讲义和缩小版练习，不能保证跑完全部官方作业、复现全部实验或精通 GPU kernel/多卡调优。要进一步独立实现，按最薄弱项继续官方 A1–A5：写明假设、跑最小实验、保存结果并核对局限。
