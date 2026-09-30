---
title: CS224N：从词向量到现代语言模型
description: 按 Stanford CS224N Winter 2026 课次组织的中文复习入口，覆盖导论、Word2Vec、反向传播、RNN、Transformer、项目方法和现代 NLP。
tags:
  - CS224N
  - NLP
  - deep-learning
---

这组页面用一条可追踪的主线串起自然语言处理：把词表示成向量，训练神经网络，再让模型预测序列，最后理解 Transformer 和现代大语言模型的训练、适配、评测与推理。

读本组内容不需要看视频。每页先讲问题和直觉，再写公式、张量形状和一个小例子；文末有容易混淆的地方与自测题。官方课件用于核对课程范围，页面文字是独立编写的中文解释，不是 Stanford 幻灯片翻译。

如果 Python 或 PyTorch 还不熟，先按 [[../01-Foundations/Python-Prerequisites|Python 基础]] → [[../01-Foundations/NumPy-and-Tensor-Shapes|NumPy 与张量形状]] → [[../01-Foundations/PyTorch-Train-Step|PyTorch 训练一步]] 阅读基础讲义，再进入 CS224N。开始时不要求会完整的 Python 语法或微积分证明；能读懂简单函数、看出向量和矩阵的 shape，并知道张量、损失和梯度大致是什么，就可以从 L01 开始。遇到代码或形状不熟，再回到对应基础页补齐。

## 页面导航

1. [[L01-课程导论与NLP演进|L01：课程导论与 NLP 演进]]：NLP 方法史与全课程地图。
2. [[L02-词向量与 Word2Vec|L02：词向量与 Word2Vec]]：共现、Skip-gram、Softmax 与负采样。
3. [[L03-神经网络与反向传播|L03：神经网络与反向传播]]：线性层、交叉熵、链式法则与梯度下降。
4. [[L04-语言模型与 RNN|L04：语言模型与 RNN]]：序列概率、下一个 token 预测、循环状态与困惑度。
5. [[L05-Attention 与 Transformer|L05：Attention 与 Transformer]]：Q/K/V、缩放点积注意力、因果遮罩与张量形状。
6. [[L06-期末项目与研究方法|L06：期末项目与研究方法]]：把研究问题变成受控、可复现的实验。
7. [[现代主题地图-L07-L14|现代主题地图：L07–L14]]：预训练、后训练、PEFT、RAG、评估、推理和多语言分词。

### L07–L14 中文讲义

- [[L07-预训练|L07 预训练]]
- [[L08-后训练|L08 后训练]]
- [[L09-高效适配与 PEFT|L09 高效适配]]
- [[L10-RAG 与语言 Agent|L10 RAG 与语言 Agent]]
- [[L11-评估与基准|L11 评估与基准]]
- [[L12-推理与解码（一）|L12 推理与解码（一）]]
- [[L13-推测解码与长上下文|L13 推测解码与长上下文]]
- [[L14-分词与多语言|L14 分词与多语言]]

## 课程编号与阅读顺序

以下编号使用 Stanford CS224N **Winter 2026 课程表**的课次，不是本目录的章节号。L01 有导论和 NLP 历史两份官方材料；L06 是期末项目方法课。

| 官方课次 | 主题 | 本地页面 |
|---|---|---|
| L01 | History of NLP / Introduction | [[L01-课程导论与NLP演进]] |
| L02 | Word Vectors | [[L02-词向量与 Word2Vec]] |
| L03 | Backpropagation and Neural Network Basics | [[L03-神经网络与反向传播]] |
| L04 | Language Models and RNNs | [[L04-语言模型与 RNN]] |
| L05 | Transformers | [[L05-Attention 与 Transformer]] |
| L06 | Final Projects: Custom and Default; Practical Tips | [[L06-期末项目与研究方法]] |
| L07 | Pretraining (Scaling, Systems, Data) | [[L07-预训练]] |
| L08 | Post-training (RLHF, SFT, DPO) | [[L08-后训练]] |
| L09 | Efficient Adaptation (Prompting + PEFT) | [[L09-高效适配与 PEFT]] |
| L10 | Agents, Tool Use, and RAG | [[L10-RAG 与语言 Agent]] |
| L11 | Benchmarking and Evaluation | [[L11-评估与基准]] |
| L12 | Reasoning 1 | [[L12-推理与解码（一）]] |
| L13 | Reasoning 2 | [[L13-推测解码与长上下文]] |
| L14 | Guest Lecture: Tokenization and Multilinguality | [[L14-分词与多语言]] |

每篇中文页都是围绕本讲主题独立编写的解释，不是课件逐页翻译。现代主题总览链接到对应 Stanford 课件；页面也标出官方课次与 PDF 文件名可能存在的版本编号差异。

官方课程主页还列出 L16 **Impact on Humanity** 和 L19 **Open Questions** 等拓展材料。这些内容不属于 L01–L14 的主线讲义；可从[官方课表](https://web.stanford.edu/class/cs224n/#schedule)打开当期材料。

## 建议的学习方式

1. 先看每页的问题定义和小例子，弄清模型要算什么。
2. 把公式中的每个符号写成具体形状，再核对矩阵乘法是否能对上。
3. 合上页面回答自测题；答不出时回到定义或用两三个 token 手算。
4. 再打开对应 Stanford PDF，核对本课强调的细节和符号。

不必一开始就记住公式。更重要的是能说清楚输入是什么、输出是什么、训练信号从哪里来，以及一个维度写错时会在哪里暴露。

## 范围与来源说明

课程编号、标题和官方材料链接依据 [Stanford CS224N Winter 2026 课程主页](https://web.stanford.edu/class/cs224n/)。本组中文文字为独立解释，未复制第三方中文笔记，也不声称是官方中文翻译。页面会分别标出官方课件与第三方资源，避免混淆来源。

第三方中文网页可作为补充阅读入口，但不是 Stanford 官方译本。其公开再分发许可未确认；本站只链接原网页，不复制或托管其内容。页面与本站解释不一致时，以 Stanford 官方课程材料为准。

- [第三方 CS224N L02 词向量笔记](https://www.dihengye.cn/cs224n/lectures/l02-word-vectors/)
- [第三方 CS224N L05 Transformer 笔记](https://www.dihengye.cn/cs224n/lectures/l05-transformers/)
