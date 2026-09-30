---
title: Language Model Systems Atlas
description: 语言模型系统图谱：从 Python 与 PyTorch 基础，到 NLP 表示、Transformer、训练系统和模型评估。
tags:
  - home
  - language-modeling
---

> **Language Model Systems Atlas · 语言模型系统图谱**<br>
> NLP · 表示学习 · Transformer · 训练与评估

这里整理学习 CS224N 与 CS336 所需的核心概念、中文参考资料、公式和实现路线。内容按主题组织，读者可从自己的基础开始，不依赖视频或固定日程。

## 学习入口

- [[01-Foundations/index|基础工具：Python、数组与 PyTorch]]
- [[02-Learning-Map/index|主题地图：从表示学习到语言模型系统]]
- [[03-CS224N/index|CS224N：深度学习自然语言处理]]
- [[04-CS336/index|CS336：从零构建语言模型]]
- [[05-Reference/Glossary|公式、张量形状与术语速查]]
- [[99-Sources/index|资料来源、版本与许可说明]]

## 主线

1. **把代码跑起来**：Python、Tensor、自动微分、训练循环。
2. **理解语言表示**：共现、Word2Vec、损失函数和反向传播。
3. **读懂序列模型**：RNN 语言模型、注意力、Transformer。
4. **亲手搭建小模型**：分词、因果注意力、优化器、训练、保存和生成。
5. **扩展到现代系统**：预训练、对齐、PEFT、RAG、评估、推理和多语言分词。

## 作品目标

完成一个可在小型文本语料上训练的 decoder-only Transformer。它应能：

- 将文本编码为 token ID，并能解码回来；
- 接受形状清楚的 batch，预测下一个 token；
- 计算损失并执行一次优化器更新；
- 保存与恢复 checkpoint；
- 从提示逐 token 采样生成文本。

这个作品用于理解模型的数据流与训练闭环，不以参数规模或 benchmark 分数衡量。

## 资料约定

- Stanford 原讲义和作业保留官方链接、课程编号与版本。
- 开源教材按其许可证保留署名和来源。
- 第三方 CS224N 2026 中文笔记快照目前仅用于本地个人阅读；来源页没有提供明确再发布许可，不应提交到公开 GitHub Pages。
- 页面公式在构建时由本地 MathJax 生成 SVG；原创示意图使用本地 SVG 文件，并配有中文替代文本和图注。
