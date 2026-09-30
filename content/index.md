---
title: 从基础到语言模型系统
description: 语言模型系统图谱：从 Python 与 PyTorch 基础，到 NLP 表示、Transformer、训练系统和模型评估。
tags:
  - home
  - language-modeling
---

<section class="atlas-hero">
<p class="atlas-eyebrow">Language Model Systems Atlas · 语言模型系统图谱</p>
<p class="atlas-lead">从词如何变成向量，到 Transformer 怎样训练与推理。用清楚的中文解释、公式和练习，沿着一条完整主线学懂语言模型。</p>
<div class="atlas-topic-row"><span>NLP 与表示学习</span><span>Transformer</span><span>训练系统</span><span>模型评估</span></div>
</section>

这里整理 CS224N 与 CS336 的核心概念、中文讲解、公式速查和实现路线。页面按知识关系组织，可从 Python 与 PyTorch 基础开始，也可以直接进入熟悉的课程主题。

## 学习入口

<div class="atlas-card-grid">
<a class="atlas-card" href="./01-foundations/">
<span class="atlas-card-kicker">FOUNDATIONS · 01</span>
<strong>Python 与 PyTorch 基础</strong>
<span class="atlas-card-summary">补齐数组、张量形状、自动微分与训练循环。</span>
</a>
<a class="atlas-card" href="./02-learning-map/">
<span class="atlas-card-kicker">ROADMAP · 02</span>
<strong>学习地图</strong>
<span class="atlas-card-summary">沿着依赖关系，把基础知识接到语言模型系统。</span>
</a>
<a class="atlas-card" href="./03-cs224n/">
<span class="atlas-card-kicker">NLP · 03</span>
<strong>CS224N：语言与表示</strong>
<span class="atlas-card-summary">从 Word2Vec、反向传播和 RNN，走到 Transformer 与现代 NLP。</span>
</a>
<a class="atlas-card" href="./04-cs336/">
<span class="atlas-card-kicker">SYSTEMS · 04</span>
<strong>CS336：构建语言模型</strong>
<span class="atlas-card-summary">从 tokenizer 和模型组件，理解训练数据、GPU 系统与对齐。</span>
</a>
<a class="atlas-card" href="./05-reference/">
<span class="atlas-card-kicker">REFERENCE · 05</span>
<strong>公式、形状与术语</strong>
<span class="atlas-card-summary">快速核对注意力、损失函数、优化器和张量维度。</span>
</a>
<a class="atlas-card" href="./99-sources/">
<span class="atlas-card-kicker">SOURCES · 99</span>
<strong>来源与许可</strong>
<span class="atlas-card-summary">查看课程版本、官方材料和本站内容边界。</span>
</a>
</div>

## 一条连贯的学习主线

1. **搭好工具基础**：用 Python、NumPy 和 PyTorch 操作张量、计算梯度、训练小模型。
2. **弄懂语言表示**：从共现和 Word2Vec 看见词向量，再用损失与反向传播更新参数。
3. **读懂序列计算**：理解 RNN 如何传递状态，以及 Attention 怎样连接序列位置。
4. **构建 Transformer**：沿着 tokenizer、embedding、注意力、优化器和训练循环追踪数据流。
5. **进入现代模型系统**：继续学习预训练、对齐、PEFT、RAG、评估与推理。

## 最终实践

把各部分连成一个可在小型语料上训练的 decoder-only Transformer。检查它能否：

- 将文本编码为 token ID 并还原；
- 接收维度清楚的 batch，预测下一个 token；
- 计算损失、执行参数更新并保存 checkpoint；
- 从提示开始逐 token 生成文本。

重点是看懂模型的数据流与训练闭环，并能定位形状、梯度或数据管道中的问题。

## 资料约定

- Stanford 原讲义和作业保留官方链接、课程编号与版本。
- 开源教材按其许可证保留署名和来源。
- 未确认再发布许可的第三方笔记和课件不打包进网站；课程文件请从官方页面获取。
- 页面公式在构建时由本地 MathJax 生成 SVG；原创示意图使用本地 SVG 文件，并配有中文替代文本和图注。
- [[99-Sources/index|查看资料来源、课程版本与许可说明]]。
