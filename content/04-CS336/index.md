---
title: CS336：从零构建语言模型
description: CS336 中文概念讲义与课程地图：从 tokenizer 和 Transformer 基础，逐步到 GPU 系统、缩放实验、训练数据与 GRPO。
tags:
  - CS336
  - language-modeling
  - transformer
---

Stanford CS336 的主线是理解并实现语言模型的完整生命周期：分词与模型结构、训练与推理、系统性能、缩放规律、训练数据，以及后训练和对齐。本目录先用 A1 把一个小型 Transformer 语言模型的关键组件串起来，再用课程地图定位后续内容。

## 本目录

- [[course-map|官方课程地图：L01–L17 与 A1–A5]]：按 Stanford Spring 2026 课程主页整理讲次和作业范围。
- [[a1-basics|A1 Basics：实现组件的概念导读]]：从 UTF-8/BPE 到 Transformer、优化器、训练与生成，包含张量形状和关键公式。
- [[a2-systems|A2 Systems：性能、显存与多 GPU 并行]]：从测量瓶颈到 checkpointing、FlashAttention、DDP 和 FSDP。
- [[a3-scaling|A3 Scaling Laws：计算预算如何分给模型和数据]]：理解 IsoFLOPs、缩放曲线和外推的边界。
- [[a4-data|A4 Data：从网页抓取到语言模型训练集]]：梳理抽取、过滤、去重和下游评估。
- [[a5-alignment-rl|A5 Alignment and Reasoning RL：从提示基线到 GRPO]]：理解奖励、策略梯度、组相对优势与离策略权衡。
- [[../06-Hands-on-LLM/index|Hands-on LLM 开源中文实战手册]]：原作者 MIT 授权的 CS336 精讲及预训练、对齐、Agent、RAG 实践笔记；完整目录见该入口。

## 怎样读这些讲义

每篇先用生活化问题解释“为什么有这个组件”，再逐步补术语、简化手算、关键公式与张量形状、端到端系统流程、取舍与常见错误，最后用带核对提示的练习自查。小例子用来学原理，不是官方作业题的解答。每篇也保留官方 handout、starter 与课程相关讲次入口；需要精确接口、规定和实验配置时，以课程当期资料为准。

| 如果你目前卡在…… | 建议先读 | 读完要能解释 |
|---|---|---|
| 文本、byte、token 分不清 | [[a1-basics|A1 Basics]] | 一段文本怎样变成 token ID，模型的预测为什么错开一位 |
| Transformer shape、loss、优化器很抽象 | [[a1-basics|A1 Basics]] | 从 $(B,T)$ 输入写到 $(B,T,V)$ logits，并算一个交叉熵 |
| 训练为何慢、GPU 为何 OOM | [[a2-systems|A2 Systems]] | 吞吐的计时边界、显存账本，以及重算/低精度/并行的代价 |
| “更大模型”该配多少数据 | [[a3-scaling|A3 Scaling Laws]] | 固定近似 FLOPs 预算比较 $N,D$，识别拟合和外推的边界 |
| 网页很多为何不能直接训练 | [[a4-data|A4 Data]] | 抽取、过滤、去重与混合如何改变覆盖和数据质量 |
| 模型如何从奖励中学习 | [[a5-alignment-rl|A5 Alignment and Reasoning RL]] | 用一组采样回答理解优势、概率比和 reward hacking |

如果 Python、线性代数或概率基础较弱，不必先背完所有公式：A1 的每个例子都可以手写 shape 或把概率代进负对数；遇到未知术语先看表格定义，再回到公式。A2–A5 会逐步复用这些基础。

## 推荐阅读顺序

1. 先看 [[course-map|课程地图]]，辨别课程讲次与作业编号；两者不是同一套编号。
2. 阅读 [[a1-basics|A1 导读]]，用 tokenizer、因果注意力和交叉熵建立从文本到下一个 token 的完整图景。
3. 按目标接着读 [[a2-systems|A2]] 到 [[a5-alignment-rl|A5]]；每章先写出自己的流程图或手算结果，再看核对提示。
4. 以官方 handout 和仓库作为作业接口、规则与实验配置的规范；按课程要求阅读原讲义、独立推导并实现。

## 官方入口

- [CS336 官方课程主页](https://cs336.stanford.edu/)
- [Assignment 1: Basics 仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment 2: Systems 仓库与讲义](https://github.com/stanford-cs336/assignment2-systems)
- [Assignment 3: Scaling 仓库与讲义](https://github.com/stanford-cs336/assignment3-scaling)
- [Assignment 4: Data 仓库与讲义](https://github.com/stanford-cs336/assignment4-data)
- [Assignment 5: Alignment and Reasoning RL 仓库与讲义](https://github.com/stanford-cs336/assignment5-alignment)

> [!warning] A1 的 AI 使用规则
> Spring 2026 官方 A1 handout 允许 AI 帮助解释高层概念或查阅低层 API 文档，但不允许 AI 工具或自动补全实现作业的任何部分。本目录提供概念说明、符号和阅读路线，不包含作业答案或实现代码。开始作业前请重新核对当前 handout 的政策，因为课程材料和政策可能更新。
