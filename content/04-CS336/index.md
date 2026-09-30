---
title: CS336：从零构建语言模型
description: CS336 官方课程路线、A1 概念导读与相关学习材料入口。
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

## 推荐阅读顺序

1. 先看 [[course-map|课程地图]]，了解课程从基础模型到训练系统、数据和后训练的顺序。
2. 阅读 [[a1-basics|A1 导读]]，建立从文本到下一个 token 的完整图景。
3. 以官方 A1 handout 和仓库作为唯一实现规范；按章节阅读原讲义、自己推导并实现。

## 官方入口

- [CS336 官方课程主页](https://cs336.stanford.edu/)
- [Assignment 1: Basics 仓库与讲义](https://github.com/stanford-cs336/assignment1-basics)
- [Assignment 2: Systems 仓库与讲义](https://github.com/stanford-cs336/assignment2-systems)
- [Assignment 3: Scaling 仓库与讲义](https://github.com/stanford-cs336/assignment3-scaling)
- [Assignment 4: Data 仓库与讲义](https://github.com/stanford-cs336/assignment4-data)
- [Assignment 5: Alignment and Reasoning RL 仓库与讲义](https://github.com/stanford-cs336/assignment5-alignment)

> [!warning] A1 的 AI 使用规则
> Spring 2026 官方 A1 handout 允许 AI 帮助解释高层概念或查阅低层 API 文档，但不允许 AI 工具或自动补全实现作业的任何部分。本目录提供概念说明、符号和阅读路线，不包含作业答案或实现代码。开始作业前请重新核对当前 handout 的政策，因为课程材料和政策可能更新。
