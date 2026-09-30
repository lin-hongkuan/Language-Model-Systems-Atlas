---
title: Hands-on LLM 原始讲义目录
description: 固定到上游 MIT 版本的五个中文模块、Word 附件、实验图和知识自测入口。
tags:
  - open-source
  - hands-on-llm
---

此处保留原仓库的目录和文档。先看[上游总览](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/readme)，再按[学习路径与使用指南](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/learning_path)选择模块。讲义正文的知识编号、实验环境和实验数字，均以固定的 [上游提交](https://github.com/FRS2003/hands-on-llm/tree/a828c96e683270a784d5d57e8aed06d3c9f9a5d3) 为准。

## 五个模块

| 模块 | 范围 | 模块导览 |
|---|---|---|
| 01 · 理论基础 | CS336 L01–L17 中文精讲：分词、Transformer、MoE、GPU、Scaling、推理、对齐、多模态 | [打开模块导览](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/readme) |
| 02 · 从零训练 | 原生 PyTorch 组件、预训练、SFT、LoRA、DPO、GRPO/RLVR 和实验对照 | [打开模块导览](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/readme) |
| 03 · PEFT 框架 | 微调与对齐选型、源码导读、分布式训练与工程记录 | [打开模块导览](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/03-peft-framework/readme) |
| 04 · Agent | 工具调用、ReAct、记忆、上下文压缩、多智能体与安全 | [打开模块导览](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/04-agent/readme) |
| 05 · RAG | 检索、融合、重排、问答评估与应用编排 | [打开模块导览](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/05-rag-application/readme) |

## 附录、附件和原作图

- [术语表、论文清单和核心知识自测](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/appendix/)
- 原始 `.docx` 文件和 CSV/TXT/log 实验记录留在各自模块的原路径，可从讲义中的附件链接下载。
- 本站收录了仓库中讲义引用的 13 处本地图片；DPO/GRPO、GQA/MQA 和训练曲线等图片均沿用作者图表。
- 数据集样本、脚本和模型配置保留在 [上游仓库](https://github.com/FRS2003/hands-on-llm/tree/a828c96e683270a784d5d57e8aed06d3c9f9a5d3)，本镜像不重打包代码与训练样本。
- 完整授权文本：[MIT License](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/license)。

本目录有 50 篇获 MIT 许可的中文 Markdown 教材、27 份配套 Word 文档、10 张作者实验图表，以及仓库自带的记录文件。三个 `license: other` 模型卡没有被镜像；README 和模块页面中的来源说明保留，点击后可在上游查看。
