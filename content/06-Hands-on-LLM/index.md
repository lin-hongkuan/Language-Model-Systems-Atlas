---
title: Hands-on LLM：开源中文实战手册
description: 镜像 FRS2003/hands-on-llm 的 MIT 授权中文讲义与配套 Word、实验图表；从 CS336 理论到预训练、微调、Agent 与 RAG。
tags:
  - open-source
  - hands-on-llm
  - CS336
---

这里完整收录了 [FRS2003 / hands-on-llm](https://github.com/FRS2003/hands-on-llm) 在 **2026-09-16** 的公开版本。内容由原作者 Runsheng Feng（FRS2003）撰写，遵循上游仓库的 **MIT License**；本镜像保留了原目录结构、配套 Word 文档和实验图表，也保留了原作者与各文档注明的引用来源。

## 从哪个入口开始

- [上游 README：内容全景、能力索引和实验结果](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/readme)
- [学习路径：从原理到训练、框架、Agent 与 RAG](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/learning_path)
- [原始目录与全部讲义列表](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/)
- [上游 MIT License 全文](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/license.txt)

原始目录保留在 [镜像目录](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/) 内。目录中的图片、`.docx` 附件、CSV 实验表和文本记录与相邻讲义保持原有层级；从模块导览和讲义页可以直接打开。遇到代码、配置或数据链接时，会跳到上游固定版本，让示例保持可追溯。

## 五个学习模块

1. [CS336 理论基础](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/readme)：从分词、Transformer、MoE、GPU 与并行，读到 scaling、推理和对齐。
2. [从零训练语言模型](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/readme)：看原生 PyTorch 组件、训练流程、实验和结果曲线。
3. [PEFT 与工业训练框架](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/03-peft-framework/readme)：读微调方法、训练框架、分布式和部署的工程取舍。
4. [Agent 工程](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/04-agent/readme)：从工具调用、记忆与上下文，理解多步 Agent 系统。
5. [RAG 实战](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/05-rag-application/readme)：从混合检索与重排，走到带引用的问答应用。

附录位于[术语、论文和核心知识自测](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/appendix/)。各模块的 `.docx` 原文件保留在对应目录，适合下载后批注或打印。

## 镜像范围与许可

此处复制了上游仓库中 MIT 授权的中文 Markdown 教材、对应 Word 文档、作者制作的训练/评测曲线，以及讲义链接到的 CSV 与文本实验记录。原文保留其作者署名和来源；镜像不是 Stanford 官方讲义或官方翻译。

出于单独许可和材料追溯边界，本镜像**不打包训练数据样本 JSON/JSONL**。目录中的 `dataset_info.json` 仅说明训练数据字段映射，不包含样本；数据样本本身保留在上游链接。其中 ShareAI 偏好样本在上游标注为 Apache-2.0，Alpaca 中文样本的实际来源未固定。三个标记 `license: other` 的模型卡也只从上游查看，不复制进本站。上游明确排除的 Stanford slides、readings、assignments 以及私有医学语料均未收录。

上游版本：[`a828c96e683270a784d5d57e8aed06d3c9f9a5d3`](https://github.com/FRS2003/hands-on-llm/tree/a828c96e683270a784d5d57e8aed06d3c9f9a5d3)。更新上游以后，可按新版本重新同步此目录。本站 `source/LICENSE.txt` 保留完整 MIT 文本及版权/免责声明；引用本模块时请同时注明上游仓库和作者。

为遵守本站不显示学习时长的要求，上游学习路径中的周次安排表在本地网页镜像中略去，其余路线说明保留。
