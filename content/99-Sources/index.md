---
title: 来源与版本边界
description: Stanford 课程版本、中文教材来源、本站内容与许可边界。
tags:
  - sources
  - licenses
---

本图谱将课程原始资料、中文教材和个人笔记分开标记。本站公开页面包含独立编写的中文说明、原创示意图和来源链接；可下载阅读不等于允许把课程材料或第三方笔记重新发布。

## Stanford 原始课程

| 课程 | 版本 | 主要来源 | 本站如何使用 |
|---|---|---|---|
| CS224N | Winter 2026 | [官方课程主页](https://web.stanford.edu/class/cs224n/) | 课程编号、主题和讲义核对；作业只链接题面与官方 starter |
| CS336 | Spring 2026 | [官方课程主页](https://cs336.stanford.edu/) | A1 实现范围与系统课程地图核对 |

Stanford 讲义和作业保留原作者/学校来源。网站不会把这些文件标成本站原创；公开发布前应沿用官方 URL，并确认所用副本的许可范围。

## 中文教材

| 资料 | 适合内容 | 许可说明 |
|---|---|---|
| [《动手学深度学习》中文 PyTorch 版](https://zh.d2l.ai/) | 张量、自动微分、反向传播、RNN、Word2Vec、注意力和 Transformer | 源码仓库标注 Apache-2.0；教材正文、插图和 PDF 的适用许可应以教材网站及具体文件声明为准；本站只作链接 |
| [Think Python 中文版](https://github.com/apachecn/think-py-2e-zh) | Python 编程基础与练习 | 随包 LICENSE 文件写 CC BY-NC-SA 4.0；README 的许可版本描述不一致，以正式 LICENSE 为准 |
| [Datawhale《深入浅出 PyTorch》](https://github.com/datawhalechina/thorough-pytorch) | Tensor、autograd、模型、损失、优化器、训练评估 | 随包 LICENSE 为 CC BY-NC-SA 4.0 |
| [Datawhale Happy-LLM](https://github.com/datawhalechina/happy-llm) | Transformer、LLM、训练与应用概览 | CC BY-NC-SA 4.0 |
| [邱锡鹏《神经网络与深度学习》](https://github.com/nndl/nndl) | 神经网络、优化、RNN、Transformer、LLM | 作者提供官方 PDF 阅读；仓库未见允许再分发的开放许可 |
| [邱锡鹏《大模型与智能体》](https://github.com/nndl/llm-beginner) | 预训练、后训练、推理、Agent 与评估 | 电子稿随作者组织 MIT 仓库提供；公开引用请保留来源与版本信息 |
| [FRS2003《Hands-on LLM · 动手学大模型》](https://github.com/FRS2003/hands-on-llm/tree/a828c96e683270a784d5d57e8aed06d3c9f9a5d3) | CS336 中文精讲、从零训练、PEFT 框架、Agent 与 RAG | 固定到 2026-09-16 的主分支提交；MIT 内容镜像与逐项排除说明见 [[../06-Hands-on-LLM/index|开源中文实战手册]] 和该目录 LICENSE |

## CS224N 中文补充

- [ApacheCN CS224N 中文笔记（2019）](https://github.com/apachecn/stanford-cs224n-notes-zh)：仅覆盖部分旧版课次，CC BY-NC-SA 4.0；不等于 CS224N 2026 完整翻译。
- [CS224N 2026 第三方学习笔记](https://www.dihengye.cn/cs224n/)：可在线读到 L02-L14 的中文讲解。原站没有找到明确开放许可或 Stanford 授权声明；本机离线快照和课件缩略图仅供个人学习，不随本站发布。

## Quartz 与部署

- Quartz 项目代码遵循仓库根目录中的 MIT LICENSE；该许可不覆盖本站独立撰写的讲义和原创示意图。
- 本站公式在构建时由 MathJax 转成页面内 SVG，浏览器不需从 CDN 加载数学引擎。
- 字体使用系统字体回退，不依赖 Google Fonts。
- 本机私人资料副本和许可未确认的第三方笔记快照位于发布目录之外，并由 Git 忽略规则和部署流程共同排除。
- [[../06-Hands-on-LLM/index|Hands-on LLM 开源材料区]]遵循 FRS2003 上游 MIT License；其第三方数据和模型卡按各自来源/许可逐项处理，不会套用根目录 MIT。
- 本站讲义与原创图示目前没有另行声明开放许可；转载或改编前请联系作者取得授权。

## 离线资料位置

未获公开许可的 PDF、starter、中文 HTML 教材和个人学习用第三方笔记快照，仅保存在维护者本机的私人资料目录中，网站读者无法访问。本网站另有独立列出、带完整 MIT License 的 Hands-on LLM 中文讲义镜像；它与本机私有快照不是同一批资料。本站不提供那些私人离线副本下载。
