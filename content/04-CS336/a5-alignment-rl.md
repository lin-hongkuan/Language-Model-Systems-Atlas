---
title: A5 Alignment and Reasoning RL：提示、奖励与 GRPO
description: 原创中文导读，解释 CS336 A5 的提示基线、策略梯度、组相对优势和离策略权衡。
tags:
  - CS336
  - A5
  - alignment
  - reinforcement-learning
---

预训练语言模型可以根据上下文继续生成 token。Reasoning RL 的目标是用可验证任务上的奖励，让模型更常生成能得到高分的回答。A5 从 zero-shot/few-shot/chain-of-thought prompting 的基线开始，再讲策略梯度和 GRPO，最后讨论若干估计器变体与 off-policy 数据复用。

本页给出概念框架和符号，不包含作业推导结果、实现代码或实验答案。请以 [Stanford A5 handout](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf)、[官方作业仓库](https://github.com/stanford-cs336/assignment5-alignment)和当前课程说明为准。

> [!warning] 作业与 AI 工具
> Spring 2026 A5 handout 允许 AI 解释概念或查询 API 文档，但禁止 AI 工具实现作业任一部分。此页是原创的概念讲解，不提供作业代码或解答。

## 1. 提示基线：先分清“改变上下文”和“更新参数”

Zero-shot 直接给任务说明；few-shot 在上下文中添加示例；chain-of-thought prompting 要求模型给出推理过程。它们改变输入上下文或生成格式，不一定改变模型参数。比较不同提示时，需要固定模型、解码配置和评估集，并明确评分器怎样从输出中提取答案。

在 A5 的 GSM8K 设置中，自动评估需要解析模型最终答案。一个回答可能推理过程正确但格式解析失败，也可能被格式解析器判为正确而推理文本有问题。把准确率、答案格式、解析失败、输出长度和代表性样例一起看，避免将 parser 的行为误认为模型能力本身。

## 2. 语言模型作为策略

把 prompt $x$ 看作初始状态。模型按自回归方式生成 token $y_1,\ldots,y_T$，每一步根据已有上下文给出下一个 token 分布：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}
\pi_\theta(y_t\mid x,y_{<t}).
$$

其中 $\theta$ 是模型参数。对完成序列 $y$ 的奖励记为 $R(x,y)$；在可验证任务里，奖励可以来自答案是否通过规则或测试。目标是提高 prompt 分布下的期望奖励：

$$
J(\theta)=\mathbb{E}_{x\sim\rho,\ y\sim\pi_\theta(\cdot\mid x)}[R(x,y)].
$$

REINFORCE 的基本梯度形式为

$$
\nabla_\theta J(\theta)=
\mathbb{E}\left[
R(x,y)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\right].
$$

直觉是：若一个回答得到较高奖励，就提高生成这条轨迹中 token 的概率；奖励低则相对降低它们的概率。这种采样目标的方差可能很大，特别是模型回答长度不同、奖励稀疏或同一 prompt 下结果随机性强时。

## 3. 基线与组相对优势

可以从奖励中减去一个不依赖当前动作的基线 $b(x)$，帮助降低估计方差而不改变期望梯度：

$$
\nabla_\theta J(\theta)=
\mathbb{E}\left[
(R(x,y)-b(x))\sum_t\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})
\right].
$$

GRPO 对同一 prompt 采样一组回答 $y_1,\ldots,y_G$，并在组内比较奖励。例如组相对优势可写为

$$
A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+\epsilon}.
$$

减去组均值让“高于同组平均”的回答为正、“低于平均”的为负；除以标准差又改变了不同 prompt 组之间的权重尺度。实现和变体会对标准差归一化、序列长度归一化与 token 聚合采取不同做法；这些选择改变优化目标或样本权重，不能只当成无影响的代码细节。

若一组样本得到相同奖励，组内标准差接近零，优势的数值处理就很重要。此时该组能提供的相对排序信号也很少。分析日志时应同时记录奖励分布、每 prompt 的成功比例和优势分布。

## 4. On-policy 与 off-policy：生成成本和分布偏差

On-policy 更新使用当前策略 $\pi_\theta$ 生成的响应。生成昂贵，但梯度所依据的样本分布与当前策略一致。Off-policy 方法会在一批响应上进行多次更新，生成成本更易摊薄；但响应来自较旧策略 $\pi_{\mathrm{old}}$，与当前策略的分布逐步拉开。

常用重要性比率为

$$
r_t(\theta)=
\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_{\mathrm{old}}(y_t\mid x,y_{<t})}.
$$

比率用于校正采样策略和当前策略之间的差异，但对极端比率进行重加权会增加梯度估计方差。PPO/GRPO 类方法常采用 clipping 限制比率对目标的影响，以换取更稳的更新；这同时带来偏差，且裁剪过紧可能压低有效学习信号。

因此 off-policy 的权衡不只是“更快或更慢”：要比较 rollout 生成用时、训练用时、总壁钟时间、响应复用次数、验证奖励、更新稳定性和不同随机种子的波动。若只看训练 step 数或训练 reward，可能错过生成成本和分布漂移。

## 5. 训练与评估中的实用问题

- **答案奖励是什么？** 解析器、测试或规则是否正确区分格式问题与内容错误？奖励来源是否可验证？
- **采样单位是什么？** 一批里有多少 prompts、每个 prompt 几条响应？batch 维度是回答数还是问题数？
- **mask 对齐正确吗？** prompt token 与 response token 哪些参与语言模型 loss？response mask、padding 和终止 token 是否一一对应？
- **归一化改变了什么？** 组内标准化、序列长度归一化、token/sequence loss aggregation 各自如何影响不同长度的回答？
- **更新有多陈旧？** off-policy 时当前策略与 rollout policy 的 log-prob 比率分布如何变化？clip 前后有多少 token 被裁剪？
- **结果是否稳健？** 多次运行的验证奖励方差多大？是否出现 reward hacking、输出变长但准确率不变、格式退化或灾难性遗忘？
- **效率是否按端到端计算？** 计入模型生成、同步权重、训练、验证和数据解析后，总时间与显存是什么？

A5 的推理链样例有助于分析行为，但并不构成对内部推理真实性的保证。奖励指标和自动评分器都可能被利用；报告任务分数时应同时抽查输出并说明评估器局限。

## 6. A5 与课程后训练主题的边界

官方课程 L15 介绍 SFT/RLHF，L16 聚焦 RLVR，L17 涉及 alignment 与 multimodality；A5 必做 handout 当前主要研究 prompting、GSM8K 上的 reasoning RL/GRPO 与估计器变体。课程主题更宽，作业不是这些主题的完整覆盖。handout 还链接了可选补充材料；不要把可选内容误写成必做 A5 范围。

Spring 2026 handout 的实验对象是 OLMo-2-0425-1B 与 GSM8K。作业从 zero-shot/few-shot/CoT 提示基线开始，然后研究 on-policy GRPO；之后比较 RFT、Dr. GRPO、MaxRL 等估计器选择，并考察 off-policy 更新和重要性比率处理。此处只概括作业主题，不预设哪种方法会获得更好的结果。

## 课程与笔记入口

- 官方相关讲次：L12 Evaluation、L15 Mid/post-training、L16 RLVR、L17 Alignment and multimodality。讲次与作业是两套编号，见 [[course-map|官方课程地图]]。
- [A5 官方 handout 与仓库](https://github.com/stanford-cs336/assignment5-alignment)
- 第三方、MIT 许可的 Hands-on-LLM 笔记文件 `05_L15-L17_对齐_GRPO_多模态.md` 把多讲主题放入一个阅读单元；这里的范围号是笔记文件分组，不是 A5 小节或官方课号。见[笔记目录](https://github.com/FRS2003/hands-on-llm/tree/main/01-foundations/notes)。
