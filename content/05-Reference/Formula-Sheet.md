---
title: 公式与张量形状速查
description: Word2Vec、softmax/交叉熵、RNN、causal attention、语言模型与优化的公式速查。
tags:
  - formula
  - shapes
  - math
---

公式在构建时由本地 MathJax 渲染为 SVG；页面不依赖数学引擎 CDN。长公式在窄屏上可以横向查看。这里保留的是定位公式和检查维度的摘要，完整推导见相应课程主题页。

## 记号

| 记号 | 含义 |
|---|---|
| $B$ | batch size |
| $T$ | token 序列长度 |
| $V$ | tokenizer 词表大小 |
| $D$ | Transformer 隐藏维度 |
| $H$ | attention head 数 |
| $d_h=D/H$ | 每个 head 的维度 |

## Word2Vec Skip-gram

中心词向量记作 $v_c$，上下文词的输出向量记作 $u_o$。点积越大，模型越倾向于认为二者来自真实上下文：

$$
P(o\mid c)=
\frac{\exp(u_o^\top v_c)}
{\sum_{w\in V}\exp(u_w^\top v_c)}.
$$

负对数似然鼓励真实上下文获得更高概率：

$$
\mathcal{L}_{\text{SG}}
=-\sum_{(c,o)\in\mathcal{D}}\log P(o\mid c).
$$

Negative sampling 将全词表归一化近似成少量正、负词对：

$$
\mathcal{L}_{\text{NS}}
=-\log\sigma(u_o^\top v_c)
-\sum_{i=1}^{k}\log\sigma(-u_{n_i}^\top v_c).
$$

前一项拉高真实词对的相似度，后一项压低噪声词对的相似度。

## Softmax 与交叉熵

logits 为 $z\in\mathbb{R}^{V}$：

$$
p_i=\operatorname{softmax}(z)_i
=\frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}.
$$

数值稳定实现会先减去最大分数 $m=\max_j z_j$，因为 softmax 对整体平移不变：

$$
p_i=\frac{\exp(z_i-m)}{\sum_{j=1}^{V}\exp(z_j-m)}.
$$

真实类别为 $y$ 时：

$$
\mathcal{L}_{\text{CE}}=-\log p_y.
$$

对于一个 batch 的 logits $Z\in\mathbb{R}^{B\times V}$，平均交叉熵对 logits 的梯度是：

$$
\frac{\partial\mathcal{L}}{\partial Z}
=\frac{P-Y_{\text{one-hot}}}{B}.
$$

## 自回归语言模型

因果语言模型用链式法则分解整个 token 序列的概率：

$$
P(x_1,\ldots,x_T)
=\prod_{t=1}^{T}P(x_t\mid x_{<t}).
$$

因此训练对齐采用输入和目标错开一位：

$$
\text{input}=[x_1,\ldots,x_{T-1}],\qquad
\text{target}=[x_2,\ldots,x_T].
$$

模型输出 logits 形状 $[B,T,V]$，target token ID 形状 $[B,T]$。对平均 token loss $H$ 的困惑度为：

$$
\operatorname{PPL}=\exp(H).
$$

只有在 tokenization、数据和评估口径可比时，PPL 才适合直接比较。

## Scaled dot-product attention

对一个 attention head：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

若输入是 $X\in\mathbb{R}^{B\times T\times D}$，拆成 $H$ 个 head 后：

$$
Q,K,V\in\mathbb{R}^{B\times H\times T\times d_h},
\qquad
S,A\in\mathbb{R}^{B\times H\times T\times T}.
$$

其中 $M$ 是 causal mask：

$$
M_{ij}=
\begin{cases}
0,&j\leq i,\\
-\infty,&j>i.
\end{cases}
$$

mask 必须在 softmax 之前加到分数上。每行 softmax 沿 key 的位置轴计算，使未来位置的概率严格为零。

## RMSNorm 与 SwiGLU

对隐藏向量 $x\in\mathbb{R}^{D}$，RMSNorm 的概念形式为：

$$
\operatorname{RMSNorm}(x)
=\frac{x}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}}\odot g.
$$

一种常见 SwiGLU 前馈形式为：

$$
\operatorname{SwiGLU}(x)
=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).
$$

两个分支先投影到中间维度 $D_{\mathrm{ff}}$，再逐元素门控，最后投回 $D$。

## AdamW

一阶与二阶移动平均为：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

偏差修正后，AdamW 将权重衰减与梯度更新解耦；精确更新、偏置项和超参数按课程 handout 实现：

$$
\theta_{t+1}
=\theta_t-\eta\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
+\lambda\theta_t
\right).
$$

## 前向 shape 检查

沿着一条 token 序列检查前向计算时，可以逐步核对下表中的形状。注意 attention 的分数和权重保留了 head 维度；各 head 的输出合并后才回到 $[B,T,D]$。

| 步骤 | 运算 | 张量形状 | 核对要点 |
|---|---|---|---|
| 1. Tokenize | token IDs | $[B,T]$ | 每个位置是一个词表索引 |
| 2. Embedding | 查表得到 $X$ | $[B,T,D]$ | 一个 token 对应一个 $D$ 维向量 |
| 3. 生成多头 Q/K/V | 线性投影并拆分 heads | $Q,K,V:[B,H,T,d_h]$ | $H d_h=D$ |
| 4. 计算注意力分数 | $S=QK^\top/\sqrt{d_h}$ | $[B,H,T,T]$ | 两个 $T$ 维分别表示 query 位置和 key 位置 |
| 5. 加因果 mask 并 softmax | $A=\operatorname{softmax}(S+M)$ | $[B,H,T,T]$ | $M$ 可广播到 batch 和 head；未来位置的权重为零 |
| 6. 加权求和并合并 heads | $AV$，拼接各 head 后投影 | $[B,T,D]$ | 每个 query 位置得到一个上下文向量 |
| 7. 词表投影 | hidden states 投影到词表 | logits $[B,T,V]$ | 每个位置对 $V$ 个 token 各有一个分数 |
| 8. Next-token loss | 与 target IDs $[B,T]$ 对齐后计算交叉熵 | 每 token 一个 loss，汇总后为标量 | target 通常是输入序列右移一位 |

排查维度时，可以沿表格从上到下追踪 $B,T,D,H,V$，再检查目标 token 是否与对应位置的 logits 对齐。

## 自查

- Linear 层为什么保持 batch 和序列两维，只改变最后一维？
- attention 分数为什么有两个长度为 $T$ 的维度？
- 哪一条测试能证明 causal mask 没让某位置看到未来 token？
- loss 不下降时，你会先检查 target 对齐、shape 还是学习率？为什么？
