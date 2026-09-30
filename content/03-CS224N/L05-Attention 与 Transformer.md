---
title: L05：Attention 与 Transformer
description: 用小例子说明 Query、Key、Value、缩放点积注意力、因果 mask、多头拆分和 Transformer block 的张量形状。
tags:
  - CS224N
  - attention
  - transformer
  - LLM
---

## 为什么需要 Attention？

处理句子中的一个位置时，模型通常需要参考句子里的其他位置。RNN 逐步传递隐藏状态，较远的信息要穿过许多递归步骤；自注意力则让每个位置直接对可见位置计算权重，再汇总它们携带的信息。

Attention 的关键不是“所有位置都一样重要”，而是根据当前查询动态分配权重。同一个词在不同上下文里可以从不同位置读取信息。

## Query、Key、Value 的计算

设输入隐藏状态为

$$
X\in\mathbb{R}^{B\times T\times d_{\text{model}}},
$$

其中 $B$ 是 batch 大小，$T$ 是序列长度。通过三组线性变换得到

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

可以把 Q 想成“当前位置要找什么”，K 想成“每个位置提供什么索引特征”，V 则是“被找到后要取出的内容”。这是帮助理解的比喻：真正参与计算的是投影后的向量和矩阵乘法。

单头缩放点积注意力为

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\!\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V.
$$

softmax 沿 Key 的位置维归一化。$M$ 是可选的 mask；不遮挡时可以省略。除以 $\sqrt{d_k}$ 是为了控制点积随向量维度增大时的尺度，避免 Softmax 过早集中在单个位置。

### 一个两位置的手算例子

把每个 Query 和 Key 简化成标量，令 $q=1$，两个 Key 为 $k_1=1,k_2=2$，对应 Value 为 $v_1=10,v_2=20$。先不考虑 mask 和缩放，logits 为 $[1,2]$。Softmax 权重为

$$
[\alpha_1,\alpha_2]
=\operatorname{softmax}([1,2])
\approx[0.269,0.731].
$$

输出是加权和

$$
\alpha_1v_1+\alpha_2v_2
\approx 0.269\cdot10+0.731\cdot20
=17.31.
$$

第二个位置的 Key 与 Query 更匹配，因此对应 Value 对结果影响更大。

## 因果遮罩：不能偷看未来

自回归语言模型在位置 $i$ 只能读取位置 $j\le i$。因果 mask 可写成

$$
M_{ij}=
\begin{cases}
0,&j\le i,\\
-\infty,&j>i.
\end{cases}
$$

加到 logits 后，未来位置经 Softmax 得到零权重。以长度为 4 的序列为例，可见关系是下三角结构：

$$
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}.
$$

这里的 1 表示允许关注，0 表示屏蔽。实际程序可能用布尔 mask，也可能在被屏蔽位置加极小值或 $-\infty$，要根据 API 约定确认其极性。

![四个序列位置的因果注意力可见矩阵原创示意图](assets/causal-attention.svg)

*图：对角线及左下区域可见，右上角代表未来位置，必须在 softmax 前屏蔽。*

## 多头注意力与完整形状

多头注意力把特征维拆成 $H$ 个头，每头维数通常为

$$
d_k=d_{\text{model}}/H.
$$

投影后将张量重排为

$$
Q,K,V\in\mathbb{R}^{B\times H\times T\times d_k}.
$$

矩阵乘法会在每个 batch、每个 head 内对序列位置配对：

| 中间量 | 形状 | 含义 |
|---|---:|---|
| $QK^\top$ | $[B,H,T,T]$ | 每个 Query 对每个 Key 位置的分数 |
| 权重 $\operatorname{softmax}(\cdot)$ | $[B,H,T,T]$ | 每行在可见位置上归一化 |
| 权重乘 $V$ | $[B,H,T,d_k]$ | 每个位置汇总 Value |
| 拼接各头 | $[B,T,Hd_k]$ | 合并头特征，常有 $Hd_k=d_{\text{model}}$ |
| 输出投影 | $[B,T,d_{\text{model}}]$ | 送入后续残差路径 |

不同头可以学习不同的投影与关注模式，但不能保证每个头都对应一个人类可命名的语言学功能。

## Transformer block 的两部分

一个常见 decoder block 包含自注意力子层和逐位置前馈网络。前馈网络对每个位置独立应用相同参数：

![带残差连接和归一化的 decoder Transformer block 结构示意图](assets/transformer-block.svg)

$$
\operatorname{FFN}(x)
=W_2\,\phi(W_1x+b_1)+b_2.
$$

隐藏维在中间通常先扩展再投影回来。残差连接把子层输入加回输出，归一化层帮助训练保持数值尺度。Pre-LayerNorm 结构的一种示意写法是

$$
x' = x+\operatorname{Attn}(\operatorname{Norm}(x)),
\qquad
y=x'+\operatorname{FFN}(\operatorname{Norm}(x')).
$$

不同模型会使用 RMSNorm、LayerNorm、不同激活函数和不同归一化位置；这些配置不是 Transformer 定义里唯一固定的一种实现。

位置编码为模型提供顺序信息。若没有位置线索，单靠对集合元素做注意力，模型难以区分“甲在乙前面”和“乙在甲前面”。可学习位置向量、正弦位置编码和 RoPE 是不同方案。

## 三类 Transformer 使用方式

- **Encoder** 通常允许一个输入位置查看整段输入，适合双向表示任务。
- **Decoder** 通常采用因果 mask，按自回归方式生成。
- **Encoder–decoder** 由编码器读取源序列，解码器生成目标序列，并可通过交叉注意力读取编码结果。

“Transformer”描述一类网络结构；“decoder-only 语言模型”是在这类结构上的一种具体安排。

## 容易混淆的地方

- **Attention 权重不是完整解释。** 它只显示这个子层中的权重分配，后续投影、残差和多层组合都会改变最终影响。
- **mask 的 0/1 约定取决于实现。** 先查 API 文档，再用一个长度为 3 的例子验证“位置 0 看不到位置 1、2”。
- **softmax 轴很关键。** 注意力权重应在 Key 的位置维归一化，而不是在 head 维或特征维。
- **形状能揭示错位。** 注意力分数最后两维应为 $[T,T]$，若得到 $[T,d_k]$，通常是转置或矩阵乘法轴错了。
- **多头不等于把完整模型复制 H 份。** 特征维会分到各头，每头通常只处理 $d_{\text{model}}/H$ 个通道。
- **因果遮罩是训练约束的一部分。** 如果未来 token 没被屏蔽，训练损失可能很好看，但生成时模型无法获得同样的信息。

## 自测

1. 给定 $Q:[B,H,T,d_k]$、$K:[B,H,T,d_k]$，注意力 logits 的形状是什么？
2. 除以 $\sqrt{d_k}$ 的目的是什么？
3. 长度为 3 时，位置 1 可以关注哪些位置？这里把位置编号从 0 开始。
4. 多头输出 $[B,H,T,d_k]$ 怎样变回模型维度？
5. 若 Softmax 沿 Query 维而不是 Key 位置维计算，会破坏什么性质？

<details>
<summary>核对要点</summary>

1. $[B,H,T,T]$。
2. 控制点积尺度，避免高维点积过大时 Softmax 过度饱和。
3. 位置 1 可关注位置 0 和 1，不能关注位置 2。
4. 转置/排列后合并 $H$ 与 $d_k$，得到 $[B,T,Hd_k]$，再做输出投影。
5. 每个 Query 对可见 Key 的权重不再按正确的 Key 轴归一化。

</details>

## 课程材料与版本

- 官方对应课程：Stanford CS224N Winter 2026，L05 Transformers。
- [官方 L05 课件 PDF](https://web.stanford.edu/class/cs224n/slides_w26/cs224n-2026-lecture05-transformers.pdf)；[课程主页与课表](https://web.stanford.edu/class/cs224n/)。
- 第三方中文补充：[L05 Attention 与 Transformer 原网页](https://www.dihengye.cn/cs224n/lectures/l05-transformers/)。该页面不是 Stanford 官方译本，公开再分发许可未确认；本站只链接原网页，未复制或托管其内容。

本页为独立中文讲解，不是 Stanford 官方译文。若课件中的符号与此页记号不同，以课件上下文为准。
