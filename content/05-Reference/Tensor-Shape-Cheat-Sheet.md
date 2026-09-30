---
title: 张量形状速查表
description: 语言模型中常见张量的轴含义、变换规则和检查方法。
tags:
  - reference
  - tensors
---

## 约定

本站使用 batch-first 布局。<code>B</code> 是批次大小，<code>T</code> 是序列长度，<code>V</code> 是词表大小，<code>D</code> 是模型隐藏维度，<code>H</code> 是注意力头数。标准多头注意力中 <code>d_h = D/H</code>。实现也可能采用 <code>[T,B,D]</code> 等排列；动手计算前先看代码。

## Transformer 主干

| 张量 | 常见 shape | 轴含义 |
|---|---|---|
| 输入 token id | <code>[B,T]</code> | 批次、序列位置；元素是整数 id |
| 目标 token id | <code>[B,T]</code> | 每个位置对应的下一个 token 标签 |
| Embedding 输出 | <code>[B,T,D]</code> | 每个 token 的 D 维向量 |
| 位置向量 | <code>[1,T,D]</code> 或 <code>[T,D]</code> | 位置轴和特征轴；可广播到 batch |
| 残差流 / 隐状态 | <code>[B,T,D]</code> | 每个位置的模型表示 |
| Q、K、V（分头后） | <code>[B,H,T,d_h]</code> | 批次、头、位置、单头通道 |
| 注意力分数 | <code>[B,H,T,T]</code> | query 位置对 key 位置的分数 |
| 因果掩码 | <code>[1,1,T,T]</code> 或 <code>[T,T]</code> | 允许或屏蔽的位置对；布尔语义依 API |
| 注意力输出（分头） | <code>[B,H,T,d_h]</code> | 每头汇总后的 value 表示 |
| 合并头后 | <code>[B,T,D]</code> | 头维并入模型通道维 |
| FFN 中间激活 | <code>[B,T,D_ff]</code> | 每个 token 的前馈层中间维度 |
| 模型 logits | <code>[B,T,V]</code> | 每个位置对词表的未归一化分数 |
| 交叉熵目标 | <code>[B,T]</code> 或 <code>[B*T]</code> | 每个位置的类别 id |
| 归约后的 loss | <code>[]</code> | 一个标量 |

Scaled dot-product attention 的核心维度：

$$
Q,K \in \mathbb{R}^{B \times H \times T \times d_h},
\qquad
QK^\top \in \mathbb{R}^{B \times H \times T \times T}.
$$

除以 $\sqrt{d_h}$ 后加掩码，沿 key 位置轴做 softmax，再乘以 $V$，得到 <code>[B,H,T,d_h]</code>。合并头后回到 <code>[B,T,D]</code>。

## 三条检查规则

1. 矩阵乘法检查中间轴：<code>[m,n] @ [n,k]</code> 得 <code>[m,k]</code>。
2. 对 <code>[B,H,T,T]</code> 的注意力分数，softmax 通常沿最后一个 key 轴；以当前实现为准。
3. 交叉熵把类别轴作为类别维。对 logits <code>[B,T,V]</code>，常展成 <code>[B*T,V]</code>，目标展成 <code>[B*T]</code>。

~~~python
logits.shape      # [B, T, V]
targets.shape     # [B, T]
loss = F.cross_entropy(
    logits.reshape(B * T, V),
    targets.reshape(B * T),
)
~~~

如果批次中有 padding，损失需忽略对应位置，统计平均 loss 时也要明确分母是否排除了 padding。

## 打印检查

~~~python
def describe(name, tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, device={tensor.device}"
    )
~~~

至少检查输入、embedding 输出、每个 Transformer block 的输入/输出、logits 和目标。相关概念见 [[NumPy-and-Tensor-Shapes|NumPy 与张量形状]]，缩写见 [[Glossary|术语表]]。
