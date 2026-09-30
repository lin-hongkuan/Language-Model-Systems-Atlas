---
title: NumPy 与张量形状：先看维度，再看公式
description: 从数组、广播和矩阵乘法建立语言模型张量的形状直觉。
tags:
  - foundations
  - numpy
  - tensors
---

张量是带有多个轴的数值数组。读模型代码时，先问三个问题：值有几维？每个轴代表什么？需要相乘或相加的轴是否兼容？

## NumPy 数组与矩阵乘法

~~~python
import numpy as np

token_ids = np.array([[4, 8, 2], [1, 5, 9]])  # shape: (2, 3)
print(token_ids.ndim)   # 2
print(token_ids.shape)  # (2, 3)
print(token_ids.dtype)  # 整数类型
~~~

<code>(2, 3)</code> 表示两行、每行三个元素。在语言模型里可解释为两条序列、每条三个 token。

矩阵乘法要让中间维度对齐：

$$
(m \times n) @ (n \times k) \longrightarrow (m \times k).
$$

NumPy 和 PyTorch 中 <code>@</code> 表示矩阵乘法，<code>*</code> 表示逐元素乘法；二者不可混用。

## 广播

广播允许长度为 1 的轴参与逐元素运算，而不必手动复制数据：

~~~python
x = np.zeros((2, 3, 4))  # B=2, T=3, D=4
b = np.ones((1, 1, 4))  # 每个特征一个偏置
y = x + b                 # shape 仍为 (2, 3, 4)
~~~

从最右轴开始对齐：两轴相同，或其中一轴为 1，才可以广播。例如 <code>(2, 3, 4)</code> 与 <code>(3, 4)</code> 可相加；与 <code>(2, 5)</code> 通常不可相加。模型里的 <code>x + position_embedding</code> 若形状分别为 <code>[B,T,D]</code>、<code>[1,T,D]</code>，位置向量会广播到每条样本。

## 语言模型的常见轴

- <code>B</code>：批次中的样本数。
- <code>T</code>：每条样本的 token 数。
- <code>V</code>：词表大小。
- <code>D</code>：隐藏维度或 <code>d_model</code>。
- <code>H</code>：注意力头数；标准多头注意力常用 <code>d_h = D / H</code>。

输入 token id 通常为 <code>[B,T]</code>。Embedding 把每个整数映射为向量，变为 <code>[B,T,D]</code>。最后模型对每个位置输出词表分数，形成 <code>[B,T,V]</code>。

完整轴约定见 [[Tensor-Shape-Cheat-Sheet|张量形状速查表]]。

## 小练习

1. 创建形状为 <code>[2,4]</code> 的整数数组，表示两条长度为四的序列。
2. 创建 <code>[T,D]</code> 的位置向量，说明它如何加到 <code>[B,T,D]</code> 表示上。
3. 用 <code>[B,T,D]</code> 与 <code>[D,V]</code> 做矩阵乘法，确认结果为 <code>[B,T,V]</code>。
4. 每个维度写出含义。若解释不了某一轴，先不要 reshape。

<code>transpose</code> 或 PyTorch <code>permute</code> 会交换轴；每次变换后都重新写 shape。<code>reshape</code> 只改变组织方式，不会判断轴的语义。<code>[B,T,D]</code> 和 <code>[T,B,D]</code> 元素数相同，但含义不同。

## 参考资料

- [NumPy Quickstart（官方）](https://numpy.org/doc/stable/user/quickstart.html)
- [PyTorch Tensors（官方）](https://pytorch.org/docs/stable/tensors.html)
- [Datawhale《深入浅出 PyTorch》源仓库](https://github.com/datawhalechina/thorough-pytorch)：可在仓库中查阅张量与自动微分章节；许可为 CC BY-NC-SA 4.0。
- [《动手学深度学习》中文版在线教材](https://zh-v2.d2l.ai/)：官方在线教材入口；代码仓库许可证不自动决定书籍正文和图示的授权范围。

> [!tip] 公式与代码对不上时
> 先看 shape，再检查内维。推导里的向量可能按列排，代码里的张量则常把 batch 和序列轴放前面；两者可以等价，但必须说明轴顺序。
