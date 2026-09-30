---
title: 术语表：Python、张量与语言模型
description: 本站基础和课程页面共用的中英文术语对照。
tags:
  - reference
  - glossary
---

不同论文和代码仓库可能为同一概念使用不同缩写，阅读时以当前材料的定义和实现为准。

## Python 与数值计算

| 中文 | English | 简明解释 |
|---|---|---|
| 列表 | list | 可变、有顺序的 Python 容器。 |
| 字典 | dictionary / dict | 由键映射到值，常用于配置和批次字段。 |
| 模块 | module | 可导入的 Python 文件或代码包。 |
| 异常 | exception | 运行时错误对象；traceback 显示错误传播路径。 |
| 数组 | array | 按轴排列的数值集合；NumPy 的主要数据结构。 |
| 张量 | tensor | 多维数值数组；PyTorch 张量还有设备和自动微分属性。 |
| 形状 | shape | 每个轴的长度，例如 <code>[B,T,D]</code>。 |
| 轴 | axis / dimension | 张量的一个方向；<code>[B,T,D]</code> 有三个轴。 |
| 广播 | broadcasting | 按兼容规则让较小张量参与逐元素运算。 |
| dtype | data type | 元素类型，如整数、<code>float32</code>、<code>bfloat16</code>。 |
| 设备 | device | 张量所在的 CPU 或加速设备。 |

## 神经网络与训练

| 中文 | English | 简明解释 |
|---|---|---|
| 参数 | parameter | 由训练算法更新的模型值，如权重和偏置。 |
| 前向传播 | forward pass | 从输入计算模型输出的过程。 |
| 计算图 | computation graph | 自动微分记录的运算依赖关系。 |
| 梯度 | gradient | 损失对参数的导数。 |
| 反向传播 | backpropagation | 沿计算图用链式法则计算梯度。 |
| 损失 | loss | 衡量模型输出与目标差异的数值。 |
| logits | logits | softmax 之前的未归一化类别分数。 |
| 交叉熵 | cross-entropy | 常见分类损失；语言模型用它评价目标 token 预测。 |
| 优化器 | optimizer | 根据梯度更新参数的算法，如 SGD、AdamW。 |
| 学习率 | learning rate | 控制参数更新幅度的超参数。 |
| 梯度裁剪 | gradient clipping | 限制梯度范数或数值范围。 |
| 批次 | batch | 一次并行处理的一组样本。 |
| 训练模式 | train mode | <code>model.train()</code> 设置的训练行为，如启用 dropout。 |
| 评估模式 | evaluation mode | <code>model.eval()</code> 设置的评估行为；不等于关闭梯度。 |
| 检查点 | checkpoint | 恢复训练所需的状态快照。 |

## NLP 与语言模型

| 中文 | English | 简明解释 |
|---|---|---|
| token | token | 分词器输出的基本单位，可以是字、子词或字节片段。 |
| 词表 | vocabulary | token 与整数 id 之间的映射集合。 |
| token id | token ID | 词表为 token 分配的整数编号。 |
| 分词器 | tokenizer | 将文本编码为 token id，并可将 id 解码为文本。 |
| 嵌入 | embedding | 把离散 id 映射成可学习向量。 |
| 隐状态 | hidden state | 网络中间层对每个位置的表示。 |
| 序列长度 | sequence length | 一个样本包含的 token 数，常记为 <code>T</code>。 |
| 因果掩码 | causal mask | 阻止某位置看到未来 token 的注意力约束。 |
| 注意力 | attention | 根据 query-key 分数加权汇总 value 的运算。 |
| 多头注意力 | multi-head attention | 在多个投影子空间并行计算注意力，再合并结果。 |
| 位置编码 | positional encoding / embedding | 向模型提供 token 顺序或位置信息。 |
| 自回归 | autoregressive | 根据已有前缀逐步预测下一个 token。 |
| 困惑度 | perplexity | 基于 token 负对数似然计算的语言模型指标；比较时需保持分词和数据设置一致。 |
| 填充 | padding | 将较短序列补齐，以便组成批次。 |
| 忽略索引 | ignore index | 损失函数用于跳过不监督位置的目标值。 |

## 本站符号

| 符号 | 含义 |
|---|---|
| <code>B</code> | 批次大小 |
| <code>T</code> | 序列长度 |
| <code>V</code> | 词表大小 |
| <code>D</code> | 隐藏维度或 <code>d_model</code> |
| <code>H</code> | 注意力头数 |
| <code>d_h</code> | 单头维度；标准 MHA 常取 <code>D/H</code> |
| <code>D_ff</code> | 前馈网络中间维度 |

更多轴顺序见 [[Tensor-Shape-Cheat-Sheet|张量形状速查表]]；概念入门见 [[NumPy-and-Tensor-Shapes|NumPy 与张量形状]]。
