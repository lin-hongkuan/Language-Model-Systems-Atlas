---
title: PyTorch 训练一步：从 token 到参数更新
description: 用语言模型训练步解释前向传播、交叉熵、反向传播和优化器。
tags:
  - foundations
  - pytorch
  - training
---

训练一步可以概括为：模型先预测，损失函数衡量预测与目标的差异，自动微分计算参数梯度，优化器再更新参数。

## 输入与目标错开一格

给定序列 <code>[BOS, 我, 喜欢, NLP, EOS]</code>，自回归模型在每个位置预测下一个 token：

~~~text
输入 x: [BOS, 我, 喜欢, NLP]
目标 y: [我,   喜欢, NLP, EOS]
~~~

批次里的 <code>input_ids</code> 和 <code>target_ids</code> 均为 <code>[B,T]</code>。模型前向输出每个位置对词表中所有 token 的分数，即 logits，形状为 <code>[B,T,V]</code>。

## 交叉熵接受什么

PyTorch 的 <code>CrossEntropyLoss</code> 接受未归一化 logits 和整数类别目标；内部会计算 log-softmax 与负对数似然。不要先对 logits 手动做 softmax 再交给它。

语言模型常将 logits 从 <code>[B,T,V]</code> 展成 <code>[B*T,V]</code>，把目标从 <code>[B,T]</code> 展成 <code>[B*T]</code>。每行预测一个位置的下一个 token。若批次里有 padding，应通过 <code>ignore_index</code> 等机制跳过，并确保 padding id 与数据管道一致。

## 训练步骨架

下面的例子假设 <code>model(input_ids)</code> 返回 <code>[B,T,V]</code>。课程作业接口可能不同，应以接口为准。本例讲解通用训练概念，不是 Stanford 作业解答。

~~~python
import torch.nn.functional as F

def train_step(model, optimizer, input_ids, target_ids, device):
    model.train()
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)  # [B, T, V]
    batch_size, seq_len, vocab_size = logits.shape
    loss = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        target_ids.reshape(batch_size * seq_len),
    )
    loss.backward()
    optimizer.step()
    return loss.detach().item()
~~~

### 顺序说明

1. <code>model.train()</code> 让 dropout 等层采用训练行为；它本身不负责打开梯度。
2. 将输入和目标放到同一设备，避免 CPU 与 CUDA 张量混算。
3. <code>zero_grad()</code> 清空上一步梯度。PyTorch 默认累加梯度，所以每个独立更新前都要清零。
4. 前向得到 logits，并和下一个 token 目标计算交叉熵。
5. <code>loss.backward()</code> 沿计算图计算梯度，填入参数的 <code>.grad</code>。
6. <code>optimizer.step()</code> 依据梯度更新参数。
7. <code>detach().item()</code> 把损失变成 Python 数字供日志记录。

如果实验要求梯度裁剪，将它放在 <code>backward()</code> 与 <code>step()</code> 之间。

## 训练、验证和生成

- 训练用 <code>model.train()</code>，计算损失并更新参数。
- 验证用 <code>model.eval()</code> 和 <code>torch.no_grad()</code>，只计算指标。
- 生成通常也用 eval 模式和无梯度上下文，逐 token 采样或取最大分数，直到遇到结束标记或达到长度上限。

<code>model.eval()</code> 改变 dropout 等层的行为；<code>torch.no_grad()</code> 停止记录梯度计算图。它们不是同一个开关。

## 第一批数据的检查

~~~python
assert input_ids.ndim == 2
assert target_ids.shape == input_ids.shape
assert logits.shape[:2] == target_ids.shape
assert target_ids.min() >= 0
assert target_ids.max() < logits.shape[-1]
assert torch.isfinite(loss)
~~~

用一个很小、能记住的批次反复训练，损失应明显下降。若没有，先查输入和目标是否错位、输出与目标形状、梯度是否为空，以及张量是否在同一设备。参见 [[Debugging-Checklist|调试清单]]。

## 官方参考与扩展阅读

- [PyTorch Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [CrossEntropyLoss API](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [CS336 Spring 2026 官方页面](https://cs336.stanford.edu/)：作业接口、依赖和工具使用规则以当期手册为准。
- [Datawhale《深入浅出 PyTorch》](https://github.com/datawhalechina/thorough-pytorch)：自动微分、训练和评估章节；仓库声明 CC BY-NC-SA 4.0。

私人离线副本保存在 Quartz 发布树之外，本站不引用这些副本。来源与许可证边界见 [[Sources-and-Materials|来源与许可说明]]。
