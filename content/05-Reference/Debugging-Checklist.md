---
title: 训练与张量调试清单
description: 从输入、形状、损失、梯度、设备到生成逐项定位常见故障。
tags:
  - reference
  - debugging
  - pytorch
---

报错时先缩小问题：用小批次、短序列和少量参数复现。先查最早异常的张量，不要同时改数据、模型和优化器。

## 输入与标签

- [ ] <code>input_ids</code>、<code>target_ids</code> 是整数类型，shape 为 <code>[B,T]</code>。
- [ ] 标签满足 <code>0 &lt;= id &lt; V</code>；<code>V</code> 与当前 tokenizer 词表大小一致。
- [ ] 自回归目标向前错一位：输入位置 <code>t</code> 对应目标位置 <code>t+1</code>。
- [ ] padding、BOS、EOS 的约定与词表、损失函数和生成代码一致。
- [ ] 测过 tokenizer 的 encode/decode 行为，明确特殊 token 是否会保留。

## 模型输出与损失

- [ ] 前向输出是 logits，而不是已经 softmax 的概率。
- [ ] logits 的 batch 和序列轴与目标一致：<code>logits.shape[:2] == targets.shape</code>。
- [ ] 类别轴大小等于 <code>V</code>，交叉熵输入的类别轴位置正确。
- [ ] loss 是有限值，没有 NaN 或 Inf，也不是意外保留的逐 token 矩阵。
- [ ] 若批次含 padding，损失跳过 padding 目标。

## 梯度与参数更新

- [ ] 每个独立更新前调用 <code>optimizer.zero_grad(...)</code>。
- [ ] <code>loss.backward()</code> 在 <code>optimizer.step()</code> 之前执行。
- [ ] 关键参数的 <code>.grad</code> 不是 <code>None</code>；预期参与计算的参数有合理梯度。
- [ ] loss 和梯度都是有限值；必要时查看梯度范数。
- [ ] 一个很小的批次能被模型记住，训练 loss 能明显下降。
- [ ] 优化器持有当前模型的可训练参数，而不是空列表或旧模型参数。

如果一个小批次都无法过拟合，先不要增加模型规模。

## 模式、设备与复现

- [ ] 模型、输入和目标处于兼容设备，未混用 CPU 与 CUDA 张量。
- [ ] 训练用 <code>model.train()</code>；验证和生成用 <code>model.eval()</code>。
- [ ] 验证和生成使用 <code>torch.no_grad()</code> 或 <code>torch.inference_mode()</code>。
- [ ] eval 模式改变 dropout 等层的行为；它本身不冻结参数。
- [ ] 需要复现时记录种子、依赖版本、配置和数据切分。

## 注意力与生成

- [ ] 因果掩码可广播到 attention scores，屏蔽约定符合所用 API。
- [ ] 未来位置干预测试通过：eval 模式下改变未来 token，不应改变更早位置的 logits。
- [ ] softmax 沿 key 位置轴归一化；每个有效 query 的权重和约为 1。
- [ ] 生成时使用前缀最后一个位置的下一个 token 分布。
- [ ] EOS 或长度上限能终止生成；采样操作按定义施加在 logits 或概率上。

## 保存与重新加载

- [ ] 检查点含模型参数和配置；要继续训练时还保存优化器状态与步数。
- [ ] 在新进程载入检查点，检查输出 shape 并生成一小段文本。
- [ ] tokenizer 与模型权重使用同一份词表和 id 顺序。
- [ ] 记录路径、设备映射和依赖版本。

训练步解释见 [[PyTorch-Train-Step|PyTorch 训练一步]]；形状参考见 [[Tensor-Shape-Cheat-Sheet|张量形状速查表]]。
