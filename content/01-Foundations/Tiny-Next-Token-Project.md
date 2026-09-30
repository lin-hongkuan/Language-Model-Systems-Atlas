---
title: 小型 next-token 训练项目：从字符到更新后的模型
description: 不下载数据，在 CPU 上训练一个极小的字符级 bigram 模型，追踪数据集、Embedding、logits、交叉熵和优化器更新。
tags:
  - foundations
  - pytorch
  - language-model
---

这是一份可以自己从头跑通的微型语言模型练习。代码只用字符级词表和当前字符作为上下文，目标是把“数据 → 模型 → loss → 梯度 → 参数更新”连成一条真实的 PyTorch 流程。它不是 Transformer，也不需要 GPU 或下载语料。

## 运行项目

在仓库根目录运行：

~~~powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/train_tiny_bigram.py
~~~

预期出现 95 个训练样本、一个从大约 0.874 降到接近 0.000 的训练集 loss，以及三条转移：猫 -> 追、球 -> 猫、追 -> 球。具体 loss 末位可能因 PyTorch/CPU 版本浮动。macOS/Linux 使用 .venv/bin/python content/01-Foundations/examples/train_tiny_bigram.py。源码：[train_tiny_bigram.py](examples/train_tiny_bigram.py)。

![序列中输入 token 与下一个 token 目标错开一格，训练用目标的交叉熵更新模型参数](assets/pytorch-next-token.svg)

*图 1. 输入和标签错一格：每个位置让模型预测下一个字符；训练时并行计算这批位置的损失。*

## 第一步：把字符变成 ID

项目的训练语料是“猫追球”重复 32 次，共 96 个字符。sorted(set(corpus)) 得到去重字符并排序；字典把三个字符映射为 0 到 2 范围的整数 ID。ID 只是查表用的地址，没有“追”比“猫”大之类的数值含义。

编码后序列长度仍为 96。因果语言模型对相邻字符建立监督，因此用前 95 个 ID 作为输入、后 95 个 ID 作为目标。第一个例子是（猫，追），第二个是（追，球）；最后一个目标是下一轮开头的猫。

## 第二步：Dataset 产出单个训练样本

NextTokenDataset 在初始化时建立 inputs = token_ids[:-1] 与 targets = token_ids[1:] 两个 tensor。__len__() 为 95；__getitem__(i) 返回两个标量 ID。

将样本堆成 batch 后，当前模型期望 input_ids:[B] 和 target_ids:[B]。与窗口数据集示例不同，这个最小模型每个样本只用一个上下文字符；所以没有序列长度 T 这一轴。

## 第三步：让 nn.Module 产生下一字符分数

模型由 embedding 查表和一个线性输出层组成：

1. nn.Embedding(V,D) 将输入 ID 从 [B] 变成稠密向量 [B,D]。
2. nn.Linear(D,V) 对每行向量算出 [B,V] 的 logits，每类对应一个词表字符。
3. 对每个样本，目标是 [B] 中的下一字符整数 ID。

当前 embedding 宽度 D=8，词表大小 V=3。线性层读 logits，不需要先 argmax 或 softmax：训练需要所有类别分数，才能算出正确答案的概率和梯度。

## 第四步：一批样本的一次训练更新

训练循环里，对每个 batch 执行以下完整步骤：

1. optimizer.zero_grad(set_to_none=True) 清除上一批的梯度。
2. logits = model(input_ids) 前向计算 [B,V] 分数。
3. F.cross_entropy(logits, target_ids) 将 logits 与 [B] 整数目标比较并对 batch 求平均。
4. loss.backward() 依据链式法则计算 embedding 和输出层的梯度。
5. optimizer.step() 用 Adam 的更新规则修改参数。

每轮训练都遍历所有 batch；共 80 轮。脚本在训练前后用同一训练数据计算平均交叉熵，最终检查模型是否记住小语料中的一阶转移。

## 为什么这个模型会学会循环

语料重复的确定性模式是“猫→追”“追→球”“球→猫”。相同前一个字符总后接同一个字符，所以一个字符上下文对这个任务已足够。训练后分别输入三个字符，取 logits 最大的类别，得到三个后继。

这里的 argmax 只用于训练完成后的演示预测。训练时若先把分数变成离散 argmax，就无法通过它对权重做正常的梯度更新。

## 改造挑战

### 挑战 A：增加一种字符

把语料改成“猫追球鸟飞”重复多次。检查词表大小如何变化；输出层类别数 V 会由词表确定，无需手写类别数。数据中存在字符级转移；运行模型，记录训练集 loss 是否下降。

### 挑战 B：扩大上下文到两个字符

bigram 这里只是“一个 token 看下一个 token”的叫法。要使输入成为两个字符的窗口，Dataset 应返回长度为 2 的 ID 序列与右移一位的目标，模型的 Embedding 输出 [B,2,D]，再将两个位置组合成用于预测的表示。你可以先用最后一个位置的向量作预测，暂不引入注意力。

### 挑战 C：加入验证样本

不要在每个 batch 更新后立刻用训练 loss 声称泛化变好。先按字符序列区段划分训练与验证数据；保证验证样本的目标不泄漏进训练集，再分别计算两个 split 的 loss。由于数据模式极小，扩大语料并设计不同转移，才能让验证更有意义。

<details><summary>改造挑战参考答案与核对方式</summary>

**挑战 A：** 新模式有 5 种不同字符，所以词表大小 V=5；重复“猫追球鸟飞”时，各字符的下一字符都唯一，包括轮末“飞→猫”。重复语料可以记住该转移，因此训练 loss 应明显下降，模型依次预测“猫→追、追→球、球→鸟、鸟→飞、飞→猫”。这仍然只验证模型记住训练样本。

**挑战 B：** 令 Dataset 的一个样本包含连续两个 ID，输入为 [t,t+1]，标签是 [t+2]。batch 后输入 shape 为 [B,2]、标签为 [B]；Embedding 输出 [B,2,D]。取最后位置向量 vectors[:,-1,:] 得到 [B,D]，再过投影层得到 [B,V]。数据序列必须至少有三个 ID 才能构造此样本。无需把两位置摊平，因为这项挑战只用最后一个上下文位置预测后继。

**挑战 C：** 先按完整的重复片段切分原始序列，例如前 24 段训练、最后 8 段验证；再分别从各自切片构造 Dataset，绝不先构造重叠窗口再随机拆分，否则相邻窗口会同时落在两个集合中。验证时切换 model.eval()，并在 torch.inference_mode() 中只计算 loss、不调用 backward 或 optimizer.step()。本小例子的重复模式在两边相同，所以它只校验代码路径，不能衡量真实泛化。

</details>

## 模型边界

此项目只演示训练环路：字符 tokenizer、没有开始/结束标记、一个字符上下文、极小重复语料、没有 train/validation/test 划分。它能记住这个确定性模式，不代表模型理解语言，也不能替代课程要求的 Transformer 实现。

完成后检查自己能否口头说出以下 shape：Dataset 单样本各是标量；一批输入 [B]、目标 [B]；Embedding [B,8]；logits [B,3]；交叉熵 loss 为标量。然后回到 [[PyTorch-Train-Step|训练一步]]，再开始 [[../03-CS224N/L02-词向量与 Word2Vec|CS224N Word2Vec]] 和 [[../04-CS336/a1-basics|CS336 A1]]。
