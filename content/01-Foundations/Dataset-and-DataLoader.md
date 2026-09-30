---
title: Dataset 与 DataLoader：把样本整理成 batch
description: 用一个 next-token 窗口数据集说明 __len__、__getitem__、错位标签、batch 堆叠和 DataLoader 的遍历方式。
tags:
  - foundations
  - pytorch
  - data
---

模型不能直接消费“一个文件夹里的一堆样本”这种抽象。`Dataset` 规定如何按索引取出单个样本，`DataLoader` 负责采样顺序、组 batch，并在形状允许时将多个样本堆成 Tensor。

跑配套脚本：

```powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/dataset_dataloader.py
```

预期：`dataset windows: 4`；第一批输入为 `[[0, 1], [1, 2]]`，对应目标为 `[[1, 2], [2, 3]]`；batch 的输入和目标 shape 都是 `(2, 2)`。macOS/Linux 把解释器换成 `.venv/bin/python`。完整脚本：[dataset_dataloader.py](examples/dataset_dataloader.py)。

## 先把序列切成相邻的小窗口

总序列是 `[0,1,2,3,4,5]`，`context_length=2`。可产生 4 对样本：

| 样本索引 | 输入 | 目标（向右错一格） |
|---:|---|---|
| 0 | `[0,1]` | `[1,2]` |
| 1 | `[1,2]` | `[2,3]` |
| 2 | `[2,3]` | `[3,4]` |
| 3 | `[3,4]` | `[4,5]` |

长度为 N 的 token 序列、上下文长度为 T，可形成 `N-T` 个窗口。每一对输入和目标都长度为 T，且目标比输入向右移一格。数据管线的职责是对齐样本；因果注意力的职责是在更复杂的 Transformer 里阻止位置提前看到未来 token。

## `Dataset` 的两个必需接口

脚本中的 `NextTokenWindowDataset` 继承 `torch.utils.data.Dataset`：

- `__len__()` 告诉 PyTorch 有多少条样本；这里是 `len(token_ids)-context_length`。
- `__getitem__(index)` 返回第 index 条 `(inputs, targets)`。在上表，索引 1 返回输入 `[1,2]` 与目标 `[2,3]`。

越界检查和参数检查同样属于数据集的一部分。示例在序列不够长、上下文长度小于 1 时明确报 `ValueError`，避免稍后得到难懂的空 batch。

## `DataLoader` 让许多样本组成批次

创建 `DataLoader(dataset, batch_size=2, shuffle=False)` 后，PyTorch 先分别调用 dataset 取两条样本，再把输入堆叠为 `[B,T]` Tensor、把目标也堆叠为 `[B,T]` Tensor。`B=2,T=2`，所以两个 shape 是 `[2,2]`。默认 `shuffle=False` 是为了输出顺序容易核对。

训练通常会用 `shuffle=True` 打乱样本读取顺序，但这不会修改 dataset 中单条样本的内部 token 顺序。语言建模的窗口常重叠；打乱的是窗口样本的排列，不是每个窗口里的 token。

另一个重要参数是 `drop_last`。当样本数不能被 batch size 整除时，最后一批可能更小。默认 `drop_last=False` 会保留尾批；某些特定 batch normalization 或多设备场景才考虑丢弃它。初学者先保留所有数据。

## 接到模型训练

每个训练迭代通常接收 `(input_ids, target_ids)`：输入经过 embedding 和模型得到 logits；目标不进入模型前向，只在算 loss 时作为正确答案。接着调用清梯度、`backward()` 和 `step()`，详见 [[PyTorch-Train-Step|PyTorch 训练一步]]。

常见错误与排查：

- 标签没有错位：打印同一条输入和目标，确认它们确实差一个 token。
- 样本 shape 不同导致无法堆叠：文本通常需截断/填充成定长，或自定义 `collate_fn`。
- token IDs 是浮点：Embedding 索引应是整数（通常 `torch.long`）。
- 训练时数据被覆盖/泄漏：先保留原始文本索引，再定义 train/validation 切分；本示例为了演示省略了验证集。

## 改成不同上下文长度

把脚本里的 `context_length=2` 改为 3，然后手工检查新的第一条样本：输入 `[0,1,2]`、目标 `[1,2,3]`；样本总数由 4 变为 3。再将 `batch_size=2` 改为 4，最后一个 batch 应只有 3 条。先预测，再运行确认。

后续用 [[Tiny-Next-Token-Project|小型 next-token 训练项目]] 把这类 batch 喂给可训练模型。
