---
title: PyTorch 核心动手：Tensor、autograd 与 nn.Module
description: 从 shape 和 dtype 开始，亲手运行 Tensor、自动求导、模型前向、交叉熵和一次 SGD 更新。
tags:
  - foundations
  - pytorch
  - autograd
---

本课把 PyTorch 拆成三件事：**Tensor 保存带 shape/dtype/device 的数据；autograd 按实际执行的运算求导；`nn.Module` 把可训练参数与前向计算组织成模型。** 先在纸上预测形状，再运行脚本核对。

开始前完成 [[Python-Environment|Python 环境搭建]]。在仓库根目录运行配套脚本：

```powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/pytorch_core.py
```

预期结果包含 `ids shape: (2, 3)`、`dy/dx at x=2: 7.0`、`logits shape: (4, 3)`、`loss is finite: True` 和 `parameter changed after step: True`。loss 的具体小数因随机初始化和库版本可能不同；脚本固定随机种子。macOS/Linux 请使用 `.venv/bin/python content/01-Foundations/examples/pytorch_core.py`。完整源码：[pytorch_core.py](examples/pytorch_core.py)。

## Tensor 是有类型、有形状的数组

Python 列表只知道“里面有些值”；PyTorch Tensor 还记录数值类型、轴尺寸，以及数据所在设备。例子中的整数 ID 用 `torch.long`，可训练参数和激活通常是浮点数。

读 shape 时给每个轴取名字。语言模型里常见：

| 形状 | 含义 | 常见例子 |
|---|---|---|
| `[]` | 标量 | 一个 loss |
| `[T]` | 一条长度为 T 的序列 | token IDs |
| `[B,T]` | B 条等长序列 | 一个 token batch |
| `[B,T,D]` | 每个 token 有 D 维表示 | embedding/隐藏状态 |
| `[B,T,V]` | 每个位置对 V 个 token 的分数 | next-token logits |

这里 `B` 是 batch size、`T` 是序列长度、`D` 是特征宽度、`V` 是词表大小。一个轴的长度本身不说明语义；由它在程序里的用途决定。

### dtype 与 device 是不同问题

- `dtype` 回答“一个元素怎样编码”：如 `torch.int64`、`torch.float32`。
- `device` 回答“这块数据放在哪里算”：如 `cpu`、`cuda:0`。
- 查表输入 `Embedding` 的 token ID 必须是整数索引；线性层权重和输入特征通常是浮点数。
- 同一个运算的参与张量一般必须在兼容的设备上。模型在 GPU、输入在 CPU 时，不会自动替你搬数据。

遇到 shape 报错，先查看 `tensor.shape`、`tensor.dtype`、`tensor.device`。不要先盲目 `reshape`：reshape 会重排元素的分组，不能凭空修正轴的语义。

## autograd：从前向运算记录到梯度

对一个标量函数 `y = x² + 3x`，在 `x=2` 时，导数是 `2x+3=7`。脚本会设置 `x.requires_grad=True`，计算 `y`，再调用 `y.backward()`；随后 PyTorch 把 `dy/dx` 放在叶子张量 `x.grad` 里。

手算结果应为 `y=10`、`x.grad=7`。这解释了训练里的自动微分：前向执行留下运算关系，链式法则把 loss 对参数的导数从后向前传回。`backward()` 是算梯度，**不会**自动更新参数。

### 梯度默认会累加

PyTorch 的 `.grad` 默认把多次反向传播的梯度加起来。因此一个常见训练步骤先让优化器清除旧梯度，再前向计算、反传、更新：

1. `optimizer.zero_grad(set_to_none=True)` 清理上一批的 `.grad`。
2. `logits = model(inputs)` 执行前向计算。
3. `loss.backward()` 计算并保存参数梯度。
4. `optimizer.step()` 根据梯度和学习率改变参数。

漏掉第 1 步会累积本不打算累积的梯度；漏掉第 4 步则模型权重不变。脚本在更新前后复制并比较了一个参数，所以 `parameter changed after step: True` 是一次直接检查。

## `nn.Module`：把参数和前向过程装进一个模型

`TinyClassifier` 继承 `nn.Module`。初始化时建立 `nn.Linear` 层；`forward(features)` 定义数据如何经过这些层。调用 `model(features)` 时，PyTorch 会分派到 `forward`，同时管理参数、设备搬移和状态。

脚本做的是一个小分类器：4 行、每行 2 个特征，经网络得到 3 个类别分数，因此输入 shape 为 `[4,2]`、logits shape 为 `[4,3]`。PyTorch 可以用 `model.parameters()` 找出所有已注册的可训练权重，优化器便据此更新它们。

类里有两层全连接变换和一个 `Tanh`。它的主要读法是：输入维数为 2，隐藏宽度为 4，输出类别数为 3。模型层定义了参数，`forward` 只写计算流程；不要在 `forward` 中每次新建需要学习的层，否则参数会在每次调用时重置或未被优化器管理。

## logits、标签与交叉熵

`logits` 是每类尚未归一化的分数。分类训练时，`F.cross_entropy(logits, labels)` 内部会稳定地做 log-softmax 和负对数似然；传入原始 logits，不需要先手动 `softmax`。

若 logits 是 `[B,V]`，类别标签是 `[B]` 的整数，标签值必须在 `0` 到 `V-1` 之间。例如脚本中 V=3，4 个标签为 `[0,1,2,1]`。对于语言模型的 `[B,T,V]`，需要把前两轴展平为 `[B*T,V]`，并把 `[B,T]` 标签展平为 `[B*T]`；每个 token 位置就成为一个分类样本。损失默认对这些位置求平均。

## `train()`、`eval()` 与关闭梯度

`model.train()` 和 `model.eval()` 控制 dropout、batch normalization 等层的行为；它们不会启用或关闭自动求导，也不等于改参数。评估/生成时通常同时使用 `model.eval()` 与 `torch.inference_mode()`，避免记录梯度图并节约内存。

微型脚本不含 dropout，因此两种状态输出行为相同；保留状态切换习惯是为了进入真实课程网络时不遗漏。

核对：第 1 题的 shapes 是 [5,2] 与 [5,3]；第 2 题输出 logits [4,4]；第 3 题通常会报 “Target 3 is out of bounds”，因为合法类别编号是 0、1、2。修正标签后，交叉熵应能返回有限的标量。

## 自己动手改三处

打开 `examples/pytorch_core.py`：

1. 把 `features` 增加到 5 行，同时给 `labels` 加上第 5 个类别编号。预测新的 shape：features `[5,2]`、logits `[5,3]`。
2. 把分类数改成 4：改模型的 `num_classes` 和标签范围。若模型输出 `[4,4]`，第二个轴表示 4 类。
3. 故意把一个标签改成 `3`（类别数仍为 3），观察交叉熵报错。根据消息把标签改回 `[0,2]` 内的有效整数。

推荐每次只改一处、再运行同一条命令。下一步读 [[Dataset-and-DataLoader|Dataset 与 DataLoader]]，看输入张量如何从样本集合中生成。

## 参考

- [PyTorch Tensor 教程](https://pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html)
- [Autograd 机制说明](https://pytorch.org/docs/stable/notes/autograd.html)
- [`nn.Module` API](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [`CrossEntropyLoss` API](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
