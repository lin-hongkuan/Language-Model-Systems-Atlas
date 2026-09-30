---
title: 分层编程练习与逐步答案
description: 通过追踪输出、改写 Python 函数、预测张量 shape、计算梯度和阅读训练循环，从入门练到可读课程 starter。
tags:
  - foundations
  - practice
  - python
  - pytorch
---

建议用法：先关掉折叠答案，在纸上写预测；再运行相应脚本，若不一致就指出是哪一个值、索引或 shape 没追上。题目从追踪、改写、组合到调试递进。对应 [[Python-Prerequisites|Python]]、[[NumPy-and-Tensor-Shapes|张量 shape]]、[[PyTorch-Core-Practice|autograd 与模型]] 和 [[Dataset-and-DataLoader|数据批处理]]。

## 第一层：读懂 Python 数据

### 练习 1：索引与切片

给定 token 顺序 猫、追、球、猫，将它编码为 [0,1,2,0]。分别写出第 0 个元素、第 1 到末尾的切片，以及错位的 next-token 输入和目标。

<details><summary>逐步答案</summary>

1. Python 从 0 开始，所以第 0 个元素是 0。
2. 起点 1 的切片包含索引 1、2、3，得到 [1,2,0]；右侧终点若省略，就延伸至末尾。
3. 输入取除最后一个外的所有元素：[0,1,2]。
4. 目标取除第一个外的所有元素：[1,2,0]。相同位置的输入预测下一个位置的 ID。

</details>

### 练习 2：写 encode 和计数器

实现一个按词表查 ID 的函数。未知 token 要么明确报错，要么映射到一个已在词表里保留的 UNK id；然后遍历 猫、追、猫、球，统计每个 token 出现次数。

先运行 [python_exercise_solutions.py](examples/python_exercise_solutions.py)：

~~~powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/python_exercise_solutions.py
~~~

预期输出：encoded: [0, 1, 0, 2]、counts: {'猫': 2, '追': 1, '球': 1}、输入/目标分别为 [0,1,0] 与 [1,0,2]，未知鸟映射为 [3]。

<details><summary>逐步答案</summary>

1. encode 先建立空列表，逐个 token 检查是否存在于 vocabulary。
2. 存在就取其 ID 并 append；不存在则检查调用者是否给了 unknown_id。若没有，主动抛 ValueError，而不是猜 ID。
3. 计数器从空字典开始；首次遇到的键先设为 0，再加 1。循环结束后字典为 {'猫': 2, '追': 1, '球': 1}。
4. 分别切除尾部和头部构造输入/目标。完整实现见同一个 python_exercise_solutions.py 文件。

</details>

## 第二层：手算 Tensor 的 shape

### 练习 3：embedding 和输出层

batch 有两条序列，每条三个 token：输入 shape 是 [2,3]。词表 V=5，embedding 宽度 D=4。预测 Embedding(5,4) 的权重 shape、输出 shape；再用线性层把 D 投影成 V，logits shape 是多少？

<details><summary>逐步答案</summary>

1. Embedding 权重按“词表中每个 ID 一行、每行 D 个数”存储，所以权重为 [5,4]。
2. 输入 [2,3] 中的每个 ID 查出一个 4 维向量，因此输出 [2,3,4]。
3. 线性层只替换最后一轴，把 D=4 投影到 V=5，所以 logits 是 [2,3,5]。
4. 每个位置有一个 5 类预测，因此标签仍是 [2,3] 的整数 ID。

</details>

### 练习 4：展平后计算交叉熵

模型输出 [B,T,V]=[2,3,5]，目标 [2,3]。cross_entropy 的类别轴应放在第二轴，重排后 logits 与标签各自什么 shape？

~~~powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/pytorch_exercise_solutions.py
~~~

预期输出包括 logits [2,3,5]、展平 logits [6,5]、展平标签 [6] 和有限的标量 loss。macOS/Linux 使用 .venv/bin/python content/01-Foundations/examples/pytorch_exercise_solutions.py。

<details><summary>逐步答案</summary>

1. B*T=6 个位置分别是 6 个分类样本，每个有 V=5 类。
2. logits reshape 为 [6,5]，每行是一个位置的 5 个类别分数。
3. 标签 reshape 为 [6]，每个整数在 0 到 4 之间。
4. F.cross_entropy 返回标量；若有效标签为 -100 等特殊忽略编号，还需显式设置 ignore_index，不能靠 reshape 隐藏它。

</details>

## 第三层：理解自动微分

### 练习 5：手算导数并区分 backward/step

设 y=x²+3x，x=2。写出 y 和 dy/dx。调用 backward() 后，参数是否已被更新？

~~~powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/pytorch_exercise_solutions.py
~~~

期望输出最后一行是 gradient of x**2 + 3*x at x=2: 7.0。

<details><summary>逐步答案</summary>

1. y=2²+3×2=10。
2. 导数为 2x+3；在 x=2 处等于 7。
3. y.backward() 把梯度写入叶子张量的 x.grad，只负责求梯度。
4. 只有与模型参数绑定的优化器执行 optimizer.step() 才会按这些梯度改变参数。一般每次 backward 前先 zero_grad()，避免无意累加。

</details>

## 第四层：数据集与训练代码阅读

### 练习 6：写窗口样本

token ID 为 [0,1,2,3,4]，窗口长度 T=2。写出样本总数和最后一条样本。

<details><summary>逐步答案</summary>

1. 样本数量 N-T = 5-2 = 3。
2. 起点 0、1、2 分别给出窗口，所以最后一个起点为 2。
3. 输入 [2,3]，目标 [3,4]。要有 T 个目标，每个输入位置对应其右侧下一个 token。

</details>

### 练习 7：排列训练一步

把以下动作排成正确顺序：参数更新；计算 logits 与 loss；清梯度；反向传播。

<details><summary>逐步答案</summary>

1. 清除旧梯度：zero_grad()。
2. 执行前向并根据答案算出 loss。
3. 对 loss 调用 backward() 写入参数梯度。
4. 调用 optimizer.step() 用梯度更新参数。

zero_grad 不清除模型参数；backward 不改模型参数。重复多个 microbatch 做梯度累积时，才会有意让多次 backward 之间不清零。

</details>

## 第五层：用脚本自查

~~~powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/dataset_dataloader.py
.\.venv\Scripts\python.exe content/01-Foundations/examples/train_tiny_bigram.py
~~~

预期：第一份脚本打印 4 个长度为 2 的窗口和两批 [2,2] 张量；第二份脚本显示训练 loss 大幅下降并学到 猫→追→球→猫。如果输出不同，优先检查你改过的上下文长度、DataLoader 顺序和词表 ID。

做完后，若能对给定代码预测输入输出 shape、说明每个标签对应哪个 token、定位梯度写入与更新的位置，就可以开始读课程 starter。继续 [[Tiny-Next-Token-Project|完整 next-token 小项目]]，然后读 [[../03-CS224N/L03-神经网络与反向传播|CS224N 反向传播]] 与 [[../04-CS336/a1-basics|CS336 A1]]。
