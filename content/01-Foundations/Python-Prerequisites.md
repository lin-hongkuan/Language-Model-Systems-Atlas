---
title: Python 前置能力：从读代码到拆问题
description: 学 CS224N 和 CS336 前需要会用的 Python 基础，以及可以自测的完成标准。
tags:
  - foundations
  - python
---

学习目标不是先把 Python 语法书读完，而是能读懂课程 starter、追踪一批数据经过哪些函数，并把训练过程拆成可检查的小步骤。

## 需要会的 Python

### 数据结构与控制流

- 数字、字符串、布尔值、空值；比较、逻辑运算和格式化字符串。
- 条件、循环，以及列表、元组、字典、集合的创建、索引和遍历。
- 能把长表达式拆成变量，能读懂列表推导式；不要求用推导式压缩复杂逻辑。

模型代码中，列表常用来收集层或批次；字典常用来表示配置、词表、批次字段和检查点。

### 函数、模块与对象

能定义函数，写清参数和返回值；理解局部变量、导入模块和异常传播；会读 traceback 并从最后几行定位错误。能看懂类似结构：

~~~python
def make_batch(token_ids, batch_size, seq_len):
    ...
    return inputs, targets

class TinyModel:
    def __init__(self, vocab_size, hidden_size):
        ...

    def forward(self, token_ids):
        ...
~~~

PyTorch 模型通常继承 <code>torch.nn.Module</code>；暂时不需要设计复杂的类层次。

### 文件、环境与调试

- 用 UTF-8 读取文本，理解相对路径相对于当前工作目录。
- 创建虚拟环境、安装依赖并运行 Python 文件或 notebook。
- 遇到异常时先看错误类型、出错行和关键变量，不要一上来重写整段程序。
- 用断言表达输入形状、取值范围等前置条件。

作业仓库要求的 Python 版本和依赖以作业说明为准，不要擅自升级作业环境。

## 自测

不查资料，尝试完成以下任务：

1. 写函数统计一段文本里每个 token 的出现次数，并返回字典。
2. 读 UTF-8 文本文件、跳过空行，并为错误输入显示清楚的提示。
3. 写出一个函数的输入、输出和形状；说明空列表输入时会怎样。
4. 看一段 traceback，指出异常类型、文件和行号。
5. 用断言检查一个二维列表的每一行长度相同。

如果卡在其中一项，先补对应语法。这里不要求掌握装饰器、元类或异步编程。

## 学习材料

- [Python 官方教程](https://docs.python.org/3/tutorial/)：查数据结构、函数、模块、文件与异常。
- [Python 官方中文教程](https://docs.python.org/zh-cn/3/tutorial/index.html)：查数据结构、函数、模块、文件与异常。
- [Think Python 英文版（作者维护）](https://allendowney.github.io/ThinkPython/)：概念讲解与练习；当前在线版与旧版中文译本不是同一版。
- [Think Python 第二版中文译本源仓库](https://github.com/apachecn/think-py-2e-zh)：由 ApacheCN 维护的第三方翻译项目，许可证信息存在版本冲突，详见 [[Sources-and-Materials|来源与许可说明]]。
- [CS224N 官方课程主页](https://web.stanford.edu/class/cs224n/)：查当前 Python 复习材料和课程公告；2025 版复习讲义是旧版本。

> [!note] 私人离线副本
> Python 文档、Think Python 中文译本和旧版 Python 复习讲义的离线副本存放在 Quartz 发布树之外。本页只链接在线来源，发布站点不会包含或链接这些私人副本；需要离线阅读时，请从自己的工作区打开副本。

## 下一步

继续读 [[NumPy-and-Tensor-Shapes|NumPy 与张量形状]]。读代码时先标出变量类型和 shape，再追踪函数调用。
