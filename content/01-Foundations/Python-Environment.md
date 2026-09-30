---
title: Python 环境与运行脚本：从终端到 PyTorch
description: 面向初学者的 Windows/macOS/Linux 环境搭建、虚拟环境、安装 CPU 版 PyTorch、运行脚本和定位常见错误。
tags:
  - foundations
  - python
  - setup
---

这一页的目标不是熟记终端命令，而是让你能稳定回答四个问题：**正在用哪个 Python？库装到了哪里？程序从哪个目录启动？错误发生在自己代码的哪一行？** 后续讲义和课程作业都依赖这几件事。

## 先理解三个名字

- **Python 解释器**：真正读取并运行 `.py` 文件的程序，例如 `python.exe`。
- **pip**：把第三方库安装到某一个 Python 环境里的工具。用 `python -m pip` 可以明确让当前这个解释器调用自己的 pip。
- **虚拟环境**：项目专属的 Python 和依赖目录。它隔开不同项目的版本，避免一个项目升级 PyTorch 后影响别的项目。

Windows 上可能同时装有 Microsoft Store Python、python.org Python、Conda Python 或 IDE 自带 Python。`python` 和 `py` 有时会找到不同解释器，因此安装和运行时尽量使用同一个环境解释器的完整路径。

## Windows：创建独立环境并安装 CPU 版 PyTorch

在 PowerShell 中先切到你保存课程资料的仓库根目录。下面这一整组命令按顺序执行；第一次安装需要网络，并会下载 PyTorch 依赖。这里用 CPU 轮子，是为了让入门例子在没有 NVIDIA 显卡的电脑上也能运行。

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

预期：创建 `.venv` 目录，pip 显示安装成功。每条命令显式使用 `.venv\Scripts\python.exe`，所以不需要先激活环境，也不受 PowerShell 执行策略阻止激活脚本的影响。

### 检查解释器与 PyTorch

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.tensor([1, 2, 3]).tolist())"
```

预期：第一行显示 Python 3.10 或更新版本；第二条输出 PyTorch 版本、`cuda available: False`（CPU 轮子）和 `[1, 2, 3]`。版本数字会随 PyTorch 官方发布更新。

运行仓库里的示例时仍使用同一个解释器。以下命令应输出 `[0, 1, 2]`：

```powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/python_basics.py
```

若你之后需要 GPU，不要照抄网上随机 CUDA 命令。先确认显卡、驱动与课程服务器环境，再使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)为自己的系统生成安装命令；安装后重新检查 `torch.cuda.is_available()`。本目录的 CPU 示例无需 GPU。

## macOS / Linux：创建独立环境

在终端切到仓库根目录，然后顺序运行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
```

预期：pip 安装成功。PyTorch 会根据当前平台选择可用构建；macOS/Linux 的 GPU 选项取决于机器硬件与驱动，不应假设都会有 CUDA。

检查版本和 CPU tensor：

```bash
.venv/bin/python --version
.venv/bin/python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.tensor([1, 2, 3]).tolist())"
```

预期：Python 3.10+、一个 PyTorch 版本、GPU 可用状态，以及 `[1, 2, 3]`。本学习路径所有示例默认在 CPU 上运行。

## 如果安装失败，先做这四个检查

1. 在执行 pip 前后都使用同一条解释器路径，例如 `.venv\Scripts\python.exe -m pip`。不要一个命令调用 `pip`、另一个命令调用 Conda 下的 `python`。
2. 确认 Python 版本。课程代码优先使用 Python 3.10+；新装环境若显示 3.8/3.9，删除并用较新解释器重建虚拟环境。
3. 确认所在目录。下面的检查告诉你解释器和当前工作目录；相对文件路径是相对工作目录解析的。

```powershell
.\.venv\Scripts\python.exe -c "import sys; from pathlib import Path; print(sys.executable); print(Path.cwd())"
```

预期：第一行路径指向当前仓库下的 `.venv`；第二行是你运行命令时所在的位置。

4. 如果终端说 `No module named torch`，通常是 torch 装到了另一个 Python。不要反复重装；比较报错终端的 Python 路径与上面的 `sys.executable`，再用对应解释器的 `-m pip install torch`。

## 跑脚本、REPL 与报错

- **脚本运行**：命令给 Python 一个文件，程序从第一行执行到结尾。接下来课程示例都采用这种方式，结果可重现。
- **交互提示符 REPL**：运行 `python` 后出现 `>>>`，适合试一两行表达式；输入 `exit()` 离开。长程序放在 `.py` 文件，不要只留在临时窗口里。
- **当前工作目录**：脚本读取相对路径时所用的起点。先运行 `pwd`（PowerShell 也支持）或在 Python 打印 `Path.cwd()`，不要猜路径。
- **traceback**：Python 列出异常调用路径。先看最后一行的异常类型和消息，再向上找第一行属于自己脚本的行号。

常见报错速查：

| 报错 | 第一件该检查的事 |
|---|---|
| `python` / `py` 无法识别 | 解释器是否已安装；重新打开终端并确认 PATH，或使用 IDE/安装器提供的完整路径 |
| `No module named torch` | 安装与运行是否用了同一解释器 |
| `FileNotFoundError` | 当前工作目录与相对文件路径是否匹配 |
| `SyntaxError` | 报错行附近的括号、引号和冒号是否成对；上一行是否忘记结束表达式 |
| `IndentationError` | 同一代码块是否统一缩进；不要混用 tab 与空格 |

## 可运行的 Python 暖身

先按上面的方式装好环境，然后运行独立脚本。它涵盖字符串、列表、字典、循环、条件、函数、未知 token 错误和错位的 next-token 样本。

```powershell
.\.venv\Scripts\python.exe content/01-Foundations/examples/python_basics.py
```

预期输出以 `sentence: 猫 喜欢 睡觉` 开头、以 `caught expected error: unknown token: 未知词` 结束；中间会显示 token id 列表 `[0, 1, 2]` 和错位的一对输入/目标。完整脚本可从 [python_basics.py](examples/python_basics.py) 打开或下载。先读 [[Python-Prerequisites|Python 前置能力]]，再亲自修改脚本中的句子和词表。

## 下一步

环境能导入 torch 后，前往 [[PyTorch-Core-Practice|PyTorch Tensor、autograd 与 nn.Module 动手课]]。若只学普通 Python，继续 [[Python-Prerequisites|Python 讲义]]。中文官方参考：[Python 教程](https://docs.python.org/zh-cn/3/tutorial/index.html) 和 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)。
