---
title: 来源、许可与在线材料索引
description: 官方课程和文档入口、参考项目的在线来源及可确认的许可边界。
tags:
  - reference
  - sources
---

本站的中文说明和代码示例为独立撰写，目的是帮助理解课程所需的 Python、张量和训练概念。本站不复制 Stanford 讲义、第三方笔记、书籍章节或图示。私人离线副本存放在 Quartz 发布树之外；本网站只链接在线来源，不打包或链接本地副本。课程作业、讲义和规范应以授课方当期版本为准。

## 官方在线文档与课程

| 主题 | 官方入口 | 用途 |
|---|---|---|
| Python | [Python 3 中文官方教程](https://docs.python.org/zh-cn/3/tutorial/index.html) | 数据结构、函数、模块、文件和异常。 |
| NumPy | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) | 数组、索引、广播和矩阵运算。 |
| PyTorch 入门 | [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) | 张量、自动微分、模型、优化和保存。 |
| PyTorch 自动微分 | [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) | 计算图和梯度行为。 |
| PyTorch 交叉熵 | [CrossEntropyLoss API](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) | logits、类别目标与忽略标签。 |
| CS224N | [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | 当期讲义、作业、日程和课程要求。 |
| CS336 | [Stanford CS336](https://cs336.stanford.edu/) | 当期讲义、作业、依赖和学术诚信要求。 |

在线文档会更新；引用时检查页面对应的版本。

## 参考项目的在线来源

| 项目 | 在线来源 | 许可证与范围 |
|---|---|---|
| Python 中文官方文档 | [Python 中文教程](https://docs.python.org/zh-cn/3/tutorial/index.html) · [官方许可页](https://docs.python.org/3/license.html) | Python 文档站说明其许可；使用具体文档时以官方许可页和页面声明为准。 |
| Think Python | [作者维护的在线教材](https://allendowney.github.io/ThinkPython/) · [第二版中文译本源仓库](https://github.com/apachecn/think-py-2e-zh) | 中文译本仓库的 LICENSE 文件标为 CC BY-NC-SA 4.0；其 README 仍标为 CC BY-NC 3.0，二者不一致。公开再发布前应由权利方澄清。 |
| Datawhale《深入浅出 PyTorch》 | [项目源仓库](https://github.com/datawhalechina/thorough-pytorch) · [LICENSE](https://github.com/datawhalechina/thorough-pytorch/blob/main/LICENSE) | 项目声明 CC BY-NC-SA 4.0；转载或改编时遵守署名、非商业和相同方式共享条款。 |
| 《动手学深度学习》 | [中文版在线教材](https://zh-v2.d2l.ai/) · [源码仓库](https://github.com/d2l-ai/d2l-zh) · [源码仓库 LICENSE](https://github.com/d2l-ai/d2l-zh/blob/master/LICENSE) | 源码仓库的 LICENSE 是 Apache 2.0；这不能单独确定书籍正文、插图或 PDF 的许可，使用这些内容时应查看教材网站和对应文件声明。 |
| Stanford CS224N | [官方课程主页](https://web.stanford.edu/class/cs224n/) | 课程文件的再发布授权需以课程页和各文件声明为准；本站只链接课程主页。 |
| Stanford CS336 | [官方课程主页](https://cs336.stanford.edu/) | 课程文件的再发布授权需以课程页和各文件声明为准；本站只链接课程主页。 |
| FRS2003 Hands-on LLM | [固定上游版本](https://github.com/FRS2003/hands-on-llm/tree/a828c96e683270a784d5d57e8aed06d3c9f9a5d3) · [MIT License](https://github.com/FRS2003/hands-on-llm/blob/a828c96e683270a784d5d57e8aed06d3c9f9a5d3/LICENSE) | 根许可证覆盖仓库作者文档；个别数据集和模型卡可能单独授权。 |

## 许可状态与本站处理

| 来源 | 已核对的信息 | 本站处理 |
|---|---|---|
| Think Python 中文译本 | 仓库 LICENSE 写 CC BY-NC-SA 4.0，README 写 CC BY-NC 3.0，版本冲突。 | 只链接作者版和翻译源仓库；许可未澄清前不复用译文。 |
| Datawhale《深入浅出 PyTorch》 | 项目声明 CC BY-NC-SA 4.0。 | 只链接项目源仓库，不复制正文或插图。 |
| D2L | 源码仓库 LICENSE 为 Apache 2.0；书籍正文、图示和 PDF 的权利应按教材站声明另行确认。 | 只链接在线教材和源仓库，不把代码仓库许可证套用于所有书籍材料。 |
| Python、NumPy、PyTorch 官方文档 | 以各自官方站点和许可页为准。 | 使用官方链接；不镜像页面。 |
| Stanford CS224N、CS336 | 课程文件具体权利声明依课程页和文件。 | 使用官方课程链接；不重新发布讲义和作业 PDF。 |
| FRS2003 Hands-on LLM | 仓库主体为 MIT；ShareAI 数据单独为 Apache-2.0；Alpaca 中文训练样本无法固定其实际来源；部分模型卡标记 `license: other`。 | 仅按 [[../06-Hands-on-LLM/index|镜像说明]]发布 MIT 中文讲义、Word、作者图表和实验记录；数据样本和非 MIT 模型卡只链接上游，不复制。 |
| 其他第三方快照或笔记 | 来源或许可不清楚时，不能据此认定可再发布。 | 不纳入本站内容。 |

这些说明是来源整理，不是法律意见。离线资料副本存放在 Quartz 发布树之外，读者需从自己的私人工作区打开；本站不提供这些副本的链接或下载。本站原创解释不代表 Stanford 官方翻译，也不能替代原课程内容。

## 本站参考页

- [[Python-Prerequisites|Python 前置能力]]
- [[NumPy-and-Tensor-Shapes|NumPy 与张量形状]]
- [[PyTorch-Train-Step|PyTorch 训练一步]]
- [术语表](https://blog.linhk.top/Language-Model-Systems-Atlas/05-reference/glossary)
- [[Tensor-Shape-Cheat-Sheet|张量形状速查表]]
- [[Debugging-Checklist|调试清单]]
