# CS336 中文精讲笔记 · LLMs from Scratch

> 📚 **本模块属于 [Hands-on-LLM 动手学大模型](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/readme) 学习库 · 返回总览看完整学习路线**

> 学习 Stanford **CS336: Language Modeling from Scratch** 系统整理的**中文精讲笔记（小白友好版）**，
> 按课程讲次 L1–L17 系统梳理，用尽量通俗的方式讲清大模型从零构建的完整链路。

## 笔记目录（`notes/`）
| 讲次 | 核心主题 | Markdown | Word |
| --- | --- | --- | --- |
| L1–L3 | Tokenization 与 BPE、训练资源核算、Transformer 基础 | [在线阅读](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/01_l1-l3_%E5%88%86%E8%AF%8D_%E8%B5%84%E6%BA%90%E6%A0%B8%E7%AE%97_transformer) | [Word](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/01_l1-l3_%E5%88%86%E8%AF%8D_%E8%B5%84%E6%BA%90%E6%A0%B8%E7%AE%97_transformer.docx) |
| L4–L8 | 注意力替代方案、MoE 混合专家、GPU 系统与性能、Triton 编程、并行 | [在线阅读](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/02_l4-l8_%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9B%BF%E4%BB%A3_moe_gpu%E7%B3%BB%E7%BB%9F_triton) | [Word](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/02_l4-l8_%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9B%BF%E4%BB%A3_moe_gpu%E7%B3%BB%E7%BB%9F_triton.docx) |
| L9–L11 | Scaling Laws、推理优化（KV Cache / 量化 / 蒸馏）、缩放案例 | [在线阅读](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/03_l9-l11_%E7%BC%A9%E6%94%BE%E5%AE%9A%E5%BE%8B_%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96) | [Word](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/03_l9-l11_%E7%BC%A9%E6%94%BE%E5%AE%9A%E5%BE%8B_%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96.docx) |
| L12–L14 | 模型评估、训练数据来源与清洗处理 | [在线阅读](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/04_l12-l14_%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0_%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE) | [Word](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/04_l12-l14_%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0_%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE.docx) |
| L15–L17 | 对齐（SFT/RLHF/DPO）、推理模型与 GRPO、多模态 | [在线阅读](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/05_l15-l17_%E5%AF%B9%E9%BD%90_grpo_%E5%A4%9A%E6%A8%A1%E6%80%81) | [Word](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/01-foundations/notes/05_l15-l17_%E5%AF%B9%E9%BD%90_grpo_%E5%A4%9A%E6%A8%A1%E6%80%81.docx) |

> 💡 每篇笔记都提供**在线 Markdown**与**可下载 Word**两个版本，Word 版已排好标题、表格与代码块样式，适合离线批注或打印。

## 知识地图
分词与资源核算 → Transformer 架构 → 注意力变体与 MoE → GPU/Triton/并行训练
→ Scaling Law 与推理优化 → 评估与数据工程 → 对齐（含 GRPO）→ 多模态。

## 说明与免责
- 本模块**只包含课程的中文精讲笔记**，不含课程官方讲义（lectures）、阅读论文（readings）与作业（assignments），这些材料的版权归 Stanford CS336 及相应作者所有。
- 笔记为学习过程中的理解，难免有疏漏，欢迎 Issue 讨论指正。
- 课程主页：[CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
