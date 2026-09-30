---
title: A2 Systems：训练性能、显存与并行
description: 原创中文导读，解释 CS336 A2 的性能测量、激活重计算、FlashAttention、DDP、状态分片和 FSDP。
tags:
  - CS336
  - A2
  - systems
  - GPU
---

A1 关注“语言模型由什么组成”；A2 关注“这些组件怎样高效地跑在 GPU 和多 GPU 系统上”。本页讲概念和分析方法，不提供官方作业实现或数值答案。组件接口、实验设置以 [Stanford A2 handout](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf) 和[官方仓库](https://github.com/stanford-cs336/assignment2-systems)为准。

> [!warning] 作业与 AI 工具
> Spring 2026 handout 限制 AI 工具实现作业代码。本页是原创概念导读，不含解题代码或测试答案。做作业前请核对 handout 的当前政策。

## 1. 先测量，再优化

优化前要先回答三个不同问题：训练 step 总共用了多久？时间花在哪些 CPU/GPU 操作上？显存峰值出现在何时、由哪些张量造成？一个 end-to-end timer 只能回答第一个问题；Nsight Systems 一类系统 profiler 有助于定位主机、CUDA 调用和 kernel 时间线；PyTorch memory profiler 则用于检查分配和峰值显存。

GPU 调用通常是异步提交的。若只用 CPU 时钟量一次函数调用，测到的可能只是“提交任务的时间”，不是 GPU 完成任务的时间。可靠测量应使用与目标设备同步的计时方法，并进行预热和重复测量。记录均值或中位数、波动、batch/序列长度、精度、模型配置、GPU 型号和软件环境；profiling 本身也会带来开销。

Roofline 思路用算术强度判断瓶颈方向：

$$
I=\frac{F}{M},\qquad
t\gtrsim \max\left(\frac{F}{P_{\mathrm{compute}}},\frac{M}{B_{\mathrm{memory}}}\right),
$$

其中 $F$ 是浮点运算量，$M$ 是读写的字节量，$P_{\mathrm{compute}}$ 是有效计算吞吐，$B_{\mathrm{memory}}$ 是有效带宽。若运算相对数据搬运多，可能受计算吞吐限制；若反复读写大量中间结果，可能受内存带宽限制。融合算子、降低精度或改用分块算法，解决的是不同瓶颈，必须通过目标 workload 的测量来判断。

**比较之前先固定条件：**相同模型、输入 shape、dtype、warmup 和测量区间；区分纯前向、前向加反向、完整 optimizer step 和端到端吞吐。只比较一个 kernel 的最快时间，不能代表训练整体更快。

## 2. 单 GPU：精度、激活与重计算

### 混合精度

低精度数据格式可以提升矩阵乘吞吐并减少激活/参数占用，但有限的指数范围和尾数精度会改变数值误差。累加精度、梯度表示和 optimizer state 的 dtype 可能不同。要同时观察速度、峰值显存、loss/梯度是否稳定；不能假设所有张量都应使用同一种 dtype。

### Activation checkpointing

反向传播需要前向时产生的一些中间激活。保存所有激活可减少反向重算，但会占用大量显存。Activation checkpointing 只保存选定边界的激活，在反向时重新计算中间部分：显存下降，计算量与 step 时间通常上升。边界选得不同，内存峰值和额外计算也不同。

可用一个粗略账本思考：若某层保存激活约占 $A$ 字节，保存 $L$ 层约为 $LA$；改为每 $k$ 层留一个边界，峰值项会下降，但每次反向要重算若干层。实际峰值还受临时张量、allocator、kernel 工作区和并行流水影响，估算要用 profiler 校验。

## 3. FlashAttention：减少注意力的中间数据搬运

给定因果多头注意力的输入

$$
Q,K,V\in\mathbb{R}^{B\times H\times T\times d_h},
$$

常规写法形成分数矩阵 $S=QK^\top/\sqrt{d_h}$ 和概率矩阵 $P=\operatorname{softmax}(S+M)$，二者 shape 都是 $(B,H,T,T)$。因此 attention 中间矩阵的空间随序列长度按 $T^2$ 增长；因果 mask 虽然禁止访问未来位置，朴素实现仍可能物化大块矩阵。

FlashAttention 的核心是按 query/key tile 分块计算，并以在线 softmax 累积每一行的归一化分母和加权 value；避免把完整 $T\times T$ 注意力矩阵写入高带宽显存（HBM），反向时按需重算部分中间值。它主要减少内存流量和中间存储；注意力的主要点积工作量仍随 $T^2$ 增长，所以它不会让长序列注意力变成线性计算。

输出 shape 仍为 $(B,H,T,d_h)$，只是中间数据的存储和搬运策略改变。比较优化实现时要检验前向输出及梯度与参考实现的数值误差，再在多个 $B,H,T,d_h$ 和 dtype 上测吞吐/峰值显存。tile 太小会增加调度和重复加载，tile 太大可能超出片上资源；寄存器占用、共享内存和 occupancy 都会影响实际速度。

## 4. 多 GPU：数据并行与状态分片

### DDP 的同步逻辑

数据并行把 batch 分给 $N$ 个 rank，每个 rank 保存一份模型副本并计算自己的梯度。若 rank $i$ 的本地 batch 大小为 $B_i$、梯度为 $g_i$，同步后的梯度应对应全局 batch 的加权平均；各 rank 再用相同更新保持参数一致。

$$
g_{\mathrm{global}}=\frac{1}{\sum_i B_i}\sum_i B_i g_i.
$$

实际工作负载 batch 相同的时候才简化为各 rank 梯度的普通平均。梯度同步需要通信；把多个小张量合并为 bucket 可减少通信调用开销，反向时逐 bucket 通信则有机会与尚未完成的计算重叠。小模型或少量 GPU 上，通信开销可能超过增加的计算吞吐。

一个环形 all-reduce 的粗略通信模型为

$$
t_{\mathrm{comm}}\approx 2(N-1)\alpha+
2\frac{N-1}{N}\frac{S}{\beta},
$$

其中 $N$ 是 rank 数，$S$ 是待同步字节数，$\alpha$ 是每阶段延迟，$\beta$ 是链路有效带宽。它只用于理解“消息延迟 + 数据量/带宽”的构成；拓扑、collective 算法和框架实现会改变实际时间。

### Optimizer state sharding 与 FSDP

DDP 通常在每张卡上复制模型参数、梯度和优化器状态。AdamW 为参数维护额外动量等状态，模型较大时这些副本会成为显存瓶颈。

- **Optimizer state sharding：**将优化器状态分散到各 rank，减少重复状态占用；参数仍需在更新及前向/反向中保持可用，实施时需要相应的同步和参数更新处理。
- **Fully Sharded Data Parallel（FSDP）：**进一步切分参数、梯度和优化器状态。计算某个模块时，需要临时 all-gather 对应权重；梯度可 reduce-scatter 回各 rank 的分片。这样降低常驻显存，但增加通信，并需要选择模块粒度和通信/计算重叠策略。

因此，“每卡显存变少”不自动等于“训练更快”。模块切得过细会带来频繁 collective 和延迟；切得过粗又可能在 all-gather 时造成显存峰值。数据并行、FSDP、tensor parallel 的适用性取决于模型大小、batch、序列长度、GPU 互连和目标吞吐。

## 5. 自己读 profile 时要问的问题

- 测量的是哪个边界：只含前向，还是包含 backward、optimizer、数据等待和通信？
- GPU 是否因 Python、数据加载或同步等待而空转？哪个 profile 时间线能支持这个判断？
- 峰值显存对应参数、梯度、优化器状态、保存的激活还是临时 attention tensor？
- 激活重计算节省多少显存，代价是多少额外 FLOPs/时间？
- attention 优化是否在目标 $T$ 上显著减少内存流量？准确度误差和速度变化分别是多少？
- 增加 GPU 后，总吞吐如何变化？每步通信占比、扩展效率和每样本时间如何变化？
- 混合精度或 kernel 修改后，loss、梯度和最终验证行为是否仍与基线相容？

这些是分析问题，不是 handout 中具体实验的解答。A2 还涉及 Triton kernel、并行实现、书面分析与性能比较，需按官方仓库自行完成。

## 课程与笔记入口

- 官方相关讲次：L02（PyTorch 与资源核算）、L05（GPU/TPU）、L06（kernels/Triton）、L07–L08（parallelism）。课次主题见 [[course-map|官方课程地图]]。
- [A2 官方 handout 与仓库](https://github.com/stanford-cs336/assignment2-systems)
- 第三方、MIT 许可的 Hands-on-LLM 笔记中可按文件名寻找 `02_L4-L8_注意力替代_MoE_GPU系统_Triton.md`；这是把数讲内容并为一个阅读单元的**笔记文件标签**，不是 Stanford 官方课号。见[笔记目录](https://github.com/FRS2003/hands-on-llm/tree/main/01-foundations/notes)。
