# 实验记录（Experiments）

> 单卡 **RTX 3080 Ti 12GB** 上，从零复现 MiniMind（63.9M）的 **继续预训练 → 监督微调（SFT）→ LoRA 参数高效微调 → DPO 偏好对齐 → GRPO 强化学习 → 可验证奖励 RLVR** 全流程。
> 所有数字均来自真实训练日志（`*/` 目录下的 `*_curve.csv`、`metrics.txt`），曲线图见 `assets/`。

## 1. 运行环境

| 项 | 配置 |
| --- | --- |
| GPU | 1 × NVIDIA RTX 3080 Ti（12 GB GDDR6X），驱动 580.76.05 |
| 框架 | Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.57.6、datasets 3.6.0 |
| 精度 | BF16 混合精度（`torch.cuda.amp.autocast`） |
| 模型 | MiniMind，hidden=768、8 层、8 头、**GQA（KV 头=4）**、词表 6400，主干 **参数量 63.91M** |
| 数据 | ModelScope `gongjy/minimind_dataset`：pretrain 全量 127.0 万条、SFT 全量 90.6 万条多轮对话 |

> 复现踩坑：该机器上 `pip` 直接拉取数百 MB 的大 wheel（torch / nvidia-cudnn 等）会零增长卡死，
> 最终改用 **curl 逐包下载 + `pip install --no-index --find-links` 离线安装**解决，详见 `../reproduced/`。

## 2. 数据准备：等间隔抽样

全量数据在单卡上训练 2 轮需要十余小时。为在可控时间内完成端到端复现并对比**数据规模效应**，
采用**等间隔抽样**（每隔 k 行取 1 条，覆盖整个文件、避免头部数据分布偏斜）构造两档子集：

| 子集 | pretrain 条数 | SFT 条数 | 占全量比例（pretrain/SFT） |
| --- | --- | --- | --- |
| small | 60,000 | 20,000 | 4.7% / 2.2% |
| medium | 150,000 | 50,000 | 11.8% / 5.5% |

抽样脚本见 `../reproduced/make_subsets.py`。

## 3. 超参数

| 阶段 | batch | 梯度累积 | 有效 batch | 学习率 | 调度 | 最大长度 | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrain | 32 | 8 | 256 | 5e-4 | cosine + warmup | 340 | 2 |
| SFT | 16 | 1 | 16 | 1e-5 | cosine | 768 | 2 |
| LoRA | 32 | 1 | 32 | 1e-4 | cosine | 340 | 3 |
| DPO | 4 | 1 | 4 | 4e-8 | — | 1024 | 1 |

优化器 AdamW、梯度裁剪 1.0；SFT 从上一阶段 pretrain 权重热启动，且**仅对 assistant 回复部分计算损失**；
LoRA 在 medium 的 full-SFT 基座上只训练注入的低秩适配器（rank 适配 Q/V 等投影），主干全部冻结。

## 4. Pretrain + SFT 结果

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/loss_curves.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：上图比较小型与中型模型从零预训练的损失变化；下图比较从预训练检查点继续做监督微调时的损失变化。中型模型训练步数更多，最终损失也更低。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/loss_curves.png" alt="63.9M MiniMind 在单张 RTX 3080 Ti 上从零预训练与监督微调的训练损失曲线。" loading="lazy" />
    </a>
  </div>
  <figcaption>上图比较小型与中型模型从零预训练的损失变化；下图比较从预训练检查点继续做监督微调时的损失变化。中型模型训练步数更多，最终损失也更低。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

| 实验 | 阶段 | 训练步数 | 耗时 | 起始→最终 loss | 显存峰值 | 平均 GPU 利用率 |
| --- | --- | --- | --- | --- | --- | --- |
| **small** | Pretrain | 3,750 | 10.3 min | 7.73 → **3.35** | 7.36 GB | 97.1% |
| **small** | SFT | 2,500 | 8.0 min | 3.91 → **3.17** | 7.36 GB | 97.1% |
| **medium** | Pretrain | 9,376 | 25.5 min | 7.31 → **2.53** | 7.36 GB | 98.8% |
| **medium** | SFT | 6,250 | 19.6 min | 2.82 → **2.21** | 7.36 GB | 98.8% |

## 5. LoRA 参数高效微调（PEFT）

在 medium full-SFT 基座上，用同一份 2 万条 SFT 子集训练 LoRA 适配器（3 轮、1875 步、**256 秒**）：

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/lora_compare.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：左图显示 LoRA 微调过程中的训练损失及全量 SFT 基线；右图对比全量 SFT 与 LoRA 的资源开销。各项数字对应图中这次实验的配置。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/lora_compare.png" alt="LoRA 微调损失曲线，以及全量 SFT 与 LoRA 的可训练参数、峰值显存、检查点体积和训练时间对比。" loading="lazy" />
    </a>
  </div>
  <figcaption>左图显示 LoRA 微调过程中的训练损失及全量 SFT 基线；右图对比全量 SFT 与 LoRA 的资源开销。各项数字对应图中这次实验的配置。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

| 指标 | 全量 SFT | LoRA | 对比 |
| --- | --- | --- | --- |
| 可训练参数量 | 63.91 M | **0.393 M** | LoRA 仅为全量的 **1/163（0.61%）** |
| 检查点体积 | 132 MB | **0.779 MB** | 缩小约 169 倍，便于多任务分发 |
| 显存峰值 | 7.36 GB | **4.60 GB** | 降低约 37%（冻结主干、优化器状态更少） |
| 20k 数据训练耗时 | 479 s（2 epoch） | **256 s（3 epoch）** | 轮次更多反而更快 |
| loss | 收敛到 2.21 | 在 2.15–2.57 间围绕基座波动 | 见下方分析 |

**分析（诚实）**：LoRA 从已经拟合好的 full-SFT 基座起步，loss 本就处于 2.2 水平，再用**同分布**数据训练不会显著下降，
而是围绕基座小幅波动——这正说明低秩适配**没有破坏基座已有能力**（无灾难性遗忘），其价值在于以极小代价做**增量/领域适配**。
推理时把 0.78MB 适配器叠加到冻结基座即可正常生成（`eval_llm.py --weight full_sft --lora_weight lora_demo`），
且模型对「你是谁」类身份认知回答更稳定。若要体现 LoRA 的领域增益，应换用医疗/身份等**专门领域小数据**（对应官方 `lora_medical`）。

## 6. DPO 偏好对齐（Direct Preference Optimization）

在 medium full-SFT 基座上，用 **17,166 对 chosen/rejected** 偏好数据做 DPO（β=0.15、lr=4e-8、batch=4、1 轮 4292 步、**744 秒**）。
DPO 需同时持有 policy 与冻结的 reference 两个模型，故显存略升到 5.71GB。

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/dpo_curve.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：DPO 损失每批波动较大，移动平均约在 0.62–0.69 之间；前后半程均值从 0.642 降到 0.620。初始基线为 −ln(2)≈0.693。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/dpo_curve.png" alt="17,166 组偏好样本上的 DPO 训练损失：灰线为逐批损失，橙线为移动平均，虚线标出初始基线与前后半程均值。" loading="lazy" />
    </a>
  </div>
  <figcaption>DPO 损失每批波动较大，移动平均约在 0.62–0.69 之间；前后半程均值从 0.642 降到 0.620。初始基线为 −ln(2)≈0.693。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

| 指标 | 数值 |
| --- | --- |
| 偏好对 / 步数 | 17,166 对 / 4,292 步 |
| 耗时 | 12.4 min |
| 理论初始 loss | -ln2 = **0.693**（chosen、rejected 等概率） |
| 区间均值 | 前半 0.6415 → 后半 **0.6201** |
| 末步 / 最低 loss | 0.4276 / 0.3908 |
| 显存峰值 / 利用率 | 5.71 GB / 98.4% |

**分析**：DPO loss 从理论值 -ln2≈0.693 起步，训练后移动平均缓慢下移，说明 chosen 相对 rejected 的对数概率差（隐式 reward margin）被逐步拉开；
因学习率刻意取极小（4e-8，防止灾难性遗忘）且 batch=4 噪声大，曲线呈「高噪声、缓慢下降」形态。
DPO **只调整输出偏好/风格、不增加知识**：对齐后回答更简短收敛（见 `generation_samples.md`），但 63M 的知识边界不变。

## 7. GRPO 强化学习（纯规则奖励 / CISPO）

在 full_sft 基座上做 GRPO，**移除官方无条件加载的 1.8B 学习型奖励模型**（fp16≈3.6GB），奖励完全由内置规则给出（回答长度 20–800、`</think>` 思维段 20–300、标签恰好 1 个、3-gram 重复惩罚），改造最小 diff 见 [`../reproduced/grpo_rule_README.md`](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/reproduced/grpo_rule_readme)。

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/grpo_curve.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：左上为规则奖励及其移动平均，右上为参考策略 KL；左下为策略损失，右下同时显示学习率衰减和回答长度变化。奖励只检查格式与长度，不判断答案内容是否正确。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/grpo_curve.png" alt="300 步 GRPO 规则奖励实验的训练曲线：奖励、参考策略 KL、策略损失，以及学习率和回答长度。" loading="lazy" />
    </a>
  </div>
  <figcaption>左上为规则奖励及其移动平均，右上为参考策略 KL；左下为策略损失，右下同时显示学习率衰减和回答长度变化。奖励只检查格式与长度，不判断答案内容是否正确。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

| 指标 | 数值 |
| --- | --- |
| prompt / 步数 | 600 / 300（batch 2 × num_gen 4） |
| 耗时 / 速度 | 21.7 min / ≈4.33 s·step（在线 rollout 为主） |
| lr / β / loss 类型 | 3e-7→3e-8 余弦 / 0.1 / CISPO |
| Reward 均值 | 全程 0.391，中段(101–200) 0.494（batch=2 噪声大） |
| KL(ref) | 均值 -0.0042、平均 |KL| 0.0051，始终贴近 0 |
| 组内优势 | max|Adv Mean|=0（中心化）、Adv Std 均值 0.93 |
| 显存峰值 | 9.9 GB（policy+reference 双模型 + 8 路生成 KV） |

**前后对比**（同 5 个 prompt、同种子/采样，脚本 `../reproduced/compare_grpo_gen.py`）：平均规则分 full_sft **0.109 → grpo 0.290**，最佳一条 0.047→1.750（满分），长度更受控、思维链格式更规范、重复更少。

**分析（诚实）**：组内优势均值严格为 0、KL 贴近 0、LR 余弦衰减均与理论一致，证明 GRPO 机制正确跑通；但规则奖励只定义"格式/长度/不重复"、**不评判内容对错**，故 63M 模型事实正确性并未提升；batch=2 使 Reward 噪声较大、有 1 条在闭合 think 前触及 512 token 上限。原始日志/逐步 CSV/对比全文见[上游 GRPO 原始记录目录](https://github.com/FRS2003/hands-on-llm/tree/a828c96e683270a784d5d57e8aed06d3c9f9a5d3/02-train-from-scratch/experiments/grpo)。

**100 条 held-out 定量评测**：评测题从全量 19,502 条等间隔抽取、并**排除 GRPO 训练用的 600 条**（脚本 `../reproduced/eval_grpo_100.py`，逐条明细 `grpo/eval_100_detail.csv`）：

| 权重 | 规则分均值 | 长度合规(20–800) | think段合规 | 标签恰1个 | 3-gram重复惩罚 | 平均字符 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| full_sft | 0.255 | 1.00 | 0.05 | 0.77 | 0.120 | 411 |
| grpo | **0.339** | 0.98 | 0.10 | 0.75 | **0.096** | 427 |

样本扩到 100 条后结论更稳健：规则分 **+0.084**、3-gram 重复度下降（0.120→0.096）、think 段长度合规率翻倍；"标签恰 1 个"比例基本持平（-0.02，属采样噪声）。提升温和且集中在"格式/重复度"，再次印证纯规则奖励不改变内容正确性。注：本次 max_new_tokens=300，think 段容易超过规则设定的 300 字符上限，故 think 段合规率绝对值偏低。

## 8. 五阶段统一生成横评（pretrain → GRPO）

用**同一组 6 个 prompt、同一随机种子与采样参数**让五个权重分别生成（pretrain 走续写模式、其余走对话模板；LoRA 是叠加在 SFT 主干上的适配器）。完整对比见独立文档 [`stage_evolution.md`](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/stage_evolution)（逐条原文 `stage_evolution.txt`，脚本 `../reproduced/compare_5stages.py`）。

| 阶段 | 平均字符 | 闭合 think 标签 | 定位 |
| --- | ---:| ---:| --- |
| pretrain | 209 | 0/6 | 只会续写、不遵循指令、事实易错乱 |
| SFT | 544 | 4/6 | 学会助手角色与 markdown/代码块结构 |
| LoRA | 522 | 5/6 | 仅训 0.61% 参数即复现 SFT 的指令遵循 |
| DPO | 529 | 5/6 | 调偏好/措辞，不增加知识 |
| GRPO | 517 | 4/6 | 强 KL 约束下小幅优化格式 |

**核心结论**：能力跃迁最大的是 **pretrain → SFT**（从"不会答题"到"会答题"），之后三个阶段都在 SFT 基础上做精细化调整，五阶段不可互相替代。

## 9. 架构与精度消融（MHA / GQA / MQA × fp32/bf16/fp16 × batch）

在同一 63M 骨架上只改 KV 头数（MHA=8 / GQA=4 / MQA=1）与精度，实测训练显存/吞吐与推理 KV cache（完整数据、图与解读见 [`ablation/`](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/ablation/readme)，脚本 `../reproduced/bench_arch.py`）：

| 维度 | 关键实测（RTX 3080 Ti） |
| --- | --- |
| 参数量 | MHA 68.63M → GQA 63.91M(-6.9%) → MQA 60.37M(-12%)，只动 K/V 投影 |
| 训练显存(bs8,bf16) | 2623 / 2490 / 2312 MB，训练侧省幅温和（激活大头与 KV 头数无关） |
| 训练吞吐(bs8,bf16) | 60.0 / 64.2 / 69.7 k tok/s，MQA 比 MHA 快约 16% |
| 混合精度 | bf16 比 fp32 省 35–49% 显存、约 1.8× 吞吐；bf16/fp16 显存相同，bf16 动态范围大更稳 |
| 推理 KV cache(4096) | 100.7 / 50.3 / 12.6 MB，**与 KV 头数成正比（8:4:1）** |

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/arch_ablation.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：上排比较不同 batch 下的峰值显存与训练吞吐；左下比较上下文长度对推理 KV cache 的影响；右下比较注意力结构的参数量。实验固定模型骨架，只改变 KV 头数或精度。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/arch_ablation.png" alt="MHA、GQA、MQA 的四组实验对比：训练显存、吞吐量、不同上下文长度下的 KV cache，以及参数量。" loading="lazy" />
    </a>
  </div>
  <figcaption>上排比较不同 batch 下的峰值显存与训练吞吐；左下比较上下文长度对推理 KV cache 的影响；右下比较注意力结构的参数量。实验固定模型骨架，只改变 KV 头数或精度。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

**一句话结论**：训练时 GQA/MQA 省得有限；推理长上下文/高并发时 KV cache 按 KV 头数等比缩小（MQA 仅 MHA 的 1/8），这才是 GQA/MQA 的核心价值，也是 LLaMA-2/3 选 GQA 的根本原因。小 batch(bs=1) 下三架构差异被固定开销抹平，算子优势要在合理 batch 下才显现。

## 10. 可验证奖励 GRPO（RLVR：答案对错自动判分）

纯规则奖励只改格式、不判对错（见 §7），这里用**答案可程序化判分**的一位加法做 RLVR：答对 +1 / 答错 0，先 SFT 冷启动到"采样能偶尔答对"的甜区，再用 GRPO 放大正确采样（完整过程、甜区探测与负结果对照见 [`verifiable_rl/`](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/verifiable_rl/readme)，脚本 `../reproduced/train_grpo_arith.py`）。

| 环节 | 配置 / 结果 |
| --- | --- |
| 冷启动 SFT | full_sft → 一位加法 1 epoch（250 步）：greedy 0.633 / 采样 0.503（甜区） |
| GRPO | B8×G4、lr 4e-6、β(KL)=0.08、300 步、CISPO、bf16，复用官方 TorchRolloutEngine |
| **独立 120 题** | **greedy 0.633→0.792（+25%）；采样三种子均值 0.503→0.603（+20%）** |

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/verifiable_rl_curve.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：在可程序化判分的一位数加法任务中，60 题留出子集的准确率随训练总体上升；虚线给出 SFT 冷启动的 0.633 基线和 GRPO 结束时的 0.792。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/verifiable_rl_curve.png" alt="一位数加法任务上的 GRPO 训练准确率：蓝线为留出题集准确率，虚线标出 SFT 起点和 GRPO 最终结果。" loading="lazy" />
    </a>
  </div>
  <figcaption>在可程序化判分的一位数加法任务中，60 题留出子集的准确率随训练总体上升；虚线给出 SFT 冷启动的 0.633 基线和 GRPO 结束时的 0.792。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

**边界（负结果同样关键）**：① 任务超出能力边界（两位加法不会进位）时组内全错、优势恒 0，RL 教不会新知识；② SFT 已到天花板（greedy 0.99）时 RL 无空间；③ lr 过大、KL 锚定太弱会**策略崩溃**（greedy 崩到 0.03–0.27、KL 偏离到 −2.9）。结论：**RL 只放大已有能力、不注入知识，且必须靠 β-KL 锚定 + 保守学习率才能稳定训练。**

## 11. GPU 画像与推理速度

<figure class="study-figure">
  <div class="study-image-viewport">
    <a class="study-image-link" href="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/gpu_profile.png" target="_blank" rel="noopener noreferrer" aria-label="打开原图：训练期间 GPU 利用率大部分时间接近满载；红线记录显存占用，绿色曲线记录 GPU 利用率。纵轴刻度分别位于左右两侧，图示约 2,700 秒的运行过程。">
      <img class="study-figure-image" src="https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/assets/gpu_profile.png" alt="一次中型模型训练运行中的 GPU 利用率与显存占用随时间变化；峰值显存约 7,363 MB，平均 GPU 利用率约 98.8%。" loading="lazy" />
    </a>
  </div>
  <figcaption>训练期间 GPU 利用率大部分时间接近满载；红线记录显存占用，绿色曲线记录 GPU 利用率。纵轴刻度分别位于左右两侧，图示约 2,700 秒的运行过程。 点击图片可在新标签页查看原尺寸。</figcaption>
</figure>

- 训练阶段 GPU 利用率长期 96–99%、温度峰值 72–75°C；显存峰值 pretrain/SFT 7.36GB、LoRA 4.60GB。
- 推理（FP16，单条 `model.generate`）解码速度约 **47–100 tokens/s**。

## 12. 关键发现

1. **数据规模的边际收益清晰可见**：pretrain 数据量 ×2.5（6 万→15 万），同架构同超参下最终 loss 由 3.35 降到 2.53；
   SFT 由 3.17 降到 2.21，medium 模型生成的语句连贯度、指令遵循度明显更好（见 `generation_samples.md`）。
2. **Pretrain 与 SFT 分工明确（消融）**：只做 pretrain 的模型只会**续写**、无法遵循指令
   （让写斐波那契函数时输出一串数字）；经过 SFT 后才学会 chat template、以助手身份分点作答。
3. **PEFT 性价比**：LoRA 只训 0.61% 参数、检查点缩小两个数量级、显存降 37%，适合多领域低成本适配与热插拔。
4. **小模型单卡可训**：63.9M 模型在 12GB 卡上 bf16 训练毫无压力，瓶颈在**数据量与训练 token 数**而非显存/算力。
6. **DPO 对齐特性**：loss 从理论值 -ln2 起步、以极小学习率缓慢下移，验证了偏好目标；DPO 改变输出风格而非知识容量，需严格控制 lr 防止遗忘。
5. **loss 与生成质量并不完全等价**：SFT loss 降到 2.2 后模型能稳定输出结构化中文，但受 63M 参数 + 仅约 5% 全量语料限制，
   仍有事实错误、重复、代码语法错误——与 Chinchilla「小模型需要足够 token」的结论一致，是后续扩数据/扩参的方向。
7. **RL 只放大已有能力、不注入知识（RLVR 实测，见 §10）**：一位加法冷启动到甜区后，可验证奖励 GRPO 把 held-out greedy 0.633→0.792；但任务超能力边界（组内全错无梯度）、SFT 已到天花板、或 lr 过大 KL 过弱（策略崩溃）时均无效——RLVR 生效前提是「SFT 已具备但未稳固」，且需 β-KL 锚定参考模型。

## 13. 目录与复现

- `small/`、`medium/`、`lora/`、`dpo/`：各自的 `*_curve.csv`（逐步 loss/lr）、`metrics.txt`、原始生成记录、GPU 采样 CSV；
- `grpo/`：GRPO 原始训练日志、逐步指标 CSV、metrics 与前后生成对比文本；
- `grpo/eval_100_*`：100 条 held-out 定量评测（指标表 + 200 行逐条明细）；
- `stage_evolution.md / .txt`：五阶段同种子生成横评（解读文档 + 逐条原文）；
- `ablation/`：MHA/GQA/MQA × 精度 × batch 消融数据、KV cache 表、解读与对照图；
- `verifiable_rl/`：可验证奖励 GRPO（RLVR）数据、完整训练/评测日志、甜区与策略崩溃对照、解读文档；
- `assets/`：loss 对比曲线、GPU 画像、LoRA 资源对比图；
- `generation_samples.md`：pretrain-only / small-SFT / medium-SFT / LoRA / DPO 生成样例对比；
- 完整复现命令见 [`../reproduced/`](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/reproduced/readme)；逐步汇总见 [`training_log.csv`](https://blog.linhk.top/Language-Model-Systems-Atlas/06-hands-on-llm/source/02-train-from-scratch/experiments/training_log.csv)。

> 说明：模型权重（主干 .pth 约 132MB、LoRA 适配器 0.78MB）按 `.gitignore` 约定不上传；数据来自 ModelScope 公开数据集，按其许可使用。
