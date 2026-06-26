# 在策略蒸馏（On-Policy Distillation）

**作者**：Kevin Lu，Thinking Machines Lab  
**日期**：2025 年 10 月 27 日  
**原文**：[thinkingmachines.ai/blog/on-policy-distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)  
**DOI**：10.64434/tml.20251026

---

## 目录

- [在策略蒸馏——兼得两者之长](#在策略蒸馏兼得两者之长)
- [实现](#实现)
  - [损失函数：反向 KL](#损失函数反向-kl)
  - [图解](#图解)
  - [伪代码](#伪代码)
- [推理蒸馏](#推理蒸馏)
  - [离策略蒸馏](#离策略蒸馏)
  - [强化学习](#强化学习)
  - [在策略蒸馏](#在策略蒸馏-1)
- [个性化蒸馏](#个性化蒸馏)
  - [训练内部助手](#训练内部助手)
  - [新知识训练会破坏已学行为](#新知识训练会破坏已学行为)
  - [在策略蒸馏恢复后训练行为](#在策略蒸馏恢复后训练行为)
- [讨论](#讨论)
  - [密集监督大幅提升计算效率](#密集监督大幅提升计算效率)
  - [蒸馏可有效复用训练数据](#蒸馏可有效复用训练数据)
  - [强化学习在语义策略空间中搜索](#强化学习在语义策略空间中搜索)
  - [在策略学习作为持续学习工具](#在策略学习作为持续学习工具)
- [结论](#结论)
- [引用](#引用)
- [参考文献](#参考文献)

---

## 引言

> "大语言模型能够在特定领域达到专家水平，这是多种能力叠加的结果：输入感知、知识检索、计划选择以及可靠执行。"

LLM 的训练通常分为三个大阶段：

1. **预训练**——培养通用能力，包括语言使用、广泛推理和世界知识
2. **中间训练（mid-training）**——注入领域知识，如代码、医疗数据库或企业内部文档
3. **后训练（post-training）**——激发特定行为，包括指令遵循、数学推理或对话

**为什么要训练小模型？**

> "在特定领域，经过充分训练的小模型往往优于更大的通用模型。"

优势包括：本地部署（隐私/安全）、持续训练能力、更低的推理成本。

---

## 在策略蒸馏——兼得两者之长

后训练有两种主要思路：

- **在策略训练（On-policy training）**：从学生模型本身采样展开轨迹，并为其分配某种奖励。以强化学习（RL）为代表。
- **离策略训练（Off-policy training）**：依赖外部来源提供的目标输出，让学生模型去模仿。以监督微调（SFT）/ 蒸馏为代表。

**RL 的致命弱点：奖励信号稀疏**

RL 每个训练 episode 只提供固定数量的信息位（bits），无论该 episode 包含多少 token。举一个数学题的例子：学生知道答案错了，但不知道错在哪一步——是运算顺序错了，还是加减法算错了？

**SFT / 蒸馏的优势与局限**

蒸馏通过让学生匹配教师模型的输出分布来传递能力。教师可以提供：
- 每一步的完整 next-token 分布（即"logit 蒸馏"）
- 采样出的完整序列

成功应用案例：[指令遵循（Alpaca）](https://crfm.stanford.edu/2023/03/13/alpaca.html)、[数理推理（OpenThoughts）](https://arxiv.org/abs/2506.04178)、[临床信息抽取](https://arxiv.org/html/2501.00031v1)、[多轮对话](https://arxiv.org/abs/2305.14233)。

然而，离策略训练有一个根本问题：**学生在教师频繁出现的上下文中学习，而不是在学生自身推理时实际遇到的上下文中学习**，从而导致误差累积（compounding error）。相关批评见 [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717)。

**在策略蒸馏：核心思想**

> "在策略蒸馏的核心思想是：从学生模型采样轨迹，然后用高性能教师对每条轨迹的每一个 token 进行打分。"

三种方法对比：

| 方法 | 采样来源 | 奖励信号 |
|------|---------|---------|
| 监督微调（SFT） | 离策略 | 密集 |
| 强化学习（RL） | 在策略 | 稀疏 |
| **在策略蒸馏** | **在策略** | **密集** |

在策略蒸馏同时拥有在策略训练的可靠性和密集奖励的计算效率。

---

## 实现

### 损失函数：反向 KL

训练目标为逐 token 的反向 KL 散度：

$$\text{KL}\Bigl(\pi_\theta \lVert \pi_\text{teacher}\Bigr) = \mathbb{E}_{x \sim \pi_\theta} \Bigl[ \log \pi_\theta(x_{t+1} \mid x_{1..t}) - \log \pi_\text{teacher}(x_{t+1} \mid x_{1..t}) \Bigr]$$

其中 `x` 从学生策略 `π_θ` 中采样，再用教师计算对应的 log 概率差作为 token 级奖励。

**反向 KL 的三个优良特性：**

1. **不可 hack**：低 KL 值始终对应教师认可的行为，无法通过投机取巧降低损失。
2. **众数搜索（mode-seeking）**：学习教师分布的某种具体行为，而非将概率分散到所有可能行为上。
3. **消除暴露偏差（exposure bias）**：学生在自己生成的 token 上下文中学习，而非教师生成的上下文，与 [DAgger](https://arxiv.org/abs/1011.0686)、[Scheduled Sampling](https://arxiv.org/abs/1506.03099) 等方法同源。

**计算优势**：由于不需要等待展开完成再计算奖励，可以使用**更短或部分轨迹**进行训练，大幅降低显存占用。

### 图解

真实案例：一个 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 学生模型在处理物理题时，错误地将其当作纯数学题，忽略了"冰块放在平底锅里会融化"这一物理事实。[Qwen3-235B](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) 教师模型对触发错误推理的关键 token 给予惩罚——这些 token 被称为**分叉词元（forking tokens）**，是引导推理方向的关键节点。

这正是 [Process Reward Models（PRM）](https://arxiv.org/abs/2305.20050) 的思想——在过程中给予逐步反馈，而非仅在结果上给分。

### 伪代码

实现分为四步：

```python
# 第一步：初始化教师客户端
teacher_client = service_client.create_sampling_client(
    base_model=teacher_config.base_model,
    model_path=teacher_config.load_checkpoint_path,
)

# 第二步：从学生采样轨迹（含 logprobs）
trajectories = do_group_rollout(student_client, env_group_builder)
sampled_logprobs = trajectories.loss_fn_inputs["logprobs"]

# 第三步：计算教师反向 KL 奖励
teacher_logprobs = teacher_client.compute_logprobs(trajectories)
reverse_kl = sampled_logprobs - teacher_logprobs
trajectories["advantages"] = -reverse_kl

# 第四步：用 RL 重要性采样损失训练
training_client.forward_backward(trajectories, loss_fn="importance_sampling")
```

详细实现见 [Tinker Cookbook 蒸馏示例](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)，训练循环见 [RL 训练脚本](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/train.py)。

---

## 推理蒸馏

**实验配置：**

- 学生模型：[Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base)
- 教师模型：[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)（后续发现用 Qwen3-8B 效果略好）
- 训练数据：[OpenThoughts-3](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M)（40 万条推理 prompt）
- 评测基准：AIME'24、GPQA-Diamond

> **注**：原文提到，此处使用的 Qwen3-32B 教师和 Qwen3-8B-Base 学生已从 Tinker 模型列表中退役，更新后的方案改用 [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 系列模型。

### 离策略蒸馏

在 40 万条 prompt 上训练 Qwen3-8B-Base 可达到 AIME'24 **60%**。前 5–10 万条 prompt 带来快速提升，之后性能沿可预测的对数线性缩放曲线增长。外推估计，要达到 70% 需约 200 万条 prompt，但缩放定律是否能一直成立并不保证。

### 强化学习

根据 [Qwen3 技术报告](https://arxiv.org/abs/2505.09388)：RL 在使用 17,920 GPU 小时、配合类似的 SFT 初始化后，AIME'24 达到 **67.6%**。

### 在策略蒸馏

从 40 万条 SFT checkpoint 出发，经过约 **150 步**（77K prompt × 每 prompt 4 个采样）的在策略蒸馏，AIME'24 达到 **70%**。

**方法性能对比：**

| 方法 | AIME'24 | GPQA-Diamond | GPU 小时 |
|------|---------|--------------|---------|
| 离策略蒸馏 | 55.0% | 55.6% | 未报告 |
| + 强化学习 | 67.6% | 61.3% | 17,920 |
| + 在策略蒸馏 | **74.4%** | **63.3%** | **1,800** |

Qwen 团队以十分之一的 RL 成本达到了 74.4%，这一结果直接启发了本文的工作。

**计算开销分析：**

| 方法 | AIME'24 | 教师 FLOPs | 学生 FLOPs | 相对 SFT-2M |
|------|---------|-----------|-----------|------------|
| SFT-400K（初始化） | 60% | 8.5×10²⁰ | 3.8×10²⁰ | — |
| SFT-2M（外推） | ~70% | 3.4×10²¹ | 1.5×10²¹ | 1× |
| 强化学习 | 68% | — | — | ≈1× |
| **在策略蒸馏** | **70%** | **8.4×10¹⁹** | **8.2×10¹⁹** | **9–30×** |

基准情况下节省约 **9 倍**计算量（已给定 SFT 数据集），GPU 小时节省约 **18 倍**，若计入完整教师成本则约达 **30 倍**。

---

## 个性化蒸馏

**问题场景**：在小模型中注入新领域知识，同时保留后训练行为（指令遵循、语气、工具调用、成本控制）。同时训练两者通常很难，轻量微调（如 LoRA）往往不够，需要更大规模的中间训练。

### 训练内部助手

两个目标：
1. **知识**：对领域的专家理解（用内部 QA 评测衡量）
2. **后训练行为**：指令遵循（用 [IF-eval](https://arxiv.org/abs/2311.07911) 衡量）

起始点：经 RL 后训练的 Qwen3-8B。

> "已有研究表明，强化学习只训练了模型的少数子网络，因此当网络继续在大量数据上训练时，RL 习得的行为容易遭到破坏。" — [Mukherjee et al., 2025](https://arxiv.org/abs/2505.11711)

### 新知识训练会破坏已学行为

基线做法：将内部文档与 [Tulu3](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) 对话 prompt（用 Qwen3-8B 重新采样）混合，作为"在策略背景数据"。

结论：**无论数据配比如何，都无法维持原始的 IF-eval 性能**。即便混入 30% 的对话数据能保留大部分指令遵循能力，但与原始表现仍有差距。[LoRA](https://arxiv.org/abs/2106.09685) 方案在保留 IF-eval 上同样失败，且学到的知识更少（见 [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)）。

### 在策略蒸馏恢复后训练行为

解决方案：以早期版本的 Qwen3-8B 作为教师，在 Tulu3 prompt 上运行在策略蒸馏，以恢复指令遵循能力。

这开启了持续学习的可能：**交替进行"新数据微调"和"行为恢复蒸馏"**，让模型在学习新知识的同时保持已有能力。

**实验结果：**

| 模型 | 内部 QA（知识） | IF-eval（对话） |
|------|--------------|--------------|
| Qwen3-8B | 18% | 85% |
| + 中间训练（100% 内部数据） | 43% | 45% |
| + 中间训练（70% 内部 + 30% 对话） | 36% | 79% |
| **+ 中间训练（70%）+ 在策略蒸馏** | **41%** | **83%** |

结论：**中间训练破坏了 Qwen3-8B 的后训练行为，但在策略蒸馏能以低成本将其恢复，同时保留中间训练所学的额外知识。**

这可以理解为：将语言模型本身用作奖励模型，高概率的行为被奖励，与[逆强化学习（IRL）](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)以及 [DPO](https://arxiv.org/abs/2305.18290) 有深刻关联。

---

## 讨论

### 密集监督大幅提升计算效率

对比实验：
1. 从 Qwen3-8B-Base（无 SFT）出发
2. 在 [DeepMath](https://arxiv.org/abs/2504.20571) 上用 LoRA rank 128 运行 RL
3. 将 RL 训练后的模型作为教师，对 Base 模型做在策略蒸馏

结论：**蒸馏达到教师性能水平的速度比 RL 快约 7–10 倍**（相同模型架构）。累积计算节省达 **50–100 倍**。

效率来源：
- 蒸馏可在较短上下文长度上学习，无需等待奖励边界的突变（对比 [高熵少数 token 驱动 RL 推理](https://arxiv.org/abs/2506.01939)）
- 当 SFT 初始化已较强时，蒸馏可用更小的 batch size
- 每个 episode 提供远多于 RL 的信息位，显著降低梯度噪声

### 蒸馏可有效复用训练数据

相比 RL，在策略蒸馏的数据效率优势在于：**蒸馏通过最小化反向 KL 来近似教师的完整分布，而非记忆单一答案**。

实验：在单个数学 prompt 上连续训练 20 步（每步 256 个展开，共 5120 个被打分序列）。尽管反复使用同一 prompt，模型仍能大致匹配教师，而无明显过拟合。

这说明在策略蒸馏可以**多轮复用相同数据**而几乎不退化——类似于 [RL on single examples](https://arxiv.org/abs/2504.20571) 的发现。

### 强化学习在语义策略空间中搜索

一个解释性框架：预训练在参数空间中探索；RL 则在**语义策略空间**中搜索。

> "与预训练不同，RL 并不在梯度步本身上花费大量算力……RL 把大部分算力用于搜索——展开策略并分配信用——而不是做更新。"

类比：科学研究中，大量时间和资源用于寻找答案、探索新思路；一旦得出结论，用自然语言传授给他人就简单得多。

> "一旦好的策略被发现，在策略蒸馏不需要建模 RL 课程中的中间策略，只需要学习最终学到的策略。"

参见 Rich Sutton 的 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)、[Phasic Policy Gradient](https://arxiv.org/abs/2009.04416)。

### 在策略学习作为持续学习工具

更广泛的应用：在获取新知识的同时不退化已有能力。

已有发现（[RL's Razor](https://arxiv.org/abs/2509.04259)）：**在策略学习（RL）比离策略学习遗忘得更少**。但 RL 只能塑造行为，无法有效教授新知识，因此不足以支撑持续学习。

**一个反直觉的实验**：在 Qwen3-32B 自己的采样样本上运行 SFT（理论上 KL=0 的数据集）。

结论：**任何大于零的学习率都会导致 IF-eval 性能下降！**

原因：KL 散度仅在期望意义上为 0，每个有限 batch 的实际分布都略有不同。在这些有限 batch 上训练会产生非零梯度，使更新后的模型策略偏离原始状态。

解决方案：**在策略蒸馏始终保持在策略**，且由于教师固定，学生会收敛到教师的期望行为，而不会像 SFT 那样在自蒸馏场景下退化。

参见 [Midtraining Bridges Pretraining and Posttraining Distributions](https://arxiv.org/abs/2510.14865)、[Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting](https://arxiv.org/abs/2510.18874)。

---

## 结论

> "我们探索了在策略蒸馏在多种场景中的应用：训练用于数学推理的小模型，以及训练持续学习的助手。"

> "在策略蒸馏兼得两者之长：在策略训练的可靠性，加上密集奖励信号的成本效益。"

> "后训练是达到前沿模型能力的关键环节。通过将学生的在策略采样与教师的密集监督相结合，在策略蒸馏能以前沿高算力 RL 运行成本的一小部分达到同等能力。"

未来方向：蒸馏的新应用、改进教师监督的新方法、提升数据效率和持续学习的途径。

> "在 Thinking Machines，我们的使命是用兼具前沿性能、适应性与个性化的 AI 模型赋能每一个人。在策略蒸馏是实现这一目标的有力工具。"

---

## 引用

**APA 格式：**

> Lu, Kevin and Thinking Machines Lab, "On-Policy Distillation", *Thinking Machines Lab: Connectionism*, Oct 2025.

**BibTeX：**

```bibtex
@article{lu2025onpolicydistillation,
  author = {Kevin Lu and Thinking Machines Lab},
  title = {On-Policy Distillation},
  journal = {Thinking Machines Lab: Connectionism},
  year = {2025},
  note = {https://thinkingmachines.ai/blog/on-policy-distillation},
  doi = {10.64434/tml.20251026},
}
```

---

## 参考文献

### 引用论文

| 序号 | 标题 | 作者 | 链接 |
|------|------|------|------|
| 1 | Alpaca: A Strong, Replicable Instruction-Following Model | Taori et al., 2023 | [blog](https://crfm.stanford.edu/2023/03/13/alpaca.html) |
| 2 | OpenThoughts: Data Recipes for Reasoning Models | Guha et al., 2025 | [arXiv:2506.04178](https://arxiv.org/abs/2506.04178) |
| 3 | Distilling Large Language Models for Efficient Clinical Information Extraction | Vedula et al., 2025 | [arXiv:2501.00031](https://arxiv.org/html/2501.00031v1) |
| 4 | Enhancing Chat Language Models by Scaling High-quality Instructional Conversations | Ding et al., 2023 | [arXiv:2305.14233](https://arxiv.org/abs/2305.14233) |
| 5 | The False Promise of Imitating Proprietary LLMs | Gudibande et al., 2023 | [arXiv:2305.15717](https://arxiv.org/abs/2305.15717) |
| 6 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) | Ross et al., 2010 | [arXiv:1011.0686](https://arxiv.org/abs/1011.0686) |
| 7 | Let's Verify Step by Step | Lightman et al., 2023 | [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) |
| 8 | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | Agarwal et al., 2023 | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) |
| 9 | MiniLLM: Knowledge Distillation of Large Language Models | Gu et al., 2023 | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| 10 | Qwen3 Technical Report | Qwen Team, 2025 | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) |
| 11 | Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks | Bengio et al., 2015 | [arXiv:1506.03099](https://arxiv.org/abs/1506.03099) |
| 12 | Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning | Wang et al., 2025 | [arXiv:2506.01939](https://arxiv.org/abs/2506.01939) |
| 13 | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al., 2021 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| 14 | Instruction-Following Evaluation for Large Language Models (IF-eval) | Zhou et al., 2023 | [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) |
| 15 | Reinforcement Learning Finetunes Small Subnetworks in Large Language Models | Mukherjee et al., 2025 | [arXiv:2505.11711](https://arxiv.org/abs/2505.11711) |
| 16 | Midtraining Bridges Pretraining and Posttraining Distributions | Liu et al., 2025 | [arXiv:2510.14865](https://arxiv.org/abs/2510.14865) |
| 17 | Tulu 3: Pushing Frontiers in Open Language Model Post-Training | Ivison et al., 2024 | [HuggingFace 数据集](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| 18 | Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting | Chen et al., 2025 | [arXiv:2510.18874](https://arxiv.org/abs/2510.18874) |
| 19 | Unfamiliar Finetuning Examples Control How Language Models Hallucinate | Kang et al., 2024 | [arXiv:2403.05612](https://arxiv.org/abs/2403.05612) |
| 20 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO) | Rafailov et al., 2023 | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| 21 | Algorithms for Inverse Reinforcement Learning | Ng and Russell, 2000 | [PDF](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf) |
| 22 | LoRA Learns Less and Forgets Less | Biderman et al., 2024 | [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) |
| 23 | Phasic Policy Gradient | Cobbe et al., 2020 | [arXiv:2009.04416](https://arxiv.org/abs/2009.04416) |
| 24 | The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks | Frankle and Carlin, 2018 | [arXiv:1803.03635](https://arxiv.org/abs/1803.03635) |
| 25 | Reinforcement Learning for Reasoning in LLMs with One Training Example | Wang et al., 2025 | [arXiv:2504.20571](https://arxiv.org/abs/2504.20571) |
| 26 | The Bitter Lesson | Rich Sutton | [blog](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) |
| 27 | RL's Razor: Why Online Reinforcement Learning Forgets Less | Shenfeld et al., 2025 | [arXiv:2509.04259](https://arxiv.org/abs/2509.04259) |

### 引用模型

| 模型 | 链接 |
|------|------|
| Qwen3-4B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Qwen3-8B-Base | [HuggingFace](https://huggingface.co/Qwen/Qwen3-8B-Base) |
| Qwen3-32B | [HuggingFace](https://huggingface.co/Qwen/Qwen3-32B) |
| Qwen3-235B-A22B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) |
| Qwen3.5-9B-Base | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B-Base) |
| Qwen3.5-9B | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-9B) |
| QwQ-32B | [HuggingFace](https://huggingface.co/Qwen/QwQ-32B) |
| DeepSeek-R1-0528-Qwen3-8B | [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) |

### 引用数据集

| 数据集 | 链接 |
|--------|------|
| OpenThoughts-3 | [HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M) |
| Tulu-3-SFT-Mixture | [HuggingFace](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |

### 相关工具

| 工具 | 链接 |
|------|------|
| Tinker Training API | [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker/) |
| Tinker Cookbook（蒸馏示例） | [GitHub](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation) |
| Tinker RL 训练脚本 | [GitHub](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/train.py) |
