# OPSD 解读：Self-Distilled Reasoner

**论文**: Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models  
**作者**: Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, Aditya Grover  
**arXiv**: https://arxiv.org/abs/2601.18734 (v3, 2026-03-20)  
**代码**: https://github.com/siyan-zhao/OPSD  
**博客**: https://siyan-zhao.github.io/blog/2026/opsd/

---

## 1. 一句话定位

同一个 LLM 扮演 student（只看题目）和 teacher（看题目 + 正确解答），在 student 自采轨迹上做逐 token forward KL 蒸馏，把本不可微的"有正确答案"信号转化为密集的分布监督，比 GRPO 的 token 效率高、比 off-policy SFT 的泛化强。

---

## 2. 要解决的问题

**GRPO 的两个代价**：
1. **采样成本高**：每 step 需要 G 条 rollout（典型 G=8），每条长达 16K token，算一次 baseline advantage。
2. **梯度稀疏**：batch 里超过 50% 的问题全 G 条都答对或全答错，advantage 为 0，这一 batch 白跑（reward diversity collapse）。

**SFT / Off-Policy Distillation 的问题**：固定数据集，分布偏移，模型自己生成的错误路径得不到监督。

**目标**：既要 on-policy（跟着 student 自己的轨迹走），又要密集监督信号（每个 token 都有梯度），还不要外部 teacher（节省存储和计算）。

---

## 3. 与前作的关系

| 属性 | SFT/Off-policy | GRPO | On-Policy Distill (GKD) | **OPSD** |
|------|---------------|------|--------------------------|---------|
| On-policy 数据 | ✗ | ✓ | ✓ | ✓ |
| 密集逐 token 监督 | ✓ | ✗（稀疏奖励） | ✓ | ✓ |
| 低采样成本 | ✓ | ✗（G rollouts） | ✓ | ✓ |
| 无外部 teacher | ✓ | ✓ | ✗（需独立 teacher） | ✓ |

**与 GKD 的关系**：GKD（On-Policy Distillation of LMs, Agarwal et al. ICLR 2024）也在 student 轨迹上做 forward KL，但需要一个**外部** teacher（通常是更大的模型）。OPSD 的贡献是把 teacher 和 student **统一成同一个模型**，靠 context 差异区分角色。

📌 **与本仓库 `on_policy_distillation` 博客的对应**：`gkd` 和 `on_policy_distillation` 两篇笔记都讲 GKD 框架，OPSD 是其在数学推理领域的 self-contained 变体。

---

## 4. 核心方法

### 4.1 双角色设定

同一模型 `π_θ`，两种 context：

| 角色 | 输入 | 模式 |
|------|------|------|
| **Student** `π_S` | `[问题]` | thinking-off（非推理模式）|
| **Teacher** `π_T` | `[问题] + [正确解答] + transition prompt` | thinking-on（推理模式）|

Teacher 看到了 ground-truth solution，因此知道最终答案在哪，能从概率上引导 student 的分布走向正确路径。Student 自采（on-policy）生成轨迹，teacher 对同一 token 序列打分。

**Teacher prompt 核心片段**（`data_collator.py`）：
```
Problem: {problem}

Here is a reference solution to this problem:
=== Reference Solution Begin ===
{solution}
=== Reference Solution End ===

After reading the reference solution above, make sure you truly understand the 
reasoning behind each step — do not copy or paraphrase it. Now, using your own 
words and independent reasoning, derive the same final answer...
```

📌 **关键设计**：Transition prompt 明确要求 teacher "不要照抄"，强迫 teacher 展示推理过程而非直接复现答案 → teacher 分布不退化为 n-gram 复制。

### 4.2 训练目标

在 student 的 on-policy rollout `y ~ π_S(·|x)` 上最小化 per-token forward KL：

$$\mathcal{L}_\text{OPSD} = \mathbb{E}_{y \sim \pi_S(\cdot \mid x)} \left[ \sum_{t=1}^{|y|} \text{KL}\!\left(\pi_T(y_t \mid x, s_\text{ref}, y_{<t}) \;\|\; \pi_S(y_t \mid x, y_{<t})\right) \right]$$

其中 `s_ref` 是正确解答，`y_<t` 是 student 已生成的 token（两边共享）。

实现中用广义 JSD（`beta=0` → forward KL）：

```python
# opsd_trainer.py: generalized_jsd_loss
# beta=0 → forward KL: KL(π_T || π_S) — teacher is target, student is trained
loss = F.kl_div(student_log_probs, teacher_log_probs,
                reduction="none", log_target=True)
```

### 4.3 Full-Vocabulary Logit Distillation

每个 token 位置对**整个词表**做软分布对比，而不是只看 student 采样到的 token。

对比：
- **OPSD（full-vocab）**：每 token O(V) 内存，但梯度信号来自整个分布（包含"本应预测但没预测"的 token）
- **Sampled-token**（GRPO/reverse KL 变体）：每 token O(1) 内存，只有采样 token 的 log-prob 得梯度

实验结论（Qwen3-4B）：
- Full-vocab：AIME25 84.1，HMMT25 60.0
- Sampled-token：AIME25 82.1，HMMT25 57.3

### 4.4 Per-Token Pointwise KL Clipping（v3 新增）

📌 **核心发现**：`<think>`、`wait` 等风格 token 的 per-token KL 比数学相关 token 高 **6-15×**，主导了梯度信号，导致训练不稳定甚至性能崩溃。

解决方案：clip 每个 token 的 KL 贡献（`jsd_token_clip=0.05`，单位：nats）：

```python
# opsd_trainer.py: generalized_jsd_loss
if token_clip is not None:
    jsd = jsd.clamp(max=token_clip)  # 每个 token 的 KL 上限
```

效果：阻止了 AIME24 上的性能崩溃（不加 clip 训到 100 步时得分大幅下滑）。

### 4.5 Thinking Mode 配置（Student off / Teacher on）

|配置|Math token KL|效果|
|----|------------|-----|
|Student TM-off / Teacher TM-on（默认）| 最高 | **最好**|
|Student TM-on / Teacher TM-on | 中等 | 中等 |
|Student TM-off / Teacher TM-off | 最低 | 最差 |

直觉：当 student 不开 thinking mode，它对数学推理 token 的分布不确定性更高（logit 更平），teacher 的分布给出的信息量更大（KL 更高）→ 梯度信号更强。对风格 token（`<think>` 等），student 已经很确定，KL 小，但即便 KL 大也被 clip 掉了。

### 4.6 Teacher 的三种形态

代码支持三种 teacher 策略（`opsd_trainer.py`）：

| 模式 | 实现 | 适用场景 |
|------|------|---------|
| **Fixed teacher**（默认推荐） | LoRA disabled → base model 做 teacher | 防 teacher collapse，requires PEFT |
| **Dynamic teacher** | 当前 student 权重直接用 | 无 PEFT 时或想让 teacher 跟随 student 进化 |
| **EMA teacher** | `ema = decay * ema + (1-decay) * student`，decay=0.999 | 软跟随，比 dynamic 更稳定 |

📌 `fixed_teacher=True` 要求 use_peft=True，通过 `model.disable_adapter()` 让 teacher 用原始 base 权重，student 的 LoRA 梯度不影响 teacher。

---

## 5. 关键代码位置

| 功能 | 文件 | 内容 |
|------|------|------|
| 核心 loss | `opsd_trainer.py:generalized_jsd_loss` | forward KL / JSD / token_clip 实现 |
| compute_loss | `opsd_trainer.py:compute_loss` | student 前向 + teacher 前向（no_grad）+ KL |
| Teacher/Student context | `data_collator.py:SelfDistillationDataCollator` | 两种 prompt 的 chat template 构造 |
| EMA teacher | `opsd_trainer.py:_update_ema` / `_ema_teacher_context` | ZeRO-3 兼容的权重交换 |
| 入口脚本 | `opsd_train.py` | `jsd_token_clip=0.05`（默认值）|
| 超参示例 | `scripts/run_opsd_1b.sh` | `beta=0`(forward KL), `fixed_teacher`, `lora_r=64` |

**Forward pass 流程**（`compute_loss`）：
```python
# 1. student forward（可微）
student_logits = model(student_input_ids)[student_prompt_len-1:-1]

# 2. teacher forward（no_grad + adapter context）
with torch.no_grad(), adapter_context:
    teacher_logits = model(teacher_input_ids)[teacher_prompt_len-1:-1]

# 3. generalized JSD loss (beta=0 → forward KL, with token_clip)
loss = generalized_jsd_loss(student_logits, teacher_logits, labels, beta=0, token_clip=0.05)
```

---

## 6. 关键配置项

```python
# Qwen3-1.7B 默认配置（scripts/run_opsd_1b.sh）
learning_rate     = 5e-6
lora_r            = 64
lora_alpha        = 128
lora_target       = q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
max_completion    = 1024   # student 最大生成长度（token 效率实验核心）
temperature       = 1.1
top_p             = 0.95
top_k             = 20
beta              = 0      # forward KL（不是 JSD，不是 reverse KL）
jsd_token_clip    = 0.05   # per-token KL 上限（关键稳定化技巧）
fixed_teacher     = True   # base model 做 teacher，LoRA 更新 student
vllm_mode         = colocate  # 同进程内跑 vLLM 做 student 采样

# 评测配置
max_new_tokens    = 38912
temperature       = 1.0
samples_per_prompt = 12    # pass@12
```

---

## 7. 实验结果

### 主要结果（Table 2，数学推理，thinking mode）

| 模型 | 方法 | AIME24 | AIME25 | HMMT25 | 平均 |
|------|------|--------|--------|--------|------|
| Qwen3-8B | Base | 75.8 | 65.6 | 43.9 | 61.8 |
| | + SFT | 72.3 | 64.2 | 42.9 | 59.8 |
| | + GRPO | 76.4 | 68.9 | 46.7 | 64.0 |
| | **+ OPSD** | **77.8** | **70.8** | 45.8 | **64.8** |
| Qwen3-4B | Base | 74.9 | 66.4 | 42.2 | 61.2 |
| | + SFT | 70.2 | 62.3 | 43.4 | 58.6 |
| | + GRPO | 75.6 | 68.1 | 44.4 | 62.7 |
| | **+ OPSD** | **76.4** | **68.3** | **46.1** | **63.6** |
| Qwen3-1.7B | Base | 51.5 | 36.7 | 23.1 | 37.1 |
| | + SFT | 48.4 | 36.3 | 22.7 | 35.8 |
| | + GRPO | 51.1 | 38.3 | 23.7 | 37.7 |
| | **+ OPSD** | **57.2** | **43.9** | **29.2** | **43.4** |

SFT 反而比 Base 差（分布偏移），GRPO 在 1.7B 上涨幅最小（reward diversity collapse 最严重），OPSD 在 1.7B 上涨幅最大（+6.3 平均）。

### Token 效率

OPSD 只用 **1024 token × 1 rollout = 1024 token/step**，GRPO 用 **16K token × 8 rollout = 128K token/step**，两者在 Qwen3-1.7B + 100 step 内的 AIME25 对比：
- OPSD step 50：43.9%，GRPO step 50：~38%
- OPSD step 100：41.1%（轻微下滑），GRPO step 100：~38%

### 消融：散度目标（Qwen3-1.7B，AIME25）

| 目标 | Base | Step 50 | Step 100 |
|------|------|---------|----------|
| **Forward KL**（默认） | 36.7 | **43.9** | 41.1 |
| Reverse KL | 36.7 | 37.5 | 35.0 |
| JSD (β=0.5) | 36.7 | 36.9 | 39.0 |

Forward KL 以 teacher 为目标（`KL(π_T || π_S)`），student 被推向覆盖 teacher 分布的全部模式，比 reverse KL（只覆盖 student 高概率区域）更适合推理任务。

---

## 8. 争议与权衡

**实验规模有限**：最大只测到 8B，对更大模型（70B+）的 reward diversity collapse 是否仍然存在未知。

**token clip 值的敏感性**：`jsd_token_clip=0.05` 是经验值。clip 太小会丢掉有效信号，clip 太大相当于没 clip。论文未提供如何根据模型/任务选取 clip 值的系统方法。

**Student TM-off 的代价**：不开 thinking mode 的 student 在推理任务上 baseline 更低（否则 GRPO 和 OPSD 的 gap 就更小了），而且评测时通常要 thinking-on。evaluation 用的是 thinking-on 的 student checkpoint，这意味着 training distribution（TM-off）和 eval distribution（TM-on）之间有 gap，训练的实际是 non-thinking 向量空间，靠 chat template 在 eval 时切换到 thinking。

**固定 teacher 的上限问题**：`fixed_teacher=True` 用的是 base model（无 LoRA），随着 student 进化，固定 teacher 的分布质量不变——当 student 已经超过 teacher 时，继续向 teacher 对齐可能有害。EMA teacher 理论上能跟随，但 decay=0.999 决定了 teacher 落后 student 约 1000 步。

---

## 9. 一句话总结

Self-Distillation 靠 prompt context 把同一个模型劈成 teacher（看答案）和 student（盲猜），在 student 轨迹上做 per-token forward KL，用 token clip 压制风格 token 主导梯度，用 fixed_teacher + LoRA 防止 teacher collapse，在 1.7B 上 100 步内超越 GRPO 约 6 个点，token 使用量少两个数量级。

---

## Q&A

**Q: OPSD 的 forward KL 和 GKD（已有笔记）的关系是什么？**

A: GKD（`gkd/analysis.md`）提出了在 student on-policy rollout 上做 forward KL 蒸馏的框架，以及 reverse KL / JSD(β) 的泛化形式。OPSD 直接继承了 GKD 的 `generalized_jsd_loss` 函数（代码注释里引用了 GKD 的 Eq.1），区别在于 teacher 来源：GKD 需要外部更大的模型作为 teacher，OPSD 把 teacher 替换为"看了答案的同一个模型"，用 context engineering 替代模型规模差距。可以理解为：OPSD = GKD 框架 + teacher self-implementation via context conditioning。

---

**Q: 为什么 SFT 比 base model 还差？**

A: SFT 用的是 ground-truth solution 数据集做有监督微调，训练分布是"给定题目，生成正确解"。问题有两个：(1) 评测时模型要独立推理，但 SFT 训练时每步都在拟合正确路径，导致对自己错误路径的恢复能力弱（exposure bias）；(2) 数学推理数据集的解答风格（solution format）可能与 Qwen3 的 CoT 风格不一致，fine-tuning 会破坏原有的 reasoning pattern。OPSD 的 on-policy 采样直接面对模型自己的错误分布，因此泛化更好。

