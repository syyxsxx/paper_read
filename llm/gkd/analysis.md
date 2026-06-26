# GKD: On-Policy Distillation of Language Models

**论文**: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes  
**机构**: Google DeepMind + Mila + University of Toronto  
**发表**: ICLR 2024  
**arXiv**: [2306.13649](https://arxiv.org/abs/2306.13649)  
**代码**: 无官方开源（实验基于内部 T5/TPU 栈）

---

## 1. 一句话定位

**要解决的问题**：传统知识蒸馏用固定数据集（ground-truth 或教师生成序列）训练学生，但学生推理时走的是自己的序列——训练和推理的分布不一致，导致误差在自回归过程中逐步累积（exposure bias）。

**核心解法**：让学生在训练时也从自己采样序列（on-policy），再用教师的 token 级概率对每个 token 打分，形成密集监督信号。这就是 **Generalized KD（GKD）**。

**GKD 还多做了两件事**：
1. 把散度从固定的 forward KL 推广到 reverse KL 和 generalized JSD(β)，应对学生容量不足时的分布失配；
2. 把 on-policy 蒸馏无缝接入 RL fine-tuning（RLAIF），形成同时优化奖励 + 保持教师知识的联合目标。

---

## 2. 与前作的关系

### 传统 KD 谱系

```
Supervised KD (Hinton et al., 2015; DistilBERT, Sanh 2019)
    固定数据集 (X,Y), 最小化 forward KL(pT ‖ pS)
    → GKD 的 λ=0 特例

SeqKD (Kim & Rush, 2016)
    用教师生成序列做 SFT（依然固定数据）
    → 也是 λ=0 的特例，只不过数据来自教师而非 ground-truth

ImitKD (Lin et al., 2020)
    识别到蒸馏=模仿学习，混合学生生成+固定数据 (λ=0.5)
    只用 forward KL，不探索纯在策略（λ=1），不集成 RL
    → GKD 的 λ=0.5 + Forward KL 特例

f-distill (Wen et al., 2023)
    用 total variation distance，混合数据
    → GKD 的特例

MiniLLM (Gu et al., 2023，同期工作)
    也利用模仿学习视角，优化 sequence 级 reverse KL
    依赖 policy gradient，需要大量稳定技巧（高方差、奖励 hacking、长度偏差）
    GKD 在 token 级操作，不反向传播采样过程，更稳定
```

**GKD 的新贡献**：
- 首次系统探索纯在策略（λ=1）蒸馏，之前工作都在 λ≤0.5
- 灵活散度选择（不限于 forward KL），任务相关性分析
- RL + on-policy GKD 的联合训练，之前无人探索

---

## 3. 核心算法

### 3.1 问题建模

自回归模型 `p(y_n | y_{<n}, x)` 在训练时看的是固定序列的前缀，但推理时看的是自己生成的前缀。当早期 token 预测出错，后续 token 所在的分布就偏离了训练分布，误差累积。

Token 级散度定义为对所有位置取平均：

$$D(p_T \| p_S^{\theta})(y|x) := \frac{1}{L_y}\sum_{n=1}^{L_y} D\bigl(p_T(\cdot|y_{<n},x) \| p_S^{\theta}(\cdot|y_{<n},x)\bigr)$$

### 3.2 GKD 目标函数

$$L_{GKD}(\theta) = (1-\lambda)\,\mathbb{E}_{(x,y)\sim(X,Y)}\bigl[D(p_T \| p_S^{\theta})(y|x)\bigr] + \lambda\,\mathbb{E}_{x\sim X}\,\mathbb{E}_{y\sim p_S(\cdot|x)}\bigl[D(p_T \| p_S^{\theta})(y|x)\bigr]$$

- `λ=0`：监督 KD（固定数据）
- `λ=1`：纯在策略 KD（全部用学生自采序列）
- `λ=0.5`：混合

**关键实现细节**：对学生采样过程 `p_S(·|x)` **不反向传播梯度**（stop gradient on sampling），使训练稳定、计算高效，同时避免高方差问题。

### 3.3 Algorithm 1

```
Given: 教师 pT，学生 pS_θ，数据集 (X, Y)
超参: λ（在策略比例），D（散度），η（学习率）

for step k = 1..K:
    u ~ Uniform(0,1)
    if u ≤ λ:
        从 X 采样输入 x，令学生自采 y ~ pS_θ(·|x)  # on-policy
    else:
        从 (X,Y) 采固定 batch                        # off-policy
    θ ← θ - η * (1/B) * Σ ∇θ D(pT ‖ pS_θ)(y|x)
```

### 3.4 散度的选择与影响

| 散度 | 行为 | 适合场景 |
|------|------|---------|
| Forward KL（`β→0`） | **均值搜索（mode-covering）**：学生在教师所有高概率区域都保持覆盖，但可能在低概率区域也分配质量 → 产生幻觉/低质量生成 | greedy 评测；推理（GSM8K） |
| Reverse KL（`β→1`） | **众数搜索（mode-seeking）**：学生只关注教师的高概率 token，生成集中但多样性低 | 指令调优（FLAN）；temperature 采样 |
| JSD(β)，`0<β<1` | 两者插值。`β→0` ≈ forward KL，`β→1` ≈ reverse KL | 摘要 XSum（JSD 0.9）；翻译 WMT（JSD 0.1） |

直觉：当学生容量比教师小很多时，forward KL 会强迫学生"覆盖"教师分布的所有模式，导致在低概率 token 上浪费容量；reverse KL 让学生专注于教师最看好的那个模式，避免低质量输出（图 A.16）。

### 3.5 RL + On-Policy GKD 联合目标

$$\mathbb{E}_{x\sim X}\Bigl[(1-\alpha)\,\mathbb{E}_{y\sim p_S^{\theta}}[r(y)] - \alpha\,\mathbb{E}_{y\sim p_S}[D(p_T \| p_S^{\theta})(y|x)]\Bigr]$$

- `α=1`：纯 GKD 蒸馏
- `α=0`：纯 RL
- 中间值：同时优化奖励（如事实一致性）+ 保持与教师的能力对齐

RL 正则项从"靠近原始学生"改为"靠近教师"，比标准 RLHF 更有效；可减少 alignment tax（对齐税）。

---

## 4. 关键实验结果

### 4.1 三大任务总览（T5-XL 教师 ~3B → 学生 77M/250M/800M）

| 任务 | 评测 | GKD vs baseline KD | 关键发现 |
|------|------|-------------------|---------|
| XSum 摘要 | ROUGE-2 | **+111%** 相对提升（T5-Small） | JSD(0.9) on-policy 最佳；5% 数据的 GKD > 100% 数据的 Supervised KD |
| WMT 翻译 | BLEU | **+70%** 相对提升 | λ=1（纯在策略）+ JSD(0.1) 最佳 |
| GSM8K 推理 | 准确率（w/ Calculator） | **+90%** 相对提升 | forward KL（greedy）好；on-policy 比例 ≥25% 就有显著收益 |

"相对提升"指 GKD 相比 baseline KD 在"提升量"上的倍数（非绝对准确率）。

### 4.2 任务无关蒸馏（指令调优，FLAN）

FLAN T5-XL → FLAN T5-Base，数据集 FLAN2021（536 万例，62 任务）：

| 方法 | MMLU 提升 | BBH 提升 |
|------|---------|---------|
| Supervised KD | ≈0 | <0.1% |
| ImitKD | ≈0 | ≈0 |
| On-policy GKD（Reverse KL） | **+2%** | **+1%** |

- Reverse KL 在指令调优中远优于 forward KL：mode-seeking 帮助学生专注于指令的"核心意图"
- 评测集 MMLU/BBH 是 held-out 任务，说明 GKD 提升的是泛化能力

### 4.3 Self-Distillation（附录 A.1）

教师 = FLAN T5-Large（在 GSM8K 上 fine-tuned），学生 = 同架构 FLAN T5-Large（未在 GSM8K 训练）。

**On-policy GKD 的学生超越了教师的表现**，说明 GKD 不仅仅是压缩，也是提升机制。

### 4.4 RL + GKD（图 5，RLAIF）

在 XSum 上同时用 RLAIF（事实一致性奖励）+ on-policy GKD：
- 比纯 RLEF（Roit et al.）获得更高 ROUGE-2
- 同时生成更符合事实的摘要（高于 3B 教师）
- `α` 控制两者平衡，工程上无需额外超参数

---

## 5. 与 Thinking Machines 博客的关系

Thinking Machines 的 [On-Policy Distillation 博客](https://thinkingmachines.ai/blog/on-policy-distillation/) 把 GKD 从 T5 时代（encoder-decoder，任务特定）推广到**现代 decoder-only LLM 场景**（Qwen3 系列），并在以下维度做了延伸：

| 维度 | GKD 论文（2023） | Thinking Machines（2025） |
|------|----------------|--------------------------|
| 模型架构 | T5（encoder-decoder） | Qwen3（decoder-only）|
| 任务 | 摘要/翻译/推理 | 推理（AIME）+ 个性化助手 |
| 散度 | 多种，任务相关 | 聚焦 Reverse KL |
| RL 结合 | RLAIF 概念验证 | 实际生产使用 |
| 教师来源 | 固定教师 | RL 训练后的模型自蒸馏 |
| 规模 | 77M–3B | 8B–235B |
| 新贡献 | GKD 框架 | 持续学习应用、效率分析（9–30×） |

GKD 的核心思想完全相同，Thinking Machines 的主要贡献是：在 LLM 规模验证、量化计算节省、以及"蒸馏作为行为恢复工具"的应用场景。

---

## 6. 争议/权衡

### 6.1 散度选择没有通用答案

论文结论是"最优散度是任务相关的"：
- XSum（多样性重要）→ JSD(0.9) or reverse KL
- WMT（确定性重要）→ JSD(0.1)
- GSM8K（greedy eval）→ forward KL
- 指令调优（多任务泛化）→ reverse KL

**实际使用时需要对目标任务进行消融**，不能直接套一个散度。

### 6.2 计算开销

On-policy 采样比使用固定数据集贵 **1.8–2.2×**（学生越小开销越高，因为教师/学生大小比更大）。但作者指出：如果模型贵到无法在训练中做在线推理，也贵到无法在推理时给用户使用——两者不独立。

### 6.3 依赖 SFT 初始化

GKD 要求学生已经能生成"质量足够的"序列（教师才能给有意义的 feedback）。随机初始化学生无法用 GKD，必须先有 SFT 阶段。这与 RLHF 的两阶段训练（SFT → RL）完全一致，但也意味着 GKD 不能完全替代 SFT。

### 6.4 token 级 vs sequence 级

GKD 在 token 级做散度最小化（per-token KL），避免了 policy gradient 的高方差；MiniLLM 在 sequence 级，理论上更合理但需要更多稳定 trick。两者在实验中 GKD 更简单实用，但 sequence 级的方法理论地位更高。

### 6.5 与 DPO 的关系

GKD 可以理解为：把"固定参考模型"从学生的初始版本（标准 RLHF/DPO 的做法）换成教师模型，并用 on-policy 数据取代对比对（preference pair）。在 KL 正则化的 RL 框架下，两者的数学结构非常相近。

---

## 7. 关键配置项

| 参数 | 值 | 说明 |
|------|----|------|
| λ（在策略比例） | 1.0（纯在策略）效果最好 | 0.5 混合次之，0.0 最差 |
| 训练温度 | γ=1（采样多样性） | 评测时 greedy 或 beam search |
| 教师温度 | greedy 评测时=1；temperature 采样评测时=0.1 | reverse KL 对高 LR 更敏感 |
| 学习率 | 0.0003（大多数实验） | T5-small 用 0.001 |
| 训练步数 | XSum: 40K；WMT: 100K；GSM8K: 40K；FLAN: 50K | |
| Batch size | 32（XSum/WMT/GSM8K）；128（FLAN） | |
| 优化器 | Adafactor | 无需显式 LR schedule weight decay |

---

## 8. 一句话总结

GKD 的核心洞察是：**把知识蒸馏视为一个模仿学习问题**——让学生在自己生成的序列（on-policy）上接受教师的逐 token 密集反馈，同时把散度从 forward KL 推广到 reverse KL / JSD(β)，解决了传统 KD 的 exposure bias 问题，在摘要/翻译/推理上均以 1.7–2.1× 的提升幅度超越 baseline，并可无缝接入 RL fine-tuning——这一框架成为后续在策略蒸馏工作（MiniLLM、Thinking Machines 博客等）的共同基础。
