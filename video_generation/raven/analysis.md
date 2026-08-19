# RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

> Yanzuo Lu, Ronglai Zuo, Jiankang Deng  
> Imperial College London  
> Preprint, 2026 · https://yanzuo.lu/raven

---

## 1. 一句话定位

训练时把 self-rollout 的中间状态**重新打包**成「clean 历史端点 + noisy 去噪状态」交错序列（RAVEN），让后续 chunk 的 loss 能反向传播到前面 chunk 缓存的历史表示；并提出 **CM-GRPO**，直接在 consistency sampler 的 Gaussian 转移核上做 GRPO，消除 Flow-GRPO 引入的 Euler-Maruyama 辅助随机过程带来的训练-推理差距。

---

## 2. 要解决的问题

自回归视频扩散模型推理时用自身生成的 chunk 作为历史，而训练时历史来源各不相同——这导致**历史分布偏差（history distribution gap）**，会随自回归步数累积。

四种已有范式的对比：

| 范式 | 历史来源 | 历史有 end-to-end 监督？ | 推理时分布匹配？ |
|---|---|---|---|
| Teacher Forcing | 真实数据（Real Data） | ✅ 有（real data 是 target） | ❌ 不匹配 |
| Diffusion Forcing | 真实数据 + 随机 SNR 扰动 | ✅ | ❌ |
| Self Forcing | self-rollout 端点（detach） | ❌ detach 截断梯度 | ✅ 匹配 |
| **RAVEN** | self-rollout 端点（attached） | ✅ 梯度可流过历史 | ✅ 匹配 |

Self Forcing 已经对齐了分布，但历史 cache 是 `sg(·)` 截断梯度的，后续 chunk loss 无法监督历史表示；RAVEN 的核心贡献就是用**重打包**消除这一梯度截断。

---

## 3. 与前作关系

- **CausVid** (CVPR 2025)：asymmetric distillation 蒸馏框架原型，从双向 teacher 蒸馏因果 student，DMD 损失。RAVEN 继承其整体架构。
- **Self Forcing** (NeurIPS 2025)：在 CausVid 上加 self-rollout，解决历史分布偏差，但历史是 detached context，无 end-to-end 监督。RAVEN 在 Self Forcing 的基础上增加 generator step 的交错序列重打包。
- **Causal Forcing** (ICML 2026)：Self Forcing + ODE 蒸馏初始化，RAVEN 以此为权重起点。
- **Flow-GRPO** (NeurIPS 2025)：flow matching RL 的原型，需 ODE→SDE 转换 + Euler-Maruyama。CM-GRPO 消除该辅助过程。

---

## 4. 核心方法

### 4.1 问题形式化

自回归视频扩散模型的生成概率：

$$
p_\theta(x_{1:T} \mid c) = \prod_{t=1}^{T} p_\theta(x_t \mid h_t, c), \quad h_t = \mathcal{H}(x_{<t})
$$

其中 `h_t` 是 KV cache 编码的历史。噪声状态 `z_t^(n) = α_n x_t + σ_n ε`。

### 4.2 RAVEN：交错序列重打包

![Fig 1: 四种训练范式的 Attention Mask](./figures/fig1_attn_mask.png)

> **Fig 1 逐列解读**：四种训练范式，行=chunk（标号 1/1/2/2/3），列=序列位置。
>
> **(a) Teacher Forcing**——历史（左侧紫色 Real Data）和当前 noisy state（顶部灰色）全部互相 attend（白色方块=有梯度注意力）。历史来自真实数据，与推理时自生成历史不匹配，同时梯度可以流过历史。
>
> **(b) Diffusion Forcing**——历史仅向当前 chunk 提供 KV（箭头方向），当前 noisy state 不反向监督历史。历史同样是真实数据，分布不匹配。
>
> **(c) Self Forcing**——历史来自 self-rollout（绿色格），分布已匹配推理；但历史通过 stop-gradient（打叉方块=无梯度注意力）进入 attention，后续 chunk 的 loss 无法流入历史表示。
>
> **(d) RAVEN（Ours）**——历史来自 self-rollout（绿色），且后续 chunk 的 noisy state 对历史有带梯度的 attention（白色方块）。关键区别：历史 chunk 的表示在同一次 causal forward pass 中被后续 chunk 的 loss 直接监督，梯度可穿越 chunk 边界流回历史。

**交错序列 `I_u` 的构造（Eq. 5）**

对于 `K` 步 consistency 采样时间步 `τ_1 > ··· > τ_K = 0`，和采样时间 `u ∈ {τ_1, ..., τ_{K-1}}`：

$$
\mathcal{I}_u = \bigl(\dot{z}_1^{(u)},\, \hat{x}_1,\, \dot{z}_2^{(u)},\, \hat{x}_2,\, \ldots,\, \hat{x}_{T-1},\, \dot{z}_T^{(u)}\bigr)
$$

其中 `ẑ_t^(u)` 是第 `t` chunk 在时间步 `u` 的 noisy 去噪状态，`x̂_t` 是对应的 clean endpoint。

- **Clean endpoint `x̂_t`**：作为第 `t+1` chunk 的历史，在 attention 中扮演 KV 来源（带梯度）
- **Noisy state `ẑ_t^(u)`**：作为 `p_θ` 的监督去噪目标，接受 reverse-KL 梯度

这两类 token 在**同一次 causal forward pass**中被处理，后向 chunk 的 loss 经 attention 传回前向 chunk 的历史表示，同时避免了反向传播整条自回归轨迹的代价（rollout 本身在 fake-score step 中已经 no-grad 完成）。

### 4.3 训练流程

![Fig 2: RAVEN 训练 Pipeline](./figures/fig2_pipeline.png)

> **Fig 2 逐段解读**：
>
> **上半部分（Fake-Score Step）**——冻结（雪花图标）的因果学生生成器做 4 个 chunk 的 AR self-rollout，KV cache 在 chunk 间传递（绿色箭头）。对每个 chunk 产出两类输出：clean endpoint（绿色实线方块，向下输出）和若干 noisy denoising state（灰色渐变方块，向右送入 Fake-Score Critic）。Critic 对加了高斯噪声的 clean endpoint 前向打分，MSE 更新 Fake-Score Critic 参数。
>
> **粗虚线分隔**：fake-score step 的 rollout 状态**不丢弃**，红色大箭头传递给 generator step 复用。
>
> **下半部分（Generator Step）**——将 rollout 的 clean endpoints + noisy states 重打包为交错序列 `I_u`（7 个方块：1, 1, 2, 2, 3, 3, 4，交替 clean/noisy），左侧 Attention Mask 图示说明 noisy chunk 可以 attend 所有前序 clean chunk（带梯度）。可训练（火焰图标）的因果学生生成器前向，得到每个 chunk 的 `x̂_θ`；经过 consistency renoise（`+noise` 步骤）到评分时间步 `s`，分别送入冻结的双向 Real-Score Teacher 和 Fake-Score Critic 计算 reverse-KL 梯度，配合 Chunk-wise Loss Scaling（后期 chunk 权重更高），更新生成器参数。

**Algorithm 1 三个阶段**：

1. **Stage 1：Self Rollout**（no-grad）—— 对每个 chunk `t`，consistency sampler 跑 `K` 步，缓存 `{ẑ_t^(τ_k)}` 和 `x̂_t`，更新 KV cache
2. **Stage 2：Fake-Score Step**—— 将 `x̂` renoise，bidirectional fake-score critic 做 MSE 监督，每步更新
3. **Stage 3：Generator Step**（每 `r` 步一次）—— 从 rollout 中采样时间步 `u`，组装 `I_u`，causal forward 生成 `x̂_θ`，consistency renoise 到评分时间步 `s`，teacher + critic 打分，reverse-KL + chunk-wise loss 更新学生

### 4.4 Chunk-wise Loss Scaling

早期 chunk（历史少）和晚期 chunk（历史丰富，需维持一致性）面临质性不同的难度。引入**未来参与分数（future participation score）**：

$$
p_j = \frac{\sum_{k=j}^{J} m_k}{\sum_{k=1}^{J} m_k}
$$

`p_j` = 第 `j` chunk 及其后续 chunk 的 token 占比，单调递减。通过权重函数 `g_η(P)` 映射为 normalized 权重 `w_j`。

**消融最优**：`Shift(α = -1)`，对应 `π_α(p) = αp/(1 + (α-1)p)` 在 `α = -1` 时，将均匀调度反转为**后期 chunk 权重更高**（约 2× 于前期）。直觉：后期 chunk 需要在更长历史条件下保持语义对齐，同时抑制误差积累。

### 4.5 CM-GRPO

![Fig 5: Chunk-wise Loss Scaling 消融](./figures/fig5_chunkwise.png)

> **Fig 5 解读**：左侧折线图横轴为 chunk 序号 j，纵轴为归一化权重 `w_j`。灰色水平线为均匀调度（均值 = 1）。蓝色/橙色/绿色系为 Mode/Logit-Normal 权重（峰值在中间），折叠在均匀线附近；红色系 Shift 曲线：`α=1`（橙红，递减）、`α=0`（绿，均匀）、`α=-1`（红实线，**Ours**，单调递增）。右侧表格确认 Shift(α=-1) 在 Total/Qual/Dyn 综合最优。

**核心思路**：consistency sampler 本身已经是 Gaussian 转移，可以直接在其转移核上做 GRPO，不需要 ODE→SDE 转换。

一次 consistency 采样步从 `u` 到 `s`：`z̃^(s) = α_s x̂_θ + σ_s ε`，对应 Gaussian 转移核：

$$
\pi_\theta\!\left(\tilde{z}^{(s)} \mid \tilde{z}^{(u)}, c\right) = \mathcal{N}\!\left(\tilde{z}^{(s)};\, \mu_\theta^{u \to s},\, \sigma_s^2 I\right), \quad \mu_\theta^{u \to s} = \alpha_s \hat{x}_\theta
$$

对轨迹打 group-relative advantage `Â_i`，log-prob 对 `x̂_θ` 的梯度为：

$$
\nabla_{\hat{x}_\theta} \!\left[-\hat{A}_i \log \pi_\theta\right] = -\hat{A}_i \alpha_s \frac{\tilde{z}_i^{(s)} - \mu_\theta^{u \to s}}{\sigma_s^2}
$$

实现为 stop-gradient 回归目标（Eq. 9）：

$$
\mathcal{L}_{\text{CM-GRPO}} = \mathbb{E}_{i,u,s} \left[ \left\lVert \hat{x}_\theta - \text{sg}\!\left(\hat{x}_\theta + \frac{\hat{A}_i \alpha_s}{2\sigma_s^2}\bigl(\tilde{z}^{(s)} - \mu_\theta^{u \to s}\bigr) \right) \right\rVert^2 \right]
$$

梯度等价于直接对 clean endpoint 施加 advantage-weighted score 更新，不引入任何辅助 SDE 或额外随机过程。**可选 KL 正则**（`β > 0`）：两 Gaussian 的 KL 闭合为：

$$
D_{\text{KL}}\!\bigl(\pi_\theta \| \pi_{\text{ref}}\bigr) = \frac{\alpha_s^2 \lVert \hat{x}_\theta - \hat{x}_{\text{ref}} \rVert^2}{2\sigma_s^2}
$$

**Reward 构成**：5 维合成奖励（各维 group 内归一化后加权求和）：
- Dynamic Degree (DD)：RAFT 光流幅度 top-5% 均值
- Motion Smoothness (MS)：AMT 时序重建误差（越小越平滑）
- Aesthetic Quality (AQ)：LAION aesthetic predictor
- Imaging Quality (IQ)：MUSIQ 多尺度质量
- Text Alignment (TA)：VideoReward（Qwen2-VL-7B 基座 + DPO 训练）

---

## 5. 关键代码位置

### 5.1 训练 ctx 流水线（`causal_wan_t2v_dmd.py`）

```
rollout → prepare_fake → fake_forward → fake_loss → prepare_gen → gen_forward → score → gen_loss
```

| 函数 | 核心操作 |
|---|---|
| `rollout()` (L157) | no-grad，一次完整 AR rollout，返回 `rollout_x0s` (clean endpoints) + `trajectory_xts` (每步 noisy states) |
| `prepare_gen()` (L264) | 从 rollout 中采样时间步 `u`，从 `trajectory_xts[random_index]` 取对应 noisy state，组装 `gen_inputs`（这里已包含 RAVEN 交错序列逻辑） |
| `gen_forward()` (L295) | causal forward，**带梯度**，生成 `gen_pred` |
| `score()` (L302) | `sampler.step_to()` consistency renoise 到评分时间步 `s`，teacher + fake_critic 分别预测 `real_score_x0s` / `fake_score_x0s` |
| `gen_loss()` (L356) | DMD loss: `(real - fake) * (fake - x0) / norm`，`norm = |x0 - real|` mean，`chunk_wise_weighting` 加权 |

**📌 `norm_per_chunk` 细节**（L371-393）：可选的按 chunk 独立归一化 normalizer，防止后续 chunk 积累的漂移把 norm 拉大、使前期 chunk 的梯度被稀释。

### 5.2 ConsistencySampler（`common/diffusion/sampler/consistency.py`）

```python
# transition_kernel: 返回 (mean, std) 对应 π_θ(z̃^(s)|z̃^(u),c)
mean = schedule.forward(pred_x_0, zeros, s)   # = α_s * x̂_θ
std  = schedule.B(effective_s)                 # = σ_s

# step_to: 采样下一步 x_s
pred_x_s = schedule.forward(pred_x_0, noises, s)   # α_s*x̂_θ + σ_s*ε

# transition_score_grad_coeff: 返回 α_s（梯度系数）
coeff = schedule.A(s)
```

### 5.3 CM-GRPO policy_loss（`causal_wan_t2v_grpo.py:280`）

```python
# score_grad = 0.5 * α_s * (-Â) * (z̃^(s) - μ_θ) / σ_s²
score_grad = 0.5 * coeff * (-advantage) * (next_xt - mean_i) / std_i**2
# stop-gradient target: x̂_θ - score_grad
target = x0.detach() - score_grad.detach()
# MSE loss 梯度 = -2 * score_grad，等价于 advantage-weighted score update
loss = F.mse_loss(x0, target, reduction="none")
```

KL 项（`beta > 0`）同形式，`(mean_i - ref_mean_i) / std_i^2`，来自两 Gaussian 的闭合 KL。

### 5.4 GRPO rollout（`causal_wan_t2v_grpo.py:85`）

- `groups_per_infer × num_infers = group_size`（默认 32）
- 对同一 prompt 生成 G=32 条独立轨迹，`_reward_fn` 先 VAE decode，再各维度奖励并行打分
- advantage = `(R_i - mean(R)) / (std(R) + ε)`，per-dimension group normalize 后再合成

### 5.5 Causal Model（`modeling/causal_model.py`）

- `CausalWanSelfAttention`：继承 WanSelfAttention，换为 FlexAttention（支持任意 block mask）
- packed sequence 格式：`latent_indexes`, `noisy_latent_relative_indexes`, `q_ranges`, `k_ranges`, `attn_type_map` 构成精细 per-chunk attention mask
- KV cache：`NaiveCache`，inference 时 past_key_values 按 layer_idx 查询，merge 进当前 Q/K/V

### 5.6 训练配置（`raven.yaml`）

| 参数 | 值 | 含义 |
|---|---|---|
| backbone | CausalWanModel (1.3B) | 因果学生，FSDP，grad checkpoint |
| fake_model | WanModel (1.3B，bidirectional) | fake-score critic |
| tea_model | WanModel (14B，bidirectional) | real-score teacher，10× 参数量 |
| chunk_size | 3 latent frames | 每 chunk 3 帧，与 Self Forcing 对齐 |
| sampling_steps | 4（trailing） | consistency sampler 4 步 |
| fake_update_interval | 1 | 每步更新 critic |
| gen_update_interval | 2 | 每 2 步更新 generator（TTUR=2） |
| chunk_wise_weighting | `dmd_losses: -1.0` | shift α=-1，后期 chunk 权重高 |
| lr_backbone | 2e-6 | AdamW，β=(0,0.999) |
| lr_fake_model | 4e-7 | critic lr 更低 |

---

## 6. 实验结果

**定量对比（VBench，Table 1）**

| 方法 | Total | Qual. | Sem. | Dyn. Deg. |
|---|---|---|---|---|
| CausVid | 83.01 | 84.18 | 78.34 | 2.340 |
| Self Forcing | 84.27 | 85.10 | 80.97 | 2.543 |
| Causal Forcing | 84.96 | 86.00 | 80.76 | 2.669 |
| Causal Forcing + CM-GRPO | 85.08 | 86.12 | 80.96 | 2.829 |
| **RAVEN** | **85.15** | **86.18** | **81.04** | **2.951** |
| **RAVEN + CM-GRPO** | **85.46** | **86.54** | **81.17** | **2.962** |

RAVEN 在所有指标上超过所有基线，**动态度增益最大**（Dyn. Deg. 2.543 → 2.951，+16%），表明监督历史表示能缓解质量-动态性 trade-off 而非仅重新分配误差。

**消融（Table 2）**：
- Teacher Forcing：Dyn. Deg. 最高（3.000），但质量和语义最低 → 分布不匹配付出代价
- Self Forcing：语义最高（81.56），动态度最低（2.347）→ detached 历史牺牲动态性
- DF + Self Rollout：接近 TF 的动态性，说明对齐分布但不监督历史只是重新分配错误
- RAVEN：综合最优，「鱼和熊掌」兼得

**CM-GRPO 消融（Table 4）**：
- EM（Euler-Maruyama）各变体在 0.1~0.8 噪声范围内接近但均稍弱于 CM-GRPO
- 最优 EM（σ=0.8, β=0）稍超 CM-GRPO 的 Semantic，但 Total/Qual/Dyn 均低于 CM-GRPO
- CM-GRPO 不需要联合调噪声水平 σ 和 KL 权重 β 两个超参，接口更简洁

**用户研究（Fig 6）**：RAVEN 在 Quality/Semantic/Overall 三个维度对比所有 4 条基线均占优（Overall 偏好率 53%~75%）。

---

## 7. 争议与权衡

**RAVEN 的核心代价**：generator step 需要重组 `I_u`，涉及对 packed token 序列的拆包/重打包（`_unpack_packed_samples` / `_repack_from_samples`，代码量约 200 行），工程复杂性显著增加。

**CM-GRPO 的局限**：KL 正则需要一个 compatible reference consistency model 输出 `x̂_ref`；当前实现中双向 teacher 不是 consistency model，因此 KL 项在当前实验中未启用（`beta=0`），留待后续工作。

**Reward 设计困难**：VLM-based reward（VideoReward）训练数据来自高步数生成器，直接用于 few-step student 时存在分布偏移；单独上调 TA 权重（到 2.0）会推高语义但压制动态度和质量（Table 3），作者采用 TA=2, DD=0.35, MS=0.75, AQ=IQ=1 平衡三者。

**训练代价**：RAVEN 70 H200·hours，CM-GRPO 额外 170 H200·hours（共 240），相比纯 SFT 方案昂贵，但相比闭源商业 RL pipeline（数千 GPU·hours）仍可接受。

**泛化性**：论文 Appendix D 指出 RAVEN 的交错序列构造和 CM-GRPO 的策略接口均可泛化到：任意 cached-history 表示方式（滑窗 attention sink、intermediate noisy state 等）、任意 consistency 或 consistency distillation 生成器（含双向模型）。

---

## 8. 一句话总结

RAVEN 的本质是**把 fake-score step 产生的 self-rollout 废物利用**——已经跑完的去噪轨迹原本丢弃，现在重打包成交错序列复用于 generator step，让后续 chunk 的 DMD 梯度流回历史缓存，一石二鸟地解决了 Self Forcing 的「分布对齐但无历史监督」困境；CM-GRPO 则把 RL 接口精确对准 consistency sampler 本身的 Gaussian 核，避免 Flow-GRPO 的辅助 SDE 把训练-推理 gap 从历史域搬到采样器域。

---

## Q&A

*(后续对话中产生的问答将追加至此处)*
