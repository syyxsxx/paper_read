# Mask Forcing: Improving Autoregressive Video Diffusion Distillation via Dual-Noise Masking Rollout

**论文**: [arXiv 2609.09123](https://arxiv.org/abs/2609.09123)  
**项目页**: [alicezrzhao.github.io/mask_forcing](https://alicezrzhao.github.io/mask_forcing/)  
**代码**: 暂未开源  
**作者**: Zhuoran Zhao, Shengju Qian, Tongtong Liang, Xianghao Kong 等（HKUST / LIGHTSPEED / UCSD）  
**时间**: 2026-09-08

---

## 1. 一句话定位

在 AR 视频扩散蒸馏（DMD self-rollout）过程中，向每步 rollout 输入注入随机掩码选出的低噪声 token，使学生分布覆盖教师更多模式，同时降低 rollout 误差累积，无需真实视频数据和额外训练阶段。

---

## 2. 要解决的问题

AR 视频扩散模型通过 Distribution Matching Distillation（DMD）将双向多步教师压缩为因果少步学生。主流方法（Self Forcing、Causal Forcing、LongLive）在 self-rollout 训练中暴露两个问题：

1. **Mode Collapse（模式坍塌）**：DMD 使用 reverse KL 目标，其 mode-seeking 特性令学生分布集中在教师高密度区域，导致生成视频过饱和、过平滑，视觉多样性差。
2. **Error Accumulation（误差累积）**：DMD 只对完整 rollout 计算损失，中间步骤无显式梯度信号；每步预测误差通过 KV cache 向后传播，越到后面 chunk 质量越差。

解决这两个问题的已有方案要么需要真实数据（DMD2、DFD），要么需要额外训练阶段（Astrolabe RL），成本高且引入新的超参。

---

## 3. 与前作的关系

| 方法 | 核心思路 | Mask Forcing 对比 |
|------|---------|-----------------|
| Teacher Forcing | 以真实帧为历史条件 | 训练-推理不一致（exposure bias） |
| Diffusion Forcing | 每帧独立噪声 | 缓解分布偏移，但不对齐训练与推理 |
| Self Forcing | self-rollout + DMD loss | 对齐训练/推理，但 mode collapse |
| Causal Forcing | ODE 初始化的 AR 教师 | 同上 |
| LongLive | frame sink + streaming 长视频 | 同上 |
| DMD2 / DFD | GAN loss / 真实数据 | 需额外数据和预训练 |
| rCM / DistillAlign | mode-covering + mode-seeking | 仍有过饱和风险 |
| **Mask Forcing** | 扰动 self-rollout 轨迹 | **无需真实数据，plug-in 兼容已有方法** |

---

## 4. 核心方法：Dual-Noise Masking Rollout

### 4.1 背景：Standard Self-Rollout DMD

AR 视频模型把视频表示为 F 个 chunk，`x = (x_1, ..., x_F)`。flow-matching 中 noisy chunk 定义为

$$
x_i^t = (1 - t)\,x_i^0 + t\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$

在 self-rollout 的第 j 个去噪步（时间戳 `t_j`），学生 `G_θ` 从 noisy chunk 预测 clean estimate：

$$
\hat{x}_{i,j}^0 = G_\theta\!\left(x_i^{t_j} \mid x_{<i},\, c,\, t_j\right)
$$

DMD 梯度来自 real/fake score 差值：

$$
\nabla_\theta \mathcal{L}_\text{DMD} = \mathbb{E}\!\left[-(s_\text{real}(x_t, t) - s_\text{fake}(x_t, t))\,\frac{dG}{d\theta}\right]
$$

Reverse KL 的 mode-seeking 特性使学生专注高概率区域，忽视教师多模态分布的低密度部分。

### 4.2 Dual-Noise Masking Rollout 步骤

在每个去噪步 j，Mask Forcing 额外采样一个**更低噪声水平** `t'_k`：

$$
\ell_j = \max\!\left(t_\text{min},\; t_j - \frac{\Delta}{N_t}\right)
$$

$$
t'_k \sim \text{Uniform}\!\left([\ell_j,\, t_j]\right)
$$

其中 `N_t = 1000` 为训练时间戳总数，`Δ` 为时间戳窗口大小（默认 250），`t_min` 防止 cleaner signal 退化为干净图像。

用同一个噪声 `ε_j ~ N(0,I)` 从上一步预测 `x̂^0_{i,j+1}` 生成两个噪声水平的样本：

$$
x_i^{t'_k} = (1 - t'_k)\,\hat{x}_{i,j+1}^0 + t'_k\,\epsilon_j \quad\text{(cleaner)}
$$

$$
x_i^{t_j} = (1 - t_j)\,\hat{x}_{i,j+1}^0 + t_j\,\epsilon_j \quad\text{(noisier)}
$$

对 chunk i 采样二值掩码 `M^i`（masking ratio α），掩码**在帧之间独立采样（per-frame），在 chunk 之间重新采样（per-chunk）**：

$$
x_i^{t_\text{mix}} = M^i \odot x_i^{t'_k} + (1 - M^i) \odot x_i^{t_j}
$$

学生以 dual-noise 输入在原始时间戳 `t_j` 进行去噪：

$$
\hat{x}_{i,j}^0 = G_\theta\!\left(x_i^{t_\text{mix}} \mid x_{<i},\, c,\, t_j\right)
$$

注意：模型接收的调度时间戳仍是 `t_j`，而输入有部分 token 处于 `t'_k < t_j` 的更低噪声，形成"局部噪声水平不一致"，迫使模型利用 cleaner token 做上下文去噪 noisier token。

![Fig 2: Dual-Noise Masking Rollout 方法概览](./figures/fig2_method.png)

> **Fig 2 逐段解读**：
>
> **(a) Standard Self-Rollout（上半部分）**——三个 chunk（`x_i^{t_j}`, `x_{i+1}^{t_j}`, `x_{i+2}^{t_j}`）依次送入 AR Student `G_θ`，生成结果加噪后继续传递。完成 rollout 后计算 DMD 梯度（real score - fake score → `∇θD_KL`）。右侧 `Mode Seeking` 分布图：**蓝色填充的 `p_real` 是三峰分布，橙色填充的 `p_fake` 只覆盖中间那一个峰**，两侧峰完全未被覆盖——这就是 mode collapse 的图示。
>
> **(b) Dual-Noise Masking Rollout（下半部分）**——从前一步预测 `x̂^0_{i,j+1}` 出发，同时生成 `x_i^{t_j}`（darker，原始噪声）和 `x_i^{t'_k}`（lighter，更低噪声），通过二值掩码 `M^i`（棋盘格图案）混合为 `x_i^{t_mix}`，送入 `G_θ` 得到新的 `x̂^0_{i,j}`。右侧 `Broader Mode Coverage`：橙色 `p_fake` 几乎完全贴合蓝色 `p_real` 的三个峰。
>
> **右下 `Timestep Schedule`**：竖直黑线自上而下标 `t_T`（右侧写 `1`）、`t_j`（实心点）、**`t'_k`（空心点）**、`t_{j-1}`（实心点）、底端 `0`。
>
> ⚠️ **注意图示与公式的口径差异**：图上把 `t'_k` 画在 `t_j` 与 `t_{j-1}` 之间，但 Eq. (6)(7) 允许 `t'_k` 落到 `t_{j-1}` 以下——`Delta=600` 时窗口宽 0.6，远大于 `T=4` 时的单步间隔 0.25。**图示只对默认 `Delta=250` 成立。**
>
> 📌 **另一个细节**：图中两个 mask 网格都是 **3 行 x 4 列 = 12 格**，`M^i` 里有 3 个深格（3/12 = 0.25），与 `alpha=0.2` 不完全对应——纯示意。

### 4.3 为什么有效：KL 分解视角

设掩码轨迹 `V = {(M_r, t'_{k,r})}_{r=1}^R`，掩码对应的 marginal 分布为 `q̄_{θ,τ}`，Appendix 证明：

$$
D_\text{KL}(\bar{q}_{\theta,\tau} \| p_\tau) = \mathbb{E}_V\!\left[D_\text{KL}(q_{\theta,\tau}^V \| p_\tau)\right] - I_\theta(V;\, X_\tau \mid c)
$$

含义：marginal 分布的 reverse KL = 各轨迹条件 KL 的均值 − 互信息 `I_θ`。当不同掩码轨迹诱导出不同的条件分布时（`I_θ > 0`），marginal 的 reverse KL 严格小于各轨迹 KL 的均值，说明**混合后的分布比单一轨迹能覆盖更多教师模式**，即使每条轨迹自身仍有 mode-seeking 特性。

### 4.4 掩码方案选择

空间轴（同一 chunk 内不同帧）和时间轴（不同 chunk 之间）各有两个选项：

| 空间 | 时间 | HPSv3 | Dynamic. |
|------|------|-------|---------|
| shared（全帧共用一个 M） | per-rollout（整个 rollout 固定 M） | 10.12 | 56 |
| shared | per-chunk（每 chunk 重采样、chunk 内各步固定） | **9.84** | **81** |
| shared | per-step（每个去噪步重采样） | 9.64 | 72 |
| per-frame（每帧独立 M） | per-rollout | 10.03 | 53 |
| **per-frame** | **per-chunk** ← 论文选用 | **9.84** | **82** |
| per-frame | per-step | 9.93 | 74 |

⚠️ **这张表最值得注意的是论文没点出的一行**：**`shared, per-chunk` 是 9.84 / 81，与选用的 `per-frame, per-chunk`（9.84 / 82）几乎完全相同**——HPSv3 一模一样，Dynamic Degree 只差 1 点。

📌 **也就是说，"空间轴上逐帧独立采样 mask"这个被写进设计要点的选择，实测贡献接近于零**；真正起作用的是**时间轴上的 per-chunk 粒度**——对比 per-rollout（56 / 53）和 per-step（72 / 74），per-chunk 的 Dynamic Degree 明显更高。

---

## 5. 关键超参

| 超参 | 默认值 | 说明 |
|------|--------|------|
| mask ratio α | 0.2 | cleaner token 占比；<0.1 扰动不足，>0.4 运动多样性下降 |
| 时间戳窗口 Δ | 250（/1000） | cleaner timestep 的采样范围；太小扰动弱，太大偏离教师 |
| t_min | schedule index 20 | 防止 cleaner signal 退化为完全干净 |
| 去噪步数 T | 4 | self-rollout 总步数 |
| 分辨率 | 832×480，81 帧/chunk=3 latent frames | |
| 训练步数 | ~1500 steps | 约 14 小时，8×GPU |
| Student | Wan2.1-T2V-1.3B | ODE 蒸馏初始化 |
| Teacher | Wan2.1-T2V-14B | 冻结，提供 real score |
| 训练数据 | VidProM dataset | 无需真实视频标注 |

---

## 6. 实验结果

### 6.1 100-prompt benchmark（chunk-wise 设置）

| Method | HPSv3↑ | Vision↑ | Instruct↑ | MQ↑ | Dynamic↑ | Total↑ |
|--------|--------|---------|----------|-----|---------|-------|
| Self Forcing | 9.55 | 10.10 | 38.50 | 15.88 | 70 | 81.89 |
| + Ours | **9.84** | **11.37** | **45.03** | **20.49** | **82** | **82.61** |
| Causal Forcing | 9.37 | 10.36 | 40.41 | 17.73 | 76 | 82.67 |
| + Ours | **10.17** | **11.58** | **46.30** | **21.54** | **82** | **82.76** |
| LongLive | 9.11 | 10.77 | 42.48 | 21.20 | 76 | 82.02 |
| + Ours | **10.14** | **11.00** | **42.65** | **22.34** | 69 | **82.75** |

完整的 VBench 三列（论文 Table 1 右半）：

| Method | Total↑ | Quality↑ | Semantic↑ |
|---|---|---|---|
| Self Forcing | 81.89 | 82.99 | 77.49 |
| + Ours | 82.61 | 83.68 | 78.74 |
| Causal Forcing | 82.67 | 83.58 | 78.98 |
| + Ours | 82.76 | 83.69 | 79.01 |
| LongLive | 82.02 | 82.87 | 78.66 |
| + Ours | 82.75 | 83.71 | 78.91 |

### 6.1b frame-wise 设置（论文 Table 1 下半，原笔记未收录）

| Method | HPSv3↑ | Vision.↑ | Instruct.↑ | MQ↑ | Dynamic.↑ | Total↑ | Quality↑ | Semantic↑ |
|---|---|---|---|---|---|---|---|---|
| Self Forcing | 9.34 | 9.45 | 35.62 | 18.27 | 53 | 80.73 | 81.72 | 76.78 |
| + Ours | 9.79 | 10.49 | 39.60 | 19.07 | 61 | 81.49 | 82.55 | 77.23 |
| Causal Forcing | 9.67 | 10.58 | 37.56 | 20.25 | **28** | 80.64 | 81.55 | 77.00 |
| + Ours | 9.96 | 10.71 | 39.60 | 23.69 | **52** | 82.28 | 83.19 | 78.62 |
| LongLive | 9.19 | 9.35 | 38.50 | 12.14 | **25** | 80.97 | 81.86 | 77.41 |
| + Ours | 9.46 | 10.55 | 42.78 | 19.32 | **76** | 81.47 | 82.33 | 78.05 |

📌 **frame-wise 的增益幅度远大于 chunk-wise**——LongLive 的 Dynamic Degree 从 25 涨到 76（**+51**），MQ 从 12.14 涨到 19.32（+7.18）。

⚠️ **但要看清 frame-wise baseline 的来源**：论文明说 *"Since Self Forcing and LongLive do not release their frame-wise ODE initialization models, **we train these models using ODE-paired data distilled from the bidirectional teacher**."* ——**这两行不是官方结果，是作者自己重训的**。frame-wise LongLive 的 MQ = 12.14、Dynamic = 25 都低得异常，重训质量无从核验。**在一个可能偏弱的 baseline 上拿到 +51，说服力要打折。**

---

三个 baseline 上 Mask Forcing 均提升 HPSv3 和 instruction following；MQ 普遍提升。**Dynamic Degree 有两处下降**：chunk-wise LongLive 76→69（−7）、长视频 LongLive 70→64（−6）。

⚠️ **论文对前者给的解释是"generic VBench rewarding drift-induced optical flow"（通用 VBench 会奖励由漂移引起的光流），但这个说法没有任何测量支撑**——没测 drift、没测光流幅度分布、也没做人评交叉验证，只引了一篇 Steady-Forcing。**对后者（长视频那处）论文完全没有解释。**

### 6.2 长视频（30s，MovieGen & VBench-Long）

| Method | HPSv3↑ | Vision↑ | Instruct↑ | MQ↑ | Dynamic↑ |
|--------|--------|---------|----------|-----|---------|
| LongLive | 8.44 | 14.31 | 62.04 | 18.84 | 70 |
| + Ours | **9.11** | **14.93** | **66.02** | **19.37** | 64 |

### 6.3 人类评估

pairwise 偏好（24 评审）：

- vs. Self Forcing：80% 偏好 Ours
- vs. Causal Forcing：79% 偏好 Ours
- vs. LongLive：83% 偏好 Ours
- vs. LongLive（长视频）：72% 偏好 Ours

### 6.4 收敛速度

CMMD（CLIP）和 VMMD（V-JEPA2）衡量分布对齐，Mask Forcing 在所有 baseline 上均加速收敛，分布偏移（红线）显著低于基线（蓝线）且更快下降。

⚠️ **但这两个指标与训练目标严格同源，必须打折看。** 论文的构造方式是：*"we first generate a fixed reference set with the **real-score teacher** on the 100-prompt set. We then generate videos for each step using the causal student on the same prompts and compute CMMD and VMMD against the teacher reference set."*

**DMD loss 的全部作用就是把 student 分布推向 real-score teacher 的分布，而 CMMD/VMMD 度量的正是 student 与同一个 teacher 的分布距离。** 所以收敛曲线上的 CMMD/VMMD 下降，本质上是**训练目标的直接读数**，不是独立的质量证据——任何让 DMD 优化更顺畅的改动都会降低它。

📌 **另外曲线本身也没那么干净**（论文 Fig 5，VMMD）：
- **Self Forcing 子图**：baseline 与 +Ours **两条线都在 500–750 步取到最小值后单调回升**（蓝 0.385@500 → 0.516@1750；红 0.301@750 → 0.387@2000）。
- **LongLive 子图**：锯齿严重，红线在 1000/1250 触底 0.206/0.211 后**跳回 0.410@1500**。
- **Causal Forcing 子图**：**step 50 处 +Ours 反而更差**（0.787 vs baseline 0.706），250 步后才反超。

### 6.5 消融（论文 Table 3/4，原笔记只有文字描述）

**Mask ratio α**（100-prompt set，baseline = Self Forcing）：

| α | HPSv3↑ | Dynamic.↑ |
|---|---|---|
| 0.1 | 10.00 | 47 |
| **0.2** ← 选用 | 9.84 | **82** |
| 0.3 | 9.82 | 65 |
| 0.4 | 10.15 | 57 |
| 0.5 | **10.17** | 44 |

**Timestep window Δ**：

| Δ | HPSv3↑ | Dynamic.↑ |
|---|---|---|
| 50 | 9.16 | **92** |
| 150 | 9.82 | 70 |
| **250** ← 选用 | 9.84 | 82 |
| 450 | **9.98** | 80 |
| 600 | 9.90 | 67 |

⚠️ **Δ 这张表的选点很可疑**：**Δ=450 的 HPSv3（9.98）高于选用的 Δ=250（9.84），Dynamic Degree（80 vs 82）只低 2 个点**。按论文自己"competitive HPSv3 + highest Dynamic"的口径，450 至少是同等合理的选择，**论文没有解释为什么选 250**。

⚠️ **更根本的问题是缺两个关键对照**：
- **没有 `α=0` 这一行**（即完全不扰动），表内无法直接看出扰动本身带来多少；
- **没有 `α=1.0`**（全部 token 都用 `t'_k`，即"纯 timestep 平移、无 mask"）——**这意味着无法区分收益来自 mask 的空间/时间异质性，还是仅仅来自噪声水平的整体平移**。而"dual-noise masking"正是本文命名的核心机制。

### 6.6 附录里的三张表（原笔记未收录）

**① vs joint distillation（论文 Table 6，100-prompt set）**：

| Method | HPSv3↑ | Vision.↑ | Instruct.↑ | MQ↑ | Dynamic.↑ |
|---|---|---|---|---|---|
| DistillAlign | 9.29 | 9.46 | 37.69 | 14.36 | 76 |
| Causal-rCM | 9.61 | 9.43 | 38.50 | 14.34 | 64 |
| **Ours** | **10.17** | **11.58** | **46.30** | **21.54** | **82** |

⚠️ **这张表少了一行关键对照**：**Causal Forcing baseline 本身就是 HPSv3 9.37 / Vision. 10.36 / MQ 17.73**（见 §6.1），**已经优于 DistillAlign 和 Causal-rCM 在 Vision. 和 MQ 上的表现**。而 DistillAlign 正是建立在 Causal Forcing 之上并声称改进的方法。**Table 6 里不放 Causal Forcing 行，这个反常就被隐去了。**

**② 多样性（论文 Table 7，8 个 seed × 100 prompt = 800 视频/方法）**：

| Method | CLIP Div.↑ | DINO Div.↑ |
|---|---|---|
| Self Forcing | 0.0773 | **0.1704** |
| + Ours | **0.0814** | 0.1645 |
| LongLive | 0.0784 | 0.1537 |
| + Ours | **0.0819** | **0.1750** |
| Causal Forcing | 0.0675 | 0.1371 |
| + Ours | **0.0720** | **0.1577** |

⚠️ **这是对"缓解 mode collapse"这个核心主张最直接的检验，而结果是混合的**：
- **Self Forcing 的 DINO Div. 下降**（0.1704 → 0.1645），论文自己承认 *"DINOv3 diversity decreases slightly"*；
- 更值得注意的是，**Self Forcing baseline 的 DINO Div.（0.1704）是全表六行里的最高值**——比任何一个 +Ours 配置都高。

**③ 长视频分区间（论文 Table 8，MovieGen，30 秒切成 5 段）**：

| Method | CLIP↑ 0-6s | 6-12s | 12-18s | 18-24s | 24-30s |
|---|---|---|---|---|---|
| LongLive | 33.70 | 33.41 | 33.29 | 33.23 | 33.15 |
| + Ours | **34.12** | **33.81** | **33.59** | **33.63** | **33.38** |

| Method | HPSv3↑ 0-6s | 6-12s | 12-18s | 18-24s | 24-30s |
|---|---|---|---|---|---|
| LongLive | 8.81 | 8.37 | 8.23 | 7.89 | 7.79 |
| + Ours | **9.65** | **9.44** | **9.05** | **8.87** | **8.68** |

📌 **两条曲线都在下降，Mask Forcing 只是把整条曲线抬高了。** 论文自己承认 *"both methods exhibit a certain decline in CLIP and HPSv3 scores as the video generation progresses, likely due to error accumulation"*——**误差累积并没有被消除。**

---

## 7. 可视化

![Fig 1: Teaser](./figures/fig1_teaser.png)

> **Fig 1 逐段解读**：
>
> **Single-Prompt Short Video（上两组）**——左侧展示帧序列，右侧是收敛曲线。行一 Self Forcing：女子捧花、水草背景色彩过于鲜艳，+ Ours 后色调自然，细节更丰富。行二 Causal Forcing：橙色夹克人物背景过饱和蓝绿色，+ Ours 后色彩层次更真实。收敛曲线：HPSv3（越高越好）红线（Ours）迅速超过蓝线；CMMD（越低越好）红线显著低于蓝线——说明 Mask Forcing 视觉质量更高同时分布更接近教师。
>
> **Single-Prompt Long Video（中间组）**——LongLive 生成的小地精棕色背景在后半段逐渐失真，+ Ours 保持细节（沙盘、炊帚纹理）更稳定。
>
> **Camera-Controlled I2V（底部组）**——给定森林参考帧 + 相机轨迹，Self Forcing 生成的视频整体偏暗、树干细节缺失；+ Ours 恢复逼真光影和植被纹理。

![Fig 3: 定性对比](./figures/fig3_qualitative.png)

> **Fig 3 逐行对比**（chunk-wise，100-prompt benchmark）：
>
> - **Self Forcing vs + Ours（行1/2）**：小提琴手视频，Self Forcing 脸部扁平、城市夜景过曝；+ Ours 皮肤细节、服饰材质、夜景层次均改善。动画兔子，Self Forcing 色块硬边明显；+ Ours 毛发和环境光更柔和。
> - **LongLive vs + Ours（行3/4）**：橙色户外服场景，LongLive 缺少高频纹理；+ Ours 织物纹理、山地背景更立体。骷髅斗篷，LongLive 金属质感平淡；+ Ours 高光和阴影更真实。
> - **Causal Forcing vs + Ours（行5/6）**：机甲战士森林，Causal Forcing 过饱和绿色；+ Ours 色彩还原准确，林间光线自然。厨房烹饪，Causal Forcing 细节丢失；+ Ours 保留炊具纹理和环境光。

---

## 8. 争议与权衡

### 8.1 理论部分与实际部署不一致（最实质的问题）

📌 **附录 6.1 的全部结论都是关于 masked-rollout 的边缘分布 `q̄_{θ,τ}`**——即对 masking trajectory `V` 求边缘得到的混合分布。

⚠️ **但 §3.2.2 明确写 *"The model follows the original denoising schedule at inference"*，即推理时不加 mask、不加 dual-noise。** 部署时生成的分布对应某一条**无掩码**的特定轨迹，**而不是那个混合分布 `q̄`**。所以 Eq. (13)/(25) 证明的"混合分布的 reverse KL 更低"，**并不直接适用于实际交付的模型**。

⚠️ **而且 Eq. (25) 本身是恒等式的直接推论，不构成对方法有效性的证明。** 它比较的是 `D_KL(q̄‖p)` 与 `E_V[D_KL(q^V‖p)]`——**同一族分布的"混合"与"平均"**，而**不是**与"不加掩码训练所得的学生分布"比较。由于互信息恒非负，**这个不等式对任意随机扰动都成立，包括有害的扰动**。理论部分因此无法区分"dual-noise masking"与"任意随机扰动"。

### 8.2 核心机制缺关键消融

**"Dual-Noise" 这个命名机制本身没有被隔离验证。** 论文从未做过以下任一对照：
- **`α = 1.0`**（全部 token 用 `t'_k`，即纯 timestep 平移、无 mask）→ 无法区分收益来自 **mask 的异质性** 还是仅来自**噪声水平平移**
- **`α = 0`**（无扰动）作为表内基准行
- **注入更高噪声**而非更低噪声（方向性对照）
- **注入纯高斯噪声**或置零/替换为可学 token（MAE 式对照）
- **用不同的 `ε`** 而非共享同一个 `ε_j` 生成两个 state

另外，**"conditioning timestep 保持原 `t_j`"** 是明确声明的设计选择（"requiring the model to denoise inputs whose local noise levels are partially inconsistent with the global timestep"），**同样无消融**——没有与"喂入 mask 感知的 per-token timestep"或"平均 timestep"作对比。

**消融只在 Self Forcing + chunk-wise 上做**，未验证 α/Δ/scheme 在 Causal Forcing、LongLive、frame-wise 上是否可迁移。

### 8.3 数字层面的问题

**① Table 1 与 Fig 9 的 LongLive 数值对不上。** Table 1 报 chunk-wise LongLive HPSv3 = **9.11**、frame-wise = **9.19**；但论文自己的 Fig 9 收敛曲线上，LongLive 蓝线在 **500 步及之后始终 ≥ 9.46**（最高 9.90@2000）。**即报告的 baseline 数值低于自己曲线上任何 ≥500 步的取值。** Fig 9 未说明对应哪个设置、取哪一步。

**② "surpasses all baseline methods across the VBench metrics" 与 Table 1 冲突。** chunk-wise 下 **Self Forcing + Ours 的 Total = 82.61 < Causal Forcing baseline 的 82.67**，Semantic = **78.74 < 78.98**。按字面读法不成立（按"各自超越自己的 baseline"读法才成立）。

**③ 摘要与引言对成因的表述不一致。** 摘要说 *"**The key contributing factor is** the mode-seeking behavior of the reverse KL"*（单一主因）；引言说 *"can be attributed to **two key factors**"*（mode-seeking + 中间预测无监督信号）。**两个机制在实验上从未被分离。**

**④ 无任何误差棒、多 seed 重复或显著性检验**（Table 7 的多样性除外）。100 个 prompt 规模偏小，Dynamic Degree 是整数百分比、方差大（同一方法 chunk-wise 76 vs frame-wise 28）。而 Table 1 里 +.03 这样的差值也被记作改进。

**⑤ 超参是在主表用的同一个 100-prompt set 上调的。** 消融（Table 3/4/5）与主结果（Table 1）用的是同一批 prompt，**没有验证/测试划分**，且选点依据（HPSv3 + Dynamic 的人工折中）也是主表指标之一。

### 8.4 宣称缺乏证据处

**① "generic VBench rewarding drift-induced optical flow"** —— 用来解释 LongLive Dynamic Degree 下降，**无任何测量支撑**（没测 drift、没测光流幅度分布、没做人评交叉验证）。

**② "improving visual quality efficiently" / "faster convergence"** —— "efficiently" 从未量化。收敛速度只有 step 轴，**没有 wall-clock 轴、没有 baseline 的训练成本、没有"达到同一 HPSv3 所需时间"的对照**。

**③ "mitigate mode collapse"** —— 唯一直接证据是 Table 7 的多样性，而其中 **Self Forcing 的 DINO Div. 下降，且 baseline 是全表最高值**（见 §6.6②）。

**④ "reducing error accumulation"** —— **没有任何直接度量误差累积的实验**（如 per-chunk 预测误差曲线、逐 chunk 与 teacher 的偏差、外推长度 vs 质量曲线）。Table 8 显示衰减仍然存在。

**⑤ "cleaner tokens act as denoising guidance"** —— 完全没有验证。**没有测过中间预测 `x̂^0_{i,j}` 的质量是否真的提升**，只有对 Self-Flow 的引用作为"consistent with the observation"。

**⑥ 附录的相机控制实验只有定性图、零数值**（无 FVD、无相机轨迹误差、无亮度漂移量化），却在 Fig 1 中作为三大卖点之一展示。

### 8.5 训练/评测长度不一致（相机控制实验）

附录明写 *"Both bidirectional and autoregressive models are trained on **5s videos** following SolarWM"*，base 是 **Wan2.2-5B-TI2V**（24 fps → 5s ≈ 121 frames）；而 Fig 10 **展示到 Frame 150**（≈151 frames ≈ 6.3s）。

⚠️ **即评测比训练长约 25%。** 而该实验观察到的失效模式（Self Forcing 逐帧变黑）**最可能的成因之一就是超出训练长度的外推**——论文却把它归因为"reverse KL mode-seeking + error accumulation"，**没有排除这个更简单的解释**。论文自己在同节末尾承认需要 *"training with long sequences and rollouts"*，等于间接确认了这一点。

### 8.6 baseline 与自引用

**① frame-wise 的 Self Forcing 与 LongLive baseline 是作者自己重训的**（见 §6.1b），数值不是官方结果。

**② Table 6 隐去了 Causal Forcing baseline 行**（见 §6.6①）。

**③ 主基准是 Causal Forcing 自家的 100-prompt set** —— 这一点对本文**不利**，属公平性上的加分项。

**④ 未与直接相关的方法比较**：**DP-DMD**（Diversity-preserved DMD, ICML 2026）只被借用作多样性评测协议，**未作为 baseline**，尽管它正是解决"DMD 多样性"这同一个问题。DFD、Astrolabe、OPSD-V、Mode Seeking meets Mean Seeking 都被批评但**都未做实验对比**。

**⑤ 自引用未披露。** 附录的相机控制实验声明 "following SolarWM (Huang et al., 2026a)"，而 **SolarWM 与本文至少有 5 位作者重叠，包括本文第一作者 Zhuoran Zhao 和第二作者 Shengju Qian**。论文以第三人称引用，**没有任何 "our prior work / concurrent work by the same authors" 式的披露**。同样地，**Astrolabe**（共享 3 位作者）在 §2.1 被明确批评（"performance is limited by reward models"）却未做任何实验对比。

**⑥ DMD vs DMD2 引用不精确。** 文中说 Self Forcing 用 "a DMD loss (Yin et al., 2024b)"（DMD v1），但 Fig 2 画的是**可训练的 Fake Score Estimator（火焰图标）**——这是 DMD2 的在线 critic 机制。论文从未说明自己的 pipeline 是否含 DMD2 的 GAN head，而 §2.1 又批评了 DMD2 的 GAN loss。

### 8.7 正面

**① 主基准选的是 baseline 自家的 benchmark**，这对自己不利，值得肯定。

**② 三个 baseline × 两种设置（chunk-wise / frame-wise）的覆盖面扎实**，不是只在一个配置上刷数。

**③ 方法确实是"零成本插件"**：不加前向、不加数据、不加训练阶段，Algorithm 1 显示**每个 chunk 只有恰好一次前向（`j=s`）带梯度**，其余前向与全部 KV cache 写入都被 detach——这个设计干净。

**④ 人评规模在这类论文里算中等偏上**（24 人、四组对比、偏好率 72–83%），虽然未说明每人评多少对、是否严格双盲。

### 8.8 工程层面

- **掩码比例 trade-off**：α=0.4~0.5 时 HPSv3 更高（10.15/10.17）但 Dynamic Degree 跌至 57/44；α=0.2 是折中点，不同任务可能需要重调。
- **Δ=600 时 Dynamic Degree 下降**：窗口过大使 cleaner token 的噪声水平与原始相差过多，模型过度依赖 cleaner 信号，限制运动探索。
- **无代码开源**：只有 project page，**没有 code 链接、没有 model/checkpoint 链接**。而**关键超参大量缺失**——learning rate、optimizer、batch size、weight decay、CFG scale、critic 更新频率、`t_min` 的实际数值、chunk 数 `F`、GPU 型号**全部没给**。复现门槛很高。

---

## 9. 一句话总结

Mask Forcing 是一个 plug-in 式的 rollout 扰动策略：在 AR 视频 DMD self-rollout 的每步去噪输入中，随机注入低噪声 token，既分散学生轨迹覆盖教师更多模式，又利用 cleaner token 作为上下文辅助去噪，无需额外数据或训练阶段即可稳定提升 Self Forcing / Causal Forcing / LongLive 等主流 AR 视频蒸馏方法的视觉质量。

---

## Q&A

**Q: "Dual-Noise Masking" 里，mask 到底在起什么作用？**

A: ⚠️ **这个问题论文没有回答，因为缺了关键对照。**

机制上有两件事同时发生：
1. **噪声水平被整体拉低**——被 mask 选中的位置用 `t'_k < t_j`
2. **噪声水平变得空间/时间异质**——同一个输入里，不同 token 处于不同噪声水平，而喂给模型的 conditioning timestep 仍是原来的 `t_j`

**论文的叙述把功劳记在第 2 点上**（"requiring the model to denoise inputs whose local noise levels are partially inconsistent with the global timestep"），**但实验无法区分这两点**——因为**没有 `α = 1.0` 的对照**（全部 token 都用 `t'_k`，即纯 timestep 平移、完全没有异质性）。

📌 **而 Table 5 提供了一个间接的反向线索**：`shared, per-chunk`（整个 chunk 内所有帧共用一个 mask）拿到 **9.84 / 81**，与逐帧独立采样的 **9.84 / 82** 几乎相同。**也就是说空间轴上的异质性贡献接近于零**，真正有效的是时间轴上的 per-chunk 粒度。这至少说明"异质性"这个解释**在空间维度上不成立**。

---

**Q: 它和 OPSD-V、ABot-World-0 这几篇放在一起，是什么关系？**

A: **四篇都在给"DMD 蒸馏出来的 few-step 因果 AR 视频模型"打补丁，但打在四个不同的位置。**

| | 改的是什么 | 核心手段 |
|---|---|---|
| **Mask Forcing** | **学生 rollout 的输入** | 在 re-noise 处混入低噪声 token，扰动学生轨迹以覆盖更多 teacher mode |
| [OPSD-V](../opsd_v/analysis.md) | **teacher 的上下文** | 把 teacher 的旧 KV cache 换成真实视频 chunk |
| [ABot-World-0](../../world_model/abot_world_0/analysis.md) | **teacher 的监督时域** | LongForcing：把 DMD 阶段 teacher 的监督横跨更长 rollout |
| [SolarWM](../../world_model/solarwm/analysis.md) | **蒸馏的阶段结构** | TF-AnyFlow 一步顶掉 Causal ODE / Causal CD 初始化阶段 |

加上 [ForgeWM](../forgewm/analysis.md)（改**阶段结构**：四阶段渐进 + 同 checkpoint 的离线 replay 精修）一共是五篇。

📌 **五篇的共同前提完全一致**：**DMD 的 teacher 是短片段的双向模型，这是长时程质量的天花板。** 但**五篇两两之间 10 对组合，做过模型质量定量对比的是 0 对**——Mask Forcing 引用了 OPSD-V 但归入"被批评的一类"且未对比；ABot 和 SolarWM 互不引用。

📌 **完整的横向对照见 [dmd_few_step_ar](../dmd_few_step_ar/analysis.md)。** 对本篇最相关的一条结论是：**Mask Forcing 与 OPSD-V 是五篇里最该、也最容易被直接对比的一对**——同 backbone（Wan2.1-T2V-1.3B）、同 NFE=4、**同 base model（Self-Forcing / LongLive）**，而且两者打的位置正交（本篇动 student 的输入，OPSD-V 动 teacher 的上下文并换掉 DMD 目标），**理论上可以叠加**。⚠️ 但两篇在"该往哪个方向调散度"上直觉相反：本篇认为 reverse KL 的 mode-seeking 有害、要往 mode-covering 推；OPSD-V 的 loss 是纯 velocity MSE（典型 mean-seeking），**却拿到了 Dynamic Degree 上升**。

⚠️ **还有一层需要注意**：Mask Forcing 的作者与 [SolarWM](../../world_model/solarwm/analysis.md) **至少 5 人重叠（含一作、二作）**，附录的相机控制实验直接 "following SolarWM"，但全文以第三人称引用、无自引用披露。

**与 [Mask Forcing 所批评的 mode-seeking 那条线**的关系：它选择**保留原始 DMD 目标**，只扰动学生 rollout，而不是像 rCM / DistillAlign / Mode-Seeking-meets-Mean-Seeking 那样去改造散度本身。论文的原话是 *"Unlike these methods, Mask Forcing retains the original DMD objective and instead perturbs the student rollouts via dual-noise masking."*

---

**Q: 想用它，要注意什么？**

A: **三件事。**

1. **它是真正的零成本插件**——不加前向、不加真实数据、不加训练阶段，直接接在现有 self-rollout DMD 训练里。Algorithm 1 显示每个 chunk 只有一次前向（`j=s`）带梯度，其余全部 detach，**显存和计算开销基本不变**。这是它最扎实的卖点。
2. **α 和 Δ 需要自己重调。** 论文的 α=0.2 / Δ=250 是在 Self Forcing + chunk-wise + 100-prompt set 上选的，**没有验证过能否迁移到其它 baseline 或设置**。而且 Δ=450 在两个指标上并不比 250 差（9.98/80 vs 9.84/82），**选点本身就有余地**。
3. ⚠️ **关键超参大量缺失且无代码**：learning rate、optimizer、batch size、CFG scale、critic 更新频率、`t_min` 实际数值、GPU 型号全部没给，只有 project page。**复现需要自己补齐这些。**

📌 **如果要验证它在你的场景里是否真的有效，最该先做的对照是 `α = 1.0`**（纯 timestep 平移、无 mask）——论文缺的正是这一个，而它直接决定了"masking"这个名字是否名副其实。
