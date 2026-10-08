# DuoMatching: Joint-Marginal Distribution Matching for Few-Step Video Generation

> Jiahao Zhan, Yan Wang, Yongrui Ma, Qunliang Xing, Ruchang Yao, Runtao Liu, Shijie Zhao†, Tianfan Xue†  
> MMLab CUHK / ByteDance / HKUST / CPII InnoHK, 2026-10  
> arXiv: 2610.03543 | [项目主页](https://johnzhan2023.github.io/DuoMatching/)

---

## 1. 一句话定位

在 few-step 视频生成的分布匹配框架里，单独依赖视频 teacher（joint DMD）会留下帧级视觉质量差距——DuoMatching 在 joint DMD 之外加入 marginal DMD（图像 teacher 逐帧监督），并通过 LatentBridge 解决视频/图像 VAE 潜变量空间不兼容问题，同时用 Latent Variation Sampling（LVS）把有限的帧监督预算均匀分配到时间变化最大的区域，整体无需额外推理开销。

---

## 2. 要解决的问题

**Few-step 视频生成现状**：Causal Forcing++ 等工作已经能用 joint DMD 做 1–4 步因果自回归视频生成，显著缓解了 AR rollout 的时序漂移（自回归误差累积）问题。

**残留质量差距**：joint DMD 优化的是跨帧联合分布 `Q_θ ∥ P_v`，对于单帧质量改善存在结构性障碍：

- 给某一帧添加缺失的羽毛细节或毛发纹理，可能与邻近帧产生不一致（条件化于其他帧的 score 会抗拒这一改变）
- 图 1 的 Q-Align 曲线：随 rollout 推进，帧质量下降；joint DMD 减轻了时序漂移，但 student 与视频 teacher 之间仍存在肉眼可见的质量差

**核心矛盾**：视频 teacher 的 joint 分布捕获时间连贯性，但在精细视觉质量和语义对齐上弱于图像 teacher；图像 teacher 对单帧质量强，但不懂时序。DuoMatching 统一两者。

---

## 3. 与前作的关系

| 工作 | 关系 |
|------|------|
| DMD (Yin et al.) | 分布匹配基础框架：KL 散度 + score difference 梯度 |
| CausVid (2024) | 首个把 DMD 用于因果自回归视频生成，引入 KV-cache reuse |
| Causal Forcing++ (2026) | 解决训练-推理 rollout mismatch（Self Forcing）+ 因果 mask 架构不匹配（Causal Forcing）；DuoMatching 以其 checkpoint 为起点 |
| One-Forcing (2026) | 用对抗监督缓解极低 NFE 时的帧模糊 |
| Reward Forcing (2026) | 引入 reward 信号偏好对齐 |
| MCM / AVDM2 | 把图像数据/score 引入视频生成训练，但侧重 AnimateDiff 类的图像派生视频模型；DuoMatching 对现代时域压缩 VAE 的视频 generator 通用 |
| SA-DMD | DMD 用于视频编辑，以源视频为锚点 |

**本文核心贡献**：从理论上分析 joint vs marginal 的关系（Appendix C 证明：若图像 teacher 在可行族中对单帧分布的近似不比视频 teacher 差，则 `D_KL(m* ∥ P_i) ≤ D_KL(m* ∥ m_v)`），并提出 LatentBridge + LVS 使 marginal DMD 在现代视频架构中可行。

---

## 4. 核心方法

### 4.1 分布匹配框架（§3.1）

视频生成目标：学 generator `G_θ`，使 `Q_θ ≈ P*`。

由于 `P*` 的 score 不可直接访问，用视频 teacher `P_v` 代理：

$$
\mathcal{J}_{\text{joint}} = D_{\text{KL}}(Q_\theta \| P_v)
$$

在实践中，joint DMD 对每帧 latent slice `z_τ^l` 估计 teacher/student score，梯度更新 `G_θ`。关键限制：每个 slice 的 score 是在给定其他所有 slice 的条件下估计的：

$$
\nabla_{z_\tau^l} \log p_\tau(\mathbf{z}_\tau) = \nabla_{z_\tau^l} \log p_\tau(z_\tau^l \mid \mathbf{z}_\tau^{\setminus l})
$$

这意味着单独改善某一帧如果影响跨帧一致性，joint score 会抑制该改变——即 joint DMD 对帧级精细质量有结构性限制。

### 4.2 DuoMatching：联合-边际分布匹配（§3.2）

设 `M` 为从视频中采一帧的算子，`m_θ = MQ_θ` 为 generator 的帧级边际分布。DuoMatching 目标：

$$
\mathcal{J}_{DM}(\theta) = D_{\text{KL}}(Q_\theta \| P_v) + \omega \, D_{\text{KL}}(m_\theta \| P_i), \quad \omega > 0
$$

其中 `P_i` 是图像 teacher（Qwen-Image）的分布。理论保证（Appendix D）：存在 `ω > 0`，使联合-边际最优解比 `P_v` 更接近 `P*`（前向 KL 意义）。

实现形式（DMD 代理损失）：

$$
\mathcal{L}_{\text{DuoMatching}} = \mathcal{L}_{\text{joint-DMD}} + \omega \, \mathcal{L}_{\text{marginal-DMD}}
$$

**marginal DMD 梯度**：

给定 video latent `z = G_θ(ξ, c)`，LatentBridge 将第 `l` 个 temporal slice `z^l` 和帧内局部索引 `i` 映射为图像 teacher 潜变量 `u^{l,(i)}`，按图像 teacher 噪声调度扰动后，用冻结图像 teacher + 可训练 fake score estimator 估计 score 差异，梯度回传经过冻结 LatentBridge 到 generator：

$$
\nabla_\theta \mathcal{L}_{\text{marginal}} = \mathbb{E}_{\xi,l,i,\tau,\epsilon}\!\left[w_{\text{img}}(\tau)\,(s_{\text{fake}}^{\text{img}} - s_{\text{real}}^{\text{img}})^\top \frac{\partial u_\tau^{l,(i)}}{\partial \theta}\right]
$$

**设计意图**：joint 项约束跨帧依赖，保持时序连贯；marginal 项直接对每帧施加图像先验，补充视频 teacher 在精细视觉/语义上的不足。

### 4.3 LatentBridge（§3.3）

**问题**：现代视频 VAE（如 Wan2.x）对时间维度做压缩：1 个 video latent slice 对应多帧 RGB。而图像 VAE 的每个 latent 对应 1 帧。两者不兼容，不能直接把视频 latent 送给图像 teacher。

**暴力方案的代价**：先 decode 视频 latent → RGB，再用图像 VAE encode——梯度需要穿透两个 VAE，显存爆炸（实验中 OOM）。

**LatentBridge**：一个轻量可微模块 `B_φ`，输入前一 slice `z^{l-1}`、当前 slice `z^l`、局部帧索引 `i`，输出图像 teacher 潜变量空间的 `u^{l,(i)}`：

$$
u^{l,(i)} = B_\phi(\mathbf{z}^{l-1}, \mathbf{z}^l, i)
$$

以 `ℓ₁` 重建损失在配对的视频/图像 latent 对上预训练：

$$
\mathcal{L}_{LatentBridge} = \mathbb{E}\!\left[\lVert B_\phi(\mathbf{z}^{l-1}, \mathbf{z}^l, i) - E_{\text{img}}(x^{l,i}) \rVert_1\right]
$$

条件化于 `z^{l-1}` 提供时序上下文（视频 VAE 是因果时序编码）；条件化于 `i` 使同一 slice 能恢复不同帧的表示。实验验证（Table 4）：LatentBridge 相比直接用压缩视频 latent（Direct）在保持 Dynamic Degree 的同时进一步提升视觉质量；相比 Decode-Encode 节省大量显存（59.4GB vs OOM）。

### 4.4 Latent Variation Sampling（LVS，§3.4）

**问题**：每步只能从视频中采 `K` 个 latent slice 做 marginal DMD（计算预算有限）。若均匀随机采样，容易集中在静态/缓慢变化区域，产生冗余监督，还可能抑制运动动态（过度强调静态帧）。

**LVS**：用相邻 slice 的均方差量化时间变化：

$$
d_l = \frac{1}{CHW}\lVert \mathbf{z}^{l+1} - \mathbf{z}^l \rVert_F^2, \quad l = 1, \ldots, L-1
$$

取前 `K-1` 个最大差值位置把序列切成 `K` 段（变化最大的边界处切割），每段均匀采一个 slice：

$$
l_k \sim \mathcal{U}\{b_{k-1}+1, \ldots, b_k\}, \quad k = 1, \ldots, K
$$

效果（Table 5）：相同 `K=4` 下，LVS 的 Dynamic/Total 双优于均匀随机采样和等长分段采样。`K=8` 时性能反而下降，说明监督布局比数量更重要。

---

## 5. 实验

### 实现细节

| 项目 | 值 |
|------|-----|
| 因果模型初始化 | Causal Forcing++ checkpoint |
| DMD 训练步数（因果） | 1,000 steps on VidProM |
| 双向模型初始化 | Wan2.1-T2V-1.3B |
| DMD 训练步数（双向） | 1,200 steps on Mixkit |
| 默认图像 teacher | Qwen-Image（冻结） |
| 输出分辨率 | 81 帧，480×832 |
| LatentBridge 训练数据 | OpenVid 中 29,400 个高美感+大运动视频 |
| `ω`（marginal DMD 权重） | 0.4 |
| `K`（LVS 采样 slice 数） | 4 |
| GPU 配置 | 8×80GB |

### 定量结果（Table 1，VBench）

| 方法 | 模式 | NFE | Semantic | Aesthetic | Imaging | Dynamic | Smoothness | Total |
|------|------|-----|----------|-----------|---------|---------|------------|-------|
| CausVid | Full | 4 | 69.23 | 64.15 | 73.00 | 36.67 | 98.84 | 80.63 |
| **Ours** | Full | 4 | **70.89** | **64.55** | **73.77** | **39.17** | 98.82 | **81.45** |
| Causal Forcing++ | Frame | 1 | 57.17 | 67.12 | 68.83 | **70.56** | 98.49 | 79.82 |
| One-Forcing | Frame | 1 | 69.25 | 65.08 | 68.28 | 66.11 | 99.03 | 82.09 |
| **Ours** | Frame | 1 | **69.95** | **68.62** | **69.42** | 69.44 | **99.07** | **82.65** |
| Causal Forcing++ | Frame | 2 | 68.42 | 62.28 | 69.82 | **94.44** | 98.20 | 80.67 |
| **Ours** | Frame | 2 | **72.84** | **66.97** | **73.48** | 93.61 | **98.39** | **83.53** |
| Causal Forcing++ | Chunk | 4 | 70.84 | 64.47 | 70.13 | **80.56** | 98.14 | 82.79 |
| **Ours** | Chunk | 4 | **71.61** | **66.90** | **72.41** | 76.67 | **99.06** | **83.51** |

DuoMatching 在所有匹配生成模式和 NFE 预算下全面领先，Semantic/Aesthetic/Imaging 提升显著，Dynamic 略有取舍（因 marginal 监督引入图像先验，但 LatentBridge+LVS 已大幅缓解）。

### 人类偏好评测（Table 2，2AFC，23 名评估者）

| 对比 | 视觉质量 | 语义对齐 | 时序/运动 | 综合 |
|------|---------|---------|---------|------|
| Ours vs CausVid | 92.17% | 89.13% | 51.30% | 91.30% |
| Ours vs Causal Forcing++ | 81.30% | 79.57% | 49.57% | 80.87% |
| Ours vs One-Forcing | 93.48% | 91.74% | 90.87% | 94.35% |
| Ours vs Reward Forcing | 83.91% | 82.61% | 60.87% | 83.91% |

综合偏好率均超 80%，时序/运动维度接近对等（49-60%），验证 marginal 匹配提升视觉质量的同时不牺牲时序。

### 消融：图像 teacher 能力（Table 3）

| Teacher | HPSv3 | Total |
|---------|-------|-------|
| 无 | — | 80.67 |
| Wan2.1-14B | 5.87 | 81.43 |
| SDXL | 6.57 | 81.47 |
| FLUX.2-4B | 8.86 | 82.64 |
| Qwen-Image | **9.05** | **83.53** |

图像 teacher 的 HPSv3（图像美感对齐）越高，视频学生的 Semantic/Aesthetic/Imaging 越好，而 Dynamic/Smoothness 几乎不受影响——LatentBridge 的时空解耦起到隔离作用。

---

## 6. 关键配置

| 配置 | 值/选择 |
|------|---------|
| 基础视频 teacher | Wan2.1-T2V（因果/双向两版本） |
| 图像 teacher | Qwen-Image（默认）；也测了 SDXL、FLUX.2-4B、Wan2.1-14B |
| LatentBridge 架构 | 轻量模块；`ℓ₁` 重建预训练；条件：前一 slice + 局部帧索引 |
| `K`（LVS 分段数） | 4（实验：K=2 过少，K=8 过多反而压运动） |
| `ω`（marginal 损失权重） | 0.4 |
| 训练数据（LatentBridge） | OpenVid 29,400 个高美感大运动视频 |
| 显存峰值（LatentBridge） | 59.4GB（vs Direct 59.4GB；Decode-Encode OOM） |

---

## 7. 争议与局限

**方法层面**：

- **marginal DMD 略微抑制 Dynamic**（Table 1，Frame 2-step：94.44 → 93.61）：图像 teacher 无时序概念，边际监督倾向于压清晰静态帧，LVS 和 LatentBridge 缓解但未完全消除
- **LatentBridge 需要额外预训练**：需要配对视频/图像 latent 数据集（OpenVid 29K）；不同视频/图像 teacher VAE 组合需重新训练 LatentBridge
- **理论保证依赖假设**（Appendix C/D）：`D_KL(m* ∥ P_i) ≤ D_KL(m* ∥ m_v)` 成立需图像 teacher 在边际分布上优于视频 teacher 的边际，实践中未必总能满足（如图像 teacher 域与视频不匹配时）
- **仅 1,000–1,200 步 DMD 训练**：在 Causal Forcing++ 的 checkpoint 上微调，结论能否推广到从头 DMD 训练尚不明确

**结论层面**：

- 实验以 Wan2.1 系列为基础，不同视频 teacher 架构（如 CogVideoX、HunyuanVideo 等）的泛化性未验证
- 双向生成实验（vs CausVid）改进幅度（80.63→81.45）相对因果生成较小，联合-边际框架在双向场景的优势有待深入分析

---

## 8. 一句话总结

DuoMatching 将 joint DMD（视频 teacher，保时序）与 marginal DMD（图像 teacher，提帧质量）统一在一个目标函数里，通过 LatentBridge 解决视频/图像 VAE 不兼容、LVS 分配时序监督预算，在 VBench 所有 NFE 档位取得领先，人类偏好率 >80%，且无需额外推理开销——核心贡献是从理论和工程两个层面打通了图像 teacher 对视频 few-step distillation 的直接帮助。

---

## Q&A

