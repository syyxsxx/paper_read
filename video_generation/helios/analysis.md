# Helios: Real Real-Time Long Video Generation Model

> arXiv:2603.04379 · ByteDance + 北京大学 · 2026-03
> 项目主页: https://pku-yuangroup.github.io/Helios-Page

---

## 1. 一句话定位

**第一个在单张 H100 上跑到 19.5 FPS 的 14B 视频生成模型**,支持分钟级长视频,**不依赖** KV-cache、稀疏/线性注意力、量化、causal masking、self-forcing 或 error-banks 中的任何一种常规加速/抗漂移手段。核心贡献是三件事同时成立:
1. **无 self-forcing 的抗漂移**(Relative RoPE + First-Frame Anchor + Frame-Aware Corrupt)
2. **不用 causal masking 的 AR 生成**(Unified History Injection + Guidance Attention)
3. **token 压缩把 14B 的算力开销压到 1.3B 级别**(Multi-Term Memory Patchification + Pyramid UPC)

---

## 2. 要解决的问题(动机)

现有实时长视频方案都是 1.3B 量级(CausVid、Self-Forcing、LongLive、Rolling Forcing 等):
- 模型太小 → 复杂运动/高频细节建模能力不足
- 抗漂移依赖 self-forcing long rollout → 训练开销巨大,限制了往 14B 扩展
- Krea-14B 把规模提上去了但 FPS 只有 6.7,且有明显漂移

根本矛盾:**大模型(14B)+ 长视频抗漂移 + 实时推理** 三角很难同时满足。Helios 的论点是:换一套算法框架,让三者都能达到。

---

## 3. 与前作的关系

```
Wan-2.1-T2V-14B (bidirectional, 50-step)
          │
          ▼
    Helios (本文,3 阶段训练)
    ├── Stage 1 (Base):     架构适配,历史注入,抗漂移
    ├── Stage 2 (Mid):      引入 Pyramid UPC,token 压缩
    └── Stage 3 (Distilled):AHD 蒸馏,50→3 步
```

**与 CausVid/Self-Forcing 系的关键差异:**

| 维度 | CausVid / Self-Forcing / LongLive | Helios |
|------|------|------|
| 因果机制 | Causal masking(块内/全序列) | **无 causal mask**,bidirectional 推理保留 |
| 抗漂移 | self-forcing rollout / error-bank / long-video training | Relative RoPE + First-Frame Anchor + Frame-Aware Corrupt |
| 加速手段 | KV-cache + 分步蒸馏(DMD/SiD) | **token 压缩**(MTMP + PyUPC) + AHD 蒸馏 |
| 长视频训练 | 短clip微调 + 推理rollout | 普通短clip(<10s),靠 Relative RoPE + Frame Anchor 外推 |
| 蒸馏 teacher | Bidirectional Wan(短视频) | **Helios-Base(长视频)** |

Helios 的核心洞见:**causal masking 并不是 AR 视频生成的必要条件**。把历史帧当 clean context 拼在 noisy 帧之前,用 Guidance Attention 管理二者的交互,双向推理完全可以生成连贯长视频。

---

## 4. 核心算法

### 4.1 Unified History Injection & Representation Control

模型输入是两段拼接:
- `XHist ∈ R^{B×C×T_hist×H×W}`:历史 clean 帧(timestep 固定为 0)
- `XNoisy ∈ R^{B×C×T_noisy×H×W}`:当前待去噪块(timestep 1000→0)

任务统一控制:
- T2V:`XHist` 全零
- I2V:仅最后一帧非零
- V2V:`XHist` 是真实历史帧

训练时随机 zero-out 一定比例历史帧,自然覆盖三种任务。

![Fig 4: Helios 架构全景图](./figures/fig4_architecture.png)

> Fig 4:左侧是 Multi-Term Memory Patchification(3 层历史:Short/Mid/Long),中间是 Pyramid Unified Predictor Corrector(多分辨率噪声→数据轨迹),右侧是 Guidance Attention(历史与噪声帧的独立自注意力 + 共享交叉注意力)。整个框架不使用 causal masking,保留双向推理能力,通过结构分离实现 T2V/I2V/V2V 统一。

### 4.2 Guidance Attention(Eq. 1-2)

历史帧和噪声帧的 KQV 分别计算,用 head-wise 可学习 `amp` 参数调制历史 key:

**Self-attention(合并 hist + noisy)**:

$$
X_{Self} = \text{Attention}([Q_{Noisy}, Q_{Hist}],\ [K_{Noisy}, K_{Hist} \cdot amp],\ [V_{Noisy}, V_{Hist}])
$$

**Cross-attention(只对 noisy)**:

$$
X_{Cross} = \text{Attention}(Q_{Noisy}, K_{Text}, V_{Text})
$$

`amp` 是 per-head 标量,允许网络自学习每个头对历史应该放大还是抑制。历史帧不做 cross-attention(已含语义)。代码见 `helios/modules/transformer_helios.py:394-408`。

---

### 4.3 Easy Anti-Drifting(三种漂移 + 三种对策)

漂移有三种典型形式:

![Fig 5: 三种漂移模式可视化](./figures/fig5_drifting.png)

> Fig 5:Position Shift(重复回到早期帧位置,场景突然重置),Color Shift(颜色/饱和度随时间累积偏移),Restoration Shift(Noise/Blur:历史帧的轻微噪声/模糊在推理时不断被放大为修复伪影)。这三种漂移对应三种独立的解决方案。

**① Relative RoPE(解决 Position Shift)**

不管视频多长,强制 `XHist` 的时间 RoPE 索引为 `[0, T_hist]`,`XNoisy` 的索引为 `[T_hist, T_hist + T_noisy]`。
- 消除了绝对索引超出训练分布的问题
- 避免 RoPE 周期性与多头注意力的共振导致的周期性重置运动

**② First-Frame Anchor(解决 Color Shift)**

`XHist` 中**永远保留第一帧**。第一帧作为全局视觉锚点,约束颜色/风格分布不漂移。实验表明去掉后 720 帧就开始出现明显颜色偏移。

**③ Frame-Aware Corrupt(解决 Restoration Shift)**

训练时对每个历史帧**独立**随机施加以下扰动之一:
- 概率 `pa`:加高斯噪声(`bmin~bmax` 幅度)
- 概率 `pb`:下采样再上采样(`cmin~cmax` 倍)
- 概率 `pc`:调整曝光(`amin~amax`)
- 概率 `pd`:保持干净

每帧独立采样(不是同一视频段统一操作),这是确保长视频稳定的关键。代码见 `helios/utils/utils_base.py:731-740`。

---

### 4.4 Deep Compression Flow — Token View

目标:把 14B 模型的 attention token 数压到 1.3B 同级。

**Multi-Term Memory Patchification(历史压缩,8×)**

历史 `XHist` 分三档,用不同步长的 Conv3D 压缩:

| 档位 | 时间帧数 | Conv3D kernel & stride |
|------|---------|------------------------|
| Short(近) | `T1=16` | `(1,2,2)` — 空间 2× |
| Mid(中) | `T2=2` | `(2,4,4)` — 时空 8× |
| Long(远) | `T3=2` | `(4,8,8)` — 时空 32× |

三档合计 token 数不随历史长度增加(固定约 `5/8 × HW` 个 token)。

代码:
- 三路 Conv3D 定义:`helios/modules/transformer_helios.py:1066-1068`
- 权重初始化(从预训练 patch embedding 扩展):`helios/modules/transformer_helios.py:1095-1101`
- forward 中拼接:`helios/modules/transformer_helios.py:1201-1256`

![Fig 7: MTMP 固定 token 预算对比](./figures/fig7_mtmp.png)

> Fig 7:三张图分别是 Token 数、训练 VRAM、推理单步时间 vs 历史帧数。Naive 方案(直接拼接)随历史长度线性增长并在 6 段时 OOM;Helios(蓝色)三项都基本平坦,历史长度可以扩展到 18 段。

**Pyramid Unified Predictor Corrector(噪声压缩,2.3×)**

把 flow matching 轨迹分 `K=3` 个分辨率 stage:

$$
x^k_t = (1 - \lambda_t) x^k + \lambda_t \, \text{Up}(x^{k-1})
$$

对应 ground-truth 速度:

$$
v^k = x^k - \text{Up}(x^{k-1})
$$

![Fig 8: Pyramid UPC 三阶段流程](./figures/fig8_pupc.png)

> Fig 8:低分辨率阶段(t=999→T1)专注结构/布局,中分辨率(T1→T3)兼顾质量与效率,高分辨率(T3→0)精修细节。每次 stage 切换时上采样 + 重新加噪(renoise)保持路径分布一致性。Stage 切换时 UniPC 的预测缓存清空重积累。

推理步数分配 `(N1, N2, N3)`,总 token 数 ≈ `HW + (H/2)(W/2) + (H/4)(W/4)` × `N/K`,约为全分辨率的 43%。调度器见 `helios/scheduler/scheduling_helios.py`。

---

### 4.5 Deep Compression Flow — Step View(Adversarial Hierarchical Distillation)

50 步(Base) → 50 步(Mid,PyUPC) → **3 步**(Distilled,AHD)

AHD 在 DMD 基础上做四点改进:

![Fig 9: AHD 蒸馏框架](./figures/fig9_ahd.png)

> Fig 9:生成器(Few-Step Generator + EMA)只看一段 section,历史完全来自真实数据(Pure Teacher Forcing)。生成的 `x0^K` 经 Dynamic Re-noise 加噪后同时喂给冻结的 Real-Score 和可学习的 Fake-Score(带多粒度 GAN Head)。Staged Backward Simulation 在 K 个分辨率 stage 上分别估 `x0^k`,只有 `x0^K` 进入 real/fake score 计算。

**① Pure Teacher Forcing(不需要 self-rollout)**

历史完全用真实数据,训练时只生成 1 段 section。对比 Self-Forcing 需要 rollout 5~20 段:
- 训练效率大幅提升 → 得以扩展到 14B
- 论文 ablation 证明抗漂移效果与 self-forcing long rollout 等效(靠 Easy Anti-Drifting 补足)

**② Staged Backward Simulation**

backward simulation 在 K 个 stage 上分别进行,产生 `{x0^k}`:

$$
x^k_0 = x^k_t - \lambda_t \cdot u^k_\theta(x^k_t, y, \lambda_t, k)
$$

**关键**:只有最终全分辨率的 `x0^K` 才送给 real/fake score(不是所有 k 层)。送中间 `x0^k` 给 pfake 会导致优化方向错误(ablation 中训练不稳定)。

**③ Coarse-to-Fine Learning**
- Staged ODE Init:在 K 个 stage 各做一轮 ODE 初始化,只需单段
- Dynamic Re-noise:Beta 分布采 timestep,参数余弦 decay,早期集中高噪(学结构),后期均匀(学细节)

**④ Adversarial Post-Training**

GAN head 布在 DiT 层 `[5, 15, 25, 35, 39]`,dim=768,对 `H/2 × W/2` crop 计算:

$$
L_D = \mathbb{E}[\log D(x^\text{real}_\tau, \tau)] + \mathbb{E}[-\log D(x^K_\tau, \tau)] + \lambda_D \cdot \|D(x^\text{real}) - D(N(x^\text{real}, \sigma_D))\|^2
$$

打破 teacher 的分布上限,让 student 直接对齐真实数据分布。

---

### 4.6 推理 Tricks

**Adaptive Sampling(自适应抗漂移)**

每段生成后,计算 latent 的均值 `μ_t` 和方差 `σ_t²`,用 EMA 跟踪全局统计:

$$
\bar{\mu}_t = \rho_\mu \bar{\mu}_{t-1} + (1-\rho_\mu)\mu_t
$$

当当前段统计偏离全局超过阈值 `δμ` 且 `δσ`,下一段历史中触发 Frame-Aware Corrupt,隐式迫使模型更依赖生成 prior 而非漂移历史。代码见 `helios/utils/utils_base.py:673-729`。

**Interactive Interpolation(交互视频)**

用户修改 prompt 时,线性插值 `e[j] = (1-λj)e(1) + λj e(2)`,M 步平滑过渡,避免 prompt 突变引起的闪烁和语义跳变。

---

## 5. 关键代码位置

| 功能 | 文件:行 |
|------|---------|
| 三路历史 patchification Conv3D 定义 | `helios/modules/transformer_helios.py:1066-1068` |
| patch 权重初始化(从预训练 embedding 扩展) | `helios/modules/transformer_helios.py:1095-1101` |
| forward 中 MTMP 拼接历史 token | `helios/modules/transformer_helios.py:1201-1256` |
| Guidance Attention key 放大(amp) | `helios/modules/transformer_helios.py:394-408` |
| HeliosAttnProcessor 完整 attention 逻辑 | `helios/modules/transformer_helios.py:196-445` |
| Discriminator3DHead(多粒度 GAN head) | `helios/modules/transformer_helios.py:96-126` |
| PyUPC 多阶段 scheduler | `helios/scheduler/scheduling_helios.py` |
| AdaptiveAntiDrifting(EMA 漂移检测) | `helios/utils/utils_base.py:673-745` |
| Frame-Aware Corrupt 实现 | `helios/utils/utils_base.py:731-740` |
| 流式 pipeline 推理 | `helios/pipelines/pipeline_helios.py` |
| Flash Normalization Triton kernel | `helios/modules/helios_kernels/triton_norm.py` |
| Flash RoPE Triton kernel | `helios/modules/helios_kernels/triton_rope.py` |

---

## 6. 关键配置项

**训练三阶段超参:**

| 配置项 | Stage 1(Base) | Stage 2(Mid) | Stage 3(Distilled) |
|--------|--------------|-------------|-------------------|
| GPU | 64 × H100 | 64 × H100 | 128 × H100 |
| LoRA rank | 128 | 256 | 256 |
| Global batch | 128 | 256 / 192 | 128 |
| LR (Gθ) | 5e-5 / 3e-5 | 1e-4 / 3e-5 | 2e-6 |
| Steps | 5.5k + 7.5k | 16k + 20k | 3759 + 2250 |
| 精度 | BFloat16 | BFloat16 | BFloat16 |

**Frame-Aware Corrupt 概率:**

| Stage | pa(噪声) | pb(下采样) | pc(曝光) | pd(干净) |
|-------|---------|-----------|---------|---------|
| 1/2 | 0.0 | 0.8 | 0.1 | 0.1 |
| 3 | 0.4 | 0.4 | 0.0 | 0.2 |

Stage 1/2 主要用下采样(模拟历史质量下降),Stage 3 加大噪声增强鲁棒性。

**MTMP 压缩系数:**`(pt, ph, pw)` = Short:(1,2,2),Mid:(2,4,4),Long:(4,8,8);帧数 `(T1,T2,T3)=(16,2,2)`。

**Stage 3 GAN head 位置:** DiT 层 `[5, 15, 25, 35, 39]`,hidden dim=768,训练 1000 步后启动。

**PyUPC 阶段划分:** `stage_range=[0, 1/3, 2/3, 1]`,K=3,推理用 UniPC solver。

**推理配置:**
- Stage 1/2:UniPC 50 步,CFG=5.0,v-prediction
- Stage 2:用 CFG-Zero-Star 替代标准 CFG
- Stage 3:x0-prediction,3 步,CFG=1.0

**Triton 加速效果(Table 6,50次前向+反向,H100):**

| 配置 | 推理时间(s) | 训练时间(s) |
|------|-----------|-----------|
| Wan-2.1-14B 原版 | 98.68 | 398.03 |
| + Flash Normalization | 89.91 | 360.77 |
| + Flash RoPE | 93.39 | 378.77 |
| + 两者 | **84.41** | **340.38** |

---

## 7. 争议 / 权衡

**1. Bidirectional vs Causal Masking**

Helios 不用 causal mask,历史帧会出现在 noisy 帧的 attention 里。Guidance Attention 通过拆分 QKV 来避免历史帧被噪声帧"污染"。但代价是:attention 矩阵更大(hist + noisy 全连接),而 causal 方案可以用 KV cache。Helios 选择用 MTMP 压缩 token 数来抵消这个开销。

**2. Pure Teacher Forcing vs Self-Forcing**

论文 ablation 表明 Pure Teacher Forcing(只用真实历史 + Easy Anti-Drifting)在长视频抗漂移上与 Self-Forcing long rollout 效果相当。但这个结论依赖 Frame-Aware Corrupt + First-Frame Anchor 共同起效。如果去掉任何一个,漂移就会很快出现。

**3. Staged Backward Simulation 的细节**

只把 `x0^K`(全分辨率最终结果)送给 real/fake score,而不是所有 `x0^k`。Paper 中 ablation 显示"w Staged Backward Simulation*"(送多尺度 `x0^k`)会导致训练不稳定。作者认为多尺度 x0 分布差异大,fake score 无法统一建模。

**4. 漂移未完全解决**

Ablation Table 5 中 Drifting-Naturalness 只有 5 分(10 分满分),与 CausVid(6)和 Rolling Forcing(7)比有差距。论文 limitation 也承认段边界 flickering 仍然存在。

**5. 分辨率限制**

所有实验都限于 384×640。更高分辨率的性能未探索。14B 无并行训练的内存节省主要靠了 token 压缩,更高分辨率会重新撑爆显存。

---

## 8. 一句话总结

Helios 的核心论点:**不用 causal masking 也能做 AR 视频生成**,用 Representation Control + Guidance Attention 把历史/噪声帧职责分开,配合三层分辨率历史压缩(MTMP)+ 多尺度噪声压缩(PyUPC)把 14B 模型的 token 算力压到 1.3B 量级,再用 Pure Teacher Forcing + Easy Anti-Drifting 取代昂贵的 self-rollout,从而第一次把 14B 长视频生成跑到单卡实时。
