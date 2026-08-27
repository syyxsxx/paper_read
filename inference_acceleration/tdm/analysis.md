# TDM: Learning Few-Step Diffusion Models by Trajectory Distribution Matching

**论文**:Learning Few-Step Diffusion Models by Trajectory Distribution Matching  
**作者**:Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, Jing Tang (HKUST + Huawei Noah's Ark Lab + NUS)  
**发表**:ICCV 2025  
**代码**:https://github.com/Luo-Yihong/TDM  
**项目页**:https://tdm-t2x.github.io/

---

## 1. 一句话定位

在 trajectory 各时间步的**分布层面**对齐 student 和 teacher 的 ODE 轨迹,而非逐实例对齐——data-free、500 iter 就能让 4-step PixArt-α 超过 50-step teacher,SDXL 蒸馏只需 2 A800 天。

---

## 2. 要解决的问题(动机)

现有扩散蒸馏分两类,各有致命缺陷:

| 方法类型 | 代表 | 优点 | 缺陷 |
|---------|------|------|------|
| **Trajectory Matching** | CM, TCD, PeRFlow | 利用完整轨迹信息,质量较好 | 逐实例匹配 ODE 轨迹需数值求解,累积误差大;受限模型容量时效果差 |
| **Distribution Matching** | DMD, SiD, Diff-Instruct | data-free,1步性能强 | 只在单步优化分布,缺少多步灵活性;hard to add NFE |

TDM 的出发点:trajectory 信息 + distribution-level 对齐可以兼得——不在实例层面跑 ODE,而是在每个轨迹时间步的**边际分布**上做 KL divergence。

---

## 3. 与前作的关系

```
Diff-Instruct / VSD (distribution matching, 单步)
    │
    ├── DMD / DMD2 (distribution matching, 多时间步直接预测 x_0)
    │       ├── 每步独立优化 → 忽略轨迹中间阶段
    │       └── fake score 共享于所有步 → 偏置(本文 Eq.9 分析)
    │
    └── SiD (distribution matching, 1步)
    
CM / LCM (trajectory matching, 随机多步)
    └── 只支持随机 sampling,确定性效果差

TCD / PeRFlow (trajectory matching, 确定性)
    └── 实例级匹配 → 求解教师 ODE 耗时且有数值误差

TDM (本文): 把 trajectory 分解成 K 段,每段做 distribution matching
    → 无需求解教师 ODE(data-free)
    → 确定性采样天然好(比随机采样更 few-step 友好)
    → 支持灵活步数(TDM-unify:K 作为 conditioning)
```

---

## 4. 核心算法/方法

### 4.1 Trajectory Distribution Matching

![Fig 4: TDM 方法图示](./figures/fig4_method.png)

> **Fig 4 逐段解读**：以 2-step generator 为例。
>
> **上方蓝色箭头(Backward Deterministic Sampling)**——student generator θ 做两步去噪：从 `x_T ~ N(0,I)` 出发，Generator θ 输出中间状态 `x_{T/2}`，再加噪，再过一次 Generator θ 得到 `x_0`。这条路径就是 student 的 ODE 轨迹。
>
> **黑色虚线箭头(Forward Diffusion Process)**——从 student 生成的干净图出发，往回加噪，得到对应各段的噪声样本 `x_{τ₁}` 和 `x_{τ₂}`。这是"扩散"方向，用于构造 distribution matching 目标所需的中间 latent 样本。
>
> **每段：Real Score - Fake Score → ∇θ KL**——对每个噪声样本 `x_τ`，分别查询 teacher score `s_φ(x_τ, τ)`（Real Score，绿色）和 fake score `s_ψ(x_τ, τ)`（Fake Score，灰色），两者之差即为梯度方向：把 student 样本向 teacher 分布推，同时用 fake score 归一化。
>
> **分段非重叠的设计**——左段 `τ₁ ∈ [T/2, T)`，右段 `τ₂ ∈ [0, T/2)`。两段不重叠，保证 fake score 可以用同一组参数无偏地建模各段分布，避免 DMD2 中 fake score 被多步分布混淆的问题。
>
> **Importance Sampling(绿色框)**——从 `q(x_τ|x_{t_i})` 采样后，用重要性权重 `q(x_τ|x_{t_i}) / q(x_τ|x̂_{t_i})` 修正 fake score 的训练信号，降低方差、确保无偏。

**核心目标(Eq. 5)**：把 student 在每段轨迹起点 `t_i` 出发、到区间 `[t_i, t_{i+1}]` 内各时间步 τ 的边际分布，与 teacher 的对应边际分布做 KL 散度：

$$
L(\theta) = \sum_{i=0}^{K-1} \sum_{\tau=t_i}^{t_{i+1}} \lambda_\tau \mathrm{KL}\!\left(p_{\theta,\tau|t_i}(\mathbf{x}_\tau) \;\Big\|\; p_{\phi,\tau}(\mathbf{x}_\tau)\right)
$$

梯度展开后等价的最小化 loss(Eq. 6):

$$
L(\theta) = \sum_{i=0}^{K-1} \sum_{\tau=t_i}^{t_{i+1}} \lambda_\tau \left\lVert \mathbf{x}_{t_i} - \mathrm{sg}(\hat{\mathbf{x}}_{t_i}) \right\rVert_2^2
$$

其中 `sg(·)` 是 stop-gradient，`x̂_{t_i}` 是对 student 样本沿 KL 梯度方向做一步修正得到的"修正目标":

$$
\hat{\mathbf{x}}_{t_i} = \mathbf{x}_{t_i} + \lambda_\tau \bigl[s_\phi(\mathbf{x}_\tau, \tau) - s_\psi(\mathbf{x}_\tau, \tau)\bigr] \frac{\partial \mathbf{x}_{t_i}}{\partial \theta}
$$

**📌 本质**：这和 DMD 的单步梯度形式完全相同(`real_score - fake_score`),只是 TDM 在 trajectory 上的 K 个区间内的每个 τ 都做一遍,而 DMD 只在一个特定 t 做。

### 4.2 Fake Score 学习与 Importance Sampling

Fake score `s_ψ` 通过对 student 生成样本进行去噪训练(Eq. 7):

$$
L(\psi) = \sum_{i=0}^{K-1} \mathbb{E}_{p_{\theta,t_i}} \mathbb{E}_{q(\mathbf{x}_\tau|\mathbf{x}_{t_i})} \omega_\tau \left\lVert f_\psi(\mathbf{x}_\tau, \tau) - \hat{\mathbf{x}}_{t_i} \right\rVert_2^2
$$

加入 importance sampling,用权重 `q(x_τ|x_{t_i}) / q(x_τ|x̂_{t_i})` 修正,使 fake score 的训练目标与 generator 的修正目标分布一致,降低方差。

**为什么非重叠区间是关键**:若区间重叠,同一 fake score `s_ψ` 必须同时建模 `p_{K₁}(x)` 和 `p_{K₂}(x)` 两个不同分布,其最优解退化为两者的混合(见 Eq. 9),引入偏置。非重叠保证每个 τ 只属于一个区间,fake score 对该时间步的分布无偏。

### 4.3 Sampling-Steps-Aware 目标(TDM-unify)

为支持推理时灵活选步数 K,将 K 注入为 student 和 fake score 的 conditioning 信号:

$$
\mathbb{E}_K \sum_{i=0}^{K-1} \sum_{\tau=t_i^K}^{t_{i+1}^K} \lambda_\tau \mathrm{KL}\!\left(p_{\theta,\tau|t_i^K}(\mathbf{x}_\tau \mid K) \;\Big\|\; p_{\phi,\tau}(\mathbf{x}_\tau)\right)
$$

其中 `t_i^K = T/K · i`。K 均匀采样自用户想支持的步数集合。训练时 K 同时输入 student DiT 和 fake score 网络作为额外条件。

消融结果:有 K-conditioning 时 1-step HPS 28.90,无时仅 26.11;4-step 同样有 +2 点收益。

### 4.4 Surrogate Training Objective(Pseudo-Huber)

受 Consistency Models(iCT)启发,把 L2 距离替换为 Pseudo-Huber metric:

$$
L(\theta) = \sum_{i=0}^{K-1} \sum_{\tau=t_i}^{t_{i+1}} \sqrt{\left\lVert \mathbf{x}_{t_i} - \mathrm{sg}(\hat{\mathbf{x}}_{t_i}) \right\rVert_2^2 + c^2} - c
$$

其中 `c = 0.00054√d`(d 为数据维度)。梯度中有归一化因子 `1 / sqrt(||·||² + c²)`,有效防止 `x - x̂` 大时梯度爆炸,稳定训练。

---

## 5. 极速收敛的直觉

![Fig 2: 收敛速度](./figures/fig2_convergence.png)

> **Fig 2 逐列解读**：固定初始噪声,对比 4-step TDM 在 0/50/100 iter 和 50-step teacher 的生成结果。
>
> - **0 iter**：纯随机初始化,Pikachu 糊成一团,猫近乎噪声。
> - **50 iter**：Pikachu 已具备完整形体、颜色、背景;猫的构图和色调基本准确。质量已接近 teacher。
> - **100 iter**：细节进一步提升,与 teacher 50 NFE 视觉等价甚至更优(少 CFG artifact)。
>
> 快速收敛的原因:TDM 的蒸馏目标是**对准分布**,每步只需做局部 denoising 而非全程 ODE 推理;distribution-level 对齐的信息密度远高于实例级匹配,梯度方向更稳定。

---

## 6. ODE 轨迹质量

![Fig 7: ODE 轨迹中间 clean 样本](./figures/fig7_trajectory.png)

> **Fig 7 逐行解读**：以"A dog reading a book"为例,可视化各中间 timestep T=749/499/249/0 的"clean x₀ 预测"。
>
> - **上行(SD 1.5 teacher trajectory)**：T=749 时狗的轮廓模糊,面部有 CFG artifact(颜色过饱和、眼神不自然);逐步清晰但 CFG 带来的过度锐化始终存在。
> - **下行(TDM learned trajectory)**：T=749 时整体构图和光影已更自然;T=249 时细节完整度与 teacher T=0 相近;**全程几乎没有 CFG artifact**。
>
> 这解释了为何 TDM 4-step 能超越 teacher 50-step:teacher 用 CFG 加速收敛但引入饱和度偏移,TDM 在分布层面对齐避免了这一副作用。

---

## 7. 实验结果

### 定量(Table 1, SD-v1.5 / SDXL / PixArt-α)

| 方法 | Backbone | Steps | HPS avg↑ | AeS↑ | CS↑ | Image-Free? |
|------|---------|-------|----------|------|-----|-------------|
| Teacher (CFG=3.5) | SD-v1.5 | 25 | 25.50 | 5.49 | 33.03 | — |
| DMD2 | SD-v1.5 | 4 | 29.49 | 5.91 | 31.53 | ✗ |
| **TDM-unify-SFT (Ours)** | SD-v1.5 | 4 | **32.12** | **6.02** | **32.77** | ✓ |
| LCM | SDXL | 4 | 27.87 | 5.84 | 30.77 | ✗ |
| DMD2 | SDXL | 4 | 30.39 | 5.88 | 31.49 | ✗ |
| **TDM (Ours)** | SDXL | 4 | **32.25** | **6.28** | **36.08** | ✓ |
| Teacher (CFG=3.5) | PixArt-α | 25 | 32.21 | 6.23 | 34.11 | — |
| LCM-1024 | PixArt-α | 4 | 30.72 | 5.64 | 17.33 | ✗ |
| **TDM-1024 (Ours)** | PixArt-α | 4 | **34.61** | **6.42** | **33.66** | ✓ |

PixArt-α 4-step TDM 全面超越 25-step teacher。

### 训练成本对比(Table 3)

| 方法 | Backbone | 训练成本 |
|------|---------|---------|
| DMD2 | SD-v1.5 | 30+ A800 Days |
| **TDM-unify-GAN** | SD-v1.5 | **4 A800 Days** |
| **TDM-unify-SFT** | SD-v1.5 | **3 A800 Days** |
| LCM | SDXL | 32 A100 Days |
| DMD2 | SDXL | 160 A100 Days |
| **TDM** | SDXL | **2 A800 Days** |
| LCM / YOSO | PixArt-α | 14.5 / 10 A100 Days |
| **TDM** | PixArt-α | **2 A800 Hours (500 iter)** |

### 视频扩展(Table 4, CogVideoX-2B)

| 方法 | NFE | VBench Total↑ |
|------|-----|---------------|
| CogVideoX-2B (teacher) | 50 | 80.91 |
| **TDM** | 4 | **81.65** |

4 NFE TDM 超越 teacher 50 NFE。

### 定性对比

![Fig 5: SDXL 定性对比](./figures/fig5_qualitative.png)

> **Fig 5 逐行对比**：3 个 prompt × 7 个方法(SDXL 50 NFE / LCM / TCD / SDXL-Lightning / Hyper-SD / DMD2 / TDM)。
>
> - **行 1("两艘海盗船在咖啡杯里")**：LCM/TCD 只生成 1 艘或形状错误;Hyper-SD 颜色偏冷;DMD2 质感还原度低;TDM 两艘船轮廓清晰,咖啡油脂纹理真实,接近 teacher。
> - **行 2("蓝色墨镜亚洲女性特写")**：LCM 面部细节糊;TCD/Lighting 肤色偏黄;DMD2 墨镜颜色正确但背景过曝;TDM 面部层次最自然,墨镜蓝色准确。
> - **行 3("宇航员骑猪,高写实电影感")**：LCM/TCD/Lighting 场景构图混乱;Hyper-SD 宇航员形变;DMD2 质感粗糙;TDM 动作自然,宇宙背景景深正确。

用户偏好研究:TDM vs SDXL-Base(25步)= 76.2% vs 23.8%;TDM vs DMD2 = 81.1% vs 18.9%;TDM vs PixArt-α-Base = 71.3% vs 28.7%。

---

## 8. 与 DMD2 的详细对比

| 维度 | DMD2 | TDM |
|------|------|-----|
| 训练目标 | 分布匹配,单步直接预测 x₀ | 分布层面轨迹匹配,K 段 |
| 是否利用轨迹中间阶段 | ❌ 忽略 | ✅ 每段 τ 都参与梯度 |
| Fake score | 共享于所有步(有偏) | 各段非重叠(无偏) + importance sampling |
| 确定性采样 | ✅ 支持 | ✅ 天然支持(更好) |
| 多步灵活性 | 需多阶段重训 | TDM-unify:K 作为 condition |
| 训练成本(SDXL) | 160 A100 Days | **2 A800 Days(25×省)** |
| FID 收敛速度 | baseline | TDM 只需 DMD2 训练时间的 4% 即可达到 DMD2 收敛 FID |
| 最终 FID | baseline | 比 DMD2 低 **20%** |

---

## 9. 关键代码位置

| 功能 | 文件 | 说明 |
|------|------|------|
| 主训练脚本 | `train_tdm_demo.py` | 完整 TDM 训练实现 |
| student/teacher/fake score 定义 | `train_tdm_demo.py:Predictor` | 封装分数预测与加噪 |
| TDM loss(生成器) | `train_tdm_demo.py:loss_instruct` | `F.mse_loss(model_latents, coop_samples)` |
| Fake score loss | `train_tdm_demo.py:loss_score` | SNR 加权 + importance sampling 权重 |
| 多步轨迹生成 | `train_tdm_demo.py:generate_new()` | K 步确定性去噪,产生 `noisy_imgs_list` |

**关键实现细节**:
```python
# 生成器 loss: student latent → 对齐 cooperative target
coop_samples = fake_latents + (sd_latents - fake_latents) + (cfg-1)*(sd_latents - sd_latents_uncond)
loss_instruct = F.mse_loss(model_latents.float(), coop_samples.detach().float())

# Fake score loss: SNR 加权 + importance sampling
loss_score = F.mse_loss(fake_latents.float(), model_latents.float()) * snr * is_weight
```

`coop_samples` 本质是:fake_latents + (real_score - fake_score) 方向的修正,即 `x̂_{t_i}`。

---

## 10. 争议/权衡

**优点**:
- Data-free,只需 prompt,不需要配对图像数据
- 极快收敛:500 iter / 2 A800 hours 就能超越 PixArt-α teacher
- 确定性采样下性能远优于随机采样方法(LCM)
- TDM-unify 一个模型支持 1/2/3/4 步

**局限**:
- Performance ceiling 受 teacher 限制——teacher 质量差(如 SD-v1.5)需先 fine-tune 才能蒸馏,需要额外数据
- Fake score 训练需要与 generator 交替更新,实现比单阶段方法复杂
- 视频扩展实验只在 CogVideoX-2B 验证,scale 较小
- 没有直接对比 SDXL + CFG 的 HPS(Table 1 中 teacher 用 CFG=3.5)

---

## 11. 一句话总结

TDM 把 distribution matching 扩展到 ODE trajectory 的每个时间段:data-free score distillation + 非重叠区间无偏 fake score + importance sampling + pseudo-Huber surrogate + K-step awareness,用 2 A800 天蒸馏出超越 teacher 的 4-step SDXL,25× 比 DMD2 省。

---

## Q&A

*(对话中产生的追问将持续追加至此处)*
