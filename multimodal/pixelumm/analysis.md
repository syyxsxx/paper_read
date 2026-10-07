# PixelUMM: Encoder-Free Unified Image and Video Understanding and Generation

> Cong Wei, Xuanchi Ren, Bryan Chu, Weiming Ren, Huan Ling, Jiahui Huang, Laura Leal-Taixé, Sanja Fidler, Wenhu Chen, Zian Wang, Jay Zhangjie Wu  
> NVIDIA / University of Waterloo, 2026-09-29  
> arXiv: 2609.38597 | [Project](https://nv-tlabs.github.io/PixelUMM/)

---

## 1. 一句话定位

PixelUMM 是首个将图像与视频的理解和生成统一在**像素空间**的 encoder-free 模型：无 ViT、无 VAE、无离散 tokenizer，图片用 2D spatial patch、视频用 3D spatiotemporal tubelet，通过单层线性投影接入 Qwen3 初始化的 Mixture-of-Transformers 骨干，并给出 8 组覆盖 patch size、decoder head、训练动态、conditioning 等维度的系统消融。

---

## 2. 要解决的问题

现有统一多模态模型（UMM）通常维护**两套视觉接口**：
- 理解侧：预训练 **ViT**（SiGLIP 等）提供语义特征
- 生成侧：**VAE** 提供重建导向的 latent

双接口的代价：
1. 每张条件图要贡献 ViT tokens + VAE tokens，视觉上下文长度约翻倍
2. 多轮对话中 tokens 随图像/视频帧数线性累积，attention 和 KV cache 开销巨大
3. 视频理解（per-frame encoder）和视频生成（causal 3D VAE）的时序表示约定不兼容，难以共享接口
4. 与 VLM 预训练 pipeline 的整合需要额外改造数据和训练流程

**PixelUMM 的问题**：能否用单一的原始像素接口，在无预训练视觉编码器的情况下，同时支持图像理解、视频理解、图像生成、视频生成四大任务？

---

## 3. 与前作的关系

| 工作 | 关系 |
|------|------|
| JiT (2025) | 首个无 VAE 的像素空间 T2I 生成（clean-pixel prediction）；PixelUMM 将其扩展到视频与理解 |
| PixelDiT (2025) | 像素空间图像生成，纯生成侧，无理解 |
| TUNA / TUNA-2 | encoder-free 统一图像理解+生成；PixelUMM 新增视频支持 |
| SenseNova-U1 | 引入 encoder-free 视觉条件（clean→理解专家，noisy→生成专家）；PixelUMM 将此机制扩展到视频 |
| BAGEL | ViT+VAE 双编码器统一模型；PixelUMM 对标并去掉两个编码器 |
| Qwen3 | PixelUMM 的骨干初始化来源 |

---

## 4. 核心方法

### 4.1 原生像素接口

图片和视频完全绕过预训练编码器，像素到骨干之间只有确定性的 patchify 和一层线性投影：

$$
\mathbf{x}^{\text{img}} \xrightarrow{\text{2D Patchify}_{16\times16}} \mathbb{R}^{N_{\text{img}} \times (16\cdot16\cdot3)} \xrightarrow{\text{img\_und/gen\_linear\_proj}} \mathbb{R}^{N_{\text{img}} \times d}
$$

$$
\mathbf{x}^{\text{video}} \xrightarrow{\text{3D Patchify}_{4\times16\times16}} \mathbb{R}^{N_{\text{video}} \times (4\cdot16\cdot16\cdot3)} \xrightarrow{\text{video\_und/gen\_linear\_proj}} \mathbb{R}^{N_{\text{video}} \times d}
$$

- 图像：patch size `p=16`，每个 patch = 768 维原始 RGB → 线性投影到 hidden dim `d`
- 视频：tubelet size `τ×p×p = 4×16×16`，每个 tubelet = 3072 维 → 投影到 `d`
- 每个模态各有**两套独立线性投影**：理解侧（输入为干净图像/视频）和生成侧（输入为加噪图像/视频）
- 输出 head 也是单层线性（RMSNorm + linear），多模态训练前**零初始化**

没有 VAE encoder、没有 ViT、没有离散 visual tokenizer。

### 4.2 Mixture-of-Transformers (MoT) 架构

![Fig 3: PixelUMM 架构图](./figures/fig3_arch.png)

> **Fig 3 逐段解读**：
>
> **(左侧蓝色 — 理解专家 Understanding Expert)**：原始图像经 2D Patchify + `img_und_linear_proj` 得到 image tokens；原始视频帧经 3D Patchify + `video_und_linear_proj` 得到 video tokens；文本走 Text Embed。三路输入共同进入 `Norm + Und. QKV` → 共享的 `Multimodal Self-Attention` → `Norm + Und. FFN`。Text Prediction Head 在上面预测下一个 text token。
>
> **(右侧粉色 — 生成专家 Generation Expert)**：加噪图像和加噪视频分别经生成侧线性投影进入 `Norm + Gen. QKV` → 同一个共享 `Multimodal Self-Attention` → `Norm + Gen. FFN` → 线性输出 head 预测干净像素。
>
> **(共享 attention，分离参数)**：绿色的 `Multimodal Self-Attention` 是唯一共享的组件；两个专家各有独立的 QKV 投影、FFN 和 normalization。路由规则：文本 token 和干净 visual token → 理解专家；加噪 visual token → 生成专家。
>
> **(无显式 timestep)**：与大多数扩散模型不同，PixelUMM 省去了 timestep embedding，网络直接从加噪像素值中隐式推断噪声水平。

### 4.3 序列建模与 Attention 设计

![Fig 5: Attention 模式](./figures/fig5_attention.png)

> **Fig 5 逐段解读**：
>
> **(a) 图像理解**：所有 image token 构成一个双向块（互相全可见）。Text token 因果注意力，可以 attend 所有前置的 image token。干净图像走理解专家。
>
> **(b) 视频理解**：每帧构成独立的双向岛。帧间保持时序因果性：Frame 2 可以看 Frame 1，反之不行。Text 可以 attend 全部前置视觉上下文。
>
> **(c) 图像/视频生成**：text prompt token 因果注意力。加噪目标 token 对 prompt 因果 attend，同时在目标块内部双向 attend（粉红方块）。干净条件不能"看进"加噪目标（masked）。

**位置编码**：三轴 Native RoPE（temporal / height / width）。一半 attention head 编码时间轴，各四分之一编码高度和宽度。`θ_H = θ_W = 10^4`，`θ_T = 10^6`。Text token 只沿时间轴前进（H=W=0）。

**视频理解的两种模式**：
- `dense_mode`：输入帧率 ≥4 FPS → 3D tubelet 块（`τ=4`），时序上下文更丰富
- `sparse_mode`：输入帧率 <4 FPS → 1 FPS 采样，每帧独立经 `img_und_linear_proj` 处理

### 4.4 训练目标

**文本目标**（cross-entropy，对 assistant token，带序列长度平方根归一化）：

$$
\mathcal{L}_{\text{CE}}
$$

**像素目标**（JiT 风格 clean-pixel prediction，v-parameterization）：

$$
\mathbf{z}_t = (1-t)\mathbf{x} + t\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

$$
\mathbf{v}_\theta = (\mathbf{z}_t - \hat{\mathbf{x}}_\theta) / \tilde{t}, \quad \mathbf{v}^* = (\mathbf{z}_t - \mathbf{x}) / \tilde{t}, \quad \tilde{t} = \max(t, 0.05)
$$

对激活像素和激活 visual token 取平均 squared velocity error。

**联合目标**：

$$
\mathcal{L} = \lambda_{\text{CE}}\mathcal{L}_{\text{CE}} + \lambda_{\text{img}}\mathcal{L}_{\text{img}} + \lambda_{\text{vid}}\mathcal{L}_{\text{vid}}
$$

### 4.5 训练流程（6 阶段）

| 阶段 | 训练分支 | 分辨率 | Steps | 主要任务 |
|------|---------|--------|-------|---------|
| Joint Stage 1 | 两侧都训 | 256² | 150K | Text + I2T + T2I |
| Gen Stage 1 | 仅生成侧 | 256² | 200K | T2I + T2V |
| Und Stage 1 | 仅理解侧 | Native | 100K | Text + 原生分辨率 I2T |
| Und Stage 2 | 仅理解侧 | 224²（视频） | 20K | + V2T |
| Und Stage 3 | 仅理解侧 | 448²（视频） | 15K | + 更高分辨率 V2T |
| Joint Stage 2 | 两侧都训 | 512² | 20K | 全任务全分辨率 |

### 4.6 多模态上下文 Conditioning

![Fig 19: Encoder-free 与 BAGEL conditioning 对比](./figures/fig19_conditioning.png)

> **Fig 19 逐段解读**：
>
> **(a) PixelUMM encoder-free 条件**：干净参考图经理解专家进入（蓝色块），可与前置文本和视觉上下文互相 attend。加噪生成目标（粉色）对 prompt（包含干净参考图）因果 attend，在目标块内部双向 attend——但干净条件被遮挡，看不到加噪目标。生成专家通过共享 attention 间接获取参考图的特征。
>
> **(b) BAGEL 双编码器条件**：每张条件图贡献两路 token：ViT token（语义，蓝紫色）+ VAE token（重建，紫色），视觉上下文长度大约翻倍。PixelUMM 的单流设计消除了这种重复，代价是需要 decoder 从头学习视觉表示。

这种单流 conditioning 支持 image-to-video 生成和视频编辑（F7 消融，Table 2-3）。

---

## 5. 八组系统消融（F1–F8）

| 消融 | 变量 | 关键结论 |
|------|------|---------|
| **F1**: 图像 patch size | 16×16 vs 32×32 | 16×16 生成 loss 更低，即使每步处理的图像数只有 32×32 的 1/4——更强的空间压缩反而让生成更难学 |
| **F2**: 视频 patch size | p32/t4 → p32/t1 | 时空压缩越弱（p32/t1：4096 像素/token）T2V loss 越低；最终选 p16/t4 与视频 VAE 惯例对齐 |
| **F3**: 输出 decoder head | Linear vs PixelShuffle vs Wan-style Up+Conv | Linear head（133 GFLOPs, 12.6M 参）在 CFG>6 时产生网格 artifacts；T-S PixelShuffle（1279 GFLOPs, 52.9M 参）是 loss 与 artifact 的最佳折中 |
| **F4**: 像素空间 vs VAE 空间训练 | flow matching in pixels vs latents | 梯度范数相近；VAE 空间 loss 数值约 4.3×（不同单位，不代表学得快）；像素空间偶发 loss spike |
| **F5**: 模型大小 | 1.7B vs 8B | 8B 以约 1/3 的训练步数达到相同 loss（3× step 效率） |
| **F6**: 算力规模 | 8 GPU vs 128 GPU | CE 损失：~2K vs 17.5K steps（快约 9×）；MSE 损失：~17K vs 23K（快约 1.3×）——文本收益远大于生成 |
| **F7**: 视觉条件 conditioning | 多任务微调前后 | 视频 benchmark 全线 +0.89 到 +3.49 pts；图像有涨有跌；结论：引入视觉条件能力不系统性损害理解 |
| **F8**: 视频理解接口 | `dense_mode` vs `sparse_mode` | 4 FPS tubelet 无一致优势，MVBench +0.07 / Video-MME −0.22 / LVBench 平 |

📌 **F1 最反直觉**：更小的 patch（16×16）每步看的图像更少，却收敛更快、loss 更低。原因：patch 越大 → 每个 token 要表示更大空间范围的像素 → 生成任务更难建模。

---

## 6. 关键结果

### 图像理解（Table 5, Joint Stage 2）

| Benchmark | PixelUMM (8B MoT) | BAGEL (7B MoT) | TUNA-2 (7B+5B) | Qwen3-VL-Inst. (8B) |
|-----------|-------------------|----------------|-----------------|----------------------|
| MMMU | 41.67 | 55.30 | 50.70 | **69.60** |
| RWQA | 71.63 | **72.80** | 67.70 | 71.50 |
| AI2D | 80.12 | **89.20** | 79.60 | 85.70 |
| DocVQA | 90.42 | — | — | **95.20** |
| CountBench | **94.30** | 82.50 | 81.70 | 89.80 |
| MME | 1809.54 | **2388.00** | — | — |

在统一模型（BAGEL/TUNA-2）中处于同一档，但与纯理解专用 VLM（Qwen3-VL-Inst.）在 MMMU 上差距约 28 pts。

### 视频理解（Table 6, sparse\_mode）

| Benchmark | PixelUMM (8B MoT) | Qwen3-VL (8B) | InternVL-3.5 (8B) |
|-----------|-------------------|----------------|-------------------|
| MVBench | 70.53 | — | 72.10 |
| Video-MME (w/o sub.) | 57.33 | **73.00** | 65.90 |
| LongVideoBench | 59.61 | **68.00** | 62.40 |
| LVBench | 40.41 | **58.00** | 44.50 |

### 图像生成（Table 7，统一模型组内对比）

| 模型 | 大小 | GenEval | DPG-Bench |
|------|------|---------|-----------|
| BAGEL | 7B MoT | 0.77 | 85.07 |
| **PixelUMM** | **8B MoT** | **0.77** | **85.74** |
| Lance | 3B MoT | 0.90 | 84.67 |
| TUNA | 7B+5B | 0.90 | **86.76** |

在统一模型中处于中等偏上水平；GenEval 与 BAGEL 持平，DPG 略超。

⚠️ **注**：发布的 checkpoint 使用线性输出 head（F3-R01），非 PixelShuffle head，在光滑区域 CFG>6 时有可见网格 artifacts。

---

## 7. 争议与局限

**无视觉先验**：从头学视觉表示是最大短板。MMMU 差距 −28 pts（vs Qwen3-VL-Inst.）反映了在需要细粒度视觉推理的任务上缺乏预训练 ViT 的代价。

**Patch artifacts**：线性 decoder head 在低纹理区域产生网格边界强度变化（CFG>6 时明显）。卷积 head（PixelShuffle）可抑制，但需要额外训练，发布版未采用。

**Dense 4FPS 无稳定增益**：视频 3D tubelet 是论文对视频理解的核心新设计，但 F8 显示在四个 benchmark 上 4FPS tubelet 与 1FPS per-frame 无一致改进。论文对此结论较为诚实，但动机受损。

**无显式 timestep 的风险**：省去 AdaLN/timestep embedding 使网络依赖像素值隐式推断噪声程度，偶发的 pixel-space loss spike（F4）可能与此相关。

**生成结果与专用模型差距未量化**：未与 Wan2.2/HunyuanVideo 等专用视频生成模型做定量对比。

---

## 8. 一句话总结

PixelUMM 用 2D/3D patchify + 单层线性投影 + Qwen3 初始化的 MoT 骨干，在零编码器（无 ViT/VAE/tokenizer）的前提下统一了图像与视频的理解和生成，在统一模型中达到有竞争力的水平，但相比专用 VLM 在语义理解上存在明显差距，且 3D tubelet 设计在视频理解上未能体现优势；8 组系统消融为像素空间统一模型的设计决策（patch 尺寸、decoder head、条件接口、算力扩展）提供了实用参考。

---

## Q&A

