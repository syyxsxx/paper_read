# SANA-Streaming 论文与代码解读

> 论文:SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer (arXiv 2605.30409v1, NVIDIA + MIT + THU + NUS + HKU, 2026-06)
> 项目主页:https://nvlabs.github.io/Sana/Streaming
> 基础模型:SANA-Video → LongSANA(LongLive-style streaming long tuning 的 SANA 变体)
> 任务:**实时流式视频到视频编辑(V2V)**,不是 T2V 生成

## 一、一句话定位

SANA-Streaming = **在 SANA-Video/LongSANA 基础上,为"流式 V2V 编辑"任务做的全栈优化**。三件核心事:**(1) 混合 DiT 架构(GDN 线性注意力 + 滑窗 softmax 注意力);(2) Cycle-Reverse 训练正则化(用反向编辑 prompt 重建源视频,绕过"无长视频编辑对"困境);(3) 混合精度量化策略搜索(MPQ)+ 自研 GDN 内核**。在 RTX 5090 单卡跑出 1280×704 @ 24 FPS 端到端,DiT 内核 58 FPS。

## 二、核心定位:这不是 T2V,是 V2V 编辑

跟前面读的 CausVid/Self-Forcing/LongLive 2.0 **任务不同**:

| | T2V / I2V 生成 | V2V 编辑 (SANA-Streaming) |
|---|----------------|---------------------------|
| 输入 | text (+ first frame) | **source video + edit instruction** |
| 输出 | 视频 | 编辑后的视频 |
| 关键挑战 | 时序一致性、运动自然 | **保留源视频的运动 + 不该改的内容,只改 instruction 指定的部分** |
| 训练数据 | (text, video) pair | **(source video, edited video) pair** — **极少且昂贵** |

→ 数据稀缺是 V2V 编辑的核心痛点,Cycle-Reverse 就是为了解决这个。

## 三、训练规模事实速查(论文 Sec B "Implementation Details")

| 维度 | 值 |
|------|---|
| 基础模型 | **SANA-Video 2B(自家开源)** → LongSANA(已经用 LongLive-style 训过长视频) |
| 模型架构 | **2B Hybrid DiT**: 20 blocks (5 × softmax + 15 × GDN),hidden 2240,20 heads |
| VAE | **LTX2 VAE**,32×32×8 压缩比,蒸馏成 causal 版本 |
| 分辨率 | **1280×704** |
| 训练数据(短视频) | 10M 原始 clips → 7M(pretraining filter)→ 1M(SFT filter)。来源:Ditto + OpenVE + 自建人像编辑数据 |
| 训练数据(长视频) | **10K × 1 分钟视频**,VLM 标注 forward/inverse edit 指令 |
| 双向短训练 | **32 × NVIDIA H100**,~100K iter,batch 2/GPU,lr=5e-5,AdamW |
| 长流式训练 | 3 阶段:**ODE init → self-forcing → streaming long**(沿用 LongLive),在 streaming long 阶段加入 cycle-reverse |
| 推理 | **4 步蒸馏**,RTX 5090 单卡 **24 FPS 端到端 / 58 FPS DiT** |

## 四、架构设计:Hybrid DiT(论文 Sec 2.1)

### 痛点

| 注意力类型 | 优点 | 缺点 |
|------------|------|------|
| Softmax 全注意力 | token 级精确交互,局部细节强 | KV cache 随长度爆炸,长视频跑不动 |
| 线性注意力(SANA-Video 用) | 历史压缩成固定大小 recurrent state,流式友好 | 局部信息不够强,**chunk-to-chunk 接缝处闪烁** |
| Softmax sliding window | 折中 | 完全丢全局信息 |

→ SANA-Streaming 的 motivation 实验(Fig 3):softmax attention map 集中在局部,linear attention map 太均匀 → **两者天然互补**。

### Hybrid 架构方案(借鉴 Qwen-Next [33])

```
20 个 DiT block,均匀插入 5 个 softmax block:

block 0  block 1  block 2  block 3  block 4  block 5 ...
  GDN     GDN     GDN     GDN      Softmax    GDN  ...
                                   (window + sink)
```

- **GDN block × 15**(主力,75%):线性注意力,做全局累积记忆
- **Softmax block × 5**(25%):**sliding window + sink chunk**,做局部精细化和首块视觉锚定
- 推理时:GDN 缓存 recurrent state(固定大小),softmax 缓存 window 内 KV(也固定大小)
- **任意视频长度都是 O(1) 显存增长**

### Gated DeltaNet (GDN) 细节

不是普通线性注意力,而是 **delta-rule 修正型** linear attention。对第 f 帧:

$$
S_f^{kv} = \alpha_f S_{f-1}^{kv} (I - \beta_f \hat{k}_f \hat{k}_f^\top) + \beta_f v_f \hat{k}_f^\top
$$
$$
S_f^z = \alpha_f S_{f-1}^z (I - \beta_f k_f k_f^\top) + \beta_f k_f^\top
$$

其中:
- $\alpha_f \in \mathbb{R}$ — decay gate(衰减门)
- $\beta_f \in \mathbb{R}^N$ — write gate(写入门)
- $\hat{k}_f$ — RoPE 后的 key
- 输出:$o_f = W_o (g_f \odot \frac{S_f^{kv} \hat{q}_f}{(S_f^z)^\top q_f + \epsilon})$,$g_f$ 是 output gate

**关键 trick — delta-rule**:不是直接把 $v_f k_f^\top$ 累加,而是先擦除现有状态在 $k_f$ 方向上的投影(乘 $I - \beta_f k k^\top$),再写新值。这是**纠错型记忆**而非简单累加,对长序列更鲁棒。

代码:`diffusion/model/nets/sana_gdn_blocks.py:372` (`class GDN`),双向版本在 `:881` (`class BidirectionalGDN`)。

### Softmax block:Window + Sink

每个 softmax block 的 query 只 attend 到:
- **自己当前 chunk**(必须)
- **persistent sink chunk**(第一个 chunk,永远保留)
- **local window**(最近几个 chunk)

不用 LongLive 2.0 那种 multi-shot sink(因为 V2V 编辑里不需要"多镜头切换"概念,源视频本身定义了所有镜头),只用最简单的**单一 sink chunk**。

代码:`diffusion/model/nets/sana_gdn_blocks.py:1135` (`_forward_softmax_attn`)。**复用 GDN 的 QKV/q_norm/k_norm/proj 参数**(无新增参数),只是替换了内层运算逻辑——这是个聪明的省参数 trick。

## 五、Cycle-Reverse Regularization(论文 Sec 2.2,论文最重要的算法贡献)

### 痛点

V2V 编辑需要(source, edited) 配对数据。但:
- **短视频对**容易:用图像编辑模型改第一帧 + I2V 生成出整段编辑后视频
- **长视频对**几乎不可能:1 分钟视频要保持几百帧的编辑一致性,合成质量根本不够

但 LongLive-style streaming long training 必须用长视频做监督。怎么办?

### 解法:只用"长源视频"训练(不需要长编辑对)

构造一个对称的训练目标:**前向编辑 + 反向编辑**(Fig 4)。

**前向分支**:
```
source_video + "Transfer the background to a modern medical bay"  
  → student(streaming rollout)
  → edited_video_chunks (生成)
  → DMD self-forcing loss (跟 LongLive-style 一样,用短视频 teacher 监督)
```

**反向分支**(关键):
```
edited_video_chunks (上面生成的) + "Transfer the background to a traditional hospital"  (反向 prompt)
  → student
  → 重建 source_chunks
  → flow matching loss(target 就是原始 source_video,这是真实数据!)
```

### 数学直觉

- 前向 loss 让 student 学会"按指令编辑",但短 teacher 在长序列上信号弱
- 反向 loss 让 student 学会"编辑后还能逆回去",**目标 = 真实 source video**,所以是**强监督**
- 想"反编辑得回去",就必须在前向时**保留足够的源视频结构信息**(motion、非编辑区域、ID 等)
- → 间接强制 student 在前向编辑时不破坏源结构 = **长程时序一致性**

### Reverse prompt 怎么来

VLM(Qwen3VL)给每个 source video 生成成对的(forward, inverse)prompt(Fig 7)。例如:
- forward: "Transfer the background to a modern medical bay"
- inverse: "Transfer the background to a traditional hospital"(回到 source 的样子)

→ 数据 pipeline 上只需要长源视频 + 一次 VLM 标注,**完全不需要长编辑对**。这是这个方法最实用的地方。

### 在训练 pipeline 里的位置

```
Stage 1: 双向短训练 (32 H100, 100K iter)
   ↓
Stage 2: ODE init (LongLive 风格,蒸馏成 4 步因果)
   ↓
Stage 3: Self-forcing training (短视频 rollout + DMD)
   ↓
Stage 4: Streaming long training (长视频 rollout + DMD + **cycle-reverse loss**)
```

只在最后一个阶段引入 cycle-reverse loss,前面阶段先把模型训稳。

## 六、Causal VAE Distillation(论文 Sec 2.2 末)

### 痛点

LTX2 VAE 原版 decoder 是**双向** temporal context,推理时需要"未来帧"才能解码当前帧。流式推理根本拿不到未来帧。

### 解法:权重重映射 + 蒸馏

LTX2 VAE 的 3-tap temporal conv:`y[t] = a·x[t-1] + b·x[t] + c·x[t+1]`(双向)

直接砍掉 c 项会丢失信号。论文的 trick(Fig 5):

```
双向权重 [a, b, c]
    ↓ 重映射
因果权重 [0, a, b+c]
            ↑
        把未来帧贡献"折叠"到当前帧
```

这样:`y[t] = a·x[t-1] + (b+c)·x[t]` — 完全因果,且保留了 a 的"前一帧"角色和 c 的"瞬时"角色。

然后用**双向 teacher 蒸馏因果 student**:
$$
\mathcal{L}_{\text{vae}} = \lambda_{\text{charb}} \mathcal{L}_{\text{charb}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{haar}} \mathcal{L}_{\text{haar}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}}
$$

四个 loss 分别:Charbonnier 重建 + perceptual + Haar 小波高频 + 中间特征蒸馏。

### 效果(论文 Table 3)

| | PSNR | LPIPS | SSIM |
|---|------|-------|------|
| Bidirectional (teacher) | 32.98 | 0.0274 | 0.923 |
| Causal (直接 padding 换 left-only,不训) | 24.66 | 0.132 | 0.785 |
| **Causal (蒸馏后)** | **32.14** | **0.0336** | **0.911** |

→ 蒸馏后几乎追平双向 teacher。

## 七、Efficient System Co-design(论文 Sec 2.3)

### 7.1 高效 GDN 内核(论文 Sec 2.3, Appendix D)

朴素 GDN 实现的问题:每帧的 $Q_f, K_f, V_f$ 跨越数百个 spatial token,要反复从 HBM 读;recurrent state $S_f, z_f$ 虽然小但需要逐帧串行。

### Triton chunkwise 三阶段流水线

把 GDN 更新公式重写成分离的 spatial / temporal 两部分:

$$
P_f = I - K_f^\top \text{diag}(\beta_f) K_f, \quad A_f = K_f^\top \text{diag}(\beta_f) V_f
$$
$$
S_f = \alpha_f P_f S_{f-1} + A_f
$$

→ $P_f, A_f$ **可帧并行计算**,只有 $S_f$ 的标量 scan 需要串行。

```
Phase A (帧并行,空间 tile reduction):
   每帧并行计算 P_f, A_f, P_f^z, b_f
   把 spatial 维度 blocked reduction,只输出"frame summaries"
   
Phase B (短串行扫描,状态保 on-chip):
   对每个 (b, h),沿 frame 维度做 recurrent scan,S_f 永远在 SRAM 里

Phase C (输出,spatial tile 流式):
   流式 load Q_f blocks,应用 compact states 得到 O_f
```

**双向 GDN 的优化**(Fig 2 + Appendix D):利用 Phase C 的线性性,把 forward scan 和 reverse scan 的历史在输出前**线性合并**,避免第二次 Q 读 + 第二次输出 kernel。

→ 端到端 1.5×~2.2× 加速(不同 GPU)。

代码:`diffusion/model/nets/sana_gdn_blocks_triton.py`、`diffusion/model/ops/fused_gdn_chunkwise.py`。

### 7.2 Mixed-Precision Quantization (MPQ) 策略搜索(论文 Sec 2.3 + Appendix A)

NVFP4 在 Blackwell 上很快,但**不能无脑全部 FP4**——有些层敏感,有些层量化开销 > 加速收益。LongLive 2.0 是手工挑层,SANA-Streaming **自动搜**。

### 搜索粒度

按"层类型 × block 位置深度"分组搜:
- **Layer-type groups**:self-attn Q/K/V/O,cross-attn Q/KV/O,FFN input/output projection,temporal FFN projection
- **Block-position groups**:shallow (0-5),middle (6-13),deep (14-19)

### 搜索目标(论文 Sec 2.3 Eq 6)

对每个候选策略 π,跑 30 秒视频生成,跟 BF16 reference 比:
$$
\text{RelRMSE}(\pi) = \frac{\sqrt{\frac{1}{N}\|\hat{x}_0^\pi - \hat{x}_0^{\text{ref}}\|_2^2}}{\text{Std}(\hat{x}_0^{\text{ref}}) + \epsilon}
$$

加上 LPIPS 双目标,在帕累托前沿挑速度 vs 质量 trade-off 最好的策略。

### 搜出来的结论(论文 Sec 2.3 末)

| 类型 | 精度 | 原因 |
|------|------|------|
| Patch embedding、output projection、timestep embedding、attention gates、MixFFN depth-wise conv | **BF16** | 最敏感或不支持低精度 |
| CA-O、SA-Q、SA-K、Temporal FFN projection | **FP4** | 误差很小 + 加速 2.06× |
| FFN input/output projection (middle/deep blocks) | **FP4** | 49% FLOPs 大头,但深层 block 对量化鲁棒 |
| FFN input/output projection (shallow blocks) | **FP8** | 浅层敏感 |
| SA-V、SA-O、CA-Q | **FP8** | value 和 post-mixing 路径对 FP4 噪声敏感 |
| CA-KV | **FP8** | 占 DiT FLOPs <1%,降级收益太小 |

→ **量化层加速 3×**,DiT 总延迟比 BF16 baseline 快 **1.59×**,质量基本无损。

### 跟 LongLive 2.0 量化策略的对比

| | LongLive 2.0 | SANA-Streaming |
|---|---|---|
| 策略选择方式 | 手工挑(论文 Sec 2.2) | **自动 AutoML-style 搜索**(论文 Sec 2.3) |
| 精度池 | NVFP4 + BF16 (sensitive ops) | **NVFP4 + FP8 + BF16(三档)** |
| 训练时是否量化 | ✅ End-to-end NVFP4 训练 | ❌ **只在推理量化**,训练 BF16 |
| KV cache 量化 | ✅ NVFP4 KV cache + K-smoothing | ❌(论文没明确做) |

注意:**SANA-Streaming 训练用 BF16**,只在推理时做 MPQ;LongLive 2.0 训练就是 NVFP4。两者出发点不同——SANA-Streaming 想"在不动训练 pipeline 的前提下榨干推理速度",LongLive 2.0 想"训练也用 NVFP4 拉动整个长视频训练"。

## 八、和前作的关系图

```
        SANA (2D image)
              ↓
        SANA-Video (线性注意力 video DiT, 短视频)
              ↓
       LongSANA (LongLive-style streaming long tuning)
              ↓
     SANA-Streaming (本论文)
       │
       ├── 架构改造: 全线性 → Hybrid (15 GDN + 5 Softmax window+sink)
       ├── 任务改造: T2V → V2V editing
       ├── 训练新增: Cycle-Reverse Regularization
       ├── VAE 改造: 双向 LTX2 → causal LTX2 (权重重映射 + 蒸馏)
       └── 系统优化: Triton GDN kernel + MPQ search
```

### 跟 LongLive 2.0 的横向对比

| | LongLive 2.0 | SANA-Streaming |
|---|--------------|-----------------|
| 任务 | 长 T2V 多镜头 | 长 V2V 编辑 |
| 基础模型 | Wan2.2-TI2V-5B (5B) | SANA-Video 2B (2B) |
| 注意力 | 全 softmax(块状因果) | **Hybrid GDN + softmax** |
| 训练 | Teacher Forcing + Error Recycling | **LongLive-style + Cycle-Reverse Reg** |
| 数据需求 | 长真实视频(120K) | **长源视频即可,不需要长编辑对**(10K) |
| 训练精度 | End-to-end NVFP4 | BF16 训练 + MPQ 推理 |
| 多镜头切换 | Multi-Shot Attention Sink + RoPE offset | 不处理(任务里源视频已定义镜头) |
| 推理硬件 | GB200(数据中心) | **RTX 5090(消费级)** |
| 推理速度 | 45.7 FPS (2 步) | 24 FPS 端到端 / 58 FPS DiT (4 步) |

## 九、关键代码位置导览

| 功能 | 文件 |
|------|------|
| GDN block 主体 | `diffusion/model/nets/sana_gdn_blocks.py:372` (class GDN) |
| 双向 GDN | `diffusion/model/nets/sana_gdn_blocks.py:881` (class BidirectionalGDN) |
| Softmax attention (hybrid 用) | `diffusion/model/nets/sana_gdn_blocks.py:1135` (_forward_softmax_attn) |
| Triton GDN kernel | `diffusion/model/nets/sana_gdn_blocks_triton.py` |
| Chunkwise fused GDN | `diffusion/model/ops/fused_gdn_chunkwise.py` |
| Streaming training model | `diffusion/longsana/model/streaming_sana_long.py` |
| Self-forcing trainer | `diffusion/longsana/trainer/self_forcing_trainer.py` |
| DMD for SANA | `diffusion/longsana/model/dmd_sana.py` |
| ODE regression init | `diffusion/longsana/model/ode_regression_sana.py` |
| LongSANA config | `configs/sana_video_config/longsana/480ms/longsana.yaml` |
| LongSANA full config | `configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml` |

注意:**cycle-reverse regularization 的实现在公开代码里没找到明确入口**(2026-06 paper,代码可能尚未完整开源,需要根据 paper 描述参考训练 pipeline)。

## 十、配置项速查(`longsana.yaml`,对照看)

```yaml
# 模型基础
sana_config: configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml
generator_ckpt: hf://Efficient-Large-Model/LongSANA_2B_480p_self_forcing/...

# 4 步去噪 timesteps(注意:LongSANA 用 5 个点,SANA-Streaming 4 步可能用前 4)
denoising_step_list:
  - 1000
  - 960
  - 889
  - 727
  - 0

# 流式训练
streaming_training: true
streaming_chunk_size: 21         # 每 chunk 21 latent 帧
streaming_max_length: 261        # 总长上限 261 帧 (≈ 1 分钟 @ 16 FPS latent)
streaming_min_new_frame: 20

# DMD 设置
dfake_gen_update_ratio: 5
distribution_loss: dmd_sana
trainer: longsana

# 模型 kwargs
model_kwargs:
  timestep_shift: 7.0
  local_attn_size: 21            # softmax window 大小(以 latent 帧为单位)

# 视频形状 [B, F, C, H, W]
image_or_video_shape:
  - 1
  - 261
  - 16
  - 60
  - 104
```

注意 `local_attn_size: 21` —— softmax block 的 sliding window 设成 21 帧(= 1 chunk),即只关注当前 chunk + sink chunk。

## 十一、争议点与权衡

### 1. Hybrid 比例的选择

5 softmax : 15 GDN(25% : 75%)是论文给的固定比例,没做消融实验说明为啥不是 4:16 或 6:14。理论上 softmax 越多 V2V 编辑越精细但内存/延迟越高。

### 2. Cycle-Reverse 的隐含假设

反向 prompt 在概念上是"源 domain"的描述,但**源 domain ≠ 源视频精确像素**。例如:
- forward: "Transfer the background to medical bay"
- inverse: "Transfer the background to traditional hospital"

如果源视频原本是 "traditional hospital",inverse 完美 — student 学到的就是真实可逆性。
但如果源视频是 "outdoor",inverse prompt 让它回到 "hospital"(不是 outdoor)— student 学到的就是**风格映射,不是像素重建**。

→ Cycle-reverse 在概念层面正则化"语义可逆",但不保证像素级一致性。这可能是为什么它叫"regularization"而不是"reconstruction"。

### 3. 不做 KV 量化的成本

跟 LongLive 2.0 比,SANA-Streaming 没做 KV cache 量化。原因可能是:
- Hybrid 架构里 softmax block 只占 25%,KV cache 已经不是主要瓶颈
- GDN 的 recurrent state 是 O(d²) 而非 O(seq×d),已经天然紧凑
- 推理目标硬件(RTX 5090)显存够用(论文报 5.56 GB VRAM)

### 4. 训练 BF16 vs 推理 MPQ 的训推 mismatch

训练 BF16、推理 MPQ 会有量化误差导致质量微降。论文用 RelRMSE+LPIPS 搜索控制误差,但理论上**训练就 NVFP4-aware**(LongLive 2.0 路线)能完全消除这个 gap。SANA-Streaming 选了更便宜的工程路径。

## 十二、一句话总结

> **SANA-Streaming = LongSANA (架构基础) + Hybrid GDN/Softmax (架构创新) + Cycle-Reverse Reg (训练创新) + Triton GDN kernel + AutoML 风格 MPQ 搜索 (系统创新)**。
> 核心算法贡献是 **Cycle-Reverse** —— 用反向编辑 prompt 把"无长视频编辑对"的难题转化成"用真实长源视频做监督",代价是只能正则化语义可逆性而非像素可逆性。系统侧 Triton GDN kernel 把线性注意力真做到 IO-aware,MPQ 搜索把推理量化策略从手工挑变成自动搜。最终在 RTX 5090 上实现 1280×704 @ 24 FPS 端到端、5.56 GB 显存的实时长视频编辑。
