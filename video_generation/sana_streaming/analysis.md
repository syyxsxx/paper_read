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

## 三.5、V2V 任务的具体实现机制

读这篇论文前必须搞清楚:**V2V 编辑模型怎么"吃"源视频 + 编辑指令两个条件**。论文文字略过了这块,但代码很清楚。

### 1. 源视频的 conditioning:channel-wise concat

`diffusion/model/nets/sana_multi_scale_video_camctrl.py:907-908`:
```python
if data_info.get("image_vae_embeds", None) is not None:
    x = torch.cat([x, data_info["image_vae_embeds"].to(self.dtype)], dim=1)
```

`image_vae_embeds` 是**源视频经 VAE encode 后的 latent**。流程:

```
推理一个 chunk:
  noisy_latent     [B, 16, T, H, W]    ← 16 通道 LTX2 VAE latent + 噪声
  source_latent    [B, 16, T, H, W]    ← 源视频经 VAE encode (逐像素对齐)
        │
        torch.cat dim=1 (channel)
        ↓
  input_to_DiT     [B, 32, T, H, W]    ← 32 通道 = 16 noisy + 16 source
        ↓
     patch_embed (Conv3D, in_channels=32, out_channels=2240)
        ↓
  video_tokens     [B, T·H·W/patch, 2240]
        ↓
     20 层 Hybrid DiT
        ↓
  flow_pred        [B, 16, T, H, W]    ← 输出还原成 16 通道
```

**几个关键点**:
- **逐像素对齐**:source 和 noisy 在同一时空位置,模型在 token 维度天然同时看到"这一点的源像素特征 + 当前去噪状态"
- **patch_embed 输入通道翻倍**(16 → 32),输出通道保持(16),这是 V2V 训练的入口改造
- **没有 ControlNet,没有 reference attention**,就是最朴素的 channel concat

### 2. Edit prompt 的 conditioning:Gemma-2-2b-it + CHI prompt

`configs/sana_video_config/Sana_2000M_480px_adamW_fsdp_longsana.yaml:80-94`:
```yaml
text_encoder:
  text_encoder_name: gemma-2-2b-it       # ← 2.6B 参数 instruct LLM
  y_norm: true
  y_norm_scale_factor: 0.01
  model_max_length: 300
  chi_prompt:                            # CHI = Chain-of-thought Hierarchy
    - 'Given a user prompt, generate an "Enhanced prompt" ...'
    - ...examples...
    - 'User Prompt: '
```

**跟传统 T2V 模型的差异**:
- **Encoder 不是 CLIP / T5**,而是 **Gemma-2-2b-it**(instruct-tuned LLM)。这是 SANA 系列标志性设计,语义理解远超 CLIP
- **CHI prompt(Chain-of-thought prompt wrapper)**:用户的 prompt 不直接喂给 Gemma,而是先包进一个固定的 instruction template(让 Gemma 把简单描述扩展成详细视觉描述),再 encode
- Gemma 输出 prompt embedding `[B, 300, 2304]`(300 = max_length,2304 = Gemma hidden dim)

### 3. Prompt 在 DiT 内部的注入路径:Cross-Attention + AdaLN

每一个 DiT block 的 forward 大致流程:

```python
# x: video tokens [B, N_tokens, 2240]
# y: prompt embedding [B, 300, 2304]
# t: timestep [B]

# 1. timestep → AdaLN modulation 参数 (6 个)
t_modulation = t_block(t)                  # [B, 6 * 2240]
shift1, scale1, gate1, shift2, scale2, gate2 = chunk(6)

# 2. Self-attention (GDN 或 softmax,看 block 类型)
x_normed = norm1(x) * (1 + scale1) + shift1
x = x + gate1 * self.attn(x_normed)         # ← Hybrid 的关键就是这里换 GDN/softmax

# 3. Cross-attention — prompt 在这里进来!!
x = x + self.cross_attn(query=x, key_value=y)
#                       ↑           ↑
#                  video token   prompt embedding

# 4. FFN (GLUMBConvTemp: 1×3×3 spatial conv + 3×1×1 temporal conv + GLU)
x_normed = norm2(x) * (1 + scale2) + shift2
x = x + gate2 * self.mlp(x_normed)
```

→ **每一层 block 都有一次 cross-attention**,prompt embedding 当 K/V,视频 token 当 Q。每层都重新"询问 prompt 该改什么"。

### 4. 三个条件的融合视图

```
                 ┌────────────────────┐
   timestep t ──→│  AdaLN modulation  │── 调节 norm1/norm2 (shift/scale/gate × 2)
                 └────────────────────┘
                 ┌────────────────────┐
   prompt y ────→│  Cross-Attention   │── 每个 block 里 video query prompt
   (Gemma)       │  Q=video, K/V=y    │
                 └────────────────────┘
                 ┌────────────────────┐
   source ──────→│  Channel concat    │── 仅在 DiT 输入处一次性塞进去
   video         │  (16 + 16 → 32 ch) │
                 └────────────────────┘
                          ↓
                 video tokens (Self-Attention 输入)
```

每个条件的"信息密度"不同:
- **Source video**:像素级精确,但只在 patch embed 一次性融入(此后靠 self-attention 在视频 token 间扩散)
- **Prompt**:语义级,每层 cross-attention 重复注入
- **Timestep**:标量,通过 AdaLN 调节每层的 norm

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

不是普通线性注意力,而是 **delta-rule 修正型** linear attention。完整推导(从 standard softmax → linear → DeltaNet → GDN)见后文 **§四.5**。这里只列核心公式。

**逐帧更新**:
$$
S_f^{kv} = \alpha_f S_{f-1}^{kv} (I - \beta_f \hat{k}_f \hat{k}_f^\top) + \beta_f v_f \hat{k}_f^\top
$$
$$
S_f^z = \alpha_f S_{f-1}^z (I - \beta_f k_f k_f^\top) + \beta_f k_f^\top
$$

**输出**:$o_f = W_o (g_f \odot \frac{S_f^{kv} \hat{q}_f}{(S_f^z)^\top q_f + \epsilon})$

其中:
- $\alpha_f \in (0,1)$ — decay gate(每帧一个标量,头共享)
- $\beta_f$ — write gate(每帧 × 每 spatial token,sigmoid 输出)
- $\hat{k}_f$ — RoPE 后的 key(只用在 numerator $S^{kv}$,**不用在 denominator $S^z$**——论文称 "mass conservation")
- $g_f$ — output gate(silu)

**关键 trick — delta-rule**:不是直接把 $v_f k_f^\top$ 累加,而是先擦除现有状态在 $k_f$ 方向上的投影(乘 $I - \beta_f k k^\top$),再写新值。这是**纠错型记忆**而非简单累加,对长序列更鲁棒。

代码:`diffusion/model/nets/sana_gdn_blocks.py:372` (`class GDN`),双向版本在 `:881` (`class BidirectionalGDN`)。

## 四.5、Linear Attention 完整推导 + GDN 代码精读

### Step 1: 从 softmax attention 到 linear attention

Softmax attention 公式:
$$
o_i = \sum_j \frac{\exp(q_i^T k_j)}{\sum_l \exp(q_i^T k_l)} v_j
$$
复杂度 O(n²d),因为必须算 n×n 的相似度矩阵。

线性注意力的 trick:把 $\exp(q^T k)$ 近似成 $\phi(q)^T \phi(k)$(SANA 用 ReLU)。代入后**φ(q) 可以从求和里提出来**:
$$
o_i = \frac{\phi(q_i)^T \sum_j v_j \phi(k_j)^T}{\phi(q_i)^T \sum_l \phi(k_l)}
$$

定义两个累积状态:
- 分子:$S^{kv} = \sum_j v_j \phi(k_j)^T \in \mathbb{R}^{d \times d}$
- 分母:$S^z = \sum_j \phi(k_j) \in \mathbb{R}^d$

则:$o_i = S^{kv} \phi(q_i) / ((S^z)^T \phi(q_i) + \epsilon)$。

**关键收益**:state 是固定 d×d 大小,不随 n 增长 → **复杂度 O(nd²),对 n 线性**。

### Step 2: 增量更新(因果版)

$$
S_i^{kv} = S_{i-1}^{kv} + v_i \phi(k_i)^T \quad \text{(累加一个 d×d 外积)}
$$

**致命问题**:无脑累加,旧信息永远擦不掉。视频里早期帧的内容会永远累积,新内容只能叠加。

### Step 3: Delta Rule — 用"预测-修正"替换累加

DeltaNet 的核心思想:**把 state 看成一个 key→value 的关联记忆库**。来一帧新的 (k_f, v_f),先用现有 state 查 k_f 应该对应什么 value,只写差额:

1. 预测:$\hat{v}_f = S_{f-1}^{kv} \phi(k_f)$
2. 残差(带 write gate β):$\delta_f = \beta_f (v_f - \hat{v}_f)$
3. 写入:$S_f^{kv} = S_{f-1}^{kv} + \delta_f \phi(k_f)^T$

展开:
$$
S_f^{kv} = S_{f-1}^{kv} \underbrace{(I - \beta_f \phi(k_f) \phi(k_f)^T)}_{\text{擦除 k 方向旧投影}} + \underbrace{\beta_f v_f \phi(k_f)^T}_{\text{写入新值}}
$$

这就是 **`(I - β k k^T)` 的来历**。两种等价理解:
- **乘性视角**:`(I - β k k^T)` 是 rank-1 投影修正算子,削弱 state 在 k 方向上的成分
- **加性视角**:`δ = β (v - v_pred)`,写新值同时擦掉旧的同方向预测

### Step 4: Gated DeltaNet (GDN) = DeltaNet + decay

加上全局 decay α,得到 SANA-Streaming 用的形式:
$$
\boxed{S_f^{kv} = \alpha_f \cdot S_{f-1}^{kv} (I - \beta_f \hat{k}_f \hat{k}_f^\top) + \beta_f v_f \hat{k}_f^\top}
$$

### 代码精读 1:`torch_recurrent_sana_gdn`(慢但语义清晰)

`sana_gdn_blocks.py:96-195`,这是教学版:

```python
state_kv = zeros(B, H, D, D)    # 初始状态
state_z  = zeros(B, H, D, 1)

for t in range(T):                          # 跨帧顺序 scan
    qt, kt, vt = q[:,:,t], k[:,:,t], v[:,:,t]
    qrt, krt   = q_rot[:,:,t], k_rot[:,:,t]
    bt, gt     = beta[:,:,t], decay[:,:,t]

    # Decay
    state_kv = state_kv * gt
    state_z  = state_z  * gt

    # Delta-rule KV update
    v_pred  = matmul(state_kv, krt)              # 预测 v
    delta_v = (vt - v_pred) * bt                 # 残差 × β
    state_kv = state_kv + matmul(delta_v, krt.T) # 写残差

    # Output
    out_num = matmul(state_kv, qrt)              # 分子: S · q_rot
    out_den = matmul(state_z.T, qt)              # 分母: (S^z)^T · q

return out_num_stacked / (out_den_stacked + eps)
```

**逐行对公式**:
| 公式 | 代码 |
|------|------|
| $\hat{v}_f = S_{f-1}^{kv} \hat{k}_f$ | `v_pred = matmul(state_kv, krt)` |
| $\delta_f = \beta_f (v_f - \hat{v}_f)$ | `delta_v = (vt - v_pred) * bt` |
| $S_f^{kv} \mathrel{+}= \delta_f \hat{k}_f^T$ | `state_kv = state_kv + matmul(delta_v, krt.T)` |
| $o_f = S_f^{kv} \hat{q}_f / ((S_f^z)^T q_f + \epsilon)$ | `out_num / out_den` |

### 代码精读 2:`torch_chunk_sana_gdn`(实际生产用的快版)

`sana_gdn_blocks.py:199-312`,关键 trick——把 update 改写成 $S_t = S_{t-1} W_t + U_t$:

```python
# 预计算两个矩阵 (帧并行,无跨帧依赖):
W_kv = decay * (I - β · k_rot @ k_rot^T)    # [B,H,T,D,D] decay × delete operator
U_kv = β · v @ k_rot^T                       # [B,H,T,D,D] new info

# 跨帧 scan 只剩一个 matmul + 加:
for t in range(T):
    state_kv = state_kv @ W_kv[:,:,t] + U_kv[:,:,t]
```

**为什么快**:
- W_t, U_t 的计算**完全帧并行**(GPU 大并发)
- scan 阶段每步一次 D×D matmul,可保持在 SRAM
- 可按 chunk 切(`chunk_size=21`)进一步并行

这正好对应论文 Sec 2.3 Eq (4)(5)。

### 没有 N×N 矩阵!

**整个 forward 里最大的中间张量是 `state_kv: [B, H, D, D]`(D ≈ 32-128)**。没有 `[N, N]`(N 是 token 数,几万)。**复杂度 O(T·S·D²),对 N 完全线性**——这就是"线性注意力"的字面含义。

### 完整 GDN forward 还有什么(`forward` at `sana_gdn_blocks.py:733`)

除了上面的核心 update rule,完整的 GDN block 还有这些额外设计:

1. **ReLU kernel**:`q = relu(q); k = relu(k)`。强制非负,保证 normalizer 不会负数,且分布稀疏
2. **K 的 short conv**:`k = depthwise_Conv1d(k along T, kernel=4)`,**让相邻几帧的 k 互相平滑**——防止 delta rule 把单帧的剧变全擦掉
3. **RoPE 只在分子**:`q_rot, k_rot` 用在 $S^{kv}$,**未 RoPE 的 q, k 用在 $S^z$ normalizer**。论文称 "mass conservation"——如果 normalizer 也加 RoPE,就不再是无偏的 mass
4. **β per-token, α per-frame**:write gate 在 spatial 维度可以不同强度,decay gate 全帧统一
5. **State 强制 FP32**:`q, k, v, β, decay = .float()`,长时间 recurrent scan 对数值精度敏感
6. **Output gate**:`out = silu(W_g · x) * out`,门控 SwiGLU 风格

### GDN vs 普通线性注意力对比

| 痛点 | 普通线性注意力 | GDN |
|------|----------------|-----|
| 老信息无法擦除 | ❌ 永远累加 | ✅ delta rule 自动擦旧投影 |
| 全局衰减 | ❌ 无 | ✅ decay gate α |
| 写入控制 | ❌ 全部硬累加 | ✅ write gate β |
| 数值稳定 | ⚠️ 容易爆 | ✅ RoPE only on numerator + FP32 scan + ReLU kernel |

**本质**:GDN 把线性注意力的累加 state 升级成一个**动态关联记忆库**。每帧来一个 (k, v),先查"用现在记忆库查 k 会得到什么 v_pred",只把差额 (v - v_pred) 写进去。这跟 80 年代 Hopfield network、fast weight memory 是同源思路(参见论文引用 [21] "Linear transformers are secretly fast weight programmers")。

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
