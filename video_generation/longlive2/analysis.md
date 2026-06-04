# LongLive 2.0 论文与代码解读

> 论文:LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation (arXiv 2605.18739v2, NVIDIA)
> 基础模型:Wan2.2-TI2V-5B
> 与之对照:CausVid (DF + DMD)、Self-Forcing (Self-rollout + DMD)、LongLive 1.0 (Long-Tuning)

## 一、一句话定位

LongLive 2.0 = **基础架构 (NVFP4 量化 + 序列并行) + 算法精简 (跳过 ODE init/中间 DMD) + 多镜头交互推理 (Multi-Shot Attention Sink)** 三件事打包的长视频 AR 扩散方案。论文的核心叙事是"工程基础设施反向简化算法 pipeline"。

## 一.5、训练规模事实速查(论文 Sec I "Implementation Details")

| 维度 | 值 |
|------|---|
| 基础模型 | **Wan2.2-TI2V-5B**(开源,Alibaba),text encoder + VAE 全程冻结 |
| 是否从头训 | **不是**,基于开源 Wan 微调 |
| 训练数据集 | **自建 120K 长视频**(论文 Sec B),按时长分三档:16-32s / 32-64s / >64s 各占 1/3。原始视频先切镜头、单独 caption,再合并成全片 caption。MANIQA 视觉质量过滤 |
| 精度 | **AR 训练和 DMD 蒸馏全程 NVFP4** —— 论文原话:"first end-to-end NVFP4 recipe for long video generation" |
| AR 训练 GPU 数 | **32 × NVIDIA GB200** |
| AR 训练超参 | SP size 4, FSDP hybrid_full, local batch 1, grad accum 2 → global batch **16**, **600 iterations**, lr=1e-5, AdamW (0.0, 0.999), EMA 0.99 from step 100 |
| AR 训练 GPU 小时 | **1920 GB200 hours**(≈ 32 GPU × 60 小时,即 2.5 天)|
| DMD LoRA GPU 数 | **16 × NVIDIA GB200** |
| DMD LoRA 超参 | local batch 2, grad accum 1 → global batch **32**, **5000 iterations**, gen lr=1e-5, critic lr=2e-6, dfake_gen_update_ratio=5, LoRA rank=128 / alpha=128 / BF16 |
| DMD LoRA GPU 小时 | **60 GB200 hours**(≈ 16 GPU × 3.75 小时) |
| Optimizer state | FP32(NVFP4 只用在 GEMM operands,标准 mixed-precision 做法) |

## 二、相对 CausVid / Self-Forcing 的关键区别

### Pipeline 简化(论文 Fig 4)

| 方法 | 训练阶段顺序 |
|------|-------------|
| **CausVid** | 双向 Diff → ODE Init → AR Initial (Few-step) → DMD → AR Model |
| **Self-Forcing** | 双向 Diff → ODE Init → DMD/SiD/GAN → AR Model (Few-step) |
| **LongLive 1.0** | 双向 Diff → ODE Init → DMD → AR Initial → **Long Tuning** → AR Model |
| **LongLive 2.0** | 双向 Diff → **AR Training (直接长视频微调)** → AR Model |
|              | 旁路:同基础模型 → **DMD (仅 LoRA)** → Few-Step LoRA 权重 |

**关键变化**:
1. **不需要 ODE 初始化**。直接在双向预训练 Wan 上做 AR teacher-forcing 长视频微调
2. **DMD 只训 LoRA**,不动 backbone。少步数能力是"可插拔"的旁路
3. **取消"中间 DMD + 长视频长 tuning"两阶段**,合并成一次 AR 长视频训练
4. 训练数据需求**逆转**:不是 data-free,而是**直接用真长视频数据**,因为高质量 infra 让长视频训练变得可行

### 训练范式

| | CausVid | Self-Forcing | LongLive 2.0 |
|---|---------|--------------|--------------|
| Generator 训练范式 | DF(数据加噪) | Self-rollout(模型自己) | **Teacher-Forcing**(clean history + noisy target) |
| 输入构造 | clean_latent + noise | rollout 累积的 KV cache | `[z_clean ; z_noisy]` 配对流 |
| Attention mask | 块内全连接 + 块间因果 | 同 + 训练时模拟自回归 | **AR Block-Sparse**: noisy chunk attend 自己 + 前面所有 clean chunks |
| 一次 forward 监督多少 chunk | 全部 21 帧 | 7 个 block(只有 1 个开梯度) | **所有 N 个 noisy chunk 同时监督** |
| 是否解决 exposure bias | ❌ | ✅(用 self-rollout) | △(用 teacher-forcing,但 paper 声称足够好且更高效) |

**注意**:LongLive 2.0 论文(Sec 2.1)特意说明:"We use clean-context teacher forcing ... rather than diffusion forcing ... to avoid the train-test gap"。它认为 teacher-forcing(让 noisy target 看 clean history)比 DF(noisy target 看 noisy history)更接近推理时"clean context + 当前 noisy"的状态。

但 Self-Forcing 论文恰好反对 teacher-forcing,认为它的 clean history 来自真实数据 ≠ 推理时来自模型自己。LongLive 2.0 的反驳是:**用更长更高质量数据训练 + error recycling 机制**(见下)来补偿这个 gap。

## 三、训练 Infra:Balanced SP + Teacher-Forcing

### 问题

长视频 (64s)训练时,21 → 96 latent 帧,token 数从 32K 飙到 150K+,单 GPU 装不下。

朴素 Sequence Parallelism 把 `[z_clean ; z_noisy]` 当一段普通序列切片分发,产生两个问题(论文 Fig 3):
1. **负载不均**:某些 rank 全是 clean 帧(没有 loss),某些 rank 全是 noisy 帧(loss 集中)
2. **VAE 编码冗余**:每个 rank 都要把整段 raw video 编码一遍

### Balanced SP 解法 (Sec 2.1)

**每个 rank 持有同一时间段的 (clean, noisy) 配对**:

```
传统 SP:                          Balanced SP:
GPU0: clean z0                    GPU0: clean z0 + noisy z0
GPU1: clean z1                    GPU1: clean z1 + noisy z1
GPU2: clean z2                    GPU2: clean z2 + noisy z2
GPU3: noisy z3   ← 负载集中        GPU3: clean z3 + noisy z3
                                  ↑ 每个 rank loss-bearing tokens 平衡
```

- 每个 rank 只 VAE-encode 自己的本地 chunk + 一小段 halo(覆盖 VAE 时间感受野)
- VAE 复杂度从 O(F) 降到 O(F/P + h)
- Ulysses All-to-All 之后,token 顺序自然变成 interleaved,直接在这个顺序上构造 AR mask,**不用再 permute 回 `[all clean; all noisy]`**
- 用 `flex_attention` 编译这个 SP-native mask

### Teacher-Forcing AR mask

```
N 个 chunks 的 paired layout (clean, noisy 交错):
[clean_0, noisy_0, clean_1, noisy_1, ..., clean_{N-1}, noisy_{N-1}]

attention 规则 (noisy_i 作为 query):
  - noisy_i 可以看 clean_0, ..., clean_{i}   (历史 clean)
  - noisy_i 可以看 noisy_i 自己              (当前 noisy)
  - noisy_i 不能看 clean_{>i} 或其他 noisy_{≠i}
```

→ **一次 forward 同时监督所有 N 个 chunk 的去噪**,这是论文标榜的训练效率核心。

### Balanced SP 是否需要改 attention?

**需要改 mask predicate,但不改 attention 内核**。论文 Sec C "Natural teacher-forcing mask" 是这块最核心的细节。

**问题**:Ulysses All-to-All 之后,全局 token 顺序变成"按 rank 拼接":

$$
[z^{(0)}_{\text{clean}}, z^{(0)}_{\text{noisy}}, z^{(1)}_{\text{clean}}, z^{(1)}_{\text{noisy}}, \dots, z^{(P-1)}_{\text{clean}}, z^{(P-1)}_{\text{noisy}}]
$$

朴素 teacher-forcing mask 是在 logical 顺序 `[all clean ; all noisy]` 上定义的,如果直接套到 interleaved 顺序就完全错位。

**两种解法**:
1. **朴素法**:每层 attention 前 permute 回 logical 顺序、计算、再 scatter 回 SP 顺序 → 每层都加 2 次 N log N 通信,极慢
2. **LongLive 2.0 法**:**在 interleaved 顺序上直接构造 mask predicate**

具体推导(论文 Eq 9-10):对 interleaved 序列里的索引 i,

$$
p(i) = \lfloor i / 2L_{\text{loc}} \rfloor \quad (\text{在哪个 rank 段}) \\
r(i) = i \bmod 2L_{\text{loc}} \quad (\text{rank 段内偏移}) \\
t(i) = p(i) \cdot L_{\text{loc}} + (r(i) \bmod L_{\text{loc}}) \quad (\text{真实时间位置})
$$

判断 i 是 clean 还是 noisy:`r(i) < L_loc` 是 clean,`r(i) ≥ L_loc` 是 noisy。

然后定义:
$$
M_{\text{nat}}(i, j) = M_{\text{TF}}(\pi(i), \pi(j))
$$

其中 $\pi$ 把 interleaved 索引映射回 logical (clean/noisy, 时间位置)。

**关键**:$\pi$ 从来不 materialize 在 Q/K/V 张量上,只是在 mask predicate 里用整数运算算一下。`flex_attention` 把这个 predicate 编译进融合 kernel,执行起来跟 dense attention 一样快。

→ **改的是 mask 函数,不改 attention 实现**。一行 flex_attention call,mask 是 Python 函数,JIT 编译。

### Balanced SP 的另外两块拼图

**1. SP-aware VAE 编码**(论文 Sec C):
- 朴素 SP:每个 rank 都 encode 整段 raw video,O(F) 重复 P 次
- Balanced SP:每个 rank 只 encode 自己负责时间段 X^(p) + **左 halo**(覆盖 VAE 时间感受野)
- Encode 完丢掉 halo 输出,保留中间精确部分
- 每 rank VAE 复杂度从 O(F) 降到 O(F/P + h)
- config 里 `vae_halo_latents: 28`,即 halo 28 latent 帧

**2. SP-aware Error Recycling**(论文 Sec C "SP-aware error recycling"):
Error recycling buffer 用 (local block position, diffusion timestep) 二维 bucket 索引。在 SP 下:
- Position 维度跟 DiT sequence 一样分片(rank p 只存自己负责的位置 bucket)
- Buffer entries 在 warmup 阶段跨 DP rank 聚合(同 SP rank 的 DP 之间共享),不跨 SP rank
- 不跨 SP 是因为别的 rank 错误对应的"位置"在当前 rank 不可见,replay 会 invalid

## 四、NVFP4 量化(论文 Sec 2.2, 3.1, 3.2)

### NVFP4 是什么

- **E2M1 4-bit 浮点**(指数 2 位、尾数 1 位)
- 分层 scaling:
  - 16 元素块共享一个 FP8 (E4M3) block-scale `α_FP8`
  - 整 tensor 共享一个 FP32 global-scale `α_FP32`
- 公式:`X̂ = X̂_FP4 · α_FP8 · α_FP32`
- 跟 INT4 比,**非均匀步长**:小值更细、大值更粗,更适合神经网络激活分布
- Blackwell GPU 原生支持

### 训练 NVFP4 配方 (Sec 2.2)

- 线性层 W 和 A 都量化到 FP4
- 权重 2D block-scale,激活/梯度 1D block-scale
- 数值敏感操作(reduction、norm 统计、optimizer state)留 FP32
- 最敏感的 weight gradient GEMM 用 **RHT (Random Hadamard Transform)** 预处理稳定数值

→ 64s 训练上 ~1.8× 加速

### DMD NVFP4 配方 (Fig 5, Sec 2.2 末)

- Generator + Real Score + Fake Score **三个模型全部 W4A4 NVFP4**
- Generator/Fake Score 训 LoRA(冻 backbone),Real Score 全冻
- 用 **adaptive scale search** 量化(在 magnitude 6 和 4 之间选低误差的)
- 公式 (4):`W ≃ Dequant(Q_search(W₀)) + (α_LoRA / r) B A`

DMD 训练显存比 BF16 降到 **0.69×**(Table 2)。

### 推理 NVFP4 (Sec 3.1)

- 用训练过的 NVFP4 backbone + LoRA 直接 W4A4 推理(比 PTQ 质量更稳)
- 可以把 LoRA fuse 进 backbone 形成 merged W4A4 模型
- 丢掉 BF16 master weights,常驻显存进一步降

## 五、Parallel KV Quantization (Sec 3.2)

### 痛点

AR 长视频生成时,每生成一个 chunk 就把这个 chunk 的 K/V 写进 cache。视频越长 cache 越大:
- 64s 视频 → 96 latent 帧 → ~150K tokens
- Wan2.2-TI2V-5B 有 30 层,每层 12 heads × 128 head_dim
- BF16 下 KV cache 显存:`2 × 150K × 12 × 128 × 2 bytes × 30 layers ≈ 27 GB`(单 batch)

显存被 KV cache 吃掉一大半。

### 量化方案的"按 chunk"是核心

不是 token-wise 量化,而是 **chunk-wise 量化**。意思是:每生成完一个 chunk(=8 latent 帧),把这个 chunk 整体 reshape 成一个矩阵,然后做 NVFP4 量化。

```python
# 单层、单 chunk 的 K/V 数据
F_c = 8                              # 一个 chunk 8 帧
L_f = 1560                           # Wan 每帧 1560 tokens (= 30×52 patch grid)
T_c = F_c * L_f = 12480              # 一个 chunk 的 token 数

K_chunk ∈ R^{T_c × H × d}            # H=12 heads, d=128 head_dim

# Reshape 成 2D 矩阵以便 micro-block 量化
K_2d = K_chunk.reshape(T_c * H, d)   # ∈ R^{(12480×12) × 128} = R^{149760 × 128}

# NVFP4 micro-block scaling (16-element blocks)
K_quantized = nvfp4_quantize(K_2d)
```

### Key 的 K-smoothing 预处理

直接量化 K 误差大,论文用 **K-smoothing**:每个 token 的 K 向量先减去 head 内 channel 均值,再量化。

$$
\bar{K}_{\ell,c}[t, h, :] = K_{\ell,c}[t, h, :] - \frac{1}{d}\sum_{u=1}^{d} K_{\ell,c}[t, h, u]
$$

为什么对 K 做这个,V 不做?
- K 通常含 systematic bias(各 head 学到的 "general direction"),减均值后分布更对称
- 对称分布对 FP4 友好(FP4 范围 [-6, 6] 是对称的)
- V 通常已经比较中心化,不需要

### 内存收益的算式

```
BF16 (基线):
  storage = 4 · T_c · H · d bytes        ← K, V 各 2 bytes × T_c·H·d 元素 = 4 T_c H d
  = 4 × 12480 × 12 × 128 = 76.7 MB / layer / chunk

NVFP4 chunk-quantized:
  - 元素本身: T_c · H · d × 0.5 bytes (4-bit FP4) × 2 (K+V) = T_c · H · d bytes
  - block scale: 每 16 元素一个 FP8 scale = T_c · H · d / 16 bytes × 2 (K+V)
  - global scale: 1 个 FP32 / tensor (忽略)
  - K-smoothing 的 mean: T_c · H × FP16 = 2 T_c H bytes (忽略)
  
  storage ≈ T_c · H · d + T_c · H · d / 8 = 9/8 · T_c · H · d bytes
  ≈ 17.6 MB / layer / chunk
```

**压缩比 ≈ 4 / 1.125 = 3.56×**(论文报 3.6×)。

### 真实存储布局(看代码 `causal_diffusion_inference.py:803-815`)

```python
# 每层一个 dict
kv_cache_pos.append({
    "k": [clone_quantized_tensor(zero_qt) for _ in range(max_blocks)],
                # ↑ list of 量化后的 chunk K,每个是一个 NVFP4 QuantizedTensor
    "v": [...],                       # 同 K
    "quantized": True,
    "block_token_size": T_c,
    "max_blocks": 例如 5,             # sliding window 装 5 个 chunk
    "num_heads": 12,
    "num_filled_blocks": 0,           # 当前装了几个 chunk
    "global_end_index": [0],          # 全局 token 末位
    "local_end_index": [0],           # cache 内 token 末位
    "pinned_start": [-1],             # multi-shot sink 用
    "pinned_len": [0],
})
```

**关键观察**:
- K/V 是 **list of 量化 chunks**,不是一个大 tensor。chunk 是量化的最小单位
- 当 cache 满需要驱逐时,**整 chunk 驱逐**,不需要重新量化任何东西
- 当 attention 需要访问时,**按需 dequantize**:一次 attention 可能访问多个 chunks(因为有 sink + sliding window)

### 为什么叫 "Parallel"

每次 attention 都要把窗口内的多个 chunks 反量化拼起来才能算 Q·K^T。如果串行 dequant,这会成为瓶颈。LongLive 2.0 写了 **parallel dequantization CUDA kernel**,把多个 chunks 同时反量化:

```
chunk 0 ──┐
chunk 1 ──┼── parallel dequant kernel ── BF16 K, V ──> attention
chunk 2 ──┤
sink chunk ──┘
```

把量化 / 反量化总开销压到 **< 2%**(论文 Sec 3.2 末)。

### Inference 显存收益(Table 3)

| 设置 | 64s 视频 Peak Memory |
|------|---------------------|
| BF16 | 112.9 GB |
| NVFP4(权重) | 96.0 GB |
| **+ NVFP4 KV Cache** | **57.6 GB ~ 19.4 GB**(差异是是否 KV 量化决定的) |
| + Async Decoding | 19.4 GB |

KV 量化是从"权重量化后"到"19.4 GB 可在单卡跑 64s"的关键一步。

## 六、Multi-Shot Attention Sink(论文 Sec 4.2, Fig 7)

LongLive 2.0 最有意思的算法 trick。

### 问题

长视频里**多镜头切换**很常见(场景 1 → 场景 2 → 场景 3)。直接套 Self-Forcing 的 sink 有问题:
- 单一 global sink:跨镜头切换时,sink 还指着场景 1 的画面 → 锚定错乱
- 滑窗只有 shot-level sink:镜头切了就丢全局身份(角色不一致)

### 解法:双层 sink

**全局 sink** `A_g`:整段视频的前 `S_g` 帧,**永远不变**。维持整段全局身份(角色、风格、场景大基调)。

**镜头 sink** `A_s`:当前镜头的前 `S_s` 帧,**每次场景切换重绑**。维持当前镜头内的时序连贯。

每一步 attention 的有效 KV 集:
$$
K_{\text{eff}}(t) = A_g \cup A_s \cup KV_{[t-W, t)}
$$
其中 `W` 是滑窗长度,重复 token 去重。

### 关键工程细节

- `A_s` 的内存开销是 **0**:只用 `(START, LEN)` 两个指针追踪。只在滑窗刚刚滚过 `A_s` 时才把它"虚拟插入"窗口前
- 跟 chunk-wise prompting 天然契合:prompt 切换 `p_k → p'_k` = 场景切换 = `A_s` 重绑 + cross-attention cache 重置
- 全局 sink 和之前的历史完全不动

### 场景切换怎么判断:**纯 prompt-based,不是检测**

这是看代码才能搞清楚的细节(论文里写得隐晦)。`pipeline/causal_diffusion_inference.py:977-990`:

```python
def _is_shot_boundary(self, raw_prompts, chunk_index):
    if chunk_index == 0:
        return False
    if chunk_index >= len(raw_prompts):
        return False
    prompt = raw_prompts[chunk_index]
    return isinstance(prompt, str) and prompt.startswith(self.scene_cut_prefix)
```

`utils/dataset.py:22`:
```python
DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "
```

**意思**:用户输入的 prompt 序列里,新场景的第一个 chunk 的 prompt **必须以 `"The scene transitions. "` 开头**,模型才能识别为场景切换。例如:

```python
prompts = [
    "A man walks slowly in a sunny park.",                              # shot 1, chunk 0
    "A man walks slowly in a sunny park.",                              # shot 1, chunk 1
    "The scene transitions. A woman drinks coffee in a cafe.",          # shot 2 starts ← 场景切换
    "A woman drinks coffee in a cafe.",                                 # shot 2, chunk 1
    "The scene transitions. A street vendor sells flowers at night.",   # shot 3 starts ← 场景切换
    ...
]
```

→ **场景切换是用户显式标注的,不是模型自动检测**。这跟训练数据集生成方式一致:论文 Sec B 说每个长视频先切 shot、各自 caption,再合并 — 自然地保留了 shot 边界。训练时数据集里就带这个标记前缀,模型学会"看到 transitions 前缀 = 新场景"。

### RoPE 需要改吗:**需要,加跨镜头相位偏移**

如果不动 RoPE,两个不同镜头的帧位置编码是连续的(例如 shot 1 的最后帧 t=14,shot 2 第一帧 t=15)。模型会以为 shot 2 是 shot 1 的"下一帧",硬要保持视觉连续性,产生"奇怪的过渡帧"。

LongLive 2.0 引入 **`multi_shot_rope_offset` φ**(config 里默认 8)。每检测到一次场景切换,RoPE 的 temporal 偏移 += φ:

`pipeline/causal_diffusion_inference.py:442-552`:
```python
current_shot_index = 0
phi = self.multi_shot_rope_offset    # = 8
self._dit_model.rope_temporal_offset = 0.0

for chunk_index in range(...):
    is_shot_boundary = self._is_shot_boundary(raw_prompts, chunk_index)
    if is_shot_boundary and phi != 0.0:
        current_shot_index += 1
        self._dit_model.rope_temporal_offset = current_shot_index * phi
        # 后续这个镜头的所有帧,RoPE 计算时都加上这个偏移
```

例子:
```
shot 1, frames 0~7:    RoPE position = 0, 1, 2, ..., 7      (offset=0)
shot 2 开始 (offset += 8):
shot 2, frames 8~15:   RoPE position = 16, 17, 18, ..., 23  (实际 8~15 + offset=8)
                                       ↑ 跳过 8 个位置
shot 3 开始 (offset += 8):
shot 3, frames 16~23:  RoPE position = 32, 33, 34, ..., 39
```

→ 这就告诉模型"shot 之间的 RoPE 位置是非连续的",学到了"看到位置跳变就是场景切换",自然产生镜头切换效果。

### 代码位置速查

`pipeline/causal_diffusion_inference.py`:
- `multi_shot_sink` config 读取: line 104
- `global_sink_size` 派生: line 106
- `multi_shot_rope_offset`: line 107-110
- `_is_shot_boundary` (prompt-based): line 977-990
- `_is_scene_cut` (sink-related): line 992-999
- `_update_sink_for_scene_cut` (legacy copy-to-front): line 1001-1018
- `_pin_current_chunk` (零拷贝镜头 sink): line 1020-1036
- RoPE offset 更新循环: line 442-554
- `_set_all_modules_global_sink_size`: line 968

### 配置项

```yaml
# train_ar.yaml
inference:
  sink_size: 0                  # shot-level sink 大小(0=禁用,实验用)
  multi_shot_sink: true         # 启用 multi-shot 模式
  multi_shot_rope_offset: 8     # 跨镜头 RoPE 偏移 φ
```

## 七、Asynchronous Streaming Decoding(Sec 3.3)

### 问题

VAE 解码是端到端 FPS 的隐形杀手。LongLive 1.0 等基线:**把所有 latent chunks 攒齐 → 一次性顺序 VAE decode**,显存 O(C · T_c),延迟随 chunk 数线性。

### 解法

- 重新工程 3D VAE,支持 **chunk-by-chunk streaming decode**,decode 完一块立刻 CPU offload
- VAE 显存降到 O(T_c)(只看当前 chunk)
- **专门拨一张 GPU 跑 VAE**,异步并行:DiT 集群正在去噪 chunk c+1 时,VAE GPU 正在解码 chunk c

### 端到端延迟变化

- 朴素:`C · (t_DiT + t_VAE)`(串行)
- 异步:`C · t_DiT + t_VAE`(VAE 隐藏在 DiT 后面)

只要 `t_DiT ≥ t_VAE` 就能完全隐藏,实际通常满足。

## 八、训练时的 Error Recycling 机制

config `train_ar.yaml` 里有一个非常重要但论文里没花太多篇幅的机制:

```yaml
error_recycling:
  enabled: true
  num_buckets: 50
  buffer_size_per_bucket: 500
  buffer_warmup_iter: 50
  context_inject_prob: 0.9
  latent_inject_prob: 0.9
  noise_inject_prob: 0.01
  clean_prob: 0.2
  clean_buffer_update_prob: 0.1
  modulate_factor: 0.3
  start_step: 0
  replacement_strategy: random
```

这是为了**补偿 teacher-forcing 的 exposure bias**。流程大致是:
1. 训练时把模型自己的预测错误(误差)按 noise level 桶化存进 buffer
2. 下一次训练时以 90% 概率把这些"模型自己的污染"注入到 clean context 里
3. 等价于把"clean history"部分替换成"带模型自身偏差的 history"
4. 让 teacher-forcing 的 clean context 分布逐步接近推理时的实际分布

→ 这是论文 teacher-forcing vs Self-Forcing 之争的"和解方案":**保留 teacher-forcing 的并行训练效率,用 error recycling 缩小 exposure gap**。

## 九、关键配置项速查表

### `configs/train_ar.yaml`(AR 长视频微调)

| 字段 | 值 | 含义 |
|------|---|------|
| `infra.sequence_parallel_size` | 4 | Balanced SP 的 GPU 数 |
| `infra.vae_halo_latents` | 28 | VAE 时间感受野的 halo 大小 |
| `model_kwargs.num_frame_per_block` | 8 | 每 chunk 8 latent 帧 |
| `model_kwargs.local_attn_size` | -1 | 训练时不限制 attention 窗口(全局看到所有 clean history) |
| `algorithm.causal` | true | 因果 attention |
| `algorithm.teacher_forcing` | true | **关键: teacher-forcing 不是 self-rollout** |
| `data.image_or_video_shape` | [1, 96, 48, 44, 80] | 96 latent 帧(对应几十秒视频) |
| `data.load_raw_video` | true | 用真实视频数据(非 data-free) |
| `inference.multi_shot_sink` | true | 启用多镜头 sink |
| `error_recycling.enabled` | true | 补偿 exposure bias |

### `configs/train_dmd.yaml`(DMD LoRA 蒸馏)

| 字段 | 值 | 含义 |
|------|---|------|
| `model_kwargs.local_attn_size` | 32 | 推理时窗口 32 latent 帧(滑窗) |
| `algorithm.all_causal` | true | 三个模型都因果 |
| `algorithm.real_guidance_scale` | 3.0 | CFG 强度 |
| `training.dfake_gen_update_ratio` | 5 | critic 每 5 步训一次 generator |
| `training.num_training_frames` | 32 | DMD 阶段只在 32 帧上训(相当于 short tuning) |
| `inference.sampling_steps` | 4 | 4 步少步推理 |
| `adapter.type` | lora | 只训 LoRA |
| `adapter.rank` | 128 | LoRA 秩 |
| `adapter.apply_to_critic` | true | critic 也只训 LoRA |

## 十、代码导览

| 功能 | 文件 |
|------|------|
| AR teacher-forcing 训练入口 | `train.py` + `trainer/diffusion.py` |
| DMD LoRA 蒸馏入口 | `train.py` + `trainer/distillation.py` |
| Balanced SP helper | `trainer/sp_helper.py` |
| 因果模型主体 | `model/base.py`, `model/diffusion.py`, `model/dmd.py` |
| Self-Forcing rollout pipeline | `pipeline/self_forcing_training.py` (DMD 阶段用) |
| 推理(含 multi-shot sink, KV 量化) | `pipeline/causal_diffusion_inference.py` |
| SP 推理 | `pipeline/causal_diffusion_inference_sp.py` |
| NVFP4 quant 工具 | `utils/` 下的相关 module |
| Wan 5B backbone | `wan_5b/` |
| FP4 kernel | `fouroversix/` |

## 十一、与前作方法关系图

```
                  Wan2.x bidirectional diffusion (base)
                          │
        ┌─────────────────┼──────────────────────────┐
        ▼                 ▼                          ▼
    CausVid           Self-Forcing                LongLive 2.0
    路径:              路径:                       路径:
    1. ODE init        1. ODE init                 1. AR Teacher-Forcing 长视频微调
    2. DMD             2. DMD/SiD/GAN              2. (旁路) DMD-LoRA 蒸馏
    3. AR              3. AR + Rolling KV          3. Multi-Shot Sink 推理
    
    支持:              支持:                       支持:
    - 因果 AR          - 因果 AR                    - 因果 AR
    - 5s 视频          - 长视频外推                  - 长视频原生
                      - data-free 训练              - 多镜头交互
                                                    - 实时 (NVFP4)
                                                    - 4→2 步可切
```

## 十二、争议点与权衡

### LongLive 2.0 vs Self-Forcing 的训推一致性之争

| 维度 | Self-Forcing 立场 | LongLive 2.0 立场 |
|------|------------------|-------------------|
| Teacher-Forcing 有 exposure bias? | 有,且无法接受 | 有,但可以用 error recycling 补偿 |
| Self-rollout 训练贵吗? | 用 stochastic exit + KV cache grad cut 可控 | 仍然太贵,放弃 |
| Data-free 重要吗? | 是核心卖点(只要 prompt) | 不重要,我们有高质量长视频数据 |
| 训练吞吐量谁更高? | 中等 | **高**(Balanced SP + NVFP4) |
| 长视频能力 | Rolling KV cache 外推 | 原生长视频训练 + Multi-Shot Sink |

两者对"训推一致性"的态度有真实的方法论分歧。LongLive 2.0 实质上是**用基础设施换数据效率**:既然可以用真长视频高效训练,就不用搞 self-rollout 的复杂梯度截断游戏。

### 不是 data-free 的代价

- 需要长视频数据集(论文用了多镜头真实视频)
- 数据质量直接影响最终质量
- 对中小团队不太友好(SF 只要 prompt)

但 NVFP4 + Balanced SP 让训练成本降下来很多,工业场景能 cover。

## 十三、一句话总结

> **LongLive 2.0 用 NVFP4 量化 + Balanced 序列并行 把长视频 teacher-forcing AR 训练做得足够便宜,跳过了 CausVid/Self-Forcing 必须的 "ODE init + 中间 DMD + 长 tuning" 多阶段流程,直接在双向 Wan 上一次性微调成"长 + 多镜头 + AR" 模型。少步数能力通过纯 LoRA 旁路加上;推理用 Multi-Shot Attention Sink (全局 sink + 每镜头 sink) 保证跨镜头视觉一致性,配合异步流式 VAE decode 实现 45.7 FPS 端到端实时长视频生成。**

附:与 Self-Forcing 的根本分歧是"用 self-rollout 解 exposure bias"vs"用真长视频数据 + error recycling 解 exposure bias"——前者算法巧妙,后者工程实在。
