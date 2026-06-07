# Cosmos 3 论文逐节解读

> 论文:Cosmos 3: Omnimodal World Models for Physical AI (arXiv 2606.02800v1, NVIDIA, 2026-06)
> 项目主页:research.nvidia.com/labs/cosmos-lab/cosmos3
> 代码:github.com/nvidia/cosmos + github.com/nvidia/cosmos-framework
> 模型权重:huggingface.co/collections/nvidia/cosmos3
> 任务定位:**Physical AI 通用 backbone**(同时做理解 + 生成,跨语言/图/视/音/动作五种模态)

## 一、定位与背景

### 一句话定位

Cosmos 3 = **NVIDIA 的"机器人/具身智能基础模型"**。把 VLM(视觉语言模型)+ 视频生成 + 世界模拟器 + 动作策略合并成**一个 Mixture-of-Transformers 架构**,五种模态(语言/图像/视频/音频/动作)共用 backbone,根据任务自动切换 AR 模式或 Diffusion 模式。

### 为什么这是个大新闻

之前 Physical AI 系统都是**多个独立模型拼凑**:
- VLM:看懂场景、做高层规划
- Video Generator / Forward Dynamics Model:模拟未来
- VLA / World-Action Model:输出控制信号

Cosmos 3 的 claim:**一个网络全部包了**。Physical AI 的"理解"和"生成"两件事本质上耦合 —— 理解需要预测未来,生成依赖结构化世界表示,所以应该 unify。

### 模型规模

| 变体 | LLM 主干 | 总参数 | 状态 |
|------|---------|--------|------|
| **Cosmos3-Edge** | 2B dense Transformer (从头训) | 4B (MoT 双塔) | 未来发布 |
| **Cosmos3-Nano** | Qwen3-VL-8B 初始化 | 16B (MoT 双塔) | ✅ 已开源 |
| **Cosmos3-Super** | Qwen3-VL-32B 初始化 | 64B (MoT 双塔) | ✅ 已开源 |

注意:**所有变体的"总参数 ≈ 2× LLM 参数"**,因为 MoT 双塔每层都有两套独立 weights(reasoner 一套,generator 一套)。

### 论文里的 SOTA 战绩(Table 1)

| 任务 | Cosmos3-Super 分数 | 击败 |
|------|-------------------|------|
| Reasoning General | 73.7 | Qwen3-VL-32B (72.8), Gemma-4-31B (69.8) |
| Reasoning Driving | 79.3 | Gemini 3.1 Pro (47.2) — 大幅领先 |
| Text-to-Image | 91.36 (Cosmos3-Super-Text2Image) | Qwen-Image-2512 (84.25), Gemini 3 Pro Image (90.85) |
| Text-to-Video | 80.0 | Veo-3.1 (79.1) |
| Image-to-Video | 82.8 | Veo-3.1 (82.6) |
| Audio Generation | 7.31 | Veo-3.1 (7.45) |
| Robot Policy | 39.7 (Cosmos3-Nano-Policy-DROID) | π0.5 (28.1) |

→ **Artificial Analysis 评测里 Text-to-Image 和 Image-to-Video 都是开源第一**(写报告时),Robot Policy 在 RoboArena 上也是第一。

---

## 第 1 节:Introduction

### 核心论证链

1. Physical AI agent 需要两个底层能力:**Understanding(理解)** + **Generation(生成/预测未来)**
2. 现有 paradigm 把这两件事拆成多个不同模型 → **fragmented architecture, 计算浪费**
3. 理解和生成本质上耦合 → 应该 unify 成一个网络
4. 引入 Cosmos 3 作为这个 unified backbone

### Cosmos 3 的 6 种"模式"(Fig 1)

按 input/output 配置自动切换:
- **Vision-Language Model (VLM)** — 多模态理解、推理
- **Image Generation Model** — 图像生成
- **Audio-Visual Generation Model** — 视频+音频联合生成
- **Policy / World-Action Model** — 同时输出动作+视频(下一帧预测)
- **Forward Dynamics Model** — 给定动作预测未来视频
- **Inverse Dynamics Model** — 给定视频反推动作

### Cosmos 3 的 3 种使用方式(Fig 2)

1. **Synthetic Data Generation** — 用 Cosmos 3 生成训练数据喂给下游模型
2. **Task-Specific Specialization** — Cosmos 3 当作基础模型 post-train 特化(论文做了 T2I/I2V/Policy 三个案例)
3. **Training Environment** — 长期愿景,Cosmos 3 充当机器人训练环境(可交互的世界模拟器)

### 一句总结

> Physical AI 的所有任务都可以统一成"读多模态输入 + 生成多模态输出",Cosmos 3 是这个统一框架的第一个真正可用的工业级模型。

---

## 第 2 节:Model Architecture

这是论文最技术化的部分,**重点章节**。架构由四块组成:**Encoders → Token Arrangement → MoT Backbone → Multimodal RoPE**。

### 2.1 Encoders(模态编码器)

每种模态用独立的 encoder 编码成 transformer 可吃的 token,**每个非语言模态加一个 learnable modality-specific embedding 标识身份**。

#### 2.1.1 Image / Video

**两套 encoder 并行**:

| 用途 | Encoder | 训练状态 |
|------|---------|---------|
| **理解(放进 AR 流)** | ViT(预训练,vision-language 对齐过)+ 16×16 patch,2 层 MLP + 2×2 token merge | 跟 backbone 联合训练 |
| **生成(放进 DM 流)** | **Wan2.2-TI2V-5B VAE encoder**(直接借用),4× temporal + 32×32 spatial 压缩(= 16×16 spatial + 2×2 patch merge) | **frozen** |

理解 + 生成用两套不同 encoder 这点很关键 —— 之前的统一模型很多硬要用一套 encoder,会顾此失彼。Cosmos 3 的选择是**理解侧用 vision-language 对齐过的 ViT(语义强),生成侧用 reconstruction-VAE(像素质量好)**。

DeepStack(Meng et al. 2024)聚合多层 ViT 特征。视频帧之间插入 text-based timestamp(Chen et al. 2024b)。

#### 2.1.2 Audio

- **Audio VAE**(Lee et al. 2025b 架构)
- 48 kHz stereo,hop size 1920 samples → **25 token / 秒音频**
- VAE **frozen**

#### 2.1.3 Action(论文里最值得专门读的子节)

**核心问题**:不同身体(autonomous vehicle / camera / robot arm / humanoid)的 native 控制空间完全不同(关节轨迹 vs 转向指令 vs body pose vs camera transform)。怎么统一?

**Cosmos 3 的设计:用共享几何成分构造紧凑动作向量**。每个动作最多包含三部分:

| 成分 | 含义 | 维度 |
|------|------|------|
| **Ego Pose** | agent 主观察坐标系的运动 | 9D = 3D translation + 6D rotation |
| **Effector Pose** | 末端执行器(夹爪/手腕)的运动 | 9D = 3D translation + 6D rotation |
| **Grasp State** | 当前抓握状态(直接编码,不是 delta) | 1D(夹爪 open/close)或 15D(5 指 × 3D 指尖位置) |

**几个设计要点**:
1. **Pose 用相对变换(delta)**:`ΔT_t = T_{t-1}^{-1} · T_t`,避免依赖具体 PID 参数/低级 actuator 接口
2. **6D 旋转**(过参数化,自由度 3,但表达 SO(3) 时连续)—— 比四元数/欧拉角更易学,Zhou et al. 2019 提的
3. **Grasp state 直接编码当前状态**(不做 delta)—— 因为抓握/松开是离散事件,delta 没意义
4. **OpenCV 约定**:z 轴沿夹爪指向,x 轴向右

**不同 embodiment 的拼装**:

```
Autonomous Vehicle: [ego 9D] + [no grasp 1D] = 10D
Camera Motion:      [ego 9D] + [effector 9D] + [no grasp 1D] × 2 = 20D
Egocentric Human:   [ego 9D] + [effector 9D × 2 wrists] + [grasp 15D × 2 hands] = 57D 总,精简到 29D
Single-Arm Robot:   [effector 9D + grasp 1D] = 10D
Dual-Arm Robot:     2 × [effector 9D + grasp 1D] = 20D
Humanoid Robot:     [ego 9D] + 2 × [effector 9D + grasp 15D] = 57D
```

**Action tokenization**(Eq 1-2):用**domain-aware** 投影矩阵,即"每个 embodiment 一对 (W_in, W_out)",但所有 embodiment **共享 MoT backbone**。

```
z = W_in^(k) · x + b_in^(k)   # 输入投影 (domain k 专属)
x = W_out^(k) · z + b_out^(k) # 输出投影 (domain k 专属)
```

6D rotation 解码用 SVD 还原成 3×3 SO(3) 矩阵。

→ 这种"backbone 共享 + I/O 投影 domain-specific"的设计是 VLA 圈子的成熟做法(参见 Zheng et al. 2026)。

### 2.2 Token Arrangement and Generation Mode

#### 2.2.1 Token Arrangement(关键设计)

任意任务的 token 序列都分两部分:

```
[ ─── AR subsequence ─── | ─── DM (Diffusion) subsequence ─── ]
   语言 + ViT 视觉 token       VAE 视觉 + 音频 + 动作 token
   走 reasoner tower            走 generator tower
   causal self-attention        full bidirectional attention
   下一 token 预测               flow matching (denoising)
```

**子序列内部规则**:
- AR 在前,DM 在后
- DM 内:每个模态先放 clean conditioning token,后放 noisy diffusion target
- 模态顺序:vision → audio → action

#### 2.2.2 Generation Mode(6 种模式)

只需改 token 序列布局,**架构和参数完全不变**:

| 模式 | 序列结构 |
|------|----------|
| **Language(纯 VLM)** | 只用 AR 子序列,不激活 DM 参数 |
| **Text-to-Image (T2I)** | `[SAR, ṽ1]`,ṽ1 是 noisy image |
| **Text-to-Video (+Audio)** | `[SAR, ṽ1:N, s̃]` |
| **Image/Video-to-Video (+Audio)** | `[SAR, v1:P, ṽ_{P+1:N}]`,前 P 个 clean 当条件 |
| **Video Transfer** | `[SAR, v_ctrl, ṽ]`,control video(edge/depth)当条件,生成 RGB |
| **Action**(3 种子模式) | Forward Dyn / Inverse Dyn / Policy(看 Fig 4)|

**Action 3 种模式**(Fig 4):
- **Forward Dynamics**:clean action → noisy video(给动作预测未来视频)
- **Inverse Dynamics**:clean video → noisy action(给视频反推动作)
- **Policy**:同时 denoise video + action(联合预测)

`SAR = [l1,...,ln, <EOS>, <BOG>]` —— `<BOG>` 是 "begin-of-generation",从 AR 切到 DM 的边界标记。

### 2.3 Mixture-of-Transformers (MoT) Architecture

#### 2.3.1 Dual-Tower Layer Structure(论文 Fig 5)

每个 transformer decoder layer 有 **两套独立参数**:

```
┌─────────────────────────────────────┐
│  Reasoner Tower(处理 AR 子序列)    │
│    - LayerNorm (独立参数)            │
│    - Q/K/V/O proj (独立参数)         │
│    - MLP (独立参数)                  │
├─────────────────────────────────────┤
│   Shared Multimodal Attention       │ ← 唯一共享的算子
├─────────────────────────────────────┤
│  Generator Tower(处理 DM 子序列)    │
│    - LayerNorm (独立参数)            │
│    - Q/K/V/O proj (独立参数)         │
│    - MLP (独立参数)                  │
└─────────────────────────────────────┘
```

**两塔都从同一个预训练 VLM 初始化**(Qwen3-VL-8B / 32B),所以一开始 reasoner 就有强语言+视觉能力,generator 在此基础上慢慢学生成。

#### 2.3.2 Dual-Stream Joint Attention(关键设计)

虽然两塔参数独立,但 attention 是**共享**的,通过下面规则实现跨流互动:

```
AR query 只看 AR key:                    causal triangular mask
                                          O_AR = Attn_causal(Q_AR, K_AR, V_AR)

DM query 看 AR + DM key:                 full bidirectional
                                          O_DM = Attn_full(Q_DM, [K_AR; K_DM], [V_AR; V_DM])
```

→ **AR 不被 DM 污染**(语言生成保持纯净),**DM 可以看到 AR**(文本 prompt + 历史视觉记忆给生成提供上下文)。

Attention mask 长这样:

```
            K_AR        K_DM
       ┌──────────┬──────────┐
Q_AR   │ 三角因果  │ 全 mask  │
       ├──────────┼──────────┤
Q_DM   │ 全看到   │ 全看到   │
       └──────────┴──────────┘
```

### 2.4 Multimodal Position Embedding(关键设计 #2)

借鉴 Qwen3-VL 的 **3D MRoPE**(Multimodal RoPE),每个 attention head 的 hidden dim 切成 (t, h, w) 三部分,各自做 RoPE。Cosmos 3 在此基础上做了**绝对时间索引**改造,关键 motivation:**视频/音频/动作可能同时生成但各自 sampling rate 不同**,必须对齐到物理时间轴。

#### 2.4.1 Position Index Allocation(Fig 6)

| Token 类型 | (t, h, w) 分配 |
|-----------|----------------|
| Language | t = h = w = monotone counter(退化成 1D RoPE) |
| ViT 视觉(AR) | 同一帧的 token 共享 t,h/w 按空间位置 |
| Diffusion 视频 | t 跟时间帧索引,h/w 跟空间网格;**每段 vision 从 0 起算**(绝对 within-video 坐标) |
| Diffusion 图像 | 当作单帧视频,只变 h/w |
| Audio token | 只有 t(h = w = 0),t 跟 audio hop 步进 |
| Action token | 只有 t,t 跟 sampling step 步进 |

**关键陷阱:AR/DM margin**:论文发现"直接让 DM 第一帧紧跟 AR 末位 → 大模型 (Super) 出现 over-saturation + checkerboard artifacts"。原因猜测:最后一个 language token 跟第一个 vision token 的 RoPE temporal embedding 几乎一样,模型迷糊。

**解法**:**在 AR 和 DM 之间插一个固定 15000 的 temporal gap**(纯加 buffer,不改架构)。这种 trick 在 Cao et al. 2025 里见过。

#### 2.4.2 Absolute Temporal Modulation(关键解决多模态时间对齐)

不同模态/不同视频源的 1 单位 temporal step 对应不同物理时间。例如:
- 60 FPS 视频 → 每 latent step = 1/(60/4) = 1/15 秒
- 24 FPS 视频 → 每 latent step = 1/(24/4) = 1/6 秒
- 25 Hz audio → 每 step = 1/25 秒

为了让所有模态对齐到**共享物理时间轴**,定义 TPS(Temporal-steps Per Second):

```
TPS_video = FPS / VAE_temporal_compression = FPS / 4
TPS_audio = 48000 / 1920 ≈ 25
TPS_action = sampling frequency
```

然后选 base TPS:`TPS_base = 24/4 = 6`(24 FPS 视频对应)。

每个 token 的实际 temporal increment 用公式(Eq 9):
```
δ_t = TPS_base / TPS
```

→ 例如 24 FPS 视频 δ=1(基线),16 FPS 视频 δ=1.5(走快 50%),30 FPS 视频 δ=0.8(走慢 20%)。**Fig 6 右**画了这个 modulation。

最终效果:同样真实时长(比如 1 秒)无论用 16/24/30 FPS 编码,**RoPE 上占用相同的 position 范围**。

### 2.5 Model Variants(Table 2)

| Variant | LLM Layers | Hidden | Attn Heads | KV Heads | Head Dim | FFN Dim |
|---------|-----------|--------|-----------|----------|----------|---------|
| Edge | 28 | 2,048 | 16 | 8 | 128 | 9,216 |
| Nano | 36 | 4,096 | 32 | 8 | 128 | 12,288 |
| Super | 64 | 5,120 | 64 | 8 | 128 | 25,600 |

- Edge 从头训(基于 Qwen3-1.7B 架构,去掉 QK normalization,FFN 用 ReLU-squared),Megatron codebase
- Nano = Qwen3-VL-8B 初始化
- Super = Qwen3-VL-32B 初始化
- 所有 variant **每层有两套参数**(reasoner + generator),所以总参数 = ~2× LLM 参数

---

## 第 3 节:Data

### 3.1 Reasoner Data(理解能力数据)

| 阶段 | 样本数 | 用途 |
|------|--------|------|
| Pre-Training | **22.0M** samples | 通用多模态理解 |
| Supervised Fine-Tuning | **2.2M** samples | Physical AI 特化 |

#### 3.1.1 Pre-Training(22M)

**两阶段数据 curation pipeline**:
1. **Semantic deduplication**:用 Qwen3-VL-Embedding-8B(text+image)和 Perception Encoder PE-Core-G14-448(video)给样本编码,K-means 聚类后用 cosine 相似度 > 0.95 去重。删掉 4.23% 数据
2. **AI-judge quality filtering**:用 Gemma-4-31B-it 给每条样本三维度打 1-5 分:
   - **Faithfulness**(response 是否有 visual 依据)
   - **Completeness**(是否完整回答)
   - **Correctness**(事实/逻辑正确性)
   - 用 **min-threshold**(三维度都要 ≥ 阈值)而非平均,保留 78% / 46%(threshold = 2 / 5)

**最终 mixture 组成**(Fig 7):OCR 42.9%、2D grounding 16.5%、Visual QA 11.3%、Image reasoning 7.5%、Text QA 6.1%、Image captioning 5.9%、Video QA 4.5%、Text instruction 3.7%、Visual instruction 1.5%。

#### 3.1.2 Supervised Fine-Tuning(2.2M)

**三大 Physical AI 领域**:

1. **General spatial understanding**(2D/3D grounding,real-world spatial QA,simulator-grounded embodied reasoning)
2. **General temporal understanding**(temporal event,physical plausibility,structured spatiotemporal scene upsampling)
3. **AV (自动驾驶)**(Action CoT 1.1M auto-labeled videos、temporal event localization、3D vehicle grounding)
4. **Robotics**(robot action CoT、embodied reasoning、healthcare robotic surgery 398K conversations)
5. **Smart Infrastructure**(warehouse spatial intelligence、dense pedestrian localization、traffic & anomaly reasoning)

### 3.2 Generator Data(生成能力数据)

**三阶段渐进 curriculum**(Fig 8),每阶段引入更多模态:

| 模式 | Pre-training | Mid-training | Post-training |
|------|--------------|--------------|---------------|
| Text-to-Image | 767M | 16M | 8M (T2I post-train) |
| Text/Image/Video-to-Video | 348M | 75M | 20K (I2V post-train) |
| Audio (Video+Audio) | 139M | 19M | — |
| (Image+Action)-to-Video / V2A / I2(A+V) | — | 8M | 58K (Policy post-train DROID) |
| Video Transfer | — | 4M | — |

#### 3.2.1 Image / Video

Pre-training:**767M 图 + 347.7M 视频(从 7.8B 图 + 3B 视频处理来)**。
- 主要分辨率:720p 和 480p 各占 ~26-36%,1080p 占 12-25%
- 处理 pipeline:Raw → 预处理 → embedding 去重 → 分类基础过滤 → annotation → 按分辨率分桶
- 引入大量 **synthetic data**(5 个 SDG 数据集)增强 Physical AI 场景:SDG-PhyxSim、SDG-RobotSim、SDG-DriveSim、SDG-SynHuman、SDG-Warehouse(论文 Appendix C)

#### 3.2.2 Audio

涵盖 music、speech、sound effects 三类。音频要跟视频同步,所以也要对齐 timestamp(基于上面 absolute temporal modulation 机制)。

#### 3.2.3 Action

跨 embodiment 收集:autonomous vehicle、camera motion、egocentric human、robot manipulation 数据。

---

## 第 4 节:Training

整个 Cosmos 3 的训练分两个大阶段:**Reasoner 训完先,然后用 Reasoner 权重初始化 Generator,再训 Generator**。这是个关键设计——把语言/视觉理解能力**transfer 到生成能力**。

### 4.1 Reasoner Training

#### 4.1.1 Pre-Training

- 初始化:Edge 从内部预训 LM 起,Nano = Qwen3-VL-8B,Super = Qwen3-VL-32B
- **不做 alignment-only 阶段**(不像 Qwen 那样先冻 LLM 训 projector),直接所有组件**一起从头训**
- **Loss**:next-token prediction,长度 ≤ 16k(per sample 限制 image=2048 token, video=8192 token)
- **Square-root normalized per-token loss weighting**(平衡短长序列贡献)
- AdamW,LM/projector lr = 5e-5,ViT lr = 5e-6(ViT 小 10×),cosine decay 到 0.1×,10% warmup
- 2 epochs over full mixture

#### 4.1.2 Supervised Fine-Tuning

- **Importance-aware sampling**:每个数据集按 importance × quality × scale 分预算
- **保留 20% pre-training data**(1:4 pretrain:SFT 比例)防止灾难性遗忘
- 800K instruction-following samples 稳定对话能力
- 8200 iter,global batch 512,lr 1e-5(LM/projector),1e-6 (ViT),1000-step warmup
- Adam (β1, β2) = (0.9, 0.95),weight decay 0.1

### 4.2 Generator Training

**Training objective(关键统一公式)**:整个 Cosmos 3 generator 用 **rectified flow matching**(不是 EDM, 不是 DDPM):

```
x_σ = σ · ε + (1 - σ) · x_0      # 直线插值
                                  # x_0 = clean target, ε ~ N(0, I), σ ∈ [0, 1]

v_θ(x_σ, σ, c) → v* = ε - x_0    # 预测常速度
```

Loss = masked MSE(conditioning tokens 不参与 loss)。

**Per-modality time sampling**:每种模态独立采 σ。
- 图像/音频/动作:**logit-normal**
- 视频:**mode sampling**(论文说效果更好)

**Rectified-flow shift reparameterization**:`σ = s · t̄ / (1 + (s-1) · t̄)`,t̄ = 1 - t,**s ≥ 1 偏向高噪声**。s 越大,越偏 noisy。

#### 4.2.1 Pre-Training(Generator)

**核心 trick 1: Multi-resolution training**(Fig 10 + Table 5):

| Tier | FPS | Frames | (w, h) at 16:9 | Eligible source | Noise shift s |
|------|-----|--------|---------------|----------------|---------------|
| 256p | 10-30 | 5-400 | (320, 192) | 全部 | 1 |
| 480p | 10-30 | 5-400 | (832, 480) | native ≥ 480p | 3 |
| 720p | 10-30 | 5-300 | (1280, 720) | native ≥ 720p | 5 |

数据 mixture 比例:`image:video-256:video-480:video-720 = 1:1:2:1`,video 内部 T2V:I2V:V2V = 70:20:10。

**核心 trick 2: Sequence Packing**:**固定 74,000 token / sequence**,把不同分辨率/时长的样本打包进同一窗口(避免 padding,最大化 GPU 利用)。Fig 10 左边画了这个 packing 过程。

**Training modes**(4 种,按 conditional frame 数区分):
- **T2I**:T = 1,纯 image,20% 采样
- **T2V**:T_cond = 0,全 noisy,56% 采样
- **I2V**:T_cond = 1,首帧 clean,16% 采样
- **V2V**:T_cond = 2,前 5 帧(2 个 latent 帧)clean,8% 采样

**FPS modulation**:训练数据 FPS 各异(16-30),用 absolute temporal modulation 对齐。Duration + FPS 也作为 metadata 附在 prompt 上,让模型可控生成指定时长。

**Optimization**:
- **只更新 generator 参数**,reasoner 冻结
- FusedAdamW,lr = 1e-4,(β1, β2) = (0.9, 0.99),weight decay 0.05,grad clip 1.0
- Text dropout 10%(为 CFG 做准备)

**训练规模**(关键数字):
- **Nano**: 31.05T tokens on **1024 × NVIDIA GB200**
- **Super**: 17.86T tokens on **2048 × NVIDIA GB200**

#### 4.2.2 Mid-Training(关键阶段,引入 action 和 transfer)

承上启下,把通用生成能力 + Physical AI 域知识 + 新模态(action / control)整合。

**Data mixture(Table 6)**:
| 流 | 模式 | 比例 |
|----|------|------|
| Image | T2I | 10% |
| Video | T2V, I2V, V2V | 32% |
| Video + Audio | T2(V+A), I2(V+A), V2(V+A) | 8% |
| Action | Forward dyn, Inverse dyn, Policy | **25%** |
| General Transfer | Edge / blur / depth / segmentation | 20% |
| Driving Transfer | World-scenario-map controls | 5% |

→ Action 占 25%、Transfer 占 25%,大大强化 Physical AI 能力。

**关键设计**:Action loss × 10×(因为 action vector 已 normalize,per-element MSE 比图像小)。

**训练规模**:
- Nano: **2.4T tokens** on 1024 × GB200
- Super: **1.9T tokens** on 2048 × GB200

#### 4.2.3 Text-to-Image Post-Training(Cosmos3-Super-Text2Image)

两阶段 SFT:
1. **Stage 1: broad T2I specialization** — 20k steps,数据 mixture 45% real / 40% synthetic / 15% text-rendering,lr=1e-4
2. **Stage 2: high-quality refinement** — 2k steps,**470K 精选 ultra-high-quality 配对**

固定 70K context window,只用 ≥720p 图像。最终 UniGenBench 全分 **91.26**(开源第一)。

#### 4.2.4 Image-to-Video Post-Training(Cosmos3-Super-Image2Video)

- 数据:过滤过的 pre-training data + 1000 人工高质量视频 + 20K synthetic
- 训练里**20% T2I image tokens**保住语义对齐
- 目标分辨率 480p,189 帧 ≈ 8 秒 @ 24 FPS
- 10k iter,lr = 1e-5,~50B tokens
- 最终 Artificial Analysis I2V 排行榜开源第一

#### 4.2.5 Robot Policy Post-Training(Cosmos3-Nano-Policy-DROID)

- Base:DROID 数据集(76K 轨迹,350h,86 任务,564 场景)
- 输入:robot 当前 proprioceptive state + 三视角图像(wrist 360×640 上方,两个外部视角 180×320 拼下方,合 540×640 canvas)
- 输出:**32 个未来 joint position actions** + 辅助 RGB 帧
- 操作频率 15 Hz
- 训练:从 Cosmos3-Nano mid-train checkpoint resume,**freshly initialized action encoder/decoder**,action 参数 lr **5× 加速**
- 推理优化:4 步 diffusion + shifted noise schedule s=5 + CFG guidance 3 + **跳过 video latent decode** → 2 × RTX Pro 6000 GPU 部署

最终 RoboLab + RoboArena 双第一。

---

## 第 5 节:Infrastructure(基础设施)

这一节讲 NVIDIA 怎么把这个庞然大物训出来。**4 个支柱**:

1. **Data engineering** — Raw 多模态数据 → WebDataset
2. **Large-scale training** — GPU 集群、并行、checkpointing
3. **Serving** — vLLM / TensorRT-LLM / vLLM-Omni
4. **Benchmark** — 评测基础设施

### 5.1 Data Infrastructure

- 大规模 data processing pipeline
- Embedding storage + semantic retrieval
- 可视化/inspection/debugging 工具

### 5.2 Training Infrastructure(重点)

#### 5.2.1 Data Loader(Joint Data-Loader,论文最有借鉴价值的工程之一)

**问题**:Cosmos 3 训练样本 token 数差 **2 个数量级**(单条文本 vs 720p 视频),传统 LLM 的 data loader 范式失效:
- 等 sample 数 → padding 爆炸、负载不均、NCCL timeout
- 等 token 数(各 rank 独立) → 各 rank 模态分布不同 → attention FLOPs 不均(attention 是 O(n²))

**解法**(Fig 12):
1. **Token-budgeted packing**:不是固定 sample 数,而是**严格 token budget T_max**,greedy pack,**完全不 padding**
2. **Joint Data-Loader**:每个 stream(image/video/action/...)独立 prefetch buffer,joint loader 复用
3. **Rank-synchronous stream selection**:**全 rank 同时刻吃同一个 stream**(globally seeded selector),保证 NCCL 步时一致,**提升 54% 吞吐**
4. **Look-ahead packing**(Fig 13):pack 时遇到放不下的样本,放进 lookaside buffer,继续往后看找小样本填洞,**提升 8% effective sequence length**

#### 5.2.2 Attention Implementation

针对 packed sequence 的 fused attention 内核,处理变长 sample 内部和跨 sample 的不同 mask 模式(AR causal + DM full)。

#### 5.2.3 Distributed Training

**HSDP + CP 组合**:
- **Hybrid Sharded Data Parallelism (HSDP)** — 在 replica group 内 shard optimizer/grad/params,group 间 replicate
- **Context Parallelism (CP)** — 沿 sequence 维度切 split(长视频时)
- 两者正交,按实验动态配置

#### 5.2.4-5.2.7 优化技术

- **Selective Activation Checkpointing** — 选择性激活重计算
- **Torch Compile for Transformer Blocks** — 编译加速 transformer block
- **Video Tokenizer** — On-the-fly VAE encoding(不预计算 latent,边训边算)
- **Checkpointing** — 异步 off-critical-path,不阻塞训练循环

#### 5.2.8 Throughput Summary

- Nano: 4.56M image tokens + 16.23M video tokens / GPU-hour
- Super 较慢

### 5.3 Serving Infrastructure

- **Plain PyTorch** — 原生推理
- **Reasoner serving**:vLLM, TensorRT-LLM
- **Generator serving**:**vLLM-Omni**(Cosmos 团队自研的扩展,支持多模态生成的高效推理)

### 5.4 Benchmark Infrastructure

集成 VLMEvalKit 等评测工具,自动跑 48+ 个 reasoner benchmark 和 generator benchmark。

---

## 第 6 节:Results

### 6.1 Reasoner Evaluation(48 benchmarks)

四大类:**General(19)、Robotics(17)、Smart Infrastructure(9)、Driving(3)**。

| 类别 | Cosmos3-Super | Qwen3-VL-32B | Gemma-4-31B | Gemini 3.1 Pro |
|------|---------------|--------------|-------------|----------------|
| General | **73.7** | 72.8 | 69.8 | 77.5 |
| Robotics | **57.8** | 52.6 | 51.0 | 58.2 |
| Smart Infra | **62.6** | 56.1 | 51.3 | 58.6 |
| Driving | **79.3** | 40.7 | 36.6 | 47.2 |

**Driving 上 Cosmos3-Super 远超 Gemini 3.1 Pro(79.3 vs 47.2)**—— 因为 SFT 阶段大量自动驾驶 CoT 数据 (1.1M auto-labeled video)。

### 6.2 Generator Evaluation

#### 6.2.1 Image Generation(UniGenBench, Artificial Analysis)

| Model | UniGenBench |
|-------|-------------|
| **Cosmos3-Super-Text2Image** | **91.36** |
| Gemini 3 Pro Image | 90.85 |
| Qwen-Image-2512 | 84.25 |
| Cosmos3-Super (mid-train, no T2I post) | 84.61 |

→ T2I post-training 把分数从 84.61 → 91.36(+ 6.75)。

Artificial Analysis 排行榜 **开源第一**(写报告时)。

#### 6.2.2 Video Generation

| Model | Text-to-Video | Image-to-Video |
|-------|--------------|----------------|
| **Cosmos3-Super-Image2Video** | 80.0 | **82.8** |
| Veo-3.1 (closed) | 79.1 | 82.6 |
| Wan2.2-A14B | 78.0 | 81.3 |
| Cosmos3-Nano | 79.4 | 82.7 |

I2V 上 Artificial Analysis 开源第一。

#### 6.2.3 Audio Generation

Cosmos3-Super 音频 7.31 vs Veo-3.1 7.45,相当接近。

#### 6.2.4 Transfer Generation(Edge/Depth/Segmentation/Blur 控制)

跨 4 种 control 输入,Cosmos 3 表现稳定,且支持驾驶专属的 world-scenario-map 控制。

#### 6.2.5 Action Generation

Forward Dynamics(FD)和 Policy 都拿到开源最佳:
- FD on RoboLab: 26.0(Cosmos3-Super, post-train),vs Ctrl-World 23.0
- Policy on RoboLab: 39.7(Cosmos3-Nano-Policy-DROID),vs π0.5 28.1

### 6.3 Generator User Guide

#### 6.3.1 Generation Guide

实用指南:CFG 范围、推理步数、prompt 怎么写等。

#### 6.3.2 Cosmos 3 Reasoner as Prompt Upsampler

**关键设计**:用 Cosmos 3 自己的 Reasoner 当 prompt enhancement layer。用户给短 prompt → Reasoner 扩成结构化 detailed prompt(JSON schema 包含 scene_imagination, temporal_caption, audio_description 等) → 喂给 Generator。

这种"自己做自己的 prompt enhancer"形成了 unified model 的一个独特优势 —— 不需要外挂 LLM。

---

## 第 7 节:Related Work

横跨 6 个领域:

1. **World Models for Physical AI** — Cosmos 1/2, World Forge, DreamerV3, GAIA-1, GENIE 等。Cosmos 3 把这些线索全部 unify
2. **Multimodal Understanding & Embodied Reasoning** — Qwen3-VL, Gemma-4, Cosmos-Reason, GR00T 等
3. **Video Generation & Visual World Simulation** — Wan2.x, Veo, Sora, Pyramid Flow, Self-Forcing, LongLive 等
4. **Action Modeling, VLAs, WAMs** — RT-2, OpenVLA, π0, MolmoAct, Cosmos-WAM 等
5. **Audio & Audio-Visual Generation** — VALL-E, AudioLDM, Veo, Lee et al. audio VAE 等
6. **Omnimodels for Understanding and Generation** — Deng et al. (closest competitor)、Chameleon、Emu 等

### Cosmos 3 vs 同期工作的差异化

| | Cosmos 3 | 多数 VLA/WAM | 多数 Video Gen |
|---|----------|--------------|----------------|
| 任务范围 | 5 模态全 unify | 仅 vision + action | 仅 text + video |
| 架构 | MoT 双塔 + Shared Attention | dedicated head | DiT 单塔 |
| 模态对齐 | 3D MRoPE + Absolute Temporal | 朴素 | 朴素或绝对时间 |
| 训练规模 | 1024-2048 GB200 | 几十~几百 H100 | 几百 H100 |

---

## 第 8 节:Conclusion

论文核心贡献的回顾:
1. **架构**:MoT 双塔 + Shared Attention,跨模态 unify
2. **数据**:大规模 curated multimodal corpora + 5 大 SDG synthetic dataset
3. **训练**:Pre + Mid + Post 三阶段 curriculum,跨模态 jointly
4. **基础设施**:GPU 集群 + Joint Data-Loader + HSDP+CP 训出来的
5. **生态**:全开源 code/weights/datasets/benchmark

---

## 个人补充:几个值得反复琢磨的设计

### A. 为什么不用一个 encoder

视觉部分用了**两个 encoder**(理解用 ViT,生成用 VAE),这跟 Chameleon / Emu 系列的"单 encoder"路线截然相反。论文没明说为什么,但合理解释是:
- **ViT** 是 contrastive/alignment 预训练出来的 → 语义强、reconstruction 弱
- **VAE** 是 reconstruction 预训练 → 像素质量好、语义浅

Unified encoder 必然在这两个目标之间妥协。Cosmos 3 的选择是**直接放弃 unified encoder**,用 MoT 双塔在 backbone 内部把两套 token 整合 —— 这其实是个工程务实主义的选择。

### B. AR / DM gap = 15000 的来历

`temporal_index += 15000` 这个数看似随意。可能的解释:
- 训练序列 ≤ 16K token,15000 几乎等于"序列长度"
- 等于人为制造一个**永远不会被任何 vision/audio/action token 占到的 position**
- 给 RoPE 一个清晰信号:"这个 position 之前是 text,之后是 vision"

是个**工程 hack**,但能解决大模型的 saturation/checkerboard 问题。

### C. Reasoner 训完冻结,只训 Generator

```
预训 VLM (Qwen3-VL-32B)
   ↓ 双塔初始化
Cosmos 3 = [Reasoner Tower + Generator Tower] (二者权重相同)
   ↓ Reasoner SFT (训 Reasoner, Generator 跟着, 但不重要)
Reasoner 强大
   ↓ Generator Pre/Mid/Post training (Reasoner frozen!)
Generator 学会生成,Reasoner 不退化
```

→ **理解能力不退化的关键**:整个 Generator 训练全程冻 Reasoner。这是个非常聪明的设计。代价是 Generator 训练时**Reasoner 的 K, V 也参与 attention**(被 DM query 看到),所以 inference 时如果只想做 generation 也得 load 整个 Reasoner。

### D. Flow matching 而非 EDM/DDPM

整个 generator 都用 rectified flow matching,**比传统 DDPM 更简单**:
- 训练目标:`v = ε - x_0`,直线插值,常速度
- 推理:Euler 步直接积分
- 容易跟 LLM-style 训练栈集成(只是回归 MSE 嘛)

这反映了 2024-2026 的扩散趋势:**简化训练 + 工业化训练栈**。

### E. Action representation 的可迁移性

ego/effector pose + grasp 这套表示**能 cover 几乎所有 embodiment**(车、相机、机械臂、人形)。**Domain-aware I/O + shared backbone** 是 VLA 圈成熟做法,Cosmos 3 把它推到 5 个 domain 同时训。

---

## 关键代码/资源位置

| 资源 | 链接 |
|------|------|
| Cosmos GitHub | github.com/nvidia/cosmos |
| Cosmos-Framework | github.com/nvidia/cosmos-framework |
| Cosmos3-Nano 权重 | huggingface.co/nvidia/Cosmos3-Nano |
| Cosmos3-Super 权重 | huggingface.co/nvidia/Cosmos3-Super |
| Cosmos3-Super-T2I | huggingface.co/nvidia/Cosmos3-Super-Text2Image |
| Cosmos3-Super-I2V | huggingface.co/nvidia/Cosmos3-Super-Image2Video |
| Cosmos3-Nano-Policy-DROID | huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID |
| Cosmos-HUE Benchmark | huggingface.co/datasets/nvidia/Cosmos-HumanEval-v1 |
| 5 个 SDG synthetic datasets | huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-* |
| Cookbook(用户示例) | cookbooks/cosmos3/ in cosmos repo |
| 架构图 | cookbooks/cosmos3/cosmos3-model-architecture.png |

---

## 一句话总结

> **Cosmos 3 = MoT 双塔架构(Reasoner + Generator) × 5 模态(语言/图/视/音/动作) × 3 训练阶段(Pre/Mid/Post),NVIDIA 用 2048 个 GB200 训出来的 Physical AI 通用基础模型**。它的关键工程贡献是:
> 1. **MoT Dual-Tower + Shared Attention** —— 每层两套独立 params 但共享 attention,理解和生成能力都不互相干扰
> 2. **3D MRoPE + Absolute Temporal Modulation** —— 把不同 FPS / 不同模态 token 对齐到物理时间轴
> 3. **统一 Action Representation** —— 9D pose + grasp 表示跨 5 种 embodiment
> 4. **三阶段 curriculum** —— 渐进引入模态(视觉 → +音频 → +动作 → +控制)
> 5. **Joint Data-Loader** —— 带 rank-sync stream selection + look-ahead packing,解决多模态训练吞吐瓶颈
>
> 思想上最重要的一点:**"理解"和"生成"在 Physical AI 里本质耦合,unify 比 fragment 强**。Artificial Analysis T2I / I2V 双开源第一 + RoboArena 第一 是这个 unify 哲学的实证。
