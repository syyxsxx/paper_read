# Cosmos 3 论文与代码解读

> 论文:Cosmos 3: Omnimodal World Models for Physical AI (arXiv 2606.02800v1, NVIDIA, 2026-06)
> 项目主页:https://research.nvidia.com/labs/cosmos-lab/cosmos3
> 代码:https://github.com/nvidia/cosmos + https://github.com/nvidia/cosmos-framework
> 模型权重:https://huggingface.co/collections/nvidia/cosmos3
> 任务:**Physical AI 通用基础模型**(理解 + 生成 + 动作,跨 5 模态)

## 一、一句话定位

Cosmos 3 = **NVIDIA 的"机器人/具身智能基础模型"**。把 VLM + 视频生成 + 世界模拟 + 动作策略合并成**一个 Mixture-of-Transformers(MoT)双塔架构**,五种模态(语言/图/视/音/动作)共用 backbone,根据 token 排列自动切换 AR 模式(reasoner)或 Diffusion 模式(generator)。Cosmos3-Super (64B) 在 Artificial Analysis 评测里 T2I + I2V 双开源第一,Cosmos3-Nano-Policy-DROID 在 RoboArena 上是策略模型第一。

![Fig 1: Cosmos 3 overview](./figures/fig1_cosmos_overview.png)

> Fig 1:Cosmos 3 = 1 个模型替代 6 类模型(VLM / Image Gen / Audio-Visual Gen / Policy-WAM / Forward Dynamics / Inverse Dynamics)。靠的不是适配器堆叠,而是统一的 token 序列 + dual-tower 架构。

## 二、要解决的问题(动机)

Physical AI agent 需要两个底层能力:**Understanding**(从局部观测推断潜在世界状态)+ **Generation**(预测未来、模拟动作后果)。当前 paradigm 把这两件事拆成多个独立模型:

| 模型类 | 干什么 | 缺什么 |
|--------|--------|--------|
| VLM | 看懂场景,出文字 plan | 不会生成未来视频 |
| Video Generator / Forward Dynamics | 模拟未来 | 不会理解 task 语义 |
| VLA / World-Action Model | 输出控制信号 | 通常是端到端黑盒,缺世界模型 |

论文核心论点:**理解和生成本质耦合** —— 理解需要预测未来,生成需要结构化世界表示。把它们 unify 成一个模型才是正确路线。

举例:家用机器人收拾餐桌,当前 paradigm 要拼一堆模型(VLM 找盘子 + VLA 出动作 + Forward Dynamics 模拟结果),fragmented architecture + 计算浪费。Cosmos 3 的目标是 **一个网络全部包了**。

## 三、与前作的关系

Cosmos 3 跨 6 个相关领域,但最相关的是 omnimodal 统一架构这条线:

| 路线 | 代表 | Cosmos 3 的差异 |
|------|------|----------------|
| **VLM-only** | Qwen3-VL, Gemma-4, Gemini 3.1 | + 生成能力,+ 动作模态 |
| **Video Gen-only** | Wan2.x, Veo, Sora, Self-Forcing, LongLive 2.0 | + 理解能力(物理常识、CoT)+ 动作 |
| **VLA / WAM** | RT-2, OpenVLA, π0, MolmoAct, Cosmos-WAM | + 高质量 video gen + 多 embodiment 动作统一 |
| **Unified gen+und** | Chameleon, Emu, Deng et al. 2025 | + Physical AI 数据特化 + MoT 双塔(而非单塔) |

incremental claim:Cosmos 3 是**第一个工业级的、跨 5 模态的 Physical AI 统一基础模型**,且每个细分任务都达到/超过该领域 SOTA。

## 四、核心算法/方法

### 4.0 整体架构鸟瞰

```
            [Language] [Image-ViT] [Image-VAE] [Audio] [Action]
                │          │           │         │        │
                ▼          ▼           ▼         ▼        ▼
              ─── AR subsequence ───  ─── DM (diffusion) subsequence ───
                       │                          │
                       ▼                          ▼
              ┌──── Reasoner Tower ────┐ ┌──── Generator Tower ────┐
              │ LN + MLP + (Q,K,V proj)│ │ LN + MLP + (Q,K,V proj) │  × L layers
              │     (独立参数)          │ │      (独立参数)          │
              │            \____________│_│____________/            │
              │            Shared Multimodal Self-Attention         │
              │              Q_AR causal / Q_DM full bidirectional  │
              └────────────────────────┘ └────────────────────────┘
                       │                          │
                       ▼                          ▼
                  next-token pred            denoised tokens (flow matching)
```

四个核心设计:**Encoders(2.1)** → **Token Arrangement(2.2)** → **MoT 双塔(2.3)** → **3D MRoPE 绝对时间(2.4)**。

### 4.1 Encoders:每模态一个,**视觉走两路**

每种模态用独立 encoder 编码,且**每个非语言模态加一个 learnable modality embedding** 帮 backbone 区分身份。

| 模态 | Encoder | 训练状态 |
|------|---------|---------|
| **Vision-AR**(理解) | 预训练 ViT,16×16 patch + 2 层 MLP + 2×2 token merge,DeepStack 多层聚合 | 跟 backbone 联合训练 |
| **Vision-DM**(生成) | Wan2.2-TI2V-5B VAE(4× temporal, 32×32 spatial) | **frozen** |
| Audio | Audio VAE(Lee et al. 2025b),48kHz/1920 hop → 25 token/sec | frozen |
| Action | Domain-aware linear 投影(每 embodiment 一对 W_in/W_out) | 跟 backbone 联合 |
| Language | Standard text tokenizer | — |

📌 **关键设计 #1:视觉拆两个 encoder**。理解和生成的目标本质冲突 —— ViT(vision-language 对齐)语义强、reconstruction 弱;VAE(reconstruction 训练)像素质量好、语义浅。Cosmos 3 直接放弃 unified encoder,**在 backbone 里用 MoT 双塔分别处理**两套 token。Chameleon/Emu 的"单一 encoder"路线 trade off 不够干净,Cosmos 3 是工程务实选择。

#### 4.1.1 Action representation(论文最值得专门读的子节)

跨 embodiment 的动作空间(关节轨迹 vs 转向 vs body pose vs camera transform)用**共享几何成分**统一:

![Fig 3: 统一动作表示](./figures/fig3_action_repr.png)

> Fig 3:不同 embodiment(车 / 相机 / 单臂 / 双臂 / 人形 / 自我中心人)用三种共享成分拼装。维度 9 + 9 + 1/15 都来自 3D translation + 6D rotation + grasp state。

三个核心成分:

| 成分 | 含义 | 维度 |
|------|------|------|
| **Ego Pose** | agent 主观察坐标系的运动 | 9D = 3D translation + 6D rotation |
| **Effector Pose** | 末端执行器(夹爪/手腕)的运动 | 9D = 3D translation + 6D rotation |
| **Grasp State** | 当前抓握状态(直接编码,**非 delta**) | 1D(夹爪 open/close)或 15D(5 指 × 3D 指尖位置) |

几个设计要点:
- **Pose 用相对变换(delta)**:$\Delta T_t = T_{t-1}^{-1} T_t$,避免依赖具体 PID 参数或 actuator 接口
- **6D 旋转**(Zhou et al. 2019):自由度 3 但表达 SO(3) 时连续,比四元数/欧拉角更易学
- **Grasp state 直接编码当前状态**,不做 delta —— 抓握/松开是离散事件,delta 没意义
- **拼装规则**:不同 embodiment 维度不同(车 10D,humanoid 57D 精简到 29D),**domain-aware I/O 投影**:

$$
z = W_{\text{in}}^{(k)} x + b_{\text{in}}^{(k)}, \quad x = W_{\text{out}}^{(k)} z + b_{\text{out}}^{(k)}
$$

每个 embodiment $k$ 一对 $(W_{\text{in}}^{(k)}, W_{\text{out}}^{(k)})$,**MoT backbone 完全共享**。6D rotation 解码时用 SVD 还原成 3×3 SO(3)。

### 4.2 Token Arrangement + Generation Mode

任意任务的 token 序列都分**两段**:

```
[ ── AR 子序列 ── | ── DM 子序列 ── ]
   language + ViT       VAE + Audio + Action
   走 Reasoner Tower    走 Generator Tower
   causal attention     full bidirectional attention
   next-token pred      flow matching denoising
```

**子序列内部规则**:
- AR 在前,DM 在后
- DM 内每模态:clean conditioning token 在前,noisy target 在后
- 模态顺序:vision → audio → action

#### 6 种 generation mode(纯靠 token 排列切换,架构不变)

| 模式 | 序列结构 |
|------|---------|
| **Language**(VLM) | 只用 AR,DM 参数不激活 |
| **T2I** | $[S_{\text{AR}}, \tilde{v}_1]$ |
| **T2V (+Audio)** | $[S_{\text{AR}}, \tilde{v}_{1:N}, \tilde{s}]$ |
| **I2V / V2V (+Audio)** | $[S_{\text{AR}}, v_{1:P}, \tilde{v}_{P+1:N}]$,P 帧 clean conditioning |
| **Video Transfer** | $[S_{\text{AR}}, v^{\text{ctrl}}_{1:N}, \tilde{v}_{1:N}]$,depth/edge/seg 当 control |
| **Action**(3 子模式) | Forward Dyn / Inverse Dyn / Policy(见下图) |

其中 $S_{\text{AR}} = [l_1, \ldots, l_n, \langle\text{EOS}\rangle, \langle\text{BOG}\rangle]$,`<BOG>` 是 "begin-of-generation"。

![Fig 4: Action 三种模式](./figures/fig4_action_modes.png)

> Fig 4:同一个 video-action 数据样本,通过设置哪些 token 是 clean / 哪些是 noisy,得到三种训练模式 —— Forward Dynamics(给动作预测视频)、Inverse Dynamics(给视频反推动作)、Policy(联合 denoise)。这种"用 mask 切换任务"是 unified 模型的精髓。

### 4.3 Mixture-of-Transformers 双塔架构

![Fig 5: MoT 架构](./figures/fig5_mot_arch.png)

> Fig 5:**每个 decoder layer 有两套独立参数**(LayerNorm + Q/K/V/O proj + MLP),Reasoner Tower 处理 AR token,Generator Tower 处理 DM token,**唯一共享的算子是中间的 Shared Multimodal Attention**。两塔都从同一个预训练 VLM(Qwen3-VL-8B/32B)初始化,所以 Reasoner 一开始就强。右边的 attention mask 直观显示了 AR causal + DM full 的混合模式。

#### Dual-Stream Joint Attention(关键)

虽然两塔参数独立,但 attention 共享,通过 mask 规则实现单向跨流:

$$
O_{\text{AR}} = \text{Attn}_{\text{causal}}\!\left(Q_{\text{AR}}, K_{\text{AR}}, V_{\text{AR}}\right)
$$

$$
O_{\text{DM}} = \text{Attn}_{\text{full}}\!\left(Q_{\text{DM}}, [K_{\text{AR}}; K_{\text{DM}}], [V_{\text{AR}}; V_{\text{DM}}]\right)
$$

→ **AR 不被 DM 污染**(语言生成纯净),**DM 看得到 AR**(prompt + 历史给生成提供上下文)。

Attention mask 长这样:
```
              K_AR        K_DM
        ┌──────────┬──────────┐
Q_AR    │ 三角 causal │ 全 mask  │
        ├──────────┼──────────┤
Q_DM    │ 全看到    │ 全看到    │
        └──────────┴──────────┘
```

📌 **关键设计 #2:为什么是 MoT 双塔而非单一 transformer**。一套参数难以同时擅长"next-token prediction"(理解)和"flow matching denoising"(生成),两者的 loss landscape 和 token 分布都不同。双塔让每条路径有自己的 weights,只在 attention 层 fuse —— 既避免互相干扰,又保留跨模态条件化能力。

### 4.4 Multimodal Position Embedding(3D MRoPE + 绝对时间)

借鉴 Qwen3-VL 的 **3D MRoPE**(每个 attention head 的 hidden dim 切成 t/h/w 三段,各自 RoPE),但 Cosmos 3 加上**绝对时间索引**改造,关键 motivation:**视频/音频/动作可能同时生成,但各自 sampling rate 不同**,必须对齐到物理时间轴。

![Fig 6: 3D MRoPE 坐标分配 + FPS modulation](./figures/fig6_mrope.png)

> Fig 6:左 = 不同模态 token 的 $(t, h, w)$ 三元组分配规则,语言用 $t=h=w$ 退化成 1D RoPE,视频在三轴都变,音频/动作只用 t。右 = FPS modulation 把不同帧率视频的 token 间距按真实时长 rescale,让"1 秒视频"在 RoPE 上占用相同 position 范围,无论 16/24/30 FPS。

#### Position 分配规则

| Token 类型 | $(t, h, w)$ |
|-----------|-------------|
| Language | $t = h = w$ = monotone counter(等于 1D RoPE) |
| ViT 视觉(AR) | 同帧共享 t,h/w 按空间网格 |
| Diffusion 视频 | t 跟帧索引,h/w 跟空间;**每段 vision 从 0 起算**(绝对 within-video 坐标) |
| Image | 当作单帧视频,只变 h/w |
| Audio | 只有 t($h = w = 0$),t 跟 audio hop 走 |
| Action | 只有 t,t 跟 sampling step 走 |

📌 **AR / DM margin 陷阱**:论文发现"直接让 DM 第一帧紧跟 AR 末位 → 大模型(Super) 出现 over-saturation + checkerboard artifacts"。猜测原因:最后一个 language token 跟第一个 vision token 的 temporal embedding 几乎一样,模型分不清。**解法:AR 和 DM 之间硬塞一个固定 15000 的 temporal gap**,纯加 buffer 不改架构。

#### 绝对时间调制公式

定义 TPS(Temporal-steps Per Second):

| Modality | TPS 计算 | 例 |
|----------|---------|----|
| Video | FPS / VAE_temporal_compression | 24 FPS / 4 = 6 |
| Audio | sample_rate / hop_size | 48000 / 1920 = 25 |
| Action | sampling frequency | 任意 |

选基准 $\text{TPS}_{\text{base}} = 24/4 = 6$(24 FPS 视频对应)。每个 token 的实际 temporal increment:

$$
\delta t = \frac{\text{TPS}_{\text{base}}}{\text{TPS}}
$$

→ 16 FPS 视频 $\delta = 1.5$(走快 50%);30 FPS 视频 $\delta = 0.8$(走慢 20%)。同样 1 秒真实时长,无论用 16/24/30 FPS 编码,**RoPE 上占同样的 position 范围**。

### 4.5 模型变体

所有变体都是 MoT 双塔(每层 2 套参数),"总参数 ≈ 2× LLM 参数":

| Variant | LLM Layers | Hidden | Attn / KV Heads | Head Dim | FFN | 总参数 | 初始化 |
|---------|-----------|--------|-----------------|----------|-----|--------|--------|
| **Edge** | 28 | 2,048 | 16 / 8 | 128 | 9,216 | 4B | 从头训 (Qwen3-1.7B 风格,去 QK norm,ReLU² FFN) |
| **Nano** | 36 | 4,096 | 32 / 8 | 128 | 12,288 | 16B | Qwen3-VL-8B |
| **Super** | 64 | 5,120 | 64 / 8 | 128 | 25,600 | 64B | Qwen3-VL-32B |

### 4.6 训练目标:Rectified Flow Matching

整个 Generator 用统一的 rectified flow,**比传统 DDPM 简单**:

$$
x_\sigma = \sigma \cdot \epsilon + (1 - \sigma) \cdot x_0
$$

$$
v_\theta(x_\sigma, \sigma, c) \to v^* = \epsilon - x_0
$$

Loss = masked MSE,conditioning token 不计入。

Per-modality time sampling:每模态独立采 $\sigma$。图/音/动作用 **logit-normal**,视频用 **mode sampling**。再过一次 shift reparameterization:

$$
\sigma = \frac{s \cdot \bar{t}}{1 + (s - 1) \cdot \bar{t}}, \quad \bar{t} = 1 - t
$$

$s \geq 1$ 偏向高噪声。$s$ 按分辨率调:256p $s = 1$,480p $s = 3$,720p $s = 5$(pre-train),mid-train 涨到 3 / 5 / 10。

## 五、数据与训练流程

整体两大阶段:**Reasoner 先训(语言/视觉理解)→ 用 Reasoner 初始化 Generator,然后 Generator 三阶段渐进训练**。

### 5.1 Reasoner Data + Training

| 阶段 | 样本 | 关键设计 |
|------|------|---------|
| Pre-Training | **22M** | Nemotron Nano 2 + 自有 2.3M,**两阶段 curation**:① Qwen3-VL-Embedding-8B 做 K-means + cosine 去重(去 4.23%),② Gemma-4-31B 当 AI judge,三维度 (Faithfulness / Completeness / Correctness) 打 1-5 分,min-threshold ≥ 2 保留 78% |
| SFT | **2.2M** | importance-aware sampling,**保留 20% pre-training data 防遗忘**(1:4 比例),三大 Physical AI 域:AV、Robotics、Smart Infra |

Pre-train hyperparams:lr 5e-5 (LLM/projector) + 5e-6 (ViT,小 10×),cosine decay,10% warmup,2 epoch,**square-root normalized per-token loss**(平衡长短序列)。

📌 **不做 alignment-only 阶段**:不像 Qwen 那样先冻 LLM 训 projector,直接所有组件一起从头训,论文说效果更好。

### 5.2 Generator Data + Training

![Fig 8: Generator 训练 curriculum](./figures/fig8_generator_curriculum.png)

> Fig 8:Generator 是 **三阶段渐进 curriculum**。每列一个 stage,每行一种模式。彩色格子是样本量,灰色 (—) 表示该 stage 不激活。**动作和 transfer 数据在 mid-training 才引入**(因为需要先有强视觉生成基础)。最后 Cosmos3-Nano/Super mid-trained 模型作为 post-training 起点,三个特化模型 Cosmos3-Super-T2I / Cosmos3-Super-I2V / Cosmos3-Nano-Policy-DROID 在右侧。

#### 5.2.1 Pre-Training(31T tokens for Nano)

![Fig 10: 多分辨率训练 + 序列打包](./figures/fig10_multires_training.png)

> Fig 10:左 = 三个分辨率 tier (256p/480p/720p),各自 max 帧数、eligible 源数据、noise shift s 不同;变长样本被打包进固定 **74,000 token / sequence**,**完全不 padding** 来最大化 GPU 利用。右 = 数据 mixture 比例:image:video-256:video-480:video-720 = 1:1:2:1;video 内部 T2V:I2V:V2V = 70:20:10。

**4 种训练模式**(按 $T_{\text{cond}}$ 区分,采样比例 20:56:16:8):
- **T2I**: $T = 1$,纯 image
- **T2V**: $T_{\text{cond}} = 0$,全 noisy
- **I2V**: $T_{\text{cond}} = 1$,首帧 clean
- **V2V**: $T_{\text{cond}} = 2$,前 5 帧(2 个 latent 帧)clean

**关键**:**只更新 Generator 参数,Reasoner 冻结**,保证理解能力不退化。

📌 **训练规模**:
- Nano: **31.05T tokens on 1024 × NVIDIA GB200**
- Super: **17.86T tokens on 2048 × NVIDIA GB200**

#### 5.2.2 Mid-Training(关键阶段,引入 action 和 transfer)

Data mixture(Table 6):

| 流 | 模式 | 比例 |
|----|------|------|
| Image | T2I | 10% |
| Video | T2V/I2V/V2V | 32% |
| Video + Audio | T2(V+A)/I2(V+A)/V2(V+A) | 8% |
| **Action** | Forward dyn / Inverse dyn / Policy | **25%** |
| **General Transfer** | Edge / blur / depth / segmentation | **20%** |
| Driving Transfer | World-scenario-map controls | 5% |

Action loss × 10×(因为 action vector 已 normalize,per-element MSE 比图像小)。

训练规模:Nano 2.4T tokens / 1024 GB200,Super 1.9T tokens / 2048 GB200。

#### 5.2.3 Post-Training:三个特化案例

| 模型 | 关键超参 | 战绩 |
|------|---------|------|
| **Cosmos3-Super-Text2Image** | 两阶段 SFT(20k step broad + 2k step refinement),470K ultra-high-quality 配对 | UniGenBench **91.26**,Artificial Analysis 开源第一 |
| **Cosmos3-Super-Image2Video** | 10k iter @ lr 1e-5,~50B token,目标 480p × 189 帧 ≈ 8 秒 | Artificial Analysis I2V 开源第一 |
| **Cosmos3-Nano-Policy-DROID** | 76K 轨迹,32 个未来 joint position + RGB 辅助,15 Hz,**action lr × 5×** | RoboLab + RoboArena 双第一 |

📌 **Reasoner 全程冻结**:Generator 训练时 Reasoner 的 K, V 仍参与 attention(被 DM query 看到),所以 inference 也要 load 整个 Reasoner —— 用空间换"理解能力不退化"。

## 六、Infrastructure 重点:Joint Data-Loader

Cosmos 3 训练涵盖 5 模态,**单条样本 token 数差 2 个数量级**(单条文本 vs 720p 视频)。传统 LLM data loader 失效:
- 等 sample 数 → padding 爆炸、负载不均、NCCL timeout
- 等 token 数(各 rank 独立) → 各 rank 模态分布不同 → attention FLOPs 不均(O(n²))

![Fig 12: Joint Data-Loader](./figures/fig12_joint_dataloader.png)

> Fig 12:每个 stream(image / video / action / audio)独立 buffer,**rank-synchronous selector 全 rank 同时刻吃同一个 stream**(globally seeded selector),后面 token-budget pack + look-ahead skip 拼成 packed batch。

四个核心机制:
1. **Token-budgeted packing**:严格 token budget $T_{\max}$,greedy pack,**完全不 padding**
2. **Joint Data-Loader**:每 stream 独立 prefetch buffer,joint loader 复用
3. **Rank-synchronous stream selection**:globally seeded selector,**所有 rank 同步选同一个 stream**,保证 NCCL 步时一致 → **+54% 吞吐**
4. **Look-ahead packing**:遇到放不下的样本放进 lookaside buffer,继续往后找小样本填洞 → **+8% effective sequence length**

其他基建亮点:
- **HSDP + CP** 组合(Hybrid Sharded Data Parallelism + Context Parallelism),前者 shard params/grads,后者 shard 长 sequence
- **Selective Activation Checkpointing**、**Torch Compile**、**Async checkpointing**(off-critical-path)
- **On-the-fly VAE encoding**(不预计算 latent,边训边算)
- **Serving**:Reasoner 用 vLLM / TensorRT-LLM,Generator 用自研 **vLLM-Omni**

## 七、关键代码/资源位置

| 资源 | 链接 |
|------|------|
| Cosmos GitHub(主仓库 + cookbooks) | github.com/nvidia/cosmos |
| Cosmos-Framework(训练 + 推理 framework) | github.com/nvidia/cosmos-framework |
| Cosmos3-Nano 权重 | huggingface.co/nvidia/Cosmos3-Nano |
| Cosmos3-Super 权重 | huggingface.co/nvidia/Cosmos3-Super |
| Cosmos3-Super-Text2Image | huggingface.co/nvidia/Cosmos3-Super-Text2Image |
| Cosmos3-Super-Image2Video | huggingface.co/nvidia/Cosmos3-Super-Image2Video |
| Cosmos3-Nano-Policy-DROID | huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID |
| Cosmos-HUE benchmark | huggingface.co/datasets/nvidia/Cosmos-HumanEval-v1 |
| 5 个 SDG synthetic datasets | huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-* |
| Cookbook(用户示例 + 架构图) | cookbooks/cosmos3/ |

## 八、争议点与权衡

1. **不是 data-free,数据成本极高**:5 个自建 SDG synthetic dataset + 22M reasoner data + 767M+348M 生成数据,中小团队复现不可能。Cosmos 3 走的是工业级 unified 路线,跟 Self-Forcing(data-free)的研究路线完全不同
2. **总参数 ≈ 2× LLM 参数**:MoT 双塔每层 2 套 weights,部署内存大。对边端不友好(所以 Edge 变体设计成 4B 总规模)
3. **三种 generation mode 在同一架构里**(VLM / Diffusion / Policy)**inference 时 Reasoner 始终 load**,不能只跑 Generator 节省内存
4. **AR / DM gap = 15000 是工程 hack**:论文坦诚承认大模型有 over-saturation/checkerboard 问题,靠加 buffer 解决,但没有架构层面的根本修复
5. **跟 Gemini 3.1 Pro 在 General reasoning 上仍有 4 分差距**(73.7 vs 77.5),开源 + 强 verticals,但通用 general 还差一点
6. **统一 action 表示对长 horizon 操作的局限性**:9D pose + grasp 这套表示在抓握/接触富的任务上是否够,论文只在 DROID 桌面操作上验证

## 九、个人补充:几个值得反复琢磨的设计

### A. 为什么 Reasoner 和 Generator 都从同一个 VLM 初始化

**两塔都从 Qwen3-VL 权重 init**,这看似冗余,但有两个好处:
- Generator 一开始就有强的视觉理解能力(VAE token 之外还能看 ViT token),不是从 scratch 学 conditioning
- Reasoner 和 Generator 在 attention 共享层面接得平滑(共享 K/V 空间初始对齐)

代价:训练前 64B 的内存需求(两塔都是 32B)。

### B. Token Arrangement 的"约定胜于配置"

整个 Cosmos 3 的灵活性建立在一个简单约定:**AR token 在前,DM token 在后,DM 内 clean 在前 noisy 在后,模态顺序 vision → audio → action**。所有 6 种 generation mode 都是这套约定的不同实例化,**没有任何架构开关**。

这是一个**unified model 的设计美学** —— 用 token 排列 + attention mask 把任务编码到数据流里,而不是用 head/adapter 编码到参数里。

### C. Flow Matching 选择的背后

Rectified flow 在整篇里全用,**不是 DDPM 不是 EDM 不是 v-prediction**。原因:
- 训练目标极简:`v = ε - x_0`,直线插值,常速度
- 推理:Euler 步直接积分
- 跟 LLM 训练栈集成天然(只是一个 MSE 回归)

2024-2026 大趋势:**简化训练 + 工业化训练栈**,Flow Matching 在大规模生产模型里成为默认选择。

### D. Action 表示的"几何抽象"

不直接用 joint position(机器人特定)或 motor command(底层硬件特定),而是上升到 **ego pose + effector pose + grasp state** 这个**任何 embodiment 都能映射**的抽象层。这跟 LLM 用 BPE token 抽象掉具体语言相似 —— **抽象层选择是 unified model 成败的关键**。

## 十、一句话总结

> **Cosmos 3 = MoT 双塔架构(Reasoner + Generator)× 5 模态(语言/图/视/音/动作)× 3 训练阶段(Pre/Mid/Post),NVIDIA 用 2048 张 GB200 训出来的 Physical AI 通用基础模型**。关键工程贡献:
> 1. **MoT Dual-Tower + Shared Attention** —— 每层两套独立 params,attention 层 fuse,理解和生成互不干扰
> 2. **3D MRoPE + Absolute Temporal Modulation** —— 不同 FPS / 不同模态 token 对齐到物理时间轴
> 3. **统一 Action Representation** —— 9D pose + grasp 跨 5 种 embodiment
> 4. **三阶段 curriculum** —— 渐进引入模态(视觉 → +音频 → +动作 → +控制)
> 5. **Joint Data-Loader** —— rank-sync stream selection (+54%) + look-ahead packing (+8%)
>
> 思想上最重要的一点:**"理解"和"生成"在 Physical AI 里本质耦合,unify 比 fragment 强**。Artificial Analysis T2I / I2V 双开源第一 + RoboArena 第一,是这个 unify 哲学的实证。
