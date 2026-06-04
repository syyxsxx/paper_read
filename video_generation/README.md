# Video Generation

视频生成方向的论文阅读笔记,主线是 **few-step 因果自回归视频扩散**(few-step causal AR video diffusion)的演进。

## 论文谱系图

```
                Wan2.x bidirectional diffusion (Alibaba)
                            │ (base model, 50-step bidirectional)
                            │
        ┌───────────────────┼─────────────────────────────┐
        ▼                   ▼                             ▼
   CausVid (2024)      Self-Forcing (2025)        LongLive 2.0 (2026)
   CVPR 2025           NeurIPS 2025               NVIDIA
   MIT + Adobe         Adobe + UT Austin
   │                   │                          │
   ODE Init            ODE Init                   AR Training (teacher forcing)
   ↓                   ↓                          ↓ (long video, multi-shot)
   AR Initial          DMD/SiD/GAN                AR Model
   ↓                   ↓                          ↓
   DMD                 AR Model                   DMD-LoRA (旁路)
   ↓                                              ↓
   AR Model            (Rolling KV)              Few-step
   
   ❌ short-only       ✅ long extrapolation     ✅ long native
   ❌ DF train-gap     ✅ self-rollout           ✅ teacher forcing + error recycling
   ❌ data-needed      ✅ data-free              ❌ data-needed (long videos)
                                                  ✅ NVFP4 infra
                                                  ✅ multi-shot sink
```

## 主线脉络

三篇论文都在做同一件事:**把 Wan 这种慢双向扩散模型改造成快因果 AR 视频生成器**。但解法不一样:

1. **CausVid**:开山之作。建立"ODE Init + DMD"两阶段范式,把 50 步双向蒸馏成 4 步因果。**问题**:DF 训练方式有 train-test gap,长视频靠分段重启实现。
2. **Self-Forcing**:针对 CausVid 的 train-test gap,把训练改成"模型自己 rollout + KV cache 累积",梯度嵌入 rollout 中。同时加 Rolling KV Cache 解决长视频外推。**特点**:data-free。
3. **LongLive 2.0**:跳出 Self-Forcing 的算法巧思,从基础设施切入。用 NVFP4 + Balanced SP 把长视频直接微调做到工程可行,绕过 ODE Init,直接 teacher-forcing。Multi-Shot Sink 支持多镜头交互。

## 关键技术词汇

| 术语 | 含义 |
|------|------|
| **ODE Init** | 用教师 50-step trajectory 的 4 个关键点蒸馏因果学生模型 |
| **DMD** | Distribution Matching Distillation,通过 real_score - fake_score 估计 KL 梯度 |
| **DF (Diffusion Forcing)** | 训练时每帧/块独立采 noise level,context 是真实数据加噪 |
| **TF (Teacher Forcing)** | 训练时 noisy target 看 clean history(可以是 GT 或自己输出) |
| **Self-Forcing** | TF 的特殊形式:clean history 来自模型自己的 rollout(经 KV cache) |
| **Block-Causal Attention** | 块内全连接、块间因果的 attention 模式 |
| **KV Cache Rollout** | 推理时每块生成完写入 KV cache,后续块直接读取 |
| **Rolling KV Cache** | 固定大小 cache + FIFO 驱逐,支持无限长视频 |
| **Attention Sink** | 永远保留头部 N 帧不被驱逐,防止滚动后注意力分布崩溃 |
| **Multi-Shot Sink** | LongLive 2.0:全局 sink + 镜头 sink 双层,支持多场景切换 |

## 论文列表

| 简称 | 标题 | 发表 | 状态 |
|------|------|------|------|
| [longlive2](./longlive2/analysis.md) | LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation | NVIDIA, 2026 | ✅ |
| self_forcing | Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion | NeurIPS 2025 (Adobe) | ⏳ |
| causvid | From Slow Bidirectional to Fast Autoregressive Video Diffusion Models | CVPR 2025 (MIT/Adobe) | ⏳ |

## 阅读建议顺序

按时间顺序读最舒服:
1. **CausVid** 先建立基本框架(双向→因果、ODE Init、DMD)
2. **Self-Forcing** 看 CausVid 的痛点(train-test gap)是怎么解的
3. **LongLive 2.0** 看怎么用工程基础设施反推算法精简

## 关键对照表(三篇横向对比)

| 维度 | CausVid | Self-Forcing | LongLive 2.0 |
|------|---------|--------------|--------------|
| 训练范式 | DF | Self-rollout | Teacher Forcing |
| 是否需要 ODE Init | ✅ | ✅ | ❌ |
| 是否 data-free | ❌ (LMDB latents) | ✅ (prompt only) | ❌ (long videos) |
| Train-test gap | ⚠️ 存在 | ✅ 消除 | ⚠️ 用 error recycling 缓解 |
| 长视频策略 | 分段重启 | Rolling KV cache | 原生长训练 + Multi-Shot Sink |
| 推理步数 | 4 步 | 4 步 | 4 步(可降到 2 步) |
| Few-step 实现 | 主训练管线 | 主训练管线 | LoRA 旁路 |
| 精度 | BF16 | BF16 | **NVFP4 (W4A4)** |
| 多镜头支持 | ❌ | ❌ | ✅ |
