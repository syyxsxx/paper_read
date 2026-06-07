# Multimodal

多模态/Omnimodal 方向论文,涵盖 VLM、VLA(Vision-Language-Action)、World Models、Omnimodal Foundation Models 等。

## 主线脉络

跟 video_generation 主要做"生成"不同,这个目录侧重**理解 + 生成 + 动作的统一框架**,典型应用是 Physical AI、机器人、具身智能。

## 论文列表

| 简称 | 标题 | 任务 | 发表 | 链接 | 状态 |
|------|------|------|------|------|------|
| [cosmos3](./cosmos3/analysis.md) | Cosmos 3: Omnimodal World Models for Physical AI | Omnimodal World Model(5 模态:语言/图/视/音/动作) | NVIDIA, 2026 | [github](https://github.com/nvidia/cosmos) | ✅ |

## 关键技术词汇

| 术语 | 含义 |
|------|------|
| **Omnimodal** | 同时处理 5+ 种模态(语言、图像、视频、音频、动作)的统一模型 |
| **Mixture-of-Transformers (MoT)** | 每层有多套独立参数(每模态/任务一套),但 attention 共享 |
| **Dual-Tower** | MoT 的特例:reasoner tower(理解)+ generator tower(生成)双塔 |
| **VLA** | Vision-Language-Action model,把动作当模态输出 |
| **WAM (World-Action Model)** | 同时输出动作和未来视频的模型 |
| **Forward Dynamics** | 给动作预测未来视频 |
| **Inverse Dynamics** | 给视频反推动作 |
| **Policy(联合)** | 同时输出动作和视频(自回归式的控制) |
| **3D MRoPE** | (t, h, w) 三维 multimodal RoPE,把时间/空间分开编码 |
| **Absolute Temporal Modulation** | 把不同 FPS 模态对齐到物理时间轴的 RoPE 改造 |
| **Rectified Flow Matching** | 用直线插值 + 常速度预测的简化扩散训练 |
