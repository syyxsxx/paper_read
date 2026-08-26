# World Model 方向

世界模型方向的论文阅读笔记。目前分成两条互不相交的支线：

- **视频世界模型**：长时生成、几何一致性、物理可控（EVOKE）
- **世界-动作模型（World-Action Model）**：把未来视觉预测当作机器人策略的辅助目标（WorldDiT）

## 论文谱系

```
支线 A：交互式视频世界模型
├── GameNGen (Google, 2024) — 扩散模型首次实时交互游戏
├── DIAMOND (2024) — world model + RL 交互
├── Self-Forcing (Berkeley, 2025) — SFD 蒸馏原型，10s 内
└── Alaya-EVOKE (USTC+Alaya, 2026-08)
    ├── World State Bank: Pi3X 点云几何持久化
    ├── Sparse Teacher: O(N) 跨 chunk 评分
    └── Self-Forced DMD: 90s 全窗口蒸馏

支线 B：世界-动作模型（机器人策略 + 未来视觉预测）
├── 大 VLM 当动作骨干（主流范式，3B–10B）
│   ├── OpenVLA / OpenVLA OFT / π0 / π0.5 / GR00T N1
│   └── ACoT VLA / MMaDA VLA / VLANeXt（LIBERO 96–98.5）
└── 不用大 VLM（sub-billion）
    ├── Diffusion Policy (2023) / Octo (2024) / DiT Policy (2024)
    ├── DreamVLA (2025) — 独立 dream 分支做视觉预测
    └── WorldDiT (Bagel Labs, 2026-07) ← 支线 B 首篇
        ├── 单个共享 DiT 同时出 action velocity 和 RGB patch velocity
        ├── action-safe attention：动作 query 读不到 noised RGB target
        └── 推理时整条视觉路径被摘除（训练有监督、部署无开销）
```

## 论文列表

| 论文 | 简称 | 支线 | 发表 | 一句话 |
|------|------|------|------|--------|
| [Alaya-EVOKE: From Linear-Scaling Supervision to Endless World](./evoke/analysis.md) | evoke | A 视频世界模型 | USTC+Alaya Lab, 2026-08 | WSB + Sparse Teacher + SFD，首个 90s 几何一致交互视频生成 |
| [WorldDiT: A Unified Diffusion Architecture for World and Action Modeling](./worlddit/analysis.md) | worlddit | B 世界-动作模型 | Bagel Labs, 2026-07 | 399M 参数共享 DiT 同时回归 7 步动作与未来归一化 RGB patch，推理时摘掉视觉路径，LIBERO 均值 94.9% 落在 sub-billion Pareto 前沿；但报告值含 60% 非 held-out episode 且无消融 |

## 核心问题

**支线 A（视频世界模型）**

1. **生成时长**：现有方法 ≤10s，EVOKE 突破到 90s
2. **几何一致性**：相机运动后场景重访时不抖动/重影
3. **交互可控**：per-chunk 文本 prompt 动态切换（Evocation）
4. **监督代价线性增长**：teacher 全序列评分显存爆炸 → Sparse Teacher 解决

**支线 B（世界-动作模型）**

1. **规模与架构的贡献无法分离**：几十亿参数的策略里，强控制来自动作设计还是预训练骨干？
2. **视觉预测目标该多重**：WorldDiT 把 RGB 损失权重压到 0.001（动作是 0.1），目标削到 1/3 patch 且逐 patch 白化 —— 到底还剩多少"世界建模"
3. **训练有、推理无**：辅助目标怎么设计才能在部署时无损摘除（答案：action-safe attention）
4. **评测协议**：LIBERO 上跨论文的 episode 数、渲染后端、是否 per-suite 微调都不统一，排名不可直接比

## 交叉引用

- **[ReWorld](../video_generation/reworld/analysis.md)**（video_generation 方向）也是交互式世界模型，与支线 A 同源
- **[MMOE](../image_generation/mmoe/analysis.md)** 的 attention-residual 跨深度复用，与 WorldDiT 的 register token / block-causal 设计属于同一层面的 DiT 内部改造
