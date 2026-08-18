# World Model 方向

交互式视频世界模型：长时生成、几何一致性、物理可控。

## 论文谱系

```
自回归世界模型
├── GameNGen (Google, 2024) — 扩散模型首次实时交互游戏
├── DIAMOND (2024) — world model + RL 交互
└── Alaya-EVOKE (USTC+Alaya, 2026) ← 本方向首篇
    ├── World State Bank: Pi3X 点云几何持久化
    ├── Sparse Teacher: O(N) 跨 chunk 评分
    └── Self-Forced DMD: 90s 全窗口蒸馏

视频生成 → 世界模型融合
├── Self-Forcing (Berkeley, 2025) — SFD 蒸馏原型，10s 内
└── EVOKE — 扩展到 90s + 几何一致
```

## 论文列表

| 论文 | 简称 | 发表 | 一句话 |
|------|------|------|--------|
| [Alaya-EVOKE: From Linear-Scaling Supervision to Endless World](./evoke/analysis.md) | evoke | USTC+Alaya Lab, 2026-08 | WSB + Sparse Teacher + SFD，首个 90s 几何一致交互视频生成 |

## 核心问题

1. **生成时长**：现有方法 ≤10s，EVOKE 突破到 90s
2. **几何一致性**：相机运动后场景重访时不抖动/重影
3. **监督代价线性增长**：teacher 全序列评分显存爆炸 → Sparse Teacher 解决
4. **交互可控**：per-chunk 文本 prompt 动态切换（Evocation）
