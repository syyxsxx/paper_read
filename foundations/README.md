# Foundations

**跨方向的基础理论与工具**方向的论文阅读笔记：被生成模型论文反复借用、本身却不属于某个应用方向的经典工作（统计检验、核方法、最优传输、信息论……）。收录标准是"至少有两篇应用方向的笔记在直接用它"，读它是为了看清那些用法的前提、偏置和盲区。

## 谱系

```
比较两个分布
│
├── 积分概率度量(IPM):sup over F 的期望差
│   ├── F = RKHS 单位球 → MMD(本方向首篇)
│   │   ├── 训练 loss:GMMN / MMD-GAN / RDM → ViRDM(video_generation)
│   │   └── 评测指标:KID / CMMD / VMMD → Mask Forcing(video_generation)
│   ├── F = Lipschitz 函数 → Wasserstein-1
│   └── F = 有界变差函数 → Kolmogorov-Smirnov
│
└── 最优传输(二次代价、熵正则)→ RWTD 的 Sinkhorn 指派(image_generation)
```

## 论文列表

| 简称 | 标题 | 主题 | 发表 | 链接 | 状态 |
|------|------|------|------|------|------|
| [mmd_two_sample](./mmd_two_sample/analysis.md) | A Kernel Method for the Two-Sample Problem | MMD 的定义、三种估计量、四个两样本检验 | MPI Tübingen+Cambridge+TU Graz+NICTA, arXiv 2008-05（定稿 JMLR 2012） | [arXiv](https://arxiv.org/abs/0805.2368) | ✅ |

## 交叉引用

- **[ViRDM](../video_generation/virdm/analysis.md)** 的训练 loss 就是 MMD 的有偏估计（高斯 RBF 核、中位数带宽、Nyström 近似）
- **[Mask Forcing](../video_generation/mask_forcing/analysis.md)** 的 CMMD / VMMD 是 CLIP / V-JEPA2 特征上的 MMD
- **[RWTD](../image_generation/rwtd/analysis.md)** 在冻结特征空间里用熵正则 OT 做分布匹配，是 MMD 之外的另一条路
