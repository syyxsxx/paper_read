# Training Infrastructure

DiT / 视频扩散模型**训练侧系统与基础设施**方向的论文阅读笔记。与 `inference_acceleration` 对偶:那边关心"推理时怎么少算",这边关心"**训练时怎么把活分匀、把通信压下去**"。

这类工作的共同特征:**不改模型、不改损失、不影响精度**,只改工作在设备之间的摆放方式,收益全在 wall-clock 和显存上。

## 方法谱系

```
混合 image-video DiT 训练的并行放置
│
├── 数据并行(DP)—— 整条序列不可分
│   ├── Naive DP:按 token 数配平        → 计算不平(max/mean 1.27–1.31×)
│   └── AdaptiveLoad:按估计算力配平      → 长视频仍是 straggler;15s 视频 OOM
│
├── 上下文并行(CP)—— 每条序列都切
│   └── USP(Ulysses × ring 网格)       → 平衡完美,但小图白白通信(256p 下占 57–71%)
│
├── DP × CP 混合,预设不相交 rank 组
│   └── KnapFormer                      → 组边界不可跨越;短序列继承组配置
│       └── ⚠️ 被 Zellige 证明存在根本权衡(定理 1 / 定理 2)
│
└── Moldable 放置,rank 集合可重叠
    └── **Zellige**(本方向首篇)
        ├── Hardware Profiler:实测 (bucket × config) 的时间,解析建模显存
        ├── Anchor/Filler 两阶段 planner:CP-SAT + ECF 消对称,33–119 ms/batch
        └── Coalesced Attention Engine:整条序列与分布式分片共存时的 kernel 合并
```

相邻领域(长上下文 LLM 训练)可对照:FlexSP、HotSPa、ByteScale、DCP。

## 核心问题

**序列长度跨度极大时,"配平"到底该按什么配?**

一条 15 秒 1080p 视频在 Wan 的 VAE + patch 配置下是 **497,760 token**,而一张 256p 图片只有几百个。而且:

- **计算随 `L` 二次增长**(dense self-attention 要算 `L²` 个 token 对)
- **显存在 memory-efficient attention 下只随 `L` 近似线性增长**

从 256p 图到 10 秒 1080p 视频,显存涨 **1,307×**,计算涨 **40,451×** —— **差 31 倍**。所以按 token 数配平必然导致计算不平,而按计算配平又会撞上"长序列不可分"的墙。

于是问题变成一个三方约束的组合优化:**step makespan 最小 × per-rank 显存上限 × 通信开销**。

## 关键术语

| 术语 | 含义 |
|------|------|
| **Step makespan** | 一个同步步的耗时 = 最慢 rank 的耗时。这是所有放置优化的真正目标函数 |
| **Moldable task** | 可塑任务:并行度不是预先固定的,而是由调度器在运行前选定 |
| **Placement / 放置** | 为每条序列选定"并行配置 + 参与 rank 集合"的过程 |
| **Parallelism config `(q,h,k)`** | query 轴 / head 轴 / key-value 轴的切分因子,`λ = qhk` 是切分度 |
| **DP(Data Parallelism)** | 整条序列分配给单 rank,不切分,零注意力通信 |
| **CP(Context Parallelism)** | 把一条序列的注意力切到多个 rank,需交换 K/V |
| **Ulysses** | head 维切分,前后各一次 all-to-all,常驻 K/V 为 `L·kv/h` |
| **Ring attention** | KV 维切分,流式传 `k−1` 次,常驻 K/V 为 `L·kv/k` |
| **All-gather CP** | query 维切分,K/V 复制到每个 rank,常驻 `L·kv` |
| **Disjoint-group placement** | rank 被预先划成互不相交的组,每组一个固定并行配置(KnapFormer 路线) |
| **Anchor / Filler** | 计算重、值得切分的长序列 / 计算轻、应保持整条的短序列 |
| **ECF(Exact Compact Formulation)** | 把 solver-等价的序列聚合成有界整数计数,消除置换对称性 —— CP-SAT 能秒解的关键 |
| **Coalescing** | 把同 rank 上的多个整条序列打包进一次 block-diagonal FlashAttention;把同配置同 rank 集的 Ulysses 序列打包进一次 all-to-all |

## 论文列表

| 简称 | 标题 | 任务 | 发表 | 链接 | 状态 |
|------|------|------|------|------|------|
| [zellige](./zellige/analysis.md) | Zellige: Moldable Sequence Placement for Mixed Image-Video DiT Training | 混合图文 DiT 训练的序列放置 | HKUST(GZ)+HIT(SZ)+HKUST, 2026-08 | [arXiv](https://arxiv.org/abs/2608.01150) | ✅ |

## 与其他方向的关系

- **`inference_acceleration`**:同样在优化"算力怎么花",但对象是推理期的去噪循环(缓存、稀疏注意力、分阶段采样)
- **`video_generation`**:那边的长视频方法(LongLive-2.0 的 NVFP4 + Balanced SP)也涉及训练基础设施,但那是模型-算法-系统协同设计;这边收录的是**纯调度层、模型无关**的工作
