# Inference Acceleration

扩散模型 / DiT **推理加速**方向的论文阅读笔记。区别于 `video_generation` 那条"few-step 因果 AR 蒸馏"主线——这里收录的是**不改模型权重(或仅推理期生效)的加速方法**,主轴是 **training-free caching**(免训练缓存)。

## 加速方法谱系

```
扩散推理加速
├── 减少计算/采样步(改模型或 schedule)
│   ├── 高阶 ODE/SDE solver(DDIM、DPM-Solver、UniPC)
│   ├── 蒸馏(LCM、DMD、CausVid …)        ← 见 video_generation 方向
│   └── 量化 / 分布式
│
├── 缓存复用(training-free)
│   ├── 均匀/固定层缓存:DeepCache、FORA、Δ-DiT、PAB
│   ├── 输入感知自适应:**TeaCache**(本方向首篇)
│   └── 内容自适应 / CFG 复用:AdaCache、FasterCache
│
└── 多分辨率分阶段采样(training-free)
    ├── Latent 空间 upsampling:RALU、SPEED、LSSGen
    └── 像素空间 SR + 低强度 noising:**MrFlow**（本方向新篇）
```

## 核心问题

扩散去噪是串行的,几十步每步过一遍完整 DiT。caching 类方法都基于同一个先验:**相邻 timestep 的模型输出高度相似**,可以缓存复用。分歧在于**"何时复用"**:

- 均匀缓存(PAB 等):固定调度,不管相邻步差异是否均匀 → 在 U 形差异曲线上必然踩坑
- 自适应缓存(TeaCache):用零成本的输入侧信号预测输出差异,动态决定复用时机

## 关键术语

| 术语 | 含义 |
|------|------|
| **Caching / Reuse** | 缓存某步的中间特征或输出,后续步直接复用,跳过计算 |
| **Uniform Cache** | 固定间隔交替 cache/reuse,不感知步间差异 |
| **Indicator** | 决定"是否复用"的判据;TeaCache 用累积相对 L1 距离 |
| **Relative L1 ($L1_{rel}$)** | 相邻步 embedding 的归一化 L1 差异 |
| **Modulated Noisy Input** | timestep embedding 经 AdaLN 调制后的 noisy input,TeaCache 选作 indicator 信号 |
| **Polynomial Rescale** | 用一元多项式把输入差异重标定成输出差异估计,消除 scaling bias |
| **Residual Cache** | 缓存 blocks 的改变量(output − input)而非输出本身 |
| **ret_steps / cutoff_steps** | 头/尾强制真算的步数,保护去噪早晚期质量 |

## 论文列表

| 简称 | 标题 | 任务 | 发表 | 链接 | 状态 |
|------|------|------|------|------|------|
| [teacache](./teacache/analysis.md) | Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model | DiT 推理加速(免训练缓存) | CVPR 2025 (UCAS+Alibaba) | [github](https://github.com/ali-vilab/TeaCache) | ✅ |
| [mrflow](./mrflow/analysis.md) | Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling | 多分辨率分阶段采样(免训练) | arXiv 2026-07 (BUAA+NTU+ICT) | [github](https://github.com/xliu-deep/MrFlow) | ✅ |
