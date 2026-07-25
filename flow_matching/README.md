# Flow Matching

Flow matching 训练方法与理论方向的论文阅读笔记。核心问题:如何训练更好的流匹配生成模型——包括表示对齐、噪声调度、自监督目标等训练层面的改进。

## 核心概念

**Flow Matching 基础公式**:

```
x_t = (1-t)*x_0 + t*x_1     -- 线性插值路径 (rectified flow)
v_θ(x_t, t) ≈ x_1 - x_0     -- 速度场建模目标
```

| 子问题 | 描述 |
|-------|------|
| **表示对齐** | 如何让生成模型学习有意义的语义表示(外部 vs 内部) |
| **噪声调度** | 时间步采样分布选择,影响训练效率与生成质量 |
| **自监督目标** | 在生成目标之外增加辅助任务提升表示质量 |

## 方法谱系

```
        表示对齐 (Representation Alignment for Generation)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    外部对齐           内部对齐          自监督统一
    REPA(DINOv2)      SRA / LayerSync   Self-Flow (本文)
    SigLIP2                 │                 │
    V-JEPA2           用模型自身中间层     双时间步不对称
          │           特征做对齐          学生-教师 EMA
    固定外部编码器     无显式 SSL 目标     无外部模型
          │                 │                 │
    Scaling 受限        效果弱于外部        超越外部对齐
```

## 论文列表

| 简称 | 标题 | 方向 | 发表 | 链接 | 状态 |
|------|------|------|------|------|------|
| [self_flow](./self_flow/analysis.md) | Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis | 训练框架 | BFL+MIT, ICML 2026 | [arXiv](https://arxiv.org/abs/2603.06507) | ✅ |

## 关键术语

| 术语 | 含义 |
|------|------|
| **REPA** | Representation Alignment — 对齐 DiT 中间层特征与 DINOv2 表示 |
| **Dual-Timestep Scheduling** | 对不同 token 施加不同噪声级别,制造信息不对称 |
| **EMA Teacher** | 指数移动平均权重副本,作为稳定的自蒸馏目标 |
| **R_M** | Masking ratio,被 secondary timestep 覆盖的 token 比例 |
| **adaLN-Zero** | Adaptive Layer Norm with zero-init,DiT 的条件调制机制 |
| **ICPlan** | Interpolant Conditional Plan,线性插值路径的数学描述 |
