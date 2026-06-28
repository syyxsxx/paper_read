# DAR: Rethinking Cross-Layer Information Routing in Diffusion Transformers

**论文**: Rethinking Cross-Layer Information Routing in Diffusion Transformers  
**机构**: 南京大学 + 阿里巴巴 + 浙江大学 + 香港城市大学  
**arXiv**: [2605.20708v2](https://arxiv.org/abs/2605.20708)  
**发表**: 2026 年 6 月  
**代码**: 无官方开源（实验基于 SiT-XL/2 + ImageNet 256×256）

---

## 1. 一句话定位

**要解决的问题**：DiT 沿用 Transformer 的逐层残差连接，但残差路由在扩散模型中有三个系统性缺陷（幅值膨胀、梯度衰减、块间冗余），导致深层块几乎无法有效优化。

**核心解法**：把"固定权重增量求和"替换为"可学习 softmax 加权的跨层非增量聚合"——每个块的输入由当前和历史所有子层输出按 timestep-aware 注意力权重决定，称为 **Diffusion-Adaptive Routing（DAR）**。

---

## 2. 与前作的关系

```
标准 DiT 残差（Peebles & Xie 2023）
    h_l = h_0 + Σ_{i<l} f_i(h_i; t)    # 单位权重固定累加
    → 深层块梯度衰减，块间余弦相似度 >0.9，深层优化几乎停滞

U-ViT / U-DiT / HunyuanDiT（手工 skip connection）
    人工指定哪几层跳连、权重固定、时间无关
    → DAR 是学习到的、timestep-adaptive 的、覆盖所有历史层

AttnRes [53]（Kimi Team, arXiv:2603.15031, 2026）——LLM 版本
    同样用 softmax-加权跨层聚合，用于 GPT-style LLM
    → 启发了 DAR，但三点关键差异见 §6
    → AttnRes 中 dynamic query 比 static 提升微弱；DiT 中却是关键

REPA（Sihyun Yu et al., ICLR 2025）
    训练目标：对齐 DiT 隐藏状态与 DINOv2-B 表示（layer 8，系数 0.5）
    纯训练目标干预，与 DAR 架构改动正交 → 叠加收益可加
```

---

## 3. 核心算法

### 3.1 三症状诊断

在 SiT-XL/2 上（L=28 个 Transformer 块，共 56 个子层，训练 600K 步）实测：

| 症状 | 现象 | 根因 |
|------|------|------|
| **前向幅值膨胀** | `RMS(h_k)` 从 block 1 的 ~15.5 膨胀到 block 28 的 ~1576（约 100×） | 残差逐层累加使幅值无界增长；PreNorm 归一化被大幅值稀释，深层 LN 实际上做了重新缩放但损失了信息 |
| **反向梯度衰减** | 第 5 块后梯度急剧下降，深层梯度约为浅层的 1/10 | 幅值膨胀导致 LN Jacobian 接近零，反向信号无法穿透 |
| **块间冗余** | 相邻块余弦相似度全程 >0.9 | 残差主导路径，子层实际贡献极小，"深层块做了什么"基本是冗余的 |

### 3.2 标准残差与 DAR 对比

**标准残差**（展开形式）：

$$h_l = h_0 + \sum_{i=0}^{l-1} f_i(h_i;\,t)$$

单位权重固定，与 timestep `t` 无关。

**DAR**（softmax 加权非增量聚合）：

$$h_l = \sum_{i=0}^{l-1} \alpha_{i \to l}(t)\cdot v_i$$

其中：

$$v_i = f_i(h_i;\,t),\quad v_0 = h_0$$

$$k_i = \mathrm{RMSNorm}(v_i)$$

$$\alpha_{i\to l}(t) = \mathrm{softmax}\!\left(\frac{q_l(t)^\top k_i}{\sqrt{d}}\right)_{i=0}^{l-1}$$

关键变化：
- 每个目标层 `l` 通过 softmax 注意力从所有历史子层输出 `{v_0, ..., v_{l-1}}` 中自适应聚合
- `h_l` **不再**包含上一层的完整 residual stream，打破了幅值无界累加
- 权重通过 timestep-aware query `q_l(t)` 决定，不同去噪步骤路由不同历史信息

### 3.3 三种 Query 参数化

| 变体 | 公式 | 特点 |
|------|------|------|
| **Static** | `q_l(t) = w_l` | 时间无关；类似固定 skip 权重 |
| **Static + t-injection** | `q_l(t) = w_l + e(t)` | 复用 DiT 已有 t-embedder；`e(t)` 零初始化，训练初期退化为纯 static，无额外参数 |
| **Dynamic** | `q_l(t) = W_q^{(l)} v_{l-1}` | query 由上一子层输出隐式编码 timestep 信息；`W_q^{(l)}` 需要学习 |

**关键发现**：在 DiT 中，t-injection 和 dynamic 均比 pure static 有显著提升（约 1.5 FID），这与 AttnRes（LLM）中 dynamic vs. static 差异微小的结论相反。原因：扩散模型的去噪轨迹本质上是 timestep-conditioned 的，不同阶段需要不同的层级信息聚合策略。

### 3.4 Chunked Aggregation（内存效率）

朴素 DAR 的 KV 存储随层数线性增长：`O(Ld)`。对 `L=56` 的深层 DiT 不可接受。

**分块策略**：
- 将 `L` 个子层划分为 `N` 个大小为 `S = L/N` 的 chunk
- 每个 chunk `n` 用最后一个子层输出 `c_n = v_{nS}` 作为 chunk summary
- 目标层 `l`（在 chunk `n` 中）的信息来源：
  - 远端：`{c_0, c_1, ..., c_{n-1}}`（历史 chunk 摘要）
  - 近端：`{v_{(n-1)S+1}, ..., v_{l-1}}`（当前 chunk 内全部子层输出）

内存从 `O(Ld)` 降为 `O((S+N)d)`，当 `S=4, N=14`（SiT-XL/2 设置）时节省显著。

**专用最终聚合器（Final Aggregator）**：最后一个 chunk 的聚合器比较特殊——它同时看到所有历史 chunk 摘要和当前 chunk 内所有子层原始输出（不仅仅是摘要），比普通块多约 2 FID 的提升。

### 3.5 最优 Chunk Size（Proposition 1）

信息论建模：路由损失 = 路由熵（来源数的对数） + 压缩失真（chunk 内信息损失） + chunk size 的边际代价：

$$\mathcal{L}(S) = \log\!\left(\frac{L}{S}\right) + S + \alpha\cdot\log S$$

全局最小值解析解：

$$S^* = \sqrt{\frac{L\cdot(1-\alpha)}{1+\alpha}}$$

对 SiT-XL/2（`L=56`，实测 `α ∈ [0.4, 0.6]`）：

$$S^* \in [3.7,\;4.9] \quad\Rightarrow\quad S=4$$

与消融实验经验最优值 `S=4` 完全吻合，验证了理论推导。

---

## 4. 关键实验结果

### 4.1 主要对比（ImageNet 256×256，SiT-XL/2 backbone）

| 方法 | 训练步数 | FID↓（SDE sampler） | 备注 |
|------|----------|---------------------|------|
| SiT（原版） | 600K | ~16.3 | baseline |
| SiT（原版） | 1.75M | 9.62 | 官方最优 |
| DAR-static c4 | 600K | **6.92** | 8.75× 更少迭代达到同等 FID |
| SiT-Plus（同参数量，2× 训练） | — | >DAR 600K | 证明提升非来自参数量 |

| 方法 | FID↓（ODE，无 CFG） | 备注 |
|------|---------------------|------|
| SiT | 9.67 | baseline |
| DAR-dynamic c4 | **7.56** | **-2.11 FID** |

| 方法 | 100K 步 FID | 等效时间节省 |
|------|------------|-------------|
| REPA alone | ~12+ | — |
| REPA alone | 6.89（200K） | baseline |
| DAR + REPA | **7.09（100K）** | **2× early-stage speedup** |

### 4.2 与手工跳连对比

- U-ViT-H/2（手工 U-Net 型跳连）：DAR 显著更优
- U-DiT-L（手工跳连）：DAR 显著更优
- 说明：学习到的、时间自适应的路由 >> 手工固定跳连

### 4.3 Triton Kernel（Appendix E）

针对在线 softmax 递归的融合 kernel，每个 source 只从 HBM 读取一次：

| 操作 | 原始实现 | Triton 融合 | 加速比 |
|------|----------|------------|--------|
| Forward（N=57） | 22.5 ms | 1.96 ms | **11.5×** |
| Backward（N=57） | 115.8 ms | 13.6 ms | **8.5×** |
| 前向激活内存 | — | -78.7% | — |
| 反向激活内存 | — | -74.6% | — |

### 4.4 DMD Post-Training

将 DAR 应用到 Qwen-Image 模型，通过 LoRA（rank 64）做分布匹配蒸馏（DMD）：
- Student LR 5e-6，Fake branch LR 2e-6
- 4 步去噪，guidance 4.0，1024² 分辨率
- 相比标准 DMD，保留更多高频细节（标准 DMD 倾向于平滑化）

---

## 5. 关键配置项

| 参数 | 值 | 说明 |
|------|----|------|
| Backbone | SiT-XL/2 | patch size 2，L=28 Transformer 块 = 56 子层 |
| Chunk size S | 4 | 由 Proposition 1 理论推导和消融双重确认 |
| Chunk 数 N | 14 | L/S = 56/4 = 14 |
| Query 最优变体 | dynamic | `q_l = W_q^{(l)} v_{l-1}` |
| t-injection | 零初始化 | 无额外参数；训练初期退化为纯 static |
| RMSNorm | 用于 key 归一化 | 保证不同量级来源可比较 |
| REPA 对齐 | DINOv2-B, layer 8, 系数 0.5 | 与 DAR 正交，叠加后收益可加 |
| DMD LoRA rank | 64 | Student/Fake branch 分开 LR |
| Triton block size | 在线 softmax 递归 | 每 source 仅读 HBM 一次 |

---

## 6. 争议/权衡

### 6.1 与 AttnRes 的三点关键差异

AttnRes [53]（Kimi Team, 2026）是 DAR 的 LLM 对应版本，两者看似相似但有关键区别：

| 维度 | AttnRes (LLM) | DAR (DiT) |
|------|--------------|-----------|
| Dynamic query 效果 | 与 static 差异微弱 | **显著优于 static（~1.5 FID）** |
| Chunk summary 选择 | 各 chunk 的平均或最后输出 | 最后子层输出（`v_{nS}`） |
| 最终聚合器 | 与普通块相同 | **专用设计**：看到当前 chunk 所有原始子层输出 |

DAR 中 dynamic 如此重要的根因：DiT 的去噪任务在不同 timestep 需要不同层的信息（高 t 需要语义、低 t 需要细节），static query 无法适应这一变化。

### 6.2 只在 SiT/ImageNet 上验证

DAR 的全部主要实验基于 SiT-XL/2 + ImageNet 256×256，无以下验证：
- 大规模文生图/文生视频 T2I/T2V（DiT 的主要工业用例）
- 更大模型（>600M 参数）
- 连续/不均匀长度序列（视频）

DMD + Qwen-Image 是初步迹象，但规模有限。

### 6.3 Chunked 聚合的信息损失

用 chunk summary（最后子层输出）代替完整细粒度历史是有损压缩。命题 1 的理论模型假设了特定的信息衰减结构，实际 DiT 子层的信息分布可能偏离假设。

### 6.4 训练稳定性

DAR 打破了残差流的累加结构，`h_l` 不再保证包含之前任何特定层的完整信息。实验报告训练稳定，但无大规模长训练验证。

### 6.5 与残差的根本差异

标准残差有一个"恒等跳过"的隐式保证（梯度高速公路）。DAR 的 softmax 权重需要学习出类似的恒等路由，初始化策略（零初始化 t-injection）是关键——但与更激进的非残差架构（如早期 ViT 无跳连）的稳定性对比没有做。

---

## 7. 关键代码位置

论文无官方开源代码。基于伪代码（Appendix E）核心逻辑：

```python
# DAR forward（伪代码，实际用 Triton 融合 kernel）
def dar_aggregate(v_list, q_l, d):
    """
    v_list: [v_0, ..., v_{l-1}]  历史子层输出
    q_l:    query，由 static/t-injection/dynamic 三种方式之一生成
    """
    keys = [rms_norm(v) for v in v_list]                   # k_i = RMSNorm(v_i)
    scores = [q_l @ k / d**0.5 for k in keys]             # attention scores
    alpha = softmax(scores)                                 # α_{i→l}
    h_l = sum(a * v for a, v in zip(alpha, v_list))        # 加权聚合
    return h_l

# Chunked 版本：
# 远端 source = chunk summaries c_n = v_{nS}
# 近端 source = 当前 chunk 内完整 v 列表
```

Triton 核心优化：在线 softmax 递归（online-softmax recurrence），避免重复读取 HBM，backward 用 recomputation 换显存。

---

## 8. 一句话总结

DAR 把 DiT 的"固定权重残差累加"替换为"timestep-aware softmax 加权跨层聚合"——用信息论推导出最优分块策略（`S*=4`）、用 Triton 实现 11.5× kernel 加速——根治了残差路由的三大缺陷（幅值膨胀/梯度衰减/块间冗余），在 600K 步内达到 SiT 1.75M 步的 FID，与 REPA 叠加后早期阶段再快 2×，是"不改 Transformer 本体、只改信息聚合路径"的架构级提速范本。
