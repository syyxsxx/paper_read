# Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

- **arXiv**: [2609.04108v2](https://arxiv.org/abs/2609.04108)（2026-09-04，19 页）
- **机构**: University of Alberta · NYU / NYU Shanghai · University of Chicago · University of Waterloo · Amii
- **作者**: Boyan Li¹\*, Bingsen Chen²³\*, Chenghao Yang⁴, Ping Nie⁵, Chen Zhao²³‡, Xi Ye¹⁶‡（\* 共同一作，‡ 通讯）

---

## 1. 一句话定位

**别把 OPD 和 RLVR 揉在一个 step 里，分两段做：先 OPD 训 60 步，再切纯 GRPO。**

这篇把现有的 OPD×RLVR 融合方法统一成一个 token 级 policy-gradient 视角，归成两类范式，然后用一个「朴素到没人认真试过」的 baseline —— **OPD-then-RL** —— 把它们全部打过去。机制解释是一句很干净的话：

> **OPD 扩张学生对「teacher 支持的解」的覆盖（pass@k），RL 在这个覆盖内部做锐化（pass@1）；联合优化把这两个角色混在一起，于是互相干扰。**

📌 **对本仓库而言，这篇的位置很特殊 —— 它正面撞上 [OPDVR](../opdvr/analysis.md)。** OPDVR 提出的恰恰是一个 **joint 的门控组合**，而本文把「joint」整类拿来做对照组并宣称 sequential 更好。两篇同期（2608.24696 vs 2609.04108）、**互不引用**。详见 [§7](#7-与-opdvr--opsa-的关系两篇同期互不引用)。

⚠️ **但先划清结论的适用范围**，这是读这篇最重要的一件事：

- **Logic 任务上它赢得非常干净** —— K&K 的 pass@1 从次优的 53.7 拉到 **92.6**，logic 平均领先其它组合方法 **11.7–26.7 分**。
- **Math 任务上它基本是平手** —— 论文自己的附录 bootstrap 检验（Table 7）显示：**相对纯 OPD 的差异不显著**（p@1 的 95% CI 是 `[−0.47, +2.13]`，含 0），相对 SRPO、KDRL-mask 也不显著；9 个对照里只有 6 个显著。
- **而摘要写的是** *"consistently outperforms pure OPD, pure RLVR, and all such joint baselines **across logic and math reasoning benchmarks**"* —— **「across ... math」这半句与它自己的 Table 7 对不上。**

![Figure 1: pass@1 vs pass@32，各后训练算法](./figures/fig1_teaser.png)

> **Fig 1**：横轴 pass@1、纵轴 pass@32，每个点是一个后训练算法。图例按本文的分类分成三组：**Pure methods**（OPD 灰菱形 / GRPO 灰叉）、**Weighted-additive**（KDRL、KDRL-Masking、HDPO、SRPO，蓝色系）、**Teacher-modulated**（TRRD、RLSD，绿色系），以及**红星 OPD-then-RL**。
>
> 红星在 **(≈80.6, ≈98.3)**，右上角孤零零一个；其余方法全挤在 pass@1 53–63 / pass@32 93–96 的一小团里；GRPO 单独掉在 (≈49, ≈68)。
>
> 🔴 **但看标题**：**"Avg across logic reasoning tasks"** —— **这张 teaser 只画了 logic 那一半**，也就是它赢 26.7 分的那一半。math 的对应图不存在。

---

## 2. 统一视角：所有组合方法只差在怎么混合两个信号

论文先把所有方法写成同一个 token 级 policy-gradient 更新（省略 importance ratio 与 PPO clipping）：

$$
\nabla_\theta J^{\mathrm{alg}}(\theta) = \mathbb{E}\left[\sum_{i,t} A_t^{\mathrm{alg},(i)}\, \nabla_\theta \log \pi_\theta\big(y_t^{(i)} \mid h_t^{(i)}\big)\right]
$$

两个纯方法分别是 `A^GRPO = Â^(i)`（轨迹级 verifier advantage）和 `A^OPD = d_t^(i)`（token 级蒸馏 advantage）。**于是所有组合方法的差别，只在于怎么把 `Â` 和 `d_t` 混成一个 `A_t`。**

### 范式 I：Weighted-Additive（加权相加）

$$
A_t^{\mathrm{add},(i)} = w_R^{(i,t)}\,\hat{A}^{(i)} + w_T^{(i,t)}\, d_t^{(i)}
$$

`w_R, w_T ≥ 0`。📌 **关键性质：`A_add` 的符号可以相对 `Â` 翻转** —— 当 teacher 与 verifier 意见相左时，teacher 项能把整个更新方向掰过来。

| 方法 | 做法 |
|---|---|
| **KDRL**（Xu et al. 2025） | 规范实例：每个 token 都加一个常数权重的蒸馏项 |
| **KDRL-mask** | **在正确的 rollout 上丢掉蒸馏项**，只在错误 rollout 上加 |
| **SRPO**† | 按正确性在两个信号之间**切换**：对了用 reward advantage，错了用蒸馏项 |
| **HDPO**† | **只在「一组 rollout 全错」时**才激活 teacher |

### 范式 II：Teacher-Modulated（教师只调幅度）

既然加性项会掰方向，那就改成用一个**恒正、带 trust-region clip 的因子**去缩放 reward advantage：

$$
A_t^{\mathrm{mod},(i)} = m\big(d_t^{(i)}\big)\cdot \hat{A}^{(i)}, \qquad m\big(d_t^{(i)}\big) > 0
$$

于是 `sign(A_mod) = sign(Â)` —— **`d_t` 退化成 token 级的 credit assignment，而不是一个竞争性的目标**：verifier 与 teacher 一致的地方放大，不一致的地方缩小。

| 方法 | 做法 |
|---|---|
| **TRRD**（Zhang et al. 2026b） | 把 teacher 信号折进 importance ratio 当 trust-region 正则，抑制偏离 teacher 的更新 |
| **RLSD**†（Yang et al. 2026） | 直接用一个 clip 过的 teacher 因子缩放 advantage 的幅度 |

> † 标记的方法原本是在 self-distillation 设定下提出的，本文沿用其组合规则但把 teacher 换成外部模型。

### 本文的替代方案：OPD-then-RL

$$
A_t^{\mathrm{alg},(i)} =
\begin{cases}
d_t^{(i)}, & \text{training step} \le S \\[4pt]
\hat{A}^{(i)}, & \text{otherwise}
\end{cases}
$$

**就是一个硬切换**，`S = 60` 步。没有新超参（除了 `S`），没有权重需要调。

📌 **另外还有一个中间形态做对照**：**KDRL-Annealing** —— 把 KDRL 的 `β` 从 0.2 线性退火到 0.002，相当于「软切换版的 OPD-then-RL」。论文特意把它放进来，用于区分「分阶段」和「硬切换」哪个是关键。

---

## 3. 实验设置

| 项 | 值 |
|---|---|
| **Student** | **Qwen3-1.7B-Base** |
| **Teacher** | **Qwen3-8B** |
| **Logic 任务** | ReasoningGym 的 **Knights & Knaves (K&K)**、**Zebra Puzzles**、**Countdown**，各自独立训练与评测 —— **teacher 在这类任务上先验暴露有限** |
| **Math 任务** | 训练用 **DeepMath-103K**，评测 **MATH500 / AMC23 / AIME24 / AIME25** —— **teacher 在数学上被重度后训练过** |
| **指标** | **pass@1（avg@32）** 与 **pass@32** |
| **切换点** | `S = 60` 步（消融见 §5.3） |
| **公平性** | 📌 **"All methods are trained under the same total step budget for fair comparison."** |
| **调参** | weighted-additive 对 `β` 高度敏感，**按任务做了针对性消融取最优 `β`**（Fig 6） |

📌 **两个任务族的选择是有意设计的**：论文明说选 K&K（**OPD 强于 GRPO**）和 Countdown（**GRPO 强于 OPD**）两种情况，*"ensuring our findings hold regardless of which stage is the stronger contributor"*。**这一点做得很规范。**

---

## 4. 结果

![Table 2: Logic 与 Math 推理任务的完整结果](./figures/tab2_main.png)

**Logic 平均（pass@1 / pass@32）**：

| 方法 | p@1 | p@32 |
|---|---|---|
| Student (Qwen3-1.7B-Base) | 1.9 | 33.0 |
| *Teacher (Qwen3-8B)* | *59.9* | *96.2* |
| GRPO | 49.4 | 68.1 |
| OPD | 53.9 | 94.9 |
| KDRL | 62.8 | 92.4 |
| SRPO | 56.0 | 94.6 |
| TRRD | 58.6 | 95.6 |
| KDRL-Annealing | 68.9 | 95.4 |
| **OPD-then-RL** | **80.6** | **98.3** |

📌 **两个很硬的读数**：① **OPD-then-RL 的 logic 平均 pass@1 (80.6) 超过了 teacher 的 59.9** —— 学生显著超过老师；② **pass@32 98.3 也超过 teacher 的 96.2**。

**Math 平均**：

| 方法 | p@1 | p@32 |
|---|---|---|
| *Teacher (Qwen3-8B)* | *50.1* | *77.1* |
| OPD | 31.0 | 55.9 |
| SRPO | 31.6 | 56.5 |
| **TRRD** | 30.1 | **58.9** |
| **KDRL** | 29.0 | **58.7** |
| **OPD-then-RL** | **31.8** | 58.5 |

⚠️ **这里的形势完全不同**：
- **pass@1 31.8 确实是最高**，但只比 SRPO 的 31.6 高 0.2、比纯 OPD 的 31.0 高 0.8；
- 🔴 **pass@32 的 58.5 排第三** —— **TRRD 58.9、KDRL 58.7 都更高**（论文表格自己把 TRRD 的 58.9 加粗了，诚实）；
- **所有方法离 teacher 的 50.1 / 77.1 都还很远** —— 数学上没有任何方法接近老师。

**逐列看 math 的子任务，OPD-then-RL 还输了三处**：AMC23 的 p@1（39.2 < SRPO 40.3）、AMC23 的 p@32（85.0 < KDRL 87.5）、AIME24 的 p@1（10.1 < SRPO 10.2）。

### 4.1 论文自己的显著性检验（Table 7，附录）—— 全文最该读的一张表

**这篇做了 paired problem-level bootstrap，给出 95% 置信区间**，在这一路文献里算是罕见的严谨。math 上「OPD-then-RL 减去对照方法」的 CI：

| 对照 | p@1 的 95% CI | p@32 的 95% CI |
|---|---|---|
| **OPD** | `[−0.47, +2.13]` | `[−1.58, +7.10]` |
| **SRPO** | `[−0.81, +1.21]` | `[−2.09, +6.45]` |
| **KDRL-mask** | `[−0.06, +4.77]` | `[−3.10, +7.27]` |
| TRRD | `[+0.54, +2.97]` ✳ | `[−5.12, +4.26]` |
| RLSD | `[+0.57, +3.01]` ✳ | `[−0.70, +8.21]` |
| HDPO | `[+0.74, +5.43]` ✳ | `[−2.52, +9.08]` |
| KDRL | `[+0.51, +5.25]` ✳ | `[−5.75, +5.34]` |
| GRPO | `[+1.06, +6.03]` ✳ | `[−3.05, +8.94]` |
| KDRL-Annealing | `[+0.81, +2.89]` ✳ | `[+0.19, +9.93]` ✳ |

（✳ = 区间不含 0，即显著）

🔴 **三条结论**：
1. **在 math 上，OPD-then-RL 相对纯 OPD 没有统计显著的优势**（p@1 与 p@32 的 CI 都含 0）。**也就是说：数学任务上，后面那段 RL 加不加，测不出差别。**
2. **相对 SRPO 和 KDRL-mask 也不显著。** 论文正文诚实地写了 *"a statistical tie with the three strongest"*。
3. **pass@32 上只有相对 KDRL-Annealing 是显著的**，对其余 8 个方法全部是平手。

📌 **这张表同时也支持了论文的另一句话**：正文说 *"On pass@32, **no method is significantly ahead of** OPD-then-RL on either task family"* —— 措辞是「没有人显著领先它」而不是「它领先所有人」，**与 CI 一致，是准确的**。

### 4.2 机制：覆盖扩张 vs 分布锐化

![Figure 2: pass@k 随 k 的变化，以及随难度的变化](./figures/fig2_passk.png)

> **Fig 2 三面板**（图例：灰圆 Base / 黑方 RL / 绿 OPD / **红 OPD-then-RL** / 紫 KDRL / 粉 TRRD）：
>
> - **左（Countdown）**：横轴 k = 1→128（log 刻度），纵轴 pass@k。**红线全程最高**（k=1 时 ≈78，k=128 时 ≈94）；灰色 Base 最低（k=1 近 0 → k=128 约 35）。
> - **中（K&K）**：红线从 k=1 的 ≈92 一路贴近 100；**黑色 RL 线全程被压在 27→40 的低位**，与其它方法差距极大。
> - **右（K&K OOD）**：横轴换成难度 `n_people`（6→14），**实线 = k=128，虚线 = k=1**。红色实线在全部难度上稳定在 100；绿色（OPD）实线从 100 降到 ≈83；红色虚线从 ≈93 降到 ≈40。
>
> 📌 **论文的机制论证就在这张图上**：*"OPD lifts the entire pass@k curve above the base model on both tasks, with the largest gain at large k. Looking at the pass@128−pass@1 gap, **OPD widens it by enlarging the set of solvable problems, while RL narrows it by redistributing probability mass toward pass@1**."*
>
> **翻译成一句话：OPD 把「能解的题」的集合做大（抬 pass@k），RL 把概率质量往对的答案上挪（抬 pass@1、压缩 pass@k 与 pass@1 的差）。顺序做，两件事各自到位；混着做，互相抵消。**

**论文还给了两类 joint 方法各自的失效方式（这段很有信息量）**：

| 范式 | 在 logic 上 | 在 math 上 |
|---|---|---|
| **Weighted-additive** | 把 RL 梯度直接注入蒸馏训练，推向 reward 驱动的锐化：pass@1 比纯 OPD **高 4.8**，但 **pass@32 掉 2.0** | teacher 的蒸馏信号更可靠，于是这种对冲**反而把 pass@1 压到纯 OPD 之下（−1.2）** |
| **Teacher-modulated** | 只放大/抑制、不改方向，**性能贴着纯 OPD**：保住了 OPD 的高 pass@32，pass@1 小幅改善 | **对 OPD 的高 pass@1 的破坏也比 weighted-additive 小** |

📌 **「Teacher-modulated 会把性能钉在纯 OPD 附近」这个实测结论，请记住它 —— [§7](#7-与-opdvr--opsa-的关系两篇同期互不引用) 会用到。**

---

## 5. 实用结论

### 5.1 切换点 `S` 怎么定

论文用三个 checkpoint（step 20 / 60 / 100）做了对照，结论很简洁：

> **「切换时刻的 OPD 验证分数，基本决定了后续 RL 能到哪」**（*"the OPD score at the switch is what propagates into RL"*）。

- **K&K 与 Zebra**：OPD 到 step 100 还在涨 → **切得越晚越好**（100 > 60 > 20）。
- **Countdown**：OPD 的验证分在 step 20 就饱和 → **三个切换点最后收敛到几乎一样**。

📌 **可操作的判据**：**盯着 OPD 的验证曲线，在它「快速上升期结束」的地方切。** 论文主实验取 60 就是这个位置 —— 拿到了大部分 OPD 收益，又给 RL 留了足够预算。**继续把 OPD 训好能抬高天花板，提前切则把上限锁死。**

### 5.2 OPD 比 SFT 是更好的 RL 冷启动

同一个 teacher 下，论文对比了 OPD 与它的 off-policy 对应物（SFT on teacher traces），结论是 **OPD 提供更强的冷启动**。理由与 GKD 那条线一致：SFT 学的是 teacher 分布上的轨迹，而推理时学生从自己的分布采样，存在 exposure bias。

---

## 6. 争议与权衡

### 6.1 摘要的适用范围比正文宽

| 出处 | 措辞 | 核对 |
|---|---|---|
| **摘要** | *"consistently outperforms pure OPD, pure RLVR, and all such joint baselines **across logic and math reasoning benchmarks**"* | 🔴 **math 上不成立** —— Table 7 显示相对纯 OPD、SRPO、KDRL-mask 都是统计平手；Math Avg 的 pass@32 还排第三（58.5 < TRRD 58.9 < KDRL 58.7） |
| **§7 结论** | *"matches or surpasses every joint variant ... **with no variant ahead of it on either axis**"* | ⚠️ 如果「两个轴」指 Fig 1 的 pass@1/pass@32 **且只算 logic**，成立；按 math 的 Avg p@32 读则不成立 |
| **§4.2 正文** | *"attains the highest average pass@1 on both task families ... **On math, where all methods sit closer together**, ... significant against six of the nine ... and a **statistical tie with the three strongest**"* | ✅ **准确**，与 Table 7 完全一致 |
| **§4.2 正文** | *"On pass@32, **no method is significantly ahead of** OPD-then-RL on either task family"* | ✅ **准确**（是"没人显著领先它"，不是"它领先所有人"） |

📌 **所以这篇的问题不是造假，而是「摘要 + teaser 只呈现赢得漂亮的那一半」**：Fig 1 明写 "Avg across logic reasoning tasks"，而 math 的同款图不存在。**正文和附录都是诚实的。**

### 6.2 结论对「teacher 与任务的相对强弱」高度敏感 —— 而这恰恰是最有价值的一条

论文自己把两个任务族按 **teacher 的先验暴露程度**设计：logic 是"teacher 没怎么见过"，math 是"teacher 被重度后训练过"。结果：

- **teacher 弱先验（logic）→ 分阶段大赢**（+11.7~26.7）
- **teacher 强先验（math）→ 分阶段与纯 OPD 打平**

📌 **反过来读，这给出了一条比论文标题更有用的判据**：**当 teacher 在目标任务上已经很强时，OPD 本身就吃掉了大部分收益，后面那段 RL 加不加区别不大；当 teacher 对任务相对陌生时，OPD 只能扩覆盖、必须靠 RL 去锐化，这时分阶段的价值才显现。** 论文没有把这条提炼成结论，但它是全文数据支持最强的模式。

### 6.3 其它

- ⚠️ **只有一个 student/teacher 组合**（Qwen3-1.7B-Base ← Qwen3-8B）。规模、家族、能力差距都没有扫描。论文在 Limitations 里承认了 teacher-student 配置的局限（只覆盖"外部更强 teacher"这一种），并把 task-specific teacher、多 teacher、self-distillation 列为 future work。
- ⚠️ **`S = 60` 的消融只有三个点**（20 / 60 / 100），且是在 logic 任务上做的。math 上切换点的敏感性没测。
- ⚠️ **pass@k 作为主指标有一个已知的解释风险**：大 k 的 pass@k 与**采样多样性/熵**强相关，而 OPD（reverse KL）和 RL 对熵的影响方向不同。论文用 pass@k 的变化来论证"覆盖扩张"，但**没有把"覆盖真的变大"与"只是熵变高、碰运气碰到了"分离开**。Fig 2 右图（OOD 难度）部分缓解了这个担心，因为难度上升时红线仍然稳在 100。
- ⚠️ **weighted-additive 的 `β` 按任务调到最优，而 OPD-then-RL 的 `S` 用的是固定 60**。这个方向对 baseline 有利（不是对自己有利），**属于公平性上的加分项**。
- ✅ **总步数预算对齐**（"All methods are trained under the same total step budget"），这是这类对照最容易翻车的地方，论文明确写了。
- ✅ **有 Limitations 章节、有 bootstrap 置信区间、有致谢与资助声明。** 在仓库的 OPD 那一簇里，**这是方法学最规范的一篇**。

---

## 7. 与 OPDVR / OPSA 的关系（三篇同期，互不引用）

### 7.1 🔴 OPDVR 恰好属于本文判定为「较弱」的那一类

**[OPDVR](../opdvr/analysis.md)（arXiv:2608.24696）的核心机制是**：

$$
R_{\mathrm{OPDVR}}(o_t) = \mathrm{sign}(R)\cdot \mathrm{ReLU}\left(R \cdot \log \frac{\pi_T(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})}\right)
$$

用 OPDVR 笔记里的话说就是 **「verifier 决定方向，teacher 决定幅度」**。

**而本文范式 II（Teacher-Modulated）的定义是**：

$$
A_t^{\mathrm{mod},(i)} = m\big(d_t^{(i)}\big)\cdot \hat{A}^{(i)},\qquad m > 0,\quad \mathrm{sign}(A^{\mathrm{mod}}) = \mathrm{sign}(\hat{A})
$$

原话：*"uses the teacher signal only to **modulate the magnitude** of the RLVR advantage, leaving the **sign determined entirely by the verifiable reward**."*

📌 **这两个是同一个东西。** OPDVR 的 `ReLU(·)` 就是那个非负的 `m`，`sign(R)` 就是 `sign(Â)`。

⚠️ **但要说准一处差别**：本文写的是 `m > 0`（严格正），而 **OPDVR 的 ReLU 会取到 0** —— 它把约一半的 token 梯度直接清零，而 TRRD / RLSD 只是缩放。**所以 OPDVR 是范式 II 的极限情形（允许调制因子归零），不是被本文直接测过的那两个实例。** 本文测的是 TRRD 和 RLSD，**没有引用也没有测 OPDVR**。

### 7.2 两篇独立地得到了同一个结论

**这是这组对照里最值得记的一点。**

| | 路径 | 结论 |
|---|---|---|
| **我在 [OPDVR 笔记](../opdvr/analysis.md) 里的推导** | 从门控结构推不动点：正确轨迹上 `π_θ ≥ π_T`、错误轨迹上 `π_θ ≤ π_T` 构成一个**单向棘轮区域**，梯度在 `π_θ = π_T` 处归零 → **teacher 是硬上界** | OPDVR 在三张主表里平均分**从未超过 teacher** |
| **本文的实测** | 直接测 teacher-modulated 这一类：*"keeping the performance **close to pure OPD**"* —— 保住 OPD 的高 pass@32、pass@1 只小幅改善 | 范式 II **被钉在纯 OPD 附近** |

📌 **两条完全不同的路径（一条是从 loss 结构推、一条是跨方法实测）指向同一件事：teacher-modulated 这类设计的天花板就是 OPD/teacher 本身。** 这比任何单篇的自述都更有说服力。

🔴 **而最锋利的一条证据是「学生有没有超过老师」**：

| | 学生 vs teacher |
|---|---|
| **[OPDVR](../opdvr/analysis.md)** | 三张主表**平均分从未超过 teacher**（49.1 vs 50.4、22.8 vs 30.9、49.4 vs 50.4） |
| **本文 OPD-then-RL（logic）** | **pass@1 80.6 vs teacher 59.9、pass@32 98.3 vs 96.2 —— 两个轴都反超** |
| **本文 OPD-then-RL（math）** | 31.8 / 58.5 vs teacher 50.1 / 77.1 —— **远未反超** |

⚠️ **注意 math 那行**：分阶段在数学上同样超不过 teacher。**所以"反超 teacher"这件事不是 sequential 这个形式带来的，而是"teacher 在 logic 上本来就不强"带来的**（见 [§6.2](#62-结论对teacher-与任务的相对强弱高度敏感--而这恰恰是最有价值的一条)）。**但它至少证明了：撤掉 teacher 之后的纯 RL 阶段，确实能把学生推到 teacher 之上 —— 这是 OPDVR 那种"teacher 始终在 loss 里"的结构做不到的。**

⚠️ **而本文给出的出路正好是 OPDVR 结构上做不到的那件事**：**换阶段，而不是换权重。** OPD 阶段把覆盖做大，然后**完全撤掉 teacher**、让纯 RL 去锐化 —— 一旦 teacher 不在 loss 里，"teacher 是上界"这个约束就自动解除了。对照 [OPDVR 笔记 Q2](../opdvr/analysis.md) 里我给的另一条出路（用 ExOPD 式的 reward extrapolation 把幅度增益推到 1 以上），**本文这条更简单也更彻底**。

### 7.3 与 OPSA 的关系

[OPSA](../opsa/analysis.md)（arXiv:2608.31046）论证 **OPD 的收益来自"压制学生自采的低概率 token"而非 teacher 的具体监督**。把三篇摆在一起：

| | 对 OPD 的判断 | 处方 |
|---|---|---|
| **OPSA** | 收益来自负信号压低概率 token，teacher 可以整个换掉 | advantage 全为负、只训 logp 最低 20% 的 token |
| **OPDVR** | 收益来自 teacher，只是符号用错了 | ReLU 门让 verifier 定符号（= 本文的范式 II） |
| **本文** | OPD 负责**扩覆盖**、RL 负责**锐化**，两者角色不同不能混 | 分两段做 |

📌 **本文的"覆盖扩张 / 锐化"二分，给 OPSA 和 OPDVR 的分歧提供了一个可能的调和**：OPSA 观察的是 pass@1 类指标下 OPD 的行为，本文指出 **OPD 真正的贡献在 pass@k 那一侧**。如果成立，那么"OPD 不 distill"与"OPD 有用"可以同时为真 —— **OPD 的作用不是把 teacher 的知识搬过来，而是把学生的解分布摊开**。

⚠️ **这是我的综合，三篇论文都没这么写过**，而且三篇的指标口径不同（OPSA 用 Avg@32，本文用 pass@1/pass@32），**要当结论用需要在统一设定下重测**。

📌 **一个具体的开放实验**：把本文的 pass@k 分析直接套到 OPSA 的 OPSA-trained 模型上 —— 如果"OPD 扩覆盖"成立，那么 OPSA（零 teacher 监督、纯负信号）应该**也**能抬 pass@k。**这一个实验能同时检验三篇。**

---

## 8. 一句话总结

**把 OPD 和 RLVR 分两段做（OPD 训 60 步再切纯 GRPO），比现有所有「在同一个 step 里融合两者」的方法都好** —— 论文先把这些融合法统一成 token 级 advantage 的两类（**weighted-additive** 会掰翻更新方向、**teacher-modulated** 只调幅度不改符号），再用 pass@k 给出机制：**OPD 扩张"可解题集合"、RL 在其中重分配概率质量，混着优化会互相干扰**；logic 上领先其它组合方法 11.7–26.7 分且学生 pass@1 80.6 反超 teacher 的 59.9。⚠️ **但摘要的 "across logic and math" 与它自己的附录 bootstrap 对不上 —— math 上相对纯 OPD / SRPO / KDRL-mask 全是统计平手（相对纯 OPD 的 p@1 CI 是 [−0.47, +2.13]），Math Avg 的 pass@32 还排第三；Fig 1 那张 teaser 只画了 logic。** 📌 而真正可迁移的判据论文没提炼出来：**teacher 在目标任务上越强，分阶段的增量越小**。方法学上它是仓库 OPD 簇里最规范的一篇（步数预算对齐、bootstrap CI、有 Limitations）。

---

## 9. 在仓库 OPD 谱系里的位置

| 层次 | 笔记 | 与本文的关系 |
|---|---|---|
| **范式源头** | [gkd](../gkd/analysis.md)、[on_policy_distillation](../on_policy_distillation/blog_zh.md) | 本文引用二者作为 OPD 的来源；§5.2 的"OPD 比 SFT 冷启动好"复用了 GKD 的 exposure-bias 论证 |
| **OPD 的机制批判** | [opsa](../opsa/analysis.md) | ⚠️ **未引用**。三篇对 OPD 收益来源给出三种解释（见 §7.3） |
| **joint 组合（本文的对照组）** | [opdvr](../opdvr/analysis.md) | 🔴 **未引用，但 OPDVR 正属于本文的范式 II**（见 §7.1）。两篇同期、结论相反 |
| **teacher 的来源** | [opsd](../opsd/analysis.md) | 本文 Limitations 明确把 self-distillation 列为未覆盖的配置 |
| **扩散侧同构** | [danceopd](../../image_generation/danceopd/analysis.md)、[flow_opd](../../image_generation/flow_opd/analysis.md)、[diffusion_nft](../../image_generation/diffusion_nft/analysis.md)、[rvm](../../video_generation/rvm/analysis.md) | 连续域上的对应问题。📌 **本文的"分阶段而非混合"在扩散侧还没有对应工作** —— 那边的 RL 后训练（如 [SenseNova-U1.5](../../multimodal/sensenova_u15/analysis.md) 的四专家 RL → OPD 蒸馏）其实**已经是分阶段的**，只是顺序相反（先 RL 再蒸馏） |

📌 **最后这条对比值得单独记**：本文主张 **OPD → RL**（先扩覆盖、后锐化），而 [SenseNova-U1.5](../../multimodal/sensenova_u15/analysis.md) 的后训练是 **RL（四个专家）→ OPD（蒸馏合并）**。**两者顺序正相反，但目的不同** —— 后者的 OPD 是用来「把多个专家合并成一个模型」，不是用来扩覆盖的。**「OPD 放在 RL 前面还是后面」在两个域里各有一种做法，而没有人对比过。**
