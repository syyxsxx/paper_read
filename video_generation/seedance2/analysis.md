# Seedance 2.0: Advancing Video Generation for World Complexity

**论文**: Seedance 2.0: Advancing Video Generation for World Complexity  
**机构**: ByteDance Seed  
**arXiv**: [2604.14148](https://arxiv.org/abs/2604.14148) (2026-04-15)  
**代码**: 无公开代码（产品技术报告）  
**模型访问**: Doubao / Jimeng / Volcano Engine (`doubao-seedance-2-0-260128`)

---

## 1. 一句话定位

**Seedance 2.0 是什么**：ByteDance Seed 的统一多模态音视频联合生成模型。支持 4 路输入模态（文本/图像/音频/视频），原生输出 480p 和 720p 的 4–15 秒视频，同步生成双声道空间音频（BGM + 环境音 + 人声三轨）。

**核心主张**：Arena.AI T2V #1（Elo 1450）+ I2V #1（Elo 1449）@ 720p；横扫 SeedVideoBench 2.0 全维度 T2V/I2V/R2V；R2V 任务类型支持 20/22 种（业界最全），含竞品全不支持的视觉特效引用和视频续写/延伸。

**论文性质**：产品技术报告，**不披露架构或训练细节**。内容重心在评测框架（SeedVideoBench 2.0）和横向竞品对比。

---

## 2. 与前作的关系

### Seedance 系列演进

```
Seedance 1.0 (2024, CVPR'24 MakePixelsDance)
    ↓
Seedance 1.5 Pro (arXiv:2512.13507, 2025-12)
    Native 音视频联合生成（单一流程：T2V/I2V + 同步音频）
    ↓
Seedance 2.0 (arXiv:2604.14148, 2026-04)
    统一多模态框架：T2V / I2V / R2V 全打通
    新增：R2V（参考视频生成）、视频编辑、续写/延伸
    新增：视觉特效引用、创意引用、组合任务
```

### 与竞品关系

- **T2V/I2V 对标**：Kling 3.0、Sora 2 Pro、Veo 3.1、Wan 2.6
- **R2V 对标**：Kling 3 Omni、Kling O1、Vidu Q2 Pro
- Seedance 2.0 R2V 独家任务：视觉特效/创意引用（3 变体）、视频续写（2 变体）、视频延伸（2 变体），共 7 个任务竞品全不支持

---

## 3. 核心能力体系

### 3.1 多模态输入框架

论文不披露模型架构，但给出了如下对外能力边界：

| 维度 | 规格 |
|------|------|
| 输入模态 | 文本、图像、音频、视频（4 路） |
| 参考上限 | 视频 ≤3 clips，图像 ≤9 张，音频 ≤3 clips |
| 输出分辨率 | 480p 和 720p（原生） |
| 输出时长 | 4–15 秒 |
| 加速版本 | Seedance 2.0 Fast（低延迟变体） |

### 3.2 R2V 任务分类体系

Seedance 2.0 最核心的新增能力是 **Reference-to-Video（R2V）** 全套任务，按任务组织如下：

**Subject Reference（主体引用）**
- Image Reference / Video Reference / Audio-Visual Reference / Audio+Image Reference

**Motion Reference（动作引用）**
- Video Motion Reference
- Video Motion Reference + Image Reference（组合）
- Video Motion Reference + First Frame（组合）

**Visual Effects / Creative Reference（视觉特效/创意引用）← 仅 Seedance 2.0 支持**
- VFX Reference / VFX + Image Reference / VFX + First Frame

**Style Reference（风格引用）**
- Style Image / Style Image+Subject / Style Video / Style Video+Subject

**Video Editing（视频编辑）**
- Video Instruction Editing / Video Reference Image Editing

**Continuation / Extension（续写/延伸）← 仅 Seedance 2.0 支持**
- Continuation / Continuation + Subject Image
- Extension（任意上传视频）/ Extension + Subject Image

**覆盖统计**：Seedance 2.0 支持 20/22 种任务；剩余 2 类（Subject Audio-Visual+Audio Reference、Video Audio Editing）当前无模型支持。

### 3.3 音频生成体系

- **双声道空间音频**：原生双耳（binaural）输出，提供空间感和立体分离
- **多轨同步**：BGM、环境音效、人声旁白三轨独立生成，精确对齐视觉节奏
- **语言覆盖**：英文、中文（普通话/方言/戏曲/综艺）、日/韩/印尼/葡/西班牙语等

### 3.4 基础生成能力亮点

- **运动建模**：人体动作自然度和时间一致性大幅提升，物理合规性强；竞速、格斗、舞蹈等大幅动作与细微表情均能生成
- **镜头语言**：支持专业级多镜头叙事、组合运镜、编辑节奏控制，I2V 新增第一/第三人称跟随 + 手持呼吸效果
- **指令遵循**：复杂长脚本精准解析；多风格（水彩/油画/工笔画/3D）与主体联合控制不破坏视觉一致性
- **视觉细节**：材质、光照、构图均衡；面部表情 + 眼神生动；服装道具设计精良

---

## 4. SeedVideoBench 2.0 评测框架

本节是产品报告的方法论核心，也是本文最主要的学术贡献。

### 4.1 框架设计

| 维度 | 方式 |
|------|------|
| 评分制度 | 1–5 分（视频/音频质量）；1–3 分（多模态任务遵循） |
| 双轨评测 | Objective（自动化流水线，如运动稳定性）+ Subjective（盲测专家打分） |
| 新增模块 | 多模态任务评测、叙事质量评测、多语言覆盖 |
| 专家来源 | 广告和游戏制作行业专业人员 |
| 真实性研究 | 独立实验让评估者区分 AI 生成与真实视频，结果反哺美学调优 |

### 4.2 叙事质量三维度（相较 SeedVideoBench 1.5 新增）

1. **Cinematographic Language（镜头语言）**：运镜是否支撑叙事，检查冗余覆盖、越轴（180°规则违反）、景别不匹配、节奏不均
2. **Plot Design（情节设计）**：模糊/简短 prompt 能否生成连贯且吸引人的内容
3. **Stylistic Aesthetics（风格美学）**：视觉整体感，含光照/构图/色调/人物服装道具协调性

### 4.3 多模态任务评测（SeedVideoBench 2.0 核心新增）

覆盖四组任务，量化"边界"——让用户无需反复试探就能知道模型支持什么：

| 组别 | 内容 |
|------|------|
| Reference tasks | 主体/动作/视觉特效/风格引用生成 |
| Editing tasks | 主体/风格/场景/音频内容编辑 |
| Extension tasks | 情节续写和无缝延伸（前向/后向时间轴） |
| Combination tasks | 真实工作流组合（如视频主体替换 + 参考图像） |

**Consistency** 双维度：参考对齐（Reference Alignment）+ 编辑区域外保留（Editing Consistency）

### 4.4 Arena.AI 排名（2026-04-08）

Arena（原 LMArena，UC Berkeley）通过真实用户 A/B 盲测产生 Elo 排行，比自动指标（FVD/CLIPScore）更接近真实使用偏好。

| 任务 | 模型 | Elo 分 | Rank Spread |
|------|------|--------|------------|
| T2V | Dreamina Seedance 2.0 720p | 1450 ±15 | 1↔1 |
| T2V | veo-3.1-audio-1080p | ~1371 | #2 |
| I2V | Dreamina Seedance 2.0 720p | 1449 ±11 | 1↔1 |
| I2V | grok-imagine-video-720p | ~1420 | #2 |

T2V 领先第 2 名 79 Elo；I2V 领先 29 Elo。在 720p 下优胜 1080p 竞品，印证质量优势而非分辨率优势。

---

## 5. 关键评测结果

### 5.1 T2V 总评（Table 1，1–5 分）

| 模型 | 运动质量 | 视频提示遵循 | 美学 | 音频质量 | 音画同步 | 音频提示遵循 |
|------|---------|------------|------|---------|---------|------------|
| Kling 3.0 | 3.10 | 2.78 | 3.36 | 2.74 | 2.78 | 2.54 |
| Sora 2 Pro | 2.69 | 2.81 | 2.82 | 2.76 | 2.65 | 2.92 |
| Veo 3.1 | 2.73 | 2.59 | 2.88 | 2.62 | 2.54 | 2.24 |
| Seedance 1.5 | 2.39 | 2.59 | 3.19 | 2.88 | 2.91 | 2.69 |
| **Seedance 2.0** | **3.75** | **3.43** | **3.67** | **3.63** | **3.75** | **3.56** |

- Seedance 2.0 是唯一所有维度 >3.4 的模型；相比 Seedance 1.5 平均提升 +0.86，运动质量最大（+1.36）
- 可用率（≥3 分）：运动质量 97.55%，所有维度均 >83%
- 满意率（≥4 分）：所有维度 >51%（竞品单维最高 43%）
- 音频是竞品最薄弱方向：多数竞品低于 2.9，Seedance 2.0 全超 3.5

**T2V 细粒度亮点（30 类运动质量）**：

| 最高分类别 | Seedance 2.0 | 次高 |
|-----------|------------|------|
| Multi-Entity Feature Match | 4.43 | Kling 3.0: 3.43 |
| Framing / Composition | 4.25 | Kling 3.0: 3.75 |
| Editing Rhythm | 4.21 | Sora 2: 2.93 |
| Abstract Challenges | 4.00 | Kling 3.0: 3.57 |
| Emotion & Expression | 4.00 | Kling 3.0: 3.64 |

相比 Seedance 1.5 提升最大：Physical Feedback（1.69→3.46）、Natural Phenomena（2.00→3.78）、Intense Sports Motion（2.00→3.79）。

### 5.2 I2V 总评（Table 9，1–5 分）

| 模型 | 运动质量 | 视频提示遵循 | 图像保留 | 音频质量 | 音画同步 | 音频提示遵循 |
|------|---------|------------|---------|---------|---------|------------|
| Kling 3.0 | 2.80 | 2.78 | 3.18 | 2.89 | 2.83 | 2.85 |
| Veo 3.1 | 2.65 | 2.87 | 2.69 | 2.68 | 2.69 | 2.79 |
| Wan 2.6 | 2.32 | 2.74 | 2.61 | 2.20 | 2.18 | 2.55 |
| **Seedance 2.0** | **3.35** | **3.46** | **3.31** | **3.61** | **3.54** | **3.70** |

- 音频维度竞品差距最大：Kling 2.6 音频质量仅 2.21，Seedance 2.0 达 3.61；大多数竞品音频可用率低于 28%
- I2V 运动满意率 43.88%，是 Kling 3.0 (12.00%) 的 3.65 倍
- 图像保留最激烈：Kling 3.0 仅落后 0.13 分（3.18 vs 3.31）

**I2V 细粒度亮点**：
- 复杂指令：Compound Multi-Instructions MQ 4.00（Kling 3.0: 2.88）
- 复杂运动：Sports VPF 3.93、Micro-Expression VPF 4.00、Combat Visual Effects MQ 3.63（Kling 3.0: 2.25）
- 复杂交互：Cross-Type Interaction VPF 4.00

### 5.3 R2V 总评（Table 24）

评分制度不同：Multimodal Task Following 和 Prompt Following 用 1–3 分制，其余 1–5 分。

| 模型 | 多模态任务遵循 | 编辑一致性 | 参考对齐 | 运动质量 | 提示遵循 |
|------|-------------|----------|---------|---------|---------|
| Kling 3.0 | 2.32 | 3.37 | 2.37 | 2.36 | 1.95 |
| Kling O1 | 2.30 | 2.89 | 2.32 | 2.30 | 1.95 |
| Vidu Q2 Pro | 2.13 | 2.29 | 1.79 | 2.38 | 2.08 |
| **Seedance 2.0** | **2.50** | **3.54** | **3.03** | **3.24** | **2.52** |

- 运动质量领先差距最大（0.86–0.94 分）
- 视频编辑：Kling O1 任务遵循略高（2.29 vs 2.20），但 Seedance 2.0 编辑一致性（3.75）和参考对齐（3.79）反超
- **最弱项——视频延伸**：Seedance 2.0 任务遵循 1.93（31.82% 达 3 分）vs Veo 3.1 的 2.78（88.89%）；但注意 Veo 3.1 只能延伸自己生成的视频，Seedance 2.0 支持任意上传视频（任务更难）

---

## 6. 争议/权衡

### 6.1 产品报告的局限性

本文不披露任何架构/训练细节：
- 评测集由 ByteDance 内部构建，数据分布和打分标准未完全公开，存在内部评测偏差风险
- "业内专家"评分存在选择性偏差
- Arena.AI 是第三方 Elo 排名，对比最为可信

### 6.2 已知短板汇总

| 短板 | 具体表现 |
|------|---------|
| 视频延伸 | 任务遵循 1.93（31.82% 达到 3 分），Veo 3.1 达 2.78（88.89%） |
| 多人场景 | 多说话人唇形同步有错误 |
| 多主体一致性 | 多主体场景主体保留不稳定 |
| 高频视觉噪声 | 轻微形变伪影 |
| 文字还原 | 文字还原精度有优化空间 |
| 复杂编辑 | 复杂编辑任务未响应、或修改非目标区域 |

### 6.3 任务范围不对称问题

R2V 评测公正性存疑：
- 视频延伸：Veo 3.1 仅延伸自生成视频（内部一致性有保证），Seedance 2.0 接受任意上传视频（更难）
- 任务支持数差异（Kling 3 Omni 9/22 vs Seedance 2.0 20/22）导致综合 R2V 排名受覆盖度影响，而非纯质量差异

### 6.4 与本方向其他技术论文的关系

| 维度 | Helios / LongLive 2.0 | Seedance 2.0 |
|------|----------------------|-------------|
| 论文类型 | 技术论文（披露架构+训练） | 产品报告（不披露） |
| 核心主张 | 推理速度/效率（14B@19.5FPS） | 能力广度（R2V + 音频） |
| 开源情况 | 部分开源 | 仅产品访问 |
| 讨论价值 | 算法设计、工程优化 | 评测框架设计、能力边界 |

Seedance 系列走"全功能商业化"路线，是否使用 causal AR/few-step distillation 均未公开。

---

## 7. 关键数据速查

### T2V 运动质量 Top 5 类别（Seedance 2.0）

| 类别 | 分数 |
|------|------|
| Multi-Entity Feature Match | 4.43 |
| Framing / Composition | 4.25 |
| Editing Rhythm | 4.21 |
| Abstract Challenges | 4.00 |
| Emotion & Expression | 4.00 |

### R2V 任务支持矩阵

| 任务组 | Seedance 2.0 | Kling 3 Omni | Vidu Q2 Pro | Kling O1 |
|--------|-------------|-------------|------------|---------|
| Subject Reference | 4/4 | 4/4 | 4/4 | 1/4 |
| Motion Reference | 3/3 | 3/3 | 3/3 | 3/3 |
| VFX/Creative Reference | **3/3** | 0/3 | 0/3 | 0/3 |
| Style Reference | 4/4 | 0/4 | 4/4 | 2/4 |
| Video Editing | 2/2 | 2/2 | 2/2 | 2/2 |
| Continuation/Extension | **4/4** | 0/4 | 0/4 | 0/4 |
| **合计** | **20/22** | **9/22** | **13/22** | **10/22** |

### Arena.AI 关键对比

| 任务 | Seedance 2.0 Elo | 第二名 Elo | 差值 |
|------|----------------|---------|------|
| T2V (720p) | 1450 ±15 | ~1371 (Veo 3.1 @ 1080p) | +79 |
| I2V (720p) | 1449 ±11 | ~1420 (Grok Video) | +29 |

---

## 8. 一句话总结

Seedance 2.0 把竞品普遍缺席的 R2V 能力（视觉特效引用 + 续写/延伸 7 个独家任务，共覆盖 20/22 种类型）和原生双声道多轨音频打包进统一多模态框架，在 Arena.AI T2V/I2V 双榜拿第一；代价是技术细节完全不公开，视频延伸任务遵循也还明显弱于 Veo 3.1（但测的是更难的开放上传场景）。
