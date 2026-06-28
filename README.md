# Paper Read

个人论文阅读笔记仓库。每篇论文一份深度解读,聚焦"算法是什么、为什么这样设计、跟前作差在哪、代码怎么实现"。

## 目录结构

```
paper_read/
├── README.md                       ← 本文件,所有论文的总索引
├── <方向>/                          ← 按研究方向分类
│   ├── README.md                   ← 该方向的论文谱系图 + 列表
│   └── <论文短名>/                  ← 每篇论文独立目录
│       ├── analysis.md             ← 深度解读(主文件)
│       ├── refs.md                 ← 相关论文/链接/原文 PDF 路径(可选)
│       ├── code_notes.md           ← 代码导览(可选)
│       └── figures/                ← 图片素材(可选)
```

### 命名约定

- **方向目录**:小写 + 下划线,例如 `video_generation`、`llm`、`multimodal`
- **论文目录**:小写论文短名,例如 `longlive2`、`self_forcing`、`causvid`。同名论文有版本号时拼后缀(如 `dpo_v1` / `dpo_v2`)
- **主分析文件**:统一叫 `analysis.md`,方便脚本/批量索引

## 论文索引

| 方向 | 论文 | 简称 | 发表 | 链接 | 一句话 |
|------|------|------|------|------|--------|
| [video_generation](./video_generation/) | LongLive-2.0: An NVFP4 Parallel Infrastructure for Long Video Generation | [longlive2](./video_generation/longlive2/analysis.md) | NVIDIA, 2026 | [github](https://github.com/NVlabs/LongLive) | NVFP4 量化 + Balanced 序列并行,用真长视频 teacher-forcing 直接微调,绕过 CausVid/Self-Forcing 的 ODE Init + DMD 多阶段,辅以 LoRA 实现 4→2 步,Multi-Shot Sink 支持多镜头实时长视频 |
| [video_generation](./video_generation/) | SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer | [sana_streaming](./video_generation/sana_streaming/analysis.md) | NVIDIA, 2026 | [project](https://nvlabs.github.io/Sana/Streaming) | Hybrid DiT (GDN 线性 + Softmax window/sink) + Cycle-Reverse 正则化(用反向编辑 prompt 绕过"无长编辑对"难题)+ Triton GDN kernel + AutoML 风格 MPQ 量化搜索,RTX 5090 单卡跑 1280×704 V2V 编辑 24 FPS |
| [multimodal](./multimodal/) | Cosmos 3: Omnimodal World Models for Physical AI | [cosmos3](./multimodal/cosmos3/analysis.md) | NVIDIA, 2026 | [github](https://github.com/nvidia/cosmos) | NVIDIA 的 Physical AI 通用 backbone。MoT 双塔架构(Reasoner + Generator)统一 5 模态(语言/图/视/音/动作),3D MRoPE + 绝对时间调制对齐多模态时间轴,unified action representation 跨 5 种 embodiment。2048×GB200 训出来的 64B Super 模型,T2I/I2V 开源第一 + RoboArena 第一 |
| [video_generation](./video_generation/) | Helios: Real Real-Time Long Video Generation Model | [helios](./video_generation/helios/analysis.md) | ByteDance+PKU, 2026 | [project](https://pku-yuangroup.github.io/Helios-Page) | 第一个单卡 H100 跑到 19.5 FPS 的 14B 视频生成模型。不用 causal mask/KV-cache/量化/self-forcing,靠 Unified History Injection + Guidance Attention 保持双向推理,Multi-Term Memory Patchification(8×) + Pyramid UPC(2.3×)把 token 算力压到 1.3B 级,Relative RoPE + First-Frame Anchor + Frame-Aware Corrupt 替代 self-rollout 抗漂移 |
| [video_generation](./video_generation/) | Seedance 2.0: Advancing Video Generation for World Complexity | [seedance2](./video_generation/seedance2/analysis.md) | ByteDance Seed, 2026 | [project](https://seed.bytedance.com/seedance2_0) | 统一多模态音视频生成（文本/图像/音频/视频 4 路输入，原生 720p 双声道三轨音频）；R2V 支持 20/22 任务类型业界最全，独家视觉特效引用 + 续写/延伸；Arena.AI T2V #1 (1450 Elo) + I2V #1 (1449 Elo)；产品技术报告，不披露架构 |
| [inference_acceleration](./inference_acceleration/) | Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model | [teacache](./inference_acceleration/teacache/analysis.md) | CVPR 2025, UCAS+Alibaba | [github](https://github.com/ali-vilab/TeaCache) | training-free 缓存加速。用 timestep-embedding 调制后 noisy input 的累积相对 L1 距离(经多项式 rescale)当 indicator,自适应跳过 DiT 计算、复用残差,2–6× 加速质量近乎无损。代价:系数需离线按模型标定,且与 few-step 蒸馏模型互斥 |
| [llm](./llm/) | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | [gkd](./llm/gkd/analysis.md) | Google DeepMind, ICLR 2024 | [arXiv](https://arxiv.org/abs/2306.13649) | GKD：把蒸馏视为模仿学习问题，学生在自采序列（on-policy）上接受教师逐 token 密集反馈，散度从 forward KL 推广到 reverse KL / JSD(β)，解决 exposure bias；摘要/翻译/推理提升 1.7–2.1×，可无缝接入 RL fine-tuning |
| [llm](./llm/) | On-Policy Distillation（博客中文译） | [on_policy_distillation](./llm/on_policy_distillation/blog_zh.md) | Thinking Machines Lab, 2025-10 | [blog](https://thinkingmachines.ai/blog/on-policy-distillation/) | 在策略蒸馏兼得 RL 在策略采样与 SFT 密集监督之长：学生自采轨迹 + 教师逐 token 反向 KL 打分，相比 RL 节省 9–30× 计算，可用于推理能力蒸馏（AIME'24 达 70%）和持续学习中的行为恢复（中间训练后接蒸馏恢复 IF-eval）|
| [image_generation](./image_generation/) | Rethinking Cross-Layer Information Routing in Diffusion Transformers | [dar](./image_generation/dar/analysis.md) | 南京大学+阿里巴巴, 2026 | [arXiv](https://arxiv.org/abs/2605.20708) | 把 DiT 的固定权重残差累加替换为 timestep-aware softmax 加权跨层聚合（DAR），信息论推导最优分块策略（S*=4），Triton kernel 11.5× 加速；600K 步达 SiT 1.75M 步 FID，与 REPA 叠加早期阶段再快 2× |
| [multimodal](./multimodal/) | Representation Forcing for Bottleneck-Free Unified Multimodal Models | [rf](./multimodal/rf/analysis.md) | 港大+ByteDance Seed, 2026 | [project](https://yuqingwang1029.github.io/RepresentationForcing) | 将 UMM 理解编码器的表示离散化为 rep token，让解码器先 AR 预测再用 in-context 引导像素空间扩散，去掉外部 VAE 瓶颈；GenEval 从 0.25（naive pixel）→0.84，匹配 VAE-based SOTA，理解能力同时优于 VAE 版本 |

## 阅读体系

每篇分析按这个结构写,方便对照:

1. **一句话定位** — 这篇论文要解决什么、提了什么核心办法
2. **与前作的关系** — 站在哪些工作之上,主要的 incremental claim 是什么
3. **核心算法/方法** — 数学层面 + 工程层面的 walk-through
4. **关键代码位置** — `file_path:line_number`,方便回查
5. **关键配置项** — 把 paper 里没明说但 config 里能看出来的细节挖出来
6. **争议/权衡** — 这篇方法的弱点、和竞争方案的真实差异
7. **一句话总结** — 阅读后留下的最有用的一句话

## 增加新论文

1. 决定方向目录(`video_generation` / `llm` / 等)
2. 在方向目录下新建论文短名目录
3. 创建 `analysis.md` 写解读
4. 更新本文件的总索引表 + 方向目录的 README

### 链接约定

索引表里的"链接"列优先级:**官方项目主页 > GitHub > arXiv**。一般加 GitHub,没有就加 arXiv。

## 当前进度

- ✅ video_generation
  - ✅ LongLive 2.0
  - ✅ SANA-Streaming
  - ✅ Helios
  - ✅ Seedance 2.0
  - ⏳ Self-Forcing(已通过对话讨论,待整理成 markdown)
  - ⏳ CausVid(已通过对话讨论,待整理成 markdown)
- ✅ multimodal
  - ✅ Cosmos 3
  - ✅ Representation Forcing（港大+ByteDance Seed, 2026）
- ✅ inference_acceleration
  - ✅ TeaCache
- ✅ llm
  - ✅ GKD（ICLR 2024，Google DeepMind）
  - ✅ On-Policy Distillation（Thinking Machines 博客翻译）
- ✅ image_generation
  - ✅ DAR（南京大学+阿里巴巴, 2026）
