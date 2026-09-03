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
| [video_generation](./video_generation/) | Bernini: A Scalable Unified Framework for Video Generation and Editing | [bernini](./video_generation/bernini/analysis.md) | ByteDance, 2026-05 | [project](https://bernini-ai.github.io) | MLLM planner(Qwen2.5-VL-7B 预测 ViT embedding 语义目标) + DiT renderer(Wan2.2-A14B flow matching)，ViT embedding 作为接口解耦双组件独立预训练；SA-3D RoPE 相位调制消除多段视觉身份混淆；CoT 推理迁移语言理解；V2V 一致性第一、S2V FaceSim 78.20 超第二名 20 分 |
| [inference_acceleration](./inference_acceleration/) | Learning Few-Step Diffusion Models by Trajectory Distribution Matching | [tdm](./inference_acceleration/tdm/analysis.md) | ICCV 2025 (HKUST+Huawei) | [github](https://github.com/Luo-Yihong/TDM) | 在 trajectory 各段的**分布层面**对齐 student/teacher ODE,data-free(仅需 prompt);500 iter / 2 A800h 让 PixArt-α 4-step 超越 50-step teacher;SDXL 蒸馏 25× 比 DMD2 省;支持 K-step flexible inference |
| [inference_acceleration](./inference_acceleration/) | TDM-R1: Reinforcing Few-Step Diffusion Models with Non-Differentiable Reward | [tdm_r1](./inference_acceleration/tdm_r1/analysis.md) | arXiv 2026-03 (HKUST+CUHK+Xiaohongshu) | [github](https://github.com/Luo-Yihong/TDM-R1) | 在 TDM few-step 模型上做 RL post-training, 解耦 Surrogate Reward（DGPO 组级 BT 偏好优化）+ Generator（TDM 反 KL 框架）, ODE 确定性轨迹消除中间步奖励估计方差; 4 NFE GenEval 0.92 超 80 NFE 基础模型与 GPT-4o |
| [inference_acceleration](./inference_acceleration/) | Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model | [teacache](./inference_acceleration/teacache/analysis.md) | CVPR 2025, UCAS+Alibaba | [github](https://github.com/ali-vilab/TeaCache) | training-free 缓存加速。用 timestep-embedding 调制后 noisy input 的累积相对 L1 距离(经多项式 rescale)当 indicator,自适应跳过 DiT 计算、复用残差,2–6× 加速质量近乎无损。代价:系数需离线按模型标定,且与 few-step 蒸馏模型互斥 |
| [inference_acceleration](./inference_acceleration/) | Multi-Resolution Flow Matching: Training-Free Diffusion Acceleration via Staged Sampling | [mrflow](./inference_acceleration/mrflow/analysis.md) | arXiv 2026-07, BUAA+NTU+ICT | [github](https://github.com/xliu-deep/MrFlow) | 四阶段分辨率 pipeline：低分辨率 latent 扩散(12 步)→ 像素空间 GAN 超分(Real-ESRGAN ×2)→ 低强度噪声注入(σ=0.12)→ 高分辨率单步精修。Qwen-Image 10.3× 加速 GenEval 0.86，叠加蒸馏可达 25× |
| [llm](./llm/) | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes | [gkd](./llm/gkd/analysis.md) | Google DeepMind, ICLR 2024 | [arXiv](https://arxiv.org/abs/2306.13649) | GKD：把蒸馏视为模仿学习问题，学生在自采序列（on-policy）上接受教师逐 token 密集反馈，散度从 forward KL 推广到 reverse KL / JSD(β)，解决 exposure bias；摘要/翻译/推理提升 1.7–2.1×，可无缝接入 RL fine-tuning |
| [llm](./llm/) | On-Policy Distillation（博客中文译） | [on_policy_distillation](./llm/on_policy_distillation/blog_zh.md) | Thinking Machines Lab, 2025-10 | [blog](https://thinkingmachines.ai/blog/on-policy-distillation/) | 在策略蒸馏兼得 RL 在策略采样与 SFT 密集监督之长：学生自采轨迹 + 教师逐 token 反向 KL 打分，相比 RL 节省 9–30× 计算，可用于推理能力蒸馏（AIME'24 达 70%）和持续学习中的行为恢复（中间训练后接蒸馏恢复 IF-eval）|
| [image_generation](./image_generation/) | Rethinking Cross-Layer Information Routing in Diffusion Transformers | [dar](./image_generation/dar/analysis.md) | 南京大学+阿里巴巴, 2026 | [arXiv](https://arxiv.org/abs/2605.20708) | 把 DiT 的固定权重残差累加替换为 timestep-aware softmax 加权跨层聚合（DAR），信息论推导最优分块策略（S*=4），Triton kernel 11.5× 加速；600K 步达 SiT 1.75M 步 FID，与 REPA 叠加早期阶段再快 2× |
| [image_generation](./image_generation/) | Qwen-Image-2.0-RL Technical Report | [qwen_image_rl](./image_generation/qwen_image_rl/analysis.md) | Qwen Team（阿里巴巴）, 2026-06 | — | 分任务 RLHF（T2I 三层奖励 + Edit 两维奖励）+ pointwise VLM 打分 + hybrid CFG（rollout 有 CFG，训练无 CFG）+ On-Policy Distillation（W₂ 速度场匹配合并 teacher）；Qwen-Image-Bench +2.61、T2I arena +78 Elo、Edit arena +93 Elo |
| [image_generation](./image_generation/) | JoyAI-Image: Awakening Spatial Intelligence in Unified Multimodal Foundation Models | [joyai_image](./image_generation/joyai_image/analysis.md) | 京东 AI Research, 2026 | [github](https://github.com/jd-opensource/JoyAI-Image) | Qwen3-VL-8B-Instruct + Wan-2.1-VAE + 16B MMDiT 统一理解/T2I/编辑;OpenSpatial-3M(3M 条 3D box-centric QA)激活空间智能;Blender 驱动双路空间编辑数据引擎;DiffusionNFT 后训练;SpatialEdit-Bench Object Overall 0.649 超所有视频模型;LongText-Bench EN=ZH=0.963 SOTA |
| [video_generation](./video_generation/) | Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence | [lingbot_video](./video_generation/lingbot_video/analysis.md) | Ant Group, 2026 | [github](https://github.com/robbyant/lingbot-video) | 首个大规模开源 MoE 视频 foundation model;Sparse MoE DiT(13B-A1.4B-E128)容量-算力解耦;70k+ 小时 VLA 具身数据;6 维解耦奖励 + Flash-GRPO + RealNFT 物理后训练;TI2V 开源第一,RBench 0.620 开源全榜第一,Physics-IQ 40.4 第一 |
| [video_generation](./video_generation/) | Video Generation Models are General-Purpose Vision Learners | [genception](./video_generation/genception/analysis.md) | Google DeepMind, 2026-07 | [project](https://genception.github.io) | 预训练 WAN 2.1 T2V 扩散模型当作视觉感知 backbone;t=0 单步前向 + 统一 RGB 空间表示(深度/法线/分割/DensePose/raymap)+ 纯 L2 loss;7,500 条合成视频以 7×~500× 更少数据匹配专用 SOTA;emergent 泛化到多实例/OOD 类别 |
| [video_generation](./video_generation/) | JoyAI-Echo: Pushing the Frontier of Long Audio-Visual Generation | [joyai_echo](./video_generation/joyai_echo/analysis.md) | 京东 Joy Future Academy, 2026-06 | [github](https://github.com/jd-opensource/JoyAI-Echo) | Slot-paired 跨模态音视频记忆库(3 anchor+4 recent,共 7 slots)+ 层级音频记忆 mask + slot-aware 跨模态 mask;SFT→OmniNFT RLHF→Bidirectional DMD 三阶段记忆感知后训练(7.5× 加速);Director Agent 闭环编辑 + 单步 SR 模块;全指标超越 Happy Oyster 和 LTX-2 系列 |
| [video_generation](./video_generation/) | Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation | [wan_dancer](./video_generation/wan_dancer/analysis.md) | Tongyi Lab, Alibaba, 2026 | [github](https://github.com/alibaba/Wan-Dancer) | 层级 Global-to-Local 解耦:Global DiT 用 Dynamic FPS RoPE 把整首音乐压进 149 帧稀疏关键帧建立编舞骨架,Local DiT 并行精化每 5 秒片段;光流损失权重(SEA-RAFT)保全快速动作细节;720p/30fps 单次生成超 1 分钟,5 种舞蹈风格,全指标超越 X-Dancer 和 MusicInfuser |
| [matting](./matting/) | SAM2Matting: Generalized Image and Video Matting | [sam2matting](./matting/sam2matting/analysis.md) | Fudan+SUFE, 2026 | [github](https://github.com/FudanCVL/SAM2Matting) | Tracker-to-Matting 范式:冻结 VOS(SAM2/SAM3)负责时序追踪,仅在图像抠图数据训练轻量 ROI Detector(学习型 ROI 替代形态学操作)+ Progressive Alpha Predictor(3 尺度级联);零样本视频 matting SOTA,SAM2.1-T 以 40 FPS @ 1080p + <4GB VRAM 实时运行 |
| [flow_matching](./flow_matching/) | Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis | [self_flow](./flow_matching/self_flow/analysis.md) | BFL+MIT, ICML 2026 | [arXiv](https://arxiv.org/abs/2603.06507) | Dual-Timestep Scheduling 对 token 施加异质噪声制造信息不对称,EMA teacher-student 自蒸馏替代外部编码器(REPA/DINOv2);首个纯自监督超越外部对齐的工作,图像/视频/音频多模态统一,收敛 2.8× 加速,T2I FID 3.61 全榜最优 |
| [multimodal](./multimodal/) | Representation Forcing for Bottleneck-Free Unified Multimodal Models | [rf](./multimodal/rf/analysis.md) | 港大+ByteDance Seed, 2026 | [project](https://yuqingwang1029.github.io/RepresentationForcing) | 将 UMM 理解编码器的表示离散化为 rep token，让解码器先 AR 预测再用 in-context 引导像素空间扩散，去掉外部 VAE 瓶颈；GenEval 从 0.25（naive pixel）→0.84，匹配 VAE-based SOTA，理解能力同时优于 VAE 版本 |
| [world_model](./world_model/) | Alaya-EVOKE: From Linear-Scaling Supervision to Endless World | [evoke](./world_model/evoke/analysis.md) | USTC+Alaya Lab, 2026-08 | — | World State Bank（Pi3X 增量点云）+ Chunk-wise Sparse Teacher（5 路注意力，O(N) 总代价）+ Self-Forced DMD（20-chunk 31.4s 全窗口分布匹配蒸馏，chunk 间梯度截断），首个可稳定生成 90s 几何一致交互视频的系统 |
| [video_generation](./video_generation/) | Video Generation with Stable Transparency via Shiftable RGB-A Distribution Learner | [wan_alpha](./video_generation/wan_alpha/analysis.md) | 天津大学+腾讯, 2026 | — | latent 空间双向扩散 loss 把 alpha 分布推离 RGB（bidiff）+ noise 空间 Gaussian 椭圆均值偏移，Wan-14B 以 DoRA+LoRA 实现 RGB-A 透明视频生成，TransPixeler 快 15×，全指标 SOTA |
| [video_generation](./video_generation/) | RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO | [raven](./video_generation/raven/analysis.md) | Imperial College London, 2026 | [project](https://yanzuo.lu/raven) | 把 fake-score step 的 self-rollout 重打包为「clean 历史端点+noisy 去噪状态」交错序列（RAVEN），让 DMD 梯度流回历史缓存；CM-GRPO 直接在 consistency 转移核上做 GRPO，消除 Flow-GRPO 的 Euler-Maruyama 训练-推理 gap，RAVEN+CM-GRPO VBench 全维度第一 |
| [video_generation](./video_generation/) | Beyond Text Conditioning: A Systematic Study of MLLM-DiT Fusion for Video Generation | [mllm_dit_fusion](./video_generation/mllm_dit_fusion/analysis.md) | 中科院+Microsoft Research 等, 2026-08 | [arXiv](https://arxiv.org/abs/2608.14043) | MLLM-DiT 耦合方式的三维设计空间扫描（表示形式 × 生成机制 × 注入方式）：EMA 码本离散语义 token + causal AR + 多层 cross-attention 胜出，VBench-Long 74.88→80.77；关键负面结论是主流的 frozen-MLLM + learnable query + connector 比不加 MLLM 还低 7 分 |
| [image_generation](./image_generation/) | MMOE: Modernizing Diffusion Transformers with Efficient Expert Design | [mmoe](./image_generation/mmoe/analysis.md) | NTU+中国电信 TeleAI, 2026-07 | [arXiv](https://arxiv.org/abs/2607.24665) | ConvNeXt 式现代化：把 routed/shared/MoE++ 轻量零计算专家 + gate-residual + attention-residual 四件 LLM MoE 武器逐个搬进 SiT。单节点 8×H100/batch 256/400k step 下 FID 5.20→3.75；拆开看涨点几乎全来自 attention-residual（−0.79），轻量专家负责把训练时长从 120h 压到 67h、激活显存降 20–32% |
| [image_generation](./image_generation/) | On-Policy Self-Distillation in Diffusion Models | [diffusion_opsd](./image_generation/diffusion_opsd/analysis.md) | ByteDance Seed+NUS+UCSD, 2026-08 | [github](https://github.com/worldbench/DiffusionOPSD) | on-policy EMA rollout 采中间状态 z_q,沿奖励梯度构造有界 y+/y- 目标(OPA)后 detach 拟合策略;Open3 把三个单奖励 LoRA teacher 蒸馏成一个 student;19/20 best held-out,比 DiffusionNFT 省 40%/63% GPU-hours |
| [image_generation](./image_generation/) | D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models | [d_opsd](./image_generation/d_opsd/analysis.md) | HKUST+Alibaba Z-Image+UCSD+CUHK, 2026-05 | [github](https://github.com/vvvvvjdy/D-OPSD) | LLM/VLM encoder 的涌现 in-context 能力(text+img 条件无需训练即生成目标变体),同模型双角色 student(text-only,on-policy rollout)/ teacher(multimodal),velocity MSE 对齐保留 few-step 能力;Z-Image-Turbo LoRA 定制 Quality-S 3.80 > PSO/SFT,全量微调 FID 40.5 优于 Base 48.7 同时 GenEval/DPG 几乎不掉 |
| [image_generation](./image_generation/) | Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher | [self_opd](./image_generation/self_opd/analysis.md) | 清华+浙大+阿里, 2026-08 | [github](https://github.com/Shiy-Zhang/Self-OPD) | 把 OPD 的 teacher 换成「自参考 + 局部分叉」:每步分出 K 条 SDE 候选各自 ODE rollout 打分,以同一父状态的纯 ODE 轨迹作 baseline 得 advantage,全分支拉/推速度回归;排斥项按与最佳方向夹角门控(去掉会在 600 步崩),方差归一化因子由逐步 KL 梯度推出而非超参;多目标改为奖励层面融合(M 个 tilt 塌缩成单个)——只用于给分支排序、从不求导,故支持黑盒 reward 且改权重不重训 |
| [image_generation](./image_generation/) | Flow-GRPO: Training Flow Matching Models via Online RL | [flow_grpo](./image_generation/flow_grpo/analysis.md) | NeurIPS 2025, 港中文+清华+快手等 | [github](https://github.com/yifan123/flow_grpo) | 扩散/flow 模型 online RL 的源头,仓库里 RVM/Self-OPD/Flow-OPD/CM-GRPO 全把它当 baseline 或攻击对象;把确定性 ODE 改写成边缘等价的 SDE,使 policy 成为各向同性高斯从而 log-prob 与 KL 都有闭式(这个闭式 KL 后来成了整条 OPD 线的共同起点),再用「训练 10 步、推理 40 步」的 Denoising Reduction 提速约 4×;SD3.5-M GenEval 0.63→0.95,但 reward 与评测指标同源、代码与论文有算法级差异(log-prob 取 mean、CFG 未提) |
| [image_generation](./image_generation/) | Flow-OPD: On-Policy Distillation for Flow Matching Models | [flow_opd](./image_generation/flow_opd/analysis.md) | USTC+UCLA+CUHK+小红书, 2026-05 | [github](https://github.com/CostaliyA/Flow-OPD) | 把 LLM 的 on-policy distillation 搬进 flow matching:先用单奖励 GRPO 养出领域专家 teacher,再按 prompt 硬路由取 teacher 速度场做逐步密集监督;关键推导是 student 与 teacher 的 SDE 转移核共享同一协方差,高斯 KL 塌缩成速度场加权 L2,于是可整个绕过 policy gradient、直接反传 MSE(梯度方差为零);MAR 用冻结美学 teacher 做全数据锚防 reward hacking;SD3.5-M 上 GenEval 0.63→0.93、OCR 0.59→0.93 并出现 teacher-surpassing |
| [image_generation](./image_generation/) | DiffusionNFT: Online Diffusion Reinforcement with Forward Process | [diffusion_nft](./image_generation/diffusion_nft/analysis.md) | ICLR 2026, 清华+NVIDIA+Stanford | [arXiv](https://arxiv.org/abs/2509.16117) | 仓库 RL 那一簇的关键拼图(被 9 篇笔记引用却一直缺专篇):指出反向过程 RL 有前向不一致/采样器锁死一阶 SDE/CFG 双模型三大结构缺陷,把 RL 整个搬到唯一的前向加噪过程上;用 reward 当最优性概率把样本切成隐式正负两集,导出正负成比例的改进方向 Δ,再用隐式参数化(同一个 v_θ 以 +β/−β 两种符号同时扮演正负策略)把 Δ 烤进单模型,训练目标退化成带负支的标准 flow matching loss;全程 CFG-free,SD3.5-M GenEval 0.24→0.98@1k 步。⚠️「25×」是三任务最好值(另两个 8×/3×),去掉负支会瞬间崩这条最关键消融只有文字没有曲线 |
| [world_model](./world_model/) | H3-World: Turning Language Understanding into World Control | [h3_world](./world_model/h3_world/analysis.md) | 腾讯+NUS+港理工, 2026-09 | [github](https://github.com/Danzer1xxxxChan/H3-World) | 主张「大视频模型里已经藏着控制接口，缺的只是时间精度」：不外接动作模块，把键鼠状态翻译成角色/相机两个可组合文本子句，每个 video latent 配一条独立指令；用 τ(A_k)=τ(V_k)−Δ 的位置平移做时间绑定(既对齐时序又保住「文本在前」的预训练顺序)，再用单出口路由掩码让每条指令只能从匹配 latent 进入视觉流；33B MiniMax-H3 上 8k 样本/10k 步/0.199% LoRA。⚠️ 全文唯一量化是光流：恒定动作下冻结 H3 得 301.8、H3-World 得 300.5(贡献只在时间绑定)，动作切换时 global 0.0/−17.3 vs 本文 +52.7/−106.0；核心的单出口路由无消融，52 个未见组合只定性测了 1 个 |
| [video_generation](./video_generation/) | DreamX-Creator 1.0: Democratizing Native Audio-Video Generation at 2K Resolution | [dreamx_creator](./video_generation/dreamx_creator/analysis.md) | DreamX Team, 阿里巴巴, 2026-08 | [arXiv](https://arxiv.org/abs/2608.31106) | 7B 原生联合音视频：前半两流独立、后半用 token+head 级 sigmoid 输出门做跨模态耦合(门同时看 target token 和 cross-attn 输出，非 query-only)，A2V/V2A/Joint 靠噪声水平高低表达条件方向而非加分支；数据流水线(PySceneDetect+两端裁3帧 / Q-Align+UniMatch+Audiobox / Synchformer+SyncNet / Qwen3-Omni→ASR→3.6 三级标注)是最可复用的部分；DeSync 0.1351 领先第二名 42%、Refiner 后 VQ 0.6930。⚠️ 全文零消融；音频美学与跨模态语义明显落后 22B/33B 对手(报告自认)；结论章称 RL/Refiner「是设计而非已完成的经验主张」、§7.3 称 refiner 双向非因果——都与正文表格矛盾 |
| [world_model](./world_model/) | Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models | [awomo](./world_model/awomo/analysis.md) | NUS+InfRec+Berkeley+HKUST, 2026-08 | [github](https://github.com/LanceZPF/cardinal-preview) | 游戏引擎 verifier(碰撞/物理/navmesh/playability)+ 开发者 acceptance = RLHEV 递归数据引擎,替代 CLIP/MLLM 模糊奖励;UWDP protocol trace Spearman 0.719 vs snapshot 0.159;UnitySceneBench Full RLHEV Primary 0.681 vs Engine-only 0.55;embodied D4RL +48.43% |
| [training_infra](./training_infra/) | Zellige: Moldable Sequence Placement for Mixed Image-Video DiT Training | [zellige](./training_infra/zellige/analysis.md) | HKUST(GZ)+HIT(SZ)+HKUST, 2026-08 | [arXiv](https://arxiv.org/abs/2608.01150) | 两条定理证明预设不相交 rank 组必然在「组间负载不均」与「组内通信冗余」之间二选一；Zellige 允许 rank 集合重叠，用 profiler + Anchor/Filler 两阶段 CP-SAT（ECF 消对称，33–119 ms/batch）+ Coalesced Attention Engine，拿到 USP 的 1.00× 完美平衡却只用其 25% 通信量，比 KnapFormer 快 1.12–1.54× |
| [world_model](./world_model/) | WorldDiT: A Unified Diffusion Architecture for World and Action Modeling | [worlddit](./world_model/worlddit/analysis.md) | Bagel Labs, 2026-07 | [HF](https://huggingface.co/bageldotcom/worlddit) | 4 层 d=1024 共享 DiT 用 flow matching 同时回归 7 步动作与未来单帧 1/3 白化 RGB patch；action-safe attention 保证推理时整条视觉路径可无损摘除。399M 参数 LIBERO 均值 94.9%，落在 sub-billion Pareto 前沿 —— 但 500 episode 里 300 个参与过 checkpoint 选择，且全文无消融 |
| [world_model](./world_model/) | H3-World: Turning Language Understanding into World Control | [h3world](./world_model/h3world/analysis.md) | Tencent+NUS+HKPolyU, 2026-09 | [github](https://github.com/Danzer1xxxxChan/H3-World) | 动作→9-bit→自然语言句子，per-latent 独立文本编码（37句/clip）+ Single-Egress Directed Mask（Ak只和Vk互见），0.199% LoRA（65.6M）将33B MiniMax-H3变成可时序切换的交互式世界模型；52/135组合零样本组合泛化 |
| [video_generation](./video_generation/) | ReWorld: An Interactive World Model with Long-Horizon Memory | [reworld](./video_generation/reworld/analysis.md) | HKUST(GZ)+Alibaba ATH, 2026-08 | [project](https://zhifeichen097.github.io/ReWorld/) | 把「跟手」和「记得住」拆成两种 attention 窗口分别训练(18 局部 head 短窗 + 6 全局 head 全历史),random head routing 让能力不绑死在特定 head 上以适配部署时的共享有界 cache;推理端 B=12 固定预算 = sink + 5 recent + 6 个按位姿检索的 landmark(定容 30,里程计准入 + 位姿冗余淘汰),chunk-drop 训练把稀疏拼接 cache 变成分布内输入;八来源 22 万 clip 统一到同一物理动作尺度;Wan2.2-TI2V-5B + 4 步 DMD LoRA,RotErr 11.95° 与 VBench 均值 0.850 均居首 |
| [video_generation](./video_generation/) | Scaling Reinforcement Learning for Diffusion Models via Velocity Matching | [rvm](./video_generation/rvm/analysis.md) | Georgia Tech+NVIDIA, 2026-08 | [project](https://jaemoo-choi.github.io/RVM/) | 主张扩散 RL 不需要 likelihood:把生成样本只加一次噪,用 reward 当带符号方向偏好做速度回归(正拉负推),无 policy ratio、无轨迹存储、无 CFG、无 clipping;证明 RAM 与 DiffusionNFT 是其在 (anchor, scale, reach) 上的特例;发现纯偏好 reward 会把视频训成静止画(Dyn. Degree 仅 2.78),提出 RAFT top-5% 光流的 DT reward 拉回 75.00;Wan2.1-1.3B 上 525 GPU-h(DanceGRPO 的 1/11.8)拿到 VBench Overall 84.13 |
| [video_generation](./video_generation/) | Parallel Decoding Distillation for Fast Image and Video Generation | [pdd](./video_generation/pdd/analysis.md) | NVIDIA+Weizmann, 2026-07 | [project](https://research.nvidia.com/labs/genair/pdd) | 不把多步合并成一大步,而是一次前向并行预测块内全部 L 个区间的平均速度(N 个线性头共享 teacher backbone),训练只是回归到 teacher 的 Runge-Kutta 近似——无 VSD/GAN/JVP/有限差分/多阶段;推理时按步长把 L 个头融合成一个线性层,开销与 teacher 完全相同,且同一套权重支持可变 NFE;首个在视频上追平 distribution-based 的纯轨迹蒸馏,Qwen-Image 4 NFE 超 teacher 且多样性 0.192 vs DMD2 的 0.095 |
| [video_generation](./video_generation/) | minWM: A Full-Stack Open-Source Framework for Real-Time Interactive Video World Models | [minwm](./video_generation/minwm/analysis.md) | 生数科技+清华+人大等, 2026-05 | [github](https://github.com/shengshu-ai/minWM) | 把「PRoPE 相机注入 → Causal Forcing/++ 三阶段蒸馏」这条已有流水线打包成开源框架并放出全阶段中间 checkpoint;方法无新算法(两个核心组件是作者自己前作)、全文零定量指标零 baseline 对比;有价值的是数据消融(感知估计位姿训不出可控性,必须真值轨迹)与全阶段 checkpoint;需打折的是「Real-Time」(反推仅 3.8–11.4 fps、可交互时长上限约 5 秒)与唯一方法公式和代码不符(issue #10 已证实) |

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
  - ✅ Bernini（ByteDance, 2026-05）
  - ✅ LingBot-Video（Ant Group, 2026）
  - ✅ GenCeption（Google DeepMind, 2026-07）
  - ✅ JoyAI-Echo（京东 Joy Future Academy, 2026-06）
  - ✅ Wan-Dancer（Tongyi Lab, Alibaba, 2026）
- ✅ matting
  - ✅ SAM2Matting（Fudan+SUFE, 2026）
  - ⏳ Self-Forcing(已通过对话讨论,待整理成 markdown)
  - ⏳ CausVid(已通过对话讨论,待整理成 markdown)
- ✅ flow_matching
  - ✅ Self-Flow（BFL+MIT, ICML 2026）
- ✅ multimodal
  - ✅ Cosmos 3
  - ✅ Representation Forcing（港大+ByteDance Seed, 2026）
- ✅ inference_acceleration
  - ✅ TDM（HKUST+Huawei, ICCV 2025）
  - ✅ TeaCache
  - ✅ MrFlow（BUAA+NTU+ICT, 2026-07）
- ✅ llm
  - ✅ GKD（ICLR 2024，Google DeepMind）
  - ✅ On-Policy Distillation（Thinking Machines 博客翻译）
- ✅ image_generation
  - ✅ DAR（南京大学+阿里巴巴, 2026）
  - ✅ Qwen-Image-2.0-RL（Qwen Team, 2026-06）
  - ✅ JoyAI-Image（京东 AI Research, 2026）
  - ✅ Self-OPD（清华+浙大+阿里, 2026-08）
  - ✅ Flow-OPD（USTC+UCLA+CUHK+小红书, 2026-05）
  - ✅ Flow-GRPO（NeurIPS 2025，港中文+清华+快手等）
- ✅ world_model
  - ✅ Alaya-EVOKE（USTC+Alaya Lab, 2026-08）
- ✅ video_generation (新增)
  - ✅ RAVEN（Imperial College London, 2026）
  - ✅ Wan-Alpha（天津大学+腾讯, 2026）
  - ✅ ReWorld（HKUST(GZ)+Alibaba ATH, 2026-08）
  - ✅ RVM（Georgia Tech+NVIDIA, 2026-08）
  - ✅ PDD（NVIDIA+Weizmann, 2026-07）
  - ✅ minWM（生数科技+清华+人大等, 2026-05）
  - ✅ Beyond Text Conditioning / BiVidGen（中科院+MSRA, 2026-08）
- ✅ image_generation (新增)
  - ✅ MMOE（NTU+TeleAI, 2026-07）
- ✅ world_model (新增)
  - ✅ WorldDiT（Bagel Labs, 2026-07）
- ✅ training_infra (新方向)
  - ✅ Zellige（HKUST(GZ)+HIT(SZ), 2026-08）
- ✅ inference_acceleration (新增)
  - ✅ TDM-R1（HKUST+CUHK+Xiaohongshu, 2026-03）
- ✅ image_generation (新增)
  - ✅ DiffusionOPSD（ByteDance Seed+NUS+UCSD, 2026-08）
  - ✅ D-OPSD（HKUST+Alibaba Z-Image+UCSD+CUHK, 2026-05）
- ✅ world_model (新增)
  - ✅ AWoMo/RLHEV（NUS+InfRec+Berkeley+HKUST, 2026-08）
  - ✅ H3-World（Tencent+NUS+HKPolyU, 2026-09）
