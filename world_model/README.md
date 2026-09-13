# World Model 方向

世界模型方向的论文阅读笔记。目前分成两条互不相交的支线：

- **视频世界模型**：长时生成、几何一致性、物理可控（EVOKE）
- **世界-动作模型（World-Action Model）**：把未来视觉预测当作机器人策略的辅助目标（WorldDiT）

## 论文谱系

```
支线 A：交互式视频世界模型
├── GameNGen (Google, 2024) — 扩散模型首次实时交互游戏
├── DIAMOND (2024) — world model + RL 交互
├── Self-Forcing (Berkeley, 2025) — SFD 蒸馏原型，10s 内
├── Alaya-EVOKE (USTC+Alaya, 2026-08)
│   ├── World State Bank: Pi3X 点云几何持久化
│   ├── Sparse Teacher: O(N) 跨 chunk 评分
│   └── Self-Forced DMD: 90s 全窗口蒸馏
└── H3-World (Tencent+NUS+HKPolyU, 2026-09)
    ├── 动作→自然语言句子（9×16=135 组合）
    ├── per-latent 独立文本绑定（37 句/clip，独立编码）
    └── Single-Egress Directed Mask：Ak 只和 Vk 双向互见

支线 B：世界-动作模型（机器人策略 + 未来视觉预测）
├── 大 VLM 当动作骨干（主流范式，3B–10B）
│   ├── OpenVLA / OpenVLA OFT / π0 / π0.5 / GR00T N1
│   └── ACoT VLA / MMaDA VLA / VLANeXt（LIBERO 96–98.5）
└── 不用大 VLM（sub-billion）
    ├── Diffusion Policy (2023) / Octo (2024) / DiT Policy (2024)
    ├── DreamVLA (2025) — 独立 dream 分支做视觉预测
    └── WorldDiT (Bagel Labs, 2026-07) ← 支线 B 首篇
        ├── 单个共享 DiT 同时出 action velocity 和 RGB patch velocity
        ├── action-safe attention：动作 query 读不到 noised RGB target
        └── 推理时整条视觉路径被摘除（训练有监督、部署无开销）
```

## 论文列表

| 论文 | 简称 | 支线 | 发表 | 一句话 |
|------|------|------|------|--------|
| [Alaya-EVOKE: From Linear-Scaling Supervision to Endless World](./evoke/analysis.md) | evoke | A 视频世界模型 | USTC+Alaya Lab, 2026-08 | WSB + Sparse Teacher + SFD，首个 90s 几何一致交互视频生成 |
| [WorldDiT: A Unified Diffusion Architecture for World and Action Modeling](./worlddit/analysis.md) | worlddit | B 世界-动作模型 | Bagel Labs, 2026-07 | 399M 参数共享 DiT 同时回归 7 步动作与未来归一化 RGB patch，推理时摘掉视觉路径，LIBERO 均值 94.9% 落在 sub-billion Pareto 前沿；但报告值含 60% 非 held-out episode 且无消融 |
| [Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models](./awomo/analysis.md) | awomo | C 可验证游戏数据引擎 | NUS+InfRec+Berkeley+HKUST, 2026-08 | 把游戏开发轨迹(intent+edit+engine check+repair+acceptance)包装成 RLHEV 递归数据飞轮，替代 CLIP 等模糊奖励；UnitySceneBench Full RLHEV Primary 0.681 vs Engine-only 0.55，Unity→held-out Unity 0.25→0.75，embodied +48.43% D4RL |
| [ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU](./abot_world_0/analysis.md) | abot_world_0 | A 交互式视频世界模型 | AMAP CV Lab, 阿里巴巴, 2026-07 | h3world 训练数据的来源。8 维键盘 multi-hot 打包×4 加性注入 + 负时间 RoPE 身份记忆 + Teacher Forcing→ODE Distillation→LongForcing 三阶段蒸馏 + LightVAE/FP8/SageAttention2/Fast-RoPE/有界 KV cache 全栈优化，单张 RTX 5090 跑 1280×704 最高 16 FPS、1.2s 首帧延迟；Table 2 保留 OOM 行证明「少步采样≠实时交互」。⚠️ 但 WorldRoamBench 是自家做的且七项无一第一、数据基建零消融无规模数字、24h 宣称只有关键帧图 |
| [GameWAM: A World Action Model for Video Games](./gamewam/analysis.md) | gamewam | D 世界-动作模型(游戏) | 复旦+LIGHTSPEED+清华 SIGS, 2026-09 | 并行 Video DiT + Action DiT 联合 flow matching,predict-long/execute-short(P=16/E=8),原生 22 维键鼠不做离散化 + per-step router 切 gameplay/GUI,三级记忆;MCU Avg Mini 50.7/All 46.6 与 ViZDoom 四图第一。真正的贡献是发现并诊断 LASI:采样噪声的低频分量被迭代生成路径放大近 7 倍后盖印到输出动作上(donor swap 94.8% / zeroing 99.25% / 闭环置换检验 p=0.001)。⚠️ 但零视频预测指标、主结果建立在未量化的 LASI 缓解上、训练片段跨度比评测上限短约 15× 致长期记忆很可能从未被训到 |
| [SolarWM: Open Data and Scalable Training for Long-Horizon Video World Models](./solarwm/analysis.md) | solarwm | A 交互式视频世界模型(开源基建) | 港中深+SLAI+NUS+NVIDIA 等, 2026-09 | 1.43M clip/25.85 TB 多源数据引擎(先全量处理后选择、rejected 带原因发布、三命名空间分离)+ 双向(fused-PRoPE)→TF-AnyFlow→DMD 三阶段在四个骨干(Wan2.2-5B/14B、LTX-2.5-22B、MiniMax-H3-33B)上实例化;只训 5s 出小时级 rollout,不用 attention sink。数据工程规范是仓库同类里最严谨的。⚠️ 但全文零量化零消融却两次宣称 SOTA,因果结果只有 5B 一个模型 |
| [H3-World: Turning Language Understanding into World Control](./h3world/analysis.md) | h3world | A 交互式视频世界模型 | Tencent+NUS+HKPolyU, 2026-09 | 把键鼠动作翻成「角色子句+相机子句」文本，逐 latent 绑定单出口路由（Ak 只和 Vk 互见），33B MiniMax-H3 上只训 0.199% LoRA（65.6M）；但全文仅一组光流数字、路由无消融、52 个未见组合只定性测了 1 个 |
| [Programmable World Model](./pwm/analysis.md) | pwm | A 交互式视频世界模型 | Alaya Lab, 2026-09 | 把世界状态维护与视觉渲染彻底解耦：确定性引擎执行可编程规则维护 OBB+属性+关系，State Compiler 投影为 Identity/Semantic/Direction 三张控制图，LingBot-World-v1 冻结主干+ControlNet 渲染；CombatStateBench Count Acc. 94%/State Acc. 98%，超 baseline 53-90pp |

## 核心问题

**支线 A（视频世界模型）**

1. **生成时长**：现有方法 ≤10s，EVOKE 突破到 90s
2. **几何一致性**：相机运动后场景重访时不抖动/重影
3. **交互可控**：per-chunk 文本 prompt 动态切换（Evocation）
4. **监督代价线性增长**：teacher 全序列评分显存爆炸 → Sparse Teacher 解决

**支线 B（世界-动作模型）**

1. **规模与架构的贡献无法分离**：几十亿参数的策略里，强控制来自动作设计还是预训练骨干？
2. **视觉预测目标该多重**：WorldDiT 把 RGB 损失权重压到 0.001（动作是 0.1），目标削到 1/3 patch 且逐 patch 白化 —— 到底还剩多少"世界建模"
3. **训练有、推理无**：辅助目标怎么设计才能在部署时无损摘除（答案：action-safe attention）
4. **评测协议**：LIBERO 上跨论文的 episode 数、渲染后端、是否 per-suite 微调都不统一，排名不可直接比

## 交叉引用

- **[ReWorld](../video_generation/reworld/analysis.md)**（video_generation 方向）也是交互式世界模型，与支线 A 同源
- **[DreamX-Creator 1.0](../video_generation/dreamx_creator/analysis.md)** 与 H3-World 同以 MiniMax-H3 为参照系：前者把它当最强开源对照（多数音频指标打不过），后者把它当改造底座
- **[MMOE](../image_generation/mmoe/analysis.md)** 的 attention-residual 跨深度复用，与 WorldDiT 的 register token / block-causal 设计属于同一层面的 DiT 内部改造
