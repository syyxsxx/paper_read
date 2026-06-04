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
  - ⏳ Self-Forcing(已通过对话讨论,待整理成 markdown)
  - ⏳ CausVid(已通过对话讨论,待整理成 markdown)
