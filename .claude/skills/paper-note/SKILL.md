---
name: paper-note
description: Add a paper reading note to this repo (paper_read) from a source PDF — extract figures as cropped PNGs, write analysis.md with GitHub-rendering LaTeX formulas and embedded figures, update the direction + root README. Use when the user wants to "add a paper", "写论文解读", "把这篇论文加进来", or turn a PDF into a structured note with images and equations.
---

# Paper Note — 把一篇论文 PDF 变成带图带公式的解读

把一篇论文(PDF + 可选代码 repo)整理成本仓库格式的阅读笔记。核心难点有两个,这个 skill 就是为它俩存在的:**(1) 从 PDF 抽出能正确显示的图片;(2) 写出能在 GitHub 上正确渲染的公式。**

## 产出物

```
<方向>/<论文短名>/
├── analysis.md          ← 主解读(必须);末尾含 Q&A 区块
└── figures/             ← 从 PDF 裁出的图(必须有图时)
    ├── fig1_xxx.png
    └── ...
```
外加:更新 `<方向>/README.md` 的论文列表 + 根 `README.md` 的索引表和进度。方向不存在就新建一个方向目录 + 方向 README(参考已有 `inference_acceleration/README.md`)。

## Step 1 — 通读 PDF,确定结构

1. 用 Read 工具读 PDF(`pages` 参数分段,每次 ≤20 页),把每张 Figure/Table 看清,记下它**在第几页、页面上的大致位置(上/中/下、左/右栏)**。
2. 决定方向目录(`video_generation` / `multimodal` / `inference_acceleration` / 新建)和论文短名(小写,如 `teacache`)。
3. 挑出值得入文的图:概念图、动机图、架构图、关键结果图。表格优先**转成 markdown 表**而不是截图(更清晰、可搜索)。

## Step 2 — 抽图(关键技术)

图在 PDF 里通常是**复合的**(矢量曲线 + 多个小图块拼成),所以 `pdfimages` 抽出来是碎片 —— **不要用它**。正确做法是**用 `pdftoppm 按区域渲染整页**。

### 工具确认
```bash
which pdftoppm        # poppler,必须有。macOS: brew install poppler
pdfinfo paper.pdf | grep -i "page size"   # 拿到页面点数,通常 612x792 (letter)
```

### 按区域裁剪
`pdftoppm -x X -y Y -W W -H H` 直接渲染页面的一个矩形区域,坐标是**目标 DPI 下的像素**。用 200 DPI(letter 页 = 1700×2200 px)。

像素 = 页面分数 × (页宽点数 / 72 × DPI)。letter @ 200DPI:`px_x = frac_x * 1700`, `px_y = frac_y * 2200`。

```bash
PDF=paper.pdf
OUT=<方向>/<短名>/figures
mkdir -p "$OUT"
# 例:抽第 2 页顶部的概念图(横跨整宽,纵向 5%~30%)
pdftoppm -png -r 200 -f 2 -l 2 -x 119 -y 110 -W 1496 -H 550 "$PDF" "$OUT/fig2_concept"
# 例:抽第 4 页左栏的架构图(左半栏,纵向 30%~50%)
pdftoppm -png -r 200 -f 4 -l 4 -x 119 -y 660 -W 714 -H 440 "$PDF" "$OUT/fig4_arch"
```
- 全宽图:`-x 119 -W 1496`(约 7%~95%)
- 左栏图:`-x 119 -W 714`;右栏图:`-x 880 -W 735`
- `pdftoppm` 会给文件名加页码后缀(如 `fig2_concept-02.png`)。

### 验证-迭代循环(必做)
裁完**立刻用 Read 工具读 PNG 确认**:图是否完整、有没有切到、有没有截进别的内容。偏了就调 `-y`/`-H` 重裁。一两轮就能对准。**别跳过这步** —— 凭估计的坐标第一次很容易偏(尤其纵向位置)。

### 去掉页码后缀
```bash
cd "$OUT" && for f in *-0*.png; do mv "$f" "$(echo "$f" | sed -E 's/-0[0-9]+//')"; done
```

## Step 3 — 写 analysis.md

### 阅读框架(沿用本仓库)
1. **一句话定位** — 解决什么、提了什么核心办法
2. **要解决的问题(动机)**
3. **与前作的关系** — incremental claim
4. **核心算法/方法** — 数学 + 工程 walk-through
5. **关键代码位置** — `file_path:line_number`
6. **关键配置项**
7. **争议/权衡** — 弱点、和竞品真实差异
8. **一句话总结**
9. **Q&A** — 后续对话中产生的问答(见下方 Q&A 规范)

### 风格
- 中文正文,技术词保留英文。标点跟随本仓库:正文用 ASCII 逗号 `,` + 中文句号 `。` + 顿号 `、`,行内技术符号用 ASCII `:` `(` `)`。
- 多用表格、ASCII 流程图、`file:line` 引用。深挖点用 `📌` 标。

### 插图
相对路径引用,每张图下面用 `>` blockquote 做**逐句图解**——不是照抄 caption,而是对图中每个面板、每条箭头、每个标注逐句说清楚它在讲什么、为什么这样设计。这是笔记最有价值的部分,读者看图时无需对照原文。

**图解结构**:用 `**Fig N 逐段解读**` 作为 blockquote 首行,然后按面板/区域分小段,每段用加粗标题标出范围,再逐句说明。

```markdown
![Fig 1: 三种架构对比](./figures/fig1_arch.png)

> **Fig 1 逐段解读**：
>
> **(a) VAE-based**——理解和生成共用同一 Transformer。生成路径：图像经冻结 VAE encoder（雪花图标）压缩为 latent，扩散后由冻结 VAE decoder 还原像素。问题：VAE latent space 为重建优化，与 UMM 整体目标不一致，有损压缩设了质量死上限。
>
> **(b) Naive pixel-space**——移除 VAE，生成直接接 Pixel head。消除了外部 bottleneck，但模型必须从同一原始像素信号同时学高层语义和低层纹理，两任务相互干扰（GenEval 仅 0.25）。
>
> **(c) Representation Forcing**——新增 Rep head：解码器先自回归预测 rep token（来自自身理解编码器的离散视觉表示），这些 token 以 in-context 方式（虚线箭头）留在序列中引导 Pixel head 生成像素，无需任何外部 VAE。
```

**图解写作要点**：
- **面板/子图**：用 `**(a) 标题**`、`**(b) 标题**` 等结构区分，每面板独立成段。
- **箭头和连接**：说明信号从哪里流向哪里，流的是什么（特征/token/梯度），为什么走这条路。
- **特殊标注**：雪花图标（冻结）、虚线框/箭头（可选路径/推理时跳过）、颜色区分等都要解释含义。
- **设计动机**：每段末点出"这样设计是为了解决什么问题"或"与前一面板相比改了什么"。
- **数量级**：如果图中有数字、坐标轴、颜色深浅，说清楚它们代表什么量级的变化。
- 架构图/流程图用逐面板格式；定性对比图（如消融 before/after）用**逐列**或**逐行**格式，每列/行说具体失效症状和改善效果。

**定性对比图示例（逐列）**：
```markdown
> **Fig 4 逐列对比**：上行 without RF，下行 with RF，五列对应五个 prompt：
>
> - **列 1（香水瓶）**：无 RF 时瓶身轮廓模糊粘连；有 RF 时瓶身独立清晰，玻璃光晕渲染准确。
> - **列 2（猕猴桃猫）**：无 RF 时猫脸五官错位；有 RF 时轮廓正确，切片有序排布，最直观体现 rep token 对空间布局的约束。
> - **列 3（猫在篮中）**：无 RF 时猫与篮边界融合；有 RF 时姿势自然，篮子编织纹理规整。
>
> 总结：rep token 预测阶段确定了物体位置和整体构图，像素扩散只需填充细节，不必同时解决"在哪里"和"长什么样"两个问题。
```

### 公式 —— GitHub 正确渲染规则
GitHub 用 MathJax 渲染 `$...$`(行内)和 `$$...$$`(独立块)。遵守:
- 独立公式**单独成行**,`$$` 各占一行,前后留空行。
- 范数写 `\lVert x \rVert_1`(比 `\|x\|_1` 更稳)。
- 求和带上下标:`\sum_{t=t_a}^{t_b-1}` ✅。
- `\text{}`、`\hat{}`、`\odot`、`\frac{}{}`、`\cdot` 都支持。
- 长函数调用括号用 `\!\left( ... \right)` 控制间距。
- **行内公式里别用裸 `_` 或 `*`**(会被 markdown 当斜体/下标),包进 `$...$` 即可。
- **中文相邻的行内 `$...$` 必须改成反引号**:GitHub MathJax 依赖空格检测行内公式边界,CJK 字符紧贴 `$` 时解析失败。具体规律:
  - 失败场景:`$\langle x \rangle$` 紧跟中文、`$\text{...}$` 内有中文、`<EOS>` 被解析成 HTML 标签、`$\tilde{x}$` 出现在中文句子中。
  - 解决方案:中文正文里所有**行内符号引用**改用反引号代码段(`` `x_0` ``、`` `\delta t` ``),只在**独立 display math** 块(`$$...$$` 单独成段,前后留空行)里保留 LaTeX —— 这类块渲染稳定。
  - 经验法则:行内提到公式符号 → 反引号;完整推导公式 → `$$`块。
- 写完**提醒用户在 GitHub 上扫一眼公式渲染**,本地无法 100% 确认。

## Q&A 规范 — 持续追加,不另开文件

analysis.md 末尾始终保留一个 `## Q&A` 区块(写 analysis 时就留好占位)。每当对话中出现值得沉淀的问答——无论是当场解答还是深入追问——立刻追加到该区块并 commit。

### 格式

```markdown
---

## Q&A

**Q: 用户的原话或精炼后的问题？**

A: 答案正文。可以包含代码块、表格、公式,风格与 analysis 正文一致。
关键结论加粗,代码引用带 `file:line`。

---

**Q: 第二个问题？**

A: ...
```

- 每条 Q&A 之间用 `---` 分隔
- 问题用用户原话或紧凑精炼版,保留问题的核心意图
- 答案要能独立阅读:不假设读者看过对话上下文
- 追加后立刻 commit:`docs(<短名>): add Q&A — <问题摘要>`

### 判断什么值得写进 Q&A

| 值得写 | 不必写 |
|--------|--------|
| 澄清 paper 里没说清楚的机制 | 纯操作类("帮我 push") |
| 跨论文的横向对比 | 已在 analysis 正文详述的内容 |
| 代码实现与 paper 描述的差异 | 一句话能回答且无技术深度 |
| 让人意外的设计取舍 | |

## Step 4 — 更新索引

1. `<方向>/README.md`:论文列表表格加一行(状态 ✅)。新方向则参考 `inference_acceleration/README.md` 建谱系图 + 术语表 + 列表。
2. 根 `README.md`:索引表加一行(含一句话摘要),"当前进度"加条目。
3. 链接列优先级:官方主页 > GitHub > arXiv。

## Step 5 — 提交

- 提交风格沿用仓库:`add(<短名>): <一句话>` 或 `docs(<短名>): ...`。
- 笔记直接提交到 `main`(本仓库既有习惯,非协作 PR 流)。
- **commit 后是否 push 到 GitHub,先问用户**(push 是对外、难撤销操作),除非用户已明确说"提交上去/push"。
- commit message 结尾按全局规范加 `Co-Authored-By` 尾注。

## 一句话
通读 PDF → 按区域 `pdftoppm` 裁图并 Read 验证 → 按八段框架写 analysis(图配引言、公式守 MathJax 规则,末尾留 Q&A 占位)→ 更新两级 README → 提交;后续对话中有价值的问答随时追加到 Q&A 区块并 commit。
