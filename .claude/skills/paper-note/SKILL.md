---
name: paper-note
description: Add a paper reading note to this repo (paper_read) from a source PDF — extract figures as cropped PNGs, write analysis.md with GitHub-rendering LaTeX formulas and embedded figures, update the direction + root README. Use when the user wants to "add a paper", "写论文解读", "把这篇论文加进来", or turn a PDF into a structured note with images and equations.
---

# Paper Note — 把一篇论文 PDF 变成带图带公式的解读

把一篇论文(PDF + 可选代码 repo)整理成本仓库格式的阅读笔记。核心难点有两个,这个 skill 就是为它俩存在的:**(1) 从 PDF 抽出能正确显示的图片;(2) 写出能在 GitHub 上正确渲染的公式。**

## 产出物

```
<方向>/<论文短名>/
├── analysis.md          ← 主解读(必须)
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

### 风格
- 中文正文,技术词保留英文。标点跟随本仓库:正文用 ASCII 逗号 `,` + 中文句号 `。` + 顿号 `、`,行内技术符号用 ASCII `:` `(` `)`。
- 多用表格、ASCII 流程图、`file:line` 引用。深挖点用 `📌` 标。

### 插图
相对路径引用,每张图配一句 `>` 引言说明它在讲什么:
```markdown
![Fig 2: 概念图](./figures/fig2_concept.png)

> Fig 2:一句话点出这张图的 takeaway(不是照抄 caption)。
```

### 公式 —— GitHub 正确渲染规则
GitHub 用 MathJax 渲染 `$...$`(行内)和 `$$...$$`(独立块)。遵守:
- 独立公式**单独成行**,`$$` 各占一行,前后留空行。
- 范数写 `\lVert x \rVert_1`(比 `\|x\|_1` 更稳)。
- 求和带上下标:`\sum_{t=t_a}^{t_b-1}` ✅。
- `\text{}`、`\hat{}`、`\odot`、`\frac{}{}`、`\cdot` 都支持。
- 长函数调用括号用 `\!\left( ... \right)` 控制间距。
- **行内公式里别用裸 `_` 或 `*`**(会被 markdown 当斜体/下标),包进 `$...$` 即可。
- 写完**提醒用户在 GitHub 上扫一眼公式渲染**,本地无法 100% 确认。

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
通读 PDF → 按区域 `pdftoppm` 裁图并 Read 验证 → 按八段框架写 analysis(图配引言、公式守 MathJax 规则)→ 更新两级 README → 提交(push 前确认)。
