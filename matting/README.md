# Matting

图像/视频抠图方向的论文阅读笔记。核心问题:从图像或视频序列中精确提取前景 alpha matte。

## 核心概念

**Alpha Matting 公式**: `I = αF + (1-α)B`

| 子任务 | 描述 |
|-------|------|
| **Image Matting** | 从单张图片精确估计像素级 alpha 值 |
| **Video Matting** | 跨帧连贯地估计 alpha,需兼顾时序一致性 |

## 两大范式

```
          Image Matting
         (细节精准, 无时序)
               │
    ┌──────────┴───────────┐
    ▼                      ▼
 Trimap-based          Automatic
  (输入 trimap)         (无额外输入)
  GFM, P3M-10k         SalientM

          Video Matting
         (时序一致 + 细节)
               │
    ┌──────────┴───────────┐
    ▼                      ▼
End-to-End             Tracker-to-Matting
(RVM, VideoMatte)      (SAM2Matting)
    │                      │
训练于视频数据          冻结 VOS + 仅图像数据
 时序建模强              追踪鲁棒 + 零样本泛化
```

## 论文列表

| 简称 | 标题 | 任务 | 发表 | 链接 | 状态 |
|------|------|------|------|------|------|
| [sam2matting](./sam2matting/analysis.md) | SAM2Matting: Generalized Image and Video Matting | 图像+视频 alpha matting | Fudan+SUFE, 2026 | [github](https://github.com/FudanCVL/SAM2Matting) | ✅ |
