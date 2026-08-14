"""Minimal Deformable DETR with knowledge distillation.

一个面向「工业质检小目标漏检」场景的极简目标检测器：

- 核心算子从零实现：手写矩阵乘法 / 手写双线性采样 / 多尺度可变形注意力
  (Multi-Scale Deformable Attention)，不依赖 mmcv 或任何预构建算子；
- 教师模型：极简 Deformable DETR；
- 学生模型：MobileNetV3-Small + 轻量检测头；
- 知识蒸馏：分类 logits KL + 回归对齐 + 编码器特征对齐。
"""

__version__ = "0.1.0"
