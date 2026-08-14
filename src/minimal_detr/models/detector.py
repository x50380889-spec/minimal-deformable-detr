"""Dense-head detector built on the from-scratch deformable encoder.

架构取舍（工程落地导向）：

- **编码端**：Deformable DETR 的多尺度可变形编码器（本项目从零实现的核心算子）；
- **解码端**：不用 DETR 原版的可变形解码器 + 匈牙利集合预测——工业小数据上
  集合预测需要数百 epoch 才能收敛；这里换成 YOLO/FCOS 风格的**密集预测头**
  （中心网格分配 + focal 分类 + L1/GIoU 回归 + center-ness），
  5~15 epoch 即可收敛，更适合端侧小数据落地。

教师/学生共用同一套结构，仅骨干与规模不同：

- 教师：SimpleCNN（stride 8/16/32）+ d_model=128 + 2 层可变形编码器；
- 学生：MobileNetV3-Small（stride 8/16）+ d_model=64 + 1 层可变形编码器。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .backbones import MobileNetV3Small, SimpleCNN
from .transformer import DeformableEncoder, SinePositionEmbedding


class DenseHead(nn.Module):
    """轻量密集预测头：共享卷积特征 → 分类 / 回归 / center-ness 三分支。

    Args:
        in_dim: 编码器输出通道数。
        hidden_dim: 共享卷积的隐层通道数。
        num_classes: 前景类别数（分类输出为 ``num_classes + 1``，含背景）。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.shared = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_conv = nn.Conv2d(hidden_dim, num_classes + 1, 1)
        self.reg_conv = nn.Conv2d(hidden_dim, 4, 1)
        self.ctr_conv = nn.Conv2d(hidden_dim, 1, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        """分类先验：初始前景概率约 0.01（避免开局全预测前景）；
        回归先验：初始框小而居中（w,h≈0.1），加速小目标匹配。"""
        with torch.no_grad():
            self.cls_conv.bias.fill_(-math.log((1.0 - 0.01) / 0.01))
            self.reg_conv.bias.zero_()
            small = math.log(math.expm1(0.1))  # softplus(b) == 0.1 的解
            self.reg_conv.bias[2] = small
            self.reg_conv.bias[3] = small
            self.ctr_conv.bias.zero_()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """前向。

        Args:
            x: ``(N, in_dim, H, W)`` 单尺度编码特征。

        Returns:
            ``(cls_logits (N, C+1, H, W), reg_raw (N, 4, H, W), ctr_logits (N, 1, H, W))``。
            回归输出为归一化 cxcywh 的原始值：cx/cy 经 sigmoid、w/h 经 softplus。
        """
        feat = self.shared(x)
        return self.cls_conv(feat), self.reg_conv(feat), self.ctr_conv(feat)


class MinimalDeformableDETR(nn.Module):
    """极简 Deformable DETR（可变形编码器 + 密集解码头，教师/学生共用结构）。

    Args:
        num_classes: 前景类别数（不含背景）。
        backbone_name: ``simple_cnn`` 或 ``mobilenet_v3_small``。
        d_model: 注意力隐藏维数。
        n_heads / n_points: 可变形注意力的头数与每尺度采样点数。
        enc_layers: 可变形编码器层数。
        width_mult: MobileNet 宽度倍率。
        in_channels: 输入图像通道数。
        dropout: 注意力 dropout。
        matmul_mode: 手写矩阵乘法收缩模式（``"einsum"`` / ``"outer"``）。
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "simple_cnn",
        d_model: int = 128,
        n_heads: int = 4,
        n_points: int = 4,
        enc_layers: int = 2,
        width_mult: float = 1.0,
        in_channels: int = 1,
        dropout: float = 0.1,
        matmul_mode: str = "einsum",
        sampling_mode: str = "manual",
    ) -> None:
        super().__init__()
        if backbone_name == "simple_cnn":
            self.backbone = SimpleCNN(in_channels=in_channels)
        elif backbone_name == "mobilenet_v3_small":
            self.backbone = MobileNetV3Small(in_channels=in_channels, width_mult=width_mult)
        else:
            raise ValueError(f"未知骨干: {backbone_name}")

        self.num_classes = num_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.enc_layers = enc_layers
        self.matmul_mode = matmul_mode
        self.sampling_mode = sampling_mode
        self.n_levels = len(self.backbone.out_channels)
        self.strides = tuple(self.backbone.strides)

        self.input_proj = nn.ModuleList(
            [nn.Conv2d(c, d_model, 1) for c in self.backbone.out_channels]
        )
        self.pos_embed = SinePositionEmbedding(d_model)
        self.encoder = DeformableEncoder(
            d_model, n_heads, self.n_levels, n_points, enc_layers, dropout,
            matmul_mode, sampling_mode,
        )
        self.head = DenseHead(d_model, d_model, num_classes)

    def forward(self, x: Tensor) -> dict[str, object]:
        """前向。

        Args:
            x: ``(N, C_in, H, W)`` 输入图像（灰度或 RGB）。

        Returns:
            dict 包含：

            - ``pred_logits``: 各尺度分类 logits 列表，元素 ``(N, C+1, H_l, W_l)``；
            - ``pred_boxes``: 各尺度回归原始值列表，元素 ``(N, 4, H_l, W_l)``；
            - ``centerness``: 各尺度 center-ness logits 列表；
            - ``encoder_features``: 编码器多尺度输出（特征蒸馏用）；
            - ``strides``: 各尺度 stride。
        """
        n = x.shape[0]
        feats = self.backbone(x)
        src_maps: list[Tensor] = []
        pos_list: list[Tensor] = []
        for proj, feat in zip(self.input_proj, feats):
            proj_feat = proj(feat)
            src_maps.append(proj_feat)  # (N, d_model, H, W)
            pos = self.pos_embed(proj_feat)  # (N, H, W, d)
            pos_list.append(pos.flatten(1, 2))  # (N, H*W, d)

        memory_maps = self.encoder(src_maps, pos_list)
        cls_list: list[Tensor] = []
        reg_list: list[Tensor] = []
        ctr_list: list[Tensor] = []
        for feat in memory_maps:
            cls_l, reg_l, ctr_l = self.head(feat)
            cls_list.append(cls_l)
            reg_list.append(reg_l)
            ctr_list.append(ctr_l)
        return {
            "pred_logits": cls_list,
            "pred_boxes": reg_list,
            "centerness": ctr_list,
            "encoder_features": memory_maps,
            "strides": self.strides,
        }


def build_detector(
    num_classes: int,
    backbone_name: str,
    d_model: int,
    n_heads: int,
    n_points: int,
    enc_layers: int,
    width_mult: float = 1.0,
    in_channels: int = 1,
    dropout: float = 0.1,
    matmul_mode: str = "einsum",
    sampling_mode: str = "manual",
) -> MinimalDeformableDETR:
    """按超参构造检测器（工厂函数）。"""
    return MinimalDeformableDETR(
        num_classes=num_classes,
        backbone_name=backbone_name,
        d_model=d_model,
        n_heads=n_heads,
        n_points=n_points,
        enc_layers=enc_layers,
        width_mult=width_mult,
        in_channels=in_channels,
        dropout=dropout,
        matmul_mode=matmul_mode,
        sampling_mode=sampling_mode,
    )


__all__ = ["MinimalDeformableDETR", "DenseHead", "build_detector"]
