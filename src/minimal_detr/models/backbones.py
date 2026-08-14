"""Backbones: a residual CNN (teacher) and MobileNetV3-Small (student).

两者都输出多尺度特征图：

- :class:`SimpleCNN`：ResNet 风格残差主干，输出 stride 4/8/16 三层特征（教师）；
- :class:`MobileNetV3Small`：按论文复现的移动端主干（h-swish + SE），
  输出 stride 8/16 两层特征（学生，可部署在端侧）。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn


def _round_channels(ch: int, width_mult: float) -> int:
    """按宽度倍率缩放通道数。"""
    return int(ch * width_mult)


class _ConvBnAct(nn.Module):
    """Conv2d + BatchNorm2d + 激活。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act: nn.Module
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "hswish":
            self.act = HardSwish()
        elif activation is None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"未知激活: {activation}")

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class HardSwish(nn.Module):
    """MobileNetV3 的 h-swish 激活：``x * ReLU6(x+3) / 6``。"""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class SqueezeExcite(nn.Module):
    """SE 通道注意力模块（MobileNetV3 版，使用 h-sigmoid）。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        out = torch.mean(x, dim=(2, 3), keepdim=True)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        gate = torch.clamp(out + 3.0, min=0.0, max=6.0) / 6.0  # h-sigmoid
        return x * gate


# ---------------------------------------------------------------------------
# 教师主干：SimpleCNN（ResNet 风格残差网络）
# ---------------------------------------------------------------------------
class _BasicBlock(nn.Module):
    """2 层卷积残差块，可选下采样。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class SimpleCNN(nn.Module):
    """轻量 ResNet 风格多尺度主干（教师网络）。

    输出 stride 8/16/32 三层特征，与 Deformable DETR 原论文（ResNet C3/C4/C5）
    的下采样节奏一致；128x128 输入下 token 数为 16²+8²+4²=336，
    显著低于 stride 4/8/16 方案的 1344 个 token（可变形注意力成本与其成正比）。

    Args:
        in_channels: 输入图像通道数（工业相机灰度图为 1）。
        channels: 四个阶段的输出通道数。

    Returns:
        ``forward`` 返回 stride 8/16/32 的三层特征列表。
    """

    def __init__(self, in_channels: int = 1, channels: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.channels = tuple(channels)
        self.strides = (8, 16, 32)
        c0, c1, c2, c3 = self.channels
        self.stem = _ConvBnAct(in_channels, c0, 3, stride=2, padding=1)
        self.stage1 = nn.Sequential(_BasicBlock(c0, c0, stride=2), _BasicBlock(c0, c0))  # stride 4（不输出）
        self.stage2 = nn.Sequential(_BasicBlock(c0, c1, stride=2), _BasicBlock(c1, c1))  # stride 8
        self.stage3 = nn.Sequential(_BasicBlock(c1, c2, stride=2), _BasicBlock(c2, c2))  # stride 16
        self.stage4 = nn.Sequential(_BasicBlock(c2, c3, stride=2), _BasicBlock(c3, c3))  # stride 32

    @property
    def out_channels(self) -> tuple[int, ...]:
        return self.channels[1:]

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)  # stride 2
        x = self.stage1(x)  # stride 4（内部用）
        f8 = self.stage2(x)  # stride 8
        f16 = self.stage3(f8)  # stride 16
        f32 = self.stage4(f16)  # stride 32
        return [f8, f16, f32]


# ---------------------------------------------------------------------------
# 学生主干：MobileNetV3-Small（按论文复现）
# ---------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    """MobileNetV3 倒残差块：expand -> depthwise -> (SE) -> project。"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation: str,
    ) -> None:
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential()
        if hidden_channels != in_channels:
            self.conv.append(
                _ConvBnAct(in_channels, hidden_channels, 1, activation=activation)
            )
        self.conv.append(
            _ConvBnAct(
                hidden_channels,
                hidden_channels,
                kernel_size,
                stride,
                padding=kernel_size // 2,
                groups=hidden_channels,
                activation=activation,
            )
        )
        if use_se:
            self.conv.append(SqueezeExcite(hidden_channels))
        self.conv.append(
            nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        return x + out if self.use_shortcut else out


class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small（学生网络），输出 stride 8/16 两层特征。

    结构与论文 Table 2 保持一致（12 个 bneck，SE + h-swish），
    仅去掉分类头并保留中间特征图。

    Args:
        in_channels: 输入通道数。
        width_mult: 宽度倍率，用于进一步压缩模型（端侧部署参数）。
    """

    # (kernel, expand, out, se, activation, stride)
    _CFG: tuple[tuple[int, int, int, bool, str, int], ...] = (
        (3, 16, 16, True, "relu", 2),
        (3, 72, 24, False, "relu", 2),
        (3, 88, 24, False, "relu", 1),
        (5, 96, 40, True, "hswish", 2),
        (5, 240, 40, True, "hswish", 1),
        (5, 240, 40, True, "hswish", 1),
        (5, 120, 48, True, "hswish", 1),
        (5, 144, 48, True, "hswish", 1),
        (5, 288, 96, True, "hswish", 2),
        (5, 576, 96, True, "hswish", 1),
        (5, 576, 96, True, "hswish", 1),
    )

    def __init__(self, in_channels: int = 1, width_mult: float = 1.0) -> None:
        super().__init__()
        self.width_mult = width_mult
        self.features = nn.Sequential(
        )
        cur = _round_channels(16, width_mult)
        self.features.append(
            _ConvBnAct(in_channels, cur, 3, stride=2, padding=1, activation="hswish")
        )
        for kernel, expand, out, use_se, act, stride in self._CFG:
            self.features.append(
                InvertedResidual(
                    in_channels=cur,
                    hidden_channels=_round_channels(expand, width_mult),
                    out_channels=_round_channels(out, width_mult),
                    kernel_size=kernel,
                    stride=stride,
                    use_se=use_se,
                    activation=act,
                )
            )
            cur = _round_channels(out, width_mult)
        # 两个输出层索引：stride 8（b1，索引 2）与 stride 16（b7，索引 8）
        # 累计下采样：stem(2) -> b0(4) -> b1(8) -> b3(16) -> b8(32)
        self._feat8_idx = 2
        self._feat16_idx = 8
        self._feat8_channels = _round_channels(24, width_mult)
        self._feat16_channels = _round_channels(48, width_mult)

    @property
    def out_channels(self) -> tuple[int, int]:
        return (self._feat8_channels, self._feat16_channels)

    @property
    def strides(self) -> tuple[int, int]:
        return (8, 16)

    def forward(self, x: Tensor) -> list[Tensor]:
        out: list[Tensor] = []
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx == self._feat8_idx:
                out.append(x)
            if idx == self._feat16_idx:
                out.append(x)
        return out


__all__ = ["SimpleCNN", "MobileNetV3Small"]
