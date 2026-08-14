"""Hand-written 2-D bilinear sampling (a from-scratch ``F.grid_sample``)."""

from __future__ import annotations

import torch
from torch import Tensor


def grid_sample_manual(
    features: Tensor,
    grid: Tensor,
    align_corners: bool = False,
) -> Tensor:
    """手写 2-D 双线性采样，行为与 ``F.grid_sample(..., mode='bilinear',
    padding_mode='zeros', align_corners=False)`` 一致（不调用官方实现）。

    思路：
    1. 把归一化坐标 ``[-1, 1]`` 映射回像素坐标；
    2. 找到采样点周围的 4 个像素并计算插值权重；
    3. 按 ``padding_mode='zeros'`` 语义处理边界：**某个角点越界时，该角点
       直接舍弃（其权重不再参与归一化）**，全部角点越界时输出 0。
       该语义与本机 PyTorch CPU 实现逐点对齐（测试中已验证）。

    Args:
        features: 特征图，形状 ``(N, C, H, W)``。
        grid: 采样坐标，形状 ``(N, S, 2)``，最后一维为 ``(x, y)``，取值 ``[-1, 1]``。
        align_corners: 是否按像素角点对齐；本实现默认 ``False``（与
            ``F.grid_sample`` 默认行为一致），像素中心映射到 ``[-1, 1]``。

    Returns:
        形状 ``(N, S, C)`` 的采样结果。

    Raises:
        ValueError: 当输入形状不合法时。
    """
    if features.ndim != 4:
        raise ValueError(f"features 必须是 4D (N,C,H,W)，收到 {tuple(features.shape)}")
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"grid 必须是 (N,S,2)，收到 {tuple(grid.shape)}")
    if features.shape[0] != grid.shape[0]:
        raise ValueError("features 与 grid 的 batch 维不一致")

    n, c, h, w = features.shape
    s = grid.shape[1]
    device = features.device

    grid = torch.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    gx = grid[..., 0]  # (N, S)
    gy = grid[..., 1]  # (N, S)

    # 归一化坐标 -> 像素坐标（align_corners=False 的映射：中心对齐）
    if align_corners:
        px = (gx + 1.0) / 2.0 * (w - 1)
        py = (gy + 1.0) / 2.0 * (h - 1)
    else:
        px = (gx + 1.0) / 2.0 * w - 0.5
        py = (gy + 1.0) / 2.0 * h - 0.5

    x0 = torch.floor(px).long()
    y0 = torch.floor(py).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (px - x0.float()).clamp(0.0, 1.0)  # 权重 (N, S)
    wy = (py - y0.float()).clamp(0.0, 1.0)

    # 越界角点标记（padding_mode='zeros' 语义：越界角点直接舍弃）
    keep_x0 = (x0 >= 0) & (x0 <= w - 1)
    keep_x1 = (x1 >= 0) & (x1 <= w - 1)
    keep_y0 = (y0 >= 0) & (y0 <= h - 1)
    keep_y1 = (y1 >= 0) & (y1 <= h - 1)

    # 索引越界时 clamp 到边缘（仅用于取值，权重由 keep_* 控制）
    x0_c = x0.clamp(0, w - 1)
    x1_c = x1.clamp(0, w - 1)
    y0_c = y0.clamp(0, h - 1)
    y1_c = y1.clamp(0, h - 1)

    f00 = _gather_2d(features, y0_c, x0_c)  # (N, S, C)
    f10 = _gather_2d(features, y0_c, x1_c)
    f01 = _gather_2d(features, y1_c, x0_c)
    f11 = _gather_2d(features, y1_c, x1_c)

    w00 = ((1 - wx) * (1 - wy)) * (keep_x0 & keep_y0)
    w10 = (wx * (1 - wy)) * (keep_x1 & keep_y0)
    w01 = ((1 - wx) * wy) * (keep_x0 & keep_y1)
    w11 = (wx * wy) * (keep_x1 & keep_y1)

    out = (
        f00 * w00.unsqueeze(-1)
        + f10 * w10.unsqueeze(-1)
        + f01 * w01.unsqueeze(-1)
        + f11 * w11.unsqueeze(-1)
    )
    return out


def _gather_2d(features: Tensor, ys: Tensor, xs: Tensor) -> Tensor:
    """按 (y, x) 坐标收集特征值，返回 ``(N, S, C)``。

    Args:
        features: ``(N, C, H, W)``。
        ys / xs: ``(N, S)`` 的长整型坐标（已 clamp 到合法范围）。
    """
    n, c, h, w = features.shape
    s = ys.shape[1]
    idx = (ys * w + xs).unsqueeze(1).expand(n, c, s)  # (N, C, S)
    flat = features.reshape(n, c, h * w)
    return flat.gather(2, idx).permute(0, 2, 1)  # (N, S, C)


__all__ = ["grid_sample_manual"]
