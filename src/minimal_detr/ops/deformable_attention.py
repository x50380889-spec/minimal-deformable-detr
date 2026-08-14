"""Multi-Scale Deformable Attention (MSDeformAttn) implemented from scratch.

该模块是 Deformable DETR (Zhu et al., ICCV 2021) 的核心注意力算子，这里
完全不依赖 mmcv 的 CUDA 算子，也不调用 ``torch.matmul`` / ``nn.Linear``：

- query 内容、采样偏移、注意力权重、value 投影、输出投影全部由
  :func:`~minimal_detr.ops.matmul.manual_linear`（手写矩阵乘法）完成；
- 多尺度特征采样由 :func:`~minimal_detr.ops.sampling.grid_sample_manual`
  （手写双线性插值）完成；
- 注意力加权求和直接展开为逐元素乘加，没有使用任何封装好的 attention API。

公式（每个 query 在每个尺度采 ``n_points`` 个点）：

.. math::
    \\text{Attn}(q, x) = \\sum_{l=1}^{L} \\sum_{p=1}^{P} A_{l p} \\cdot
    x_l(\\phi_l(\\hat{q}) + \\Delta p_{l p})

其中 :math:`A_{lp}` 是对 ``(L*P)`` 个采样点做 softmax 得到的注意力权重，
:math:`\\Delta p_{lp}` 是网络预测的采样偏移，:math:`\\phi_l` 把归一化参考点
映射到第 :math:`l` 个尺度的坐标。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .matmul import manual_linear
from .sampling import grid_sample_manual


class MSDeformAttn(nn.Module):
    """多尺度可变形注意力（从零实现，无 mmcv / 无 torch.matmul）。

    Args:
        d_model: 输入/输出特征维数。
        n_levels: 多尺度特征层数。
        n_heads: 注意力头数。
        n_points: 每个尺度采样的关键点数。
        d_value: value 的投影维数，默认与 ``d_model`` 相同。
        dropout: 输出 dropout 概率。
        offsets_scale: 采样偏移的尺度系数（相对特征图边长）。
        matmul_mode: 手写矩阵乘法的收缩模式（``"einsum"`` 或 ``"outer"``）。
        sampling_mode: 双线性采样实现（``"manual"`` 为手写实现，默认；
            ``"native"`` 使用等价的 ``F.grid_sample``，训练更快）。

    Inputs:
        query: 查询特征，形状 ``(N, Lq, d_model)``。
        value_list: 多尺度特征列表，每个元素 ``(N, d_model, H_l, W_l)``。
        reference_points: 归一化参考点，形状 ``(N, Lq, 2)``，取值 ``[0, 1]``
            （相对整图坐标，各尺度内部自动缩放）。

    Returns:
        加权聚合后的输出，形状 ``(N, Lq, d_model)``。
    """

    def __init__(
        self,
        d_model: int,
        n_levels: int,
        n_heads: int,
        n_points: int,
        d_value: int | None = None,
        dropout: float = 0.0,
        offsets_scale: float = 1.0,
        matmul_mode: str = "einsum",
        sampling_mode: str = "manual",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_value = d_value or d_model
        self.dropout = nn.Dropout(dropout)
        self.offsets_scale = offsets_scale
        self.matmul_mode = matmul_mode
        self.sampling_mode = sampling_mode

        # ---- 手写参数（刻意不用 nn.Linear，便于展示 matmul 逻辑）----
        self.q_proj_w = nn.Parameter(torch.empty(n_heads * self.d_value, d_model))
        self.q_proj_b = nn.Parameter(torch.zeros(n_heads * self.d_value))
        self.offset_proj_w = nn.Parameter(
            torch.empty(n_heads * n_levels * n_points * 2, d_model)
        )
        self.offset_proj_b = nn.Parameter(torch.zeros(n_heads * n_levels * n_points * 2))
        self.attn_proj_w = nn.Parameter(
            torch.empty(n_heads * n_levels * n_points, d_model)
        )
        self.attn_proj_b = nn.Parameter(torch.zeros(n_heads * n_levels * n_points))
        self.value_proj_w = nn.Parameter(torch.empty(n_heads * self.d_value, d_model))
        self.value_proj_b = nn.Parameter(torch.zeros(n_heads * self.d_value))
        self.out_proj_w = nn.Parameter(torch.empty(d_model, n_heads * self.d_value))
        self.out_proj_b = nn.Parameter(torch.zeros(d_model))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier 初始化权重，偏置置零。"""
        for w in (self.q_proj_w, self.offset_proj_w, self.attn_proj_w,
                  self.value_proj_w, self.out_proj_w):
            nn.init.xavier_uniform_(w)
        for b in (self.q_proj_b, self.offset_proj_b, self.attn_proj_b,
                  self.value_proj_b, self.out_proj_b):
            nn.init.constant_(b, 0.0)

    def forward(
        self,
        query: Tensor,
        value_list: list[Tensor],
        reference_points: Tensor,
    ) -> Tensor:
        """前向计算，见类注释。"""
        n, lq, _ = query.shape
        n_heads, n_levels, n_points = self.n_heads, self.n_levels, self.n_points
        d_value = self.d_value
        device = query.device

        # 1) query content: (N, Lq, n_heads, d_value)
        q = manual_linear(query, self.q_proj_w, self.q_proj_b, mode=self.matmul_mode)
        q = q.view(n, lq, n_heads, d_value)

        # 2) sampling offsets: (N, Lq, n_heads, n_levels, n_points, 2)
        offsets = manual_linear(query, self.offset_proj_w, self.offset_proj_b, mode=self.matmul_mode)
        offsets = offsets.view(n, lq, n_heads, n_levels, n_points, 2)

        # 3) attention weights: softmax over (n_levels * n_points) per head
        attn = manual_linear(query, self.attn_proj_w, self.attn_proj_b, mode=self.matmul_mode)
        attn = attn.view(n, lq, n_heads, n_levels * n_points)
        attn = torch.softmax(attn, dim=-1).view(n, lq, n_heads, n_levels, n_points)

        # 4) per-level sampling
        sampled_levels: list[Tensor] = []
        for lvl in range(n_levels):
            feat = value_list[lvl]  # (N, d_model, H_l, W_l)
            if feat.shape[1] != self.d_model:
                raise ValueError(
                    f"value_list[{lvl}] 通道数 {feat.shape[1]} != d_model {self.d_model}"
                )
            _, _, h_l, w_l = feat.shape

            # value projection: (N, H_l, W_l, n_heads * d_value)
            v = manual_linear(
                feat.permute(0, 2, 3, 1),
                self.value_proj_w,
                self.value_proj_b,
                mode=self.matmul_mode,
            )

            # 采样位置：ref(整图归一化) + offset(以该层像素为单位)
            # 其中 offset 按该层边长归一化，得到各层自己的归一化坐标，再转到 [-1,1]
            scale = torch.tensor([w_l, h_l], dtype=query.dtype, device=device)
            loc_norm = reference_points.unsqueeze(2).unsqueeze(3) + (
                self.offsets_scale * offsets[:, :, :, lvl] / scale
            )  # (N, Lq, n_heads, n_points, 2)
            loc_grid = loc_norm * 2.0 - 1.0  # grid_sample 坐标约定

            s_total = lq * n_heads * n_points
            grid = loc_grid.reshape(n, s_total, 2)
            # 一次性采样所有头：value 通道按 [head, d_value] 排列，
            # 采样结果 (N, S, nH*dv) reshape 为 (N, lq, nH, nP, nH, dv) 后取 head 对角线；
            # 高级索引结果为 (nH, N, lq, nP, dv)，用 permute 还原为 (N, lq, nH, nP, dv)
            v_perm = v.permute(0, 3, 1, 2)  # (N, nH*dv, H_l, W_l)
            if self.sampling_mode == "manual":
                sampled = grid_sample_manual(v_perm, grid)  # (N, S, nH*dv)
            elif self.sampling_mode == "native":
                sampled = F.grid_sample(
                    v_perm,
                    grid.reshape(n, s_total, 1, 2),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )  # (N, nH*dv, S, 1)
                sampled = sampled.permute(0, 2, 3, 1).reshape(n, s_total, -1)
            else:
                raise ValueError(f"未知采样模式: {self.sampling_mode} (可选 manual/native)")
            sampled = sampled.view(n, lq, n_heads, n_points, n_heads, d_value)
            head_idx = torch.arange(n_heads, device=device)
            sampled = sampled[:, :, head_idx, :, head_idx, :].permute(1, 2, 0, 3, 4)
            sampled_levels.append(sampled)

        # 5) 注意力加权求和（显式乘加，不使用封装 attention API）
        # sampled_levels: (N, Lq, nH, nP, d_value) -> stacked: (N, Lq, nH, nL, nP, d_value)
        stacked = torch.stack(sampled_levels, dim=3)
        out = (stacked * attn.unsqueeze(-1)).sum(dim=(3, 4))  # (N, Lq, n_heads, d_value)
        out = out.reshape(n, lq, n_heads * d_value)

        # 6) 输出投影
        out = manual_linear(out, self.out_proj_w, self.out_proj_b, mode=self.matmul_mode)
        return self.dropout(out)


__all__ = ["MSDeformAttn"]
