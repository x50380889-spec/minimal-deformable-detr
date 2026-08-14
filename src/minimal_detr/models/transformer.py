"""Transformer building blocks (encoder/decoder) for the minimal Deformable DETR."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

from ..ops.deformable_attention import MSDeformAttn
from ..ops.matmul import manual_linear


class SinePositionEmbedding(nn.Module):
    """2-D 正弦位置编码（DETR 风格，temperature=10000）。"""

    def __init__(self, d_model: int, temperature: float = 10000.0) -> None:
        super().__init__()
        self.num_pos_feats = d_model // 2
        self.temperature = temperature

    def forward(self, features: Tensor) -> Tensor:
        """生成位置编码。

        Args:
            features: ``(N, C, H, W)``，只使用其空间尺寸。

        Returns:
            ``(N, H, W, d_model)`` 的位置编码。
        """
        n, _, h, w = features.shape
        device, dtype = features.device, features.dtype
        y = torch.arange(1, h + 1, device=device, dtype=dtype).view(1, h, 1)
        x = torch.arange(1, w + 1, device=device, dtype=dtype).view(1, 1, w)
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x.unsqueeze(-1) / dim_t  # (1,1,W,F)
        pos_y = y.unsqueeze(-1) / dim_t  # (1,H,1,F)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(3)
        pos = torch.cat((pos_y.expand(n, h, w, -1), pos_x.expand(n, h, w, -1)), dim=-1)
        return pos


class ManualMultiHeadSelfAttention(nn.Module):
    """手写多头自注意力（缩放点积），投影与注意力矩阵全部基于手写 matmul。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        matmul_mode: str = "einsum",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.matmul_mode = matmul_mode
        self.q_w = nn.Parameter(torch.empty(d_model, d_model))
        self.q_b = nn.Parameter(torch.zeros(d_model))
        self.k_w = nn.Parameter(torch.empty(d_model, d_model))
        self.k_b = nn.Parameter(torch.zeros(d_model))
        self.v_w = nn.Parameter(torch.empty(d_model, d_model))
        self.v_b = nn.Parameter(torch.zeros(d_model))
        self.o_w = nn.Parameter(torch.empty(d_model, d_model))
        self.o_b = nn.Parameter(torch.zeros(d_model))
        for w in (self.q_w, self.k_w, self.v_w, self.o_w):
            nn.init.xavier_uniform_(w)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """前向。

        Args:
            x: ``(N, L, d_model)``。

        Returns:
            ``(输出 (N, L, d_model), 注意力权重 (N, H, L, L))``。
        """
        n, length, _ = x.shape
        h, d = self.n_heads, self.head_dim
        q = manual_linear(x, self.q_w, self.q_b, mode=self.matmul_mode).view(n, length, h, d)
        k = manual_linear(x, self.k_w, self.k_b, mode=self.matmul_mode).view(n, length, h, d)
        v = manual_linear(x, self.v_w, self.v_b, mode=self.matmul_mode).view(n, length, h, d)
        # (N*H, L, d) 批量手写 matmul
        q = q.transpose(1, 2).reshape(n * h, length, d)
        k = k.transpose(1, 2).reshape(n * h, length, d)
        v = v.transpose(1, 2).reshape(n * h, length, d)
        scores = manual_bmm(q, k.transpose(1, 2), mode=self.matmul_mode) / math.sqrt(d)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = manual_bmm(attn, v, mode=self.matmul_mode)  # (N*H, L, d)
        out = out.view(n, h, length, d).transpose(1, 2).reshape(n, length, h * d)
        out = manual_linear(out, self.o_w, self.o_b, mode=self.matmul_mode)
        return out, attn.view(n, h, length, length)


def manual_bmm(a: Tensor, b: Tensor, mode: str = "einsum") -> Tensor:
    """转发到 ops.matmul 的手写实现（保持模块内 API 简洁）。"""
    from ..ops.matmul import manual_bmm as _manual_bmm

    return _manual_bmm(a, b, mode=mode)


class _FFN(nn.Module):
    """前馈网络：Linear -> ReLU -> Dropout -> Linear（标准实现）。"""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DeformableEncoderLayer(nn.Module):
    """编码器层：MSDeformAttn 自注意力 + FFN（带残差与 LayerNorm）。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
        dropout: float = 0.1,
        matmul_mode: str = "einsum",
        sampling_mode: str = "manual",
    ) -> None:
        super().__init__()
        self.attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points,
            dropout=dropout, matmul_mode=matmul_mode, sampling_mode=sampling_mode,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model, d_model * 4, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, value_list: list[Tensor], ref_points: Tensor) -> Tensor:
        """前向。

        Args:
            src: 展平后的多尺度 token，``(N, sum(H_l*W_l), d_model)``。
            value_list: 各尺度特征（用于采样），元素 ``(N, H_l*W_l, d_model)``。
            ref_points: 每个 token 的归一化参考点，``(N, L, 2)``。
        """
        src2 = self.attn(src, value_list, ref_points)
        src = self.norm1(src + src2)
        src = self.norm2(src + self.ffn(src))
        return src


class DeformableEncoder(nn.Module):
    """多尺度可变形编码器：逐层 MSDeformAttn，输出保持逐尺度划分。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
        n_layers: int,
        dropout: float = 0.1,
        matmul_mode: str = "einsum",
        sampling_mode: str = "manual",
    ) -> None:
        super().__init__()
        self.n_levels = n_levels
        self.layers = nn.ModuleList(
            [
                DeformableEncoderLayer(
                    d_model, n_heads, n_levels, n_points, dropout, matmul_mode, sampling_mode
                )
                for _ in range(n_layers)
            ]
        )
        self.level_embed = nn.Parameter(torch.zeros(n_levels, d_model))
        nn.init.normal_(self.level_embed, std=0.02)

    def _build_ref_points(self, shapes: Sequence[tuple[int, int]], n: int, device: torch.device) -> Tensor:
        """为每个尺度的每个 token 生成归一化参考点（中心对齐）。"""
        grids: list[Tensor] = []
        for h_l, w_l in shapes:
            ys = (torch.arange(h_l, device=device) + 0.5) / h_l
            xs = (torch.arange(w_l, device=device) + 0.5) / w_l
            grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), dim=-1).reshape(-1, 2)
            grids.append(grid)
        ref = torch.cat(grids, dim=0)  # (L, 2)
        return ref.unsqueeze(0).expand(n, -1, -1)

    def forward(self, src_maps: list[Tensor], pos_list: list[Tensor]) -> list[Tensor]:
        """前向。

        Args:
            src_maps: 各尺度输入特征图（4D），元素 ``(N, d_model, H_l, W_l)``。
            pos_list: 对应位置编码，元素 ``(N, H_l*W_l, d_model)``。

        Returns:
            逐尺度的编码结果列表（4D 特征图，供解码器/特征蒸馏使用）。
        """
        n = src_maps[0].shape[0]
        device = src_maps[0].device
        shapes: list[tuple[int, int]] = []
        for src in src_maps:
            side = src.shape[-1]
            if src.shape[-2] != side:
                raise ValueError("当前编码器要求方形特征图（stride 相等的下采样）")
            shapes.append((side, side))

        # 加入位置编码与尺度嵌入，然后拼接为一条序列
        flat: list[Tensor] = []
        for lvl, (src, pos) in enumerate(zip(src_maps, pos_list)):
            tokens = src.flatten(2).transpose(1, 2)  # (N, H*W, d_model)
            flat.append(tokens + pos + self.level_embed[lvl].view(1, 1, -1))
        src = torch.cat(flat, dim=1)
        ref_points = self._build_ref_points(shapes, n, device)

        for layer in self.layers:
            src = layer(src, src_maps, ref_points)

        # 按尺度切回并还原为 4D 特征图
        out: list[Tensor] = []
        start = 0
        for src_map in src_maps:
            end = start + src_map.shape[2] * src_map.shape[3]
            tokens = src[:, start:end]
            out.append(tokens.transpose(1, 2).reshape(n, -1, src_map.shape[2], src_map.shape[3]))
            start = end
        return out


class DeformableDecoderLayer(nn.Module):
    """解码器层：手写自注意力 + MSDeformAttn 交叉注意力 + FFN。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
        dropout: float = 0.1,
        matmul_mode: str = "einsum",
    ) -> None:
        super().__init__()
        self.self_attn = ManualMultiHeadSelfAttention(d_model, n_heads, dropout, matmul_mode)
        self.cross_attn = MSDeformAttn(
            d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points,
            dropout=dropout, matmul_mode=matmul_mode,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model, d_model * 4, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory_list: list[Tensor], ref_points: Tensor) -> Tensor:
        """前向。

        Args:
            tgt: 查询（object queries），``(N, Q, d_model)``。
            memory_list: 编码器输出的各尺度特征。
            ref_points: 查询的归一化参考点，``(N, Q, 2)``。
        """
        tgt2, _ = self.self_attn(tgt)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.cross_attn(tgt, memory_list, ref_points)
        tgt = self.norm2(tgt + tgt2)
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt


class DeformableDecoder(nn.Module):
    """可变形解码器：逐层预测参考点并进行交叉注意力。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
        n_layers: int,
        dropout: float = 0.1,
        matmul_mode: str = "einsum",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeformableDecoderLayer(
                    d_model, n_heads, n_levels, n_points, dropout, matmul_mode
                )
                for _ in range(n_layers)
            ]
        )
        self.anchor_proj = nn.Linear(d_model, 2)

    def forward(self, tgt: Tensor, memory_list: list[Tensor]) -> list[Tensor]:
        """前向，返回每一层的输出列表（当前仅取最后一层用于预测）。"""
        outputs: list[Tensor] = []
        for layer in self.layers:
            ref_points = torch.sigmoid(self.anchor_proj(tgt))
            tgt = layer(tgt, memory_list, ref_points)
            outputs.append(tgt)
        return outputs


__all__ = [
    "SinePositionEmbedding",
    "ManualMultiHeadSelfAttention",
    "DeformableEncoder",
    "DeformableDecoder",
]
