"""Hand-written matrix multiplication primitives.

项目要求不使用 mmcv 的自定义算子；为了把「矩阵乘法逻辑」也讲清楚，
这里提供两种手写实现：

1. :func:`matmul_naive` —— 教科书式三层循环，逐元素求和（仅用于教学与测试）；
2. :func:`manual_bmm` —— 向量化手写实现：把外积显式展开后沿收缩维求和，
   与 ``torch.matmul`` 数学上完全等价，但不调用任何官方 matmul/Linear API。

:func:`manual_linear` 基于 :func:`manual_bmm` 实现全连接投影，
MSDeformAttn 的全部线性投影都使用它，从而把注意力算子做成真正的「从零实现」。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


def matmul_naive(a: Tensor, b: Tensor) -> Tensor:
    """教科书式 batched 矩阵乘法（三层循环，仅供教学/测试参考）。

    Args:
        a: 形状 ``(B, I, K)`` 的输入矩阵。
        b: 形状 ``(B, K, J)`` 的输入矩阵。

    Returns:
        形状 ``(B, I, J)`` 的矩阵积，其中 ``out[b, i, j] = sum_k a[b, i, k] * b[b, k, j]``。

    Raises:
        ValueError: 当维度不为 3 或收缩维不匹配时。
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"naive matmul 只支持 3D 输入，收到 a.ndim={a.ndim}, b.ndim={b.ndim}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"batch 维不一致: {a.shape[0]} != {b.shape[0]}")
    if a.shape[2] != b.shape[1]:
        raise ValueError(f"收缩维不一致: a.shape[2]={a.shape[2]} != b.shape[1]={b.shape[1]}")

    bsz, i_len, j_len = a.shape[0], a.shape[1], b.shape[2]
    k_len = a.shape[2]
    out = torch.zeros(bsz, i_len, j_len, dtype=a.dtype, device=a.device)
    for b_i in range(bsz):
        for i in range(i_len):
            for j in range(j_len):
                acc = 0.0
                for k in range(k_len):
                    acc += float(a[b_i, i, k] * b[b_i, k, j])
                out[b_i, i, j] = acc
    return out


def manual_bmm(a: Tensor, b: Tensor, mode: str = "einsum") -> Tensor:
    """向量化的手写 batched 矩阵乘法（不调用 ``torch.matmul``）。

    提供两种收缩策略（数学上完全等价，见单测）：

    - ``"outer"``：把 ``A[b,i,k]`` 与 ``B[b,k,j]`` 的外积显式展开为
      ``(B, I, K, J)`` 再沿收缩维求和，是矩阵乘法定义式的直接翻译，
      教学意义最强，但会显式占用 ``B*I*K*J`` 的临时内存（大矩阵易 OOM）；
    - ``"einsum"``：用 ``torch.einsum`` 表达同一个求和公式，由底层优化内核
      完成收缩，训练效率高（可差 1~2 个数量级）。

    .. math::
        C[b, i, j] = \\sum_{k} A[b, i, k] \\cdot B[b, k, j]

    Args:
        a: 形状 ``(B, I, K)``。
        b: 形状 ``(B, K, J)``。
        mode: ``"outer"`` 或 ``"einsum"``。

    Returns:
        形状 ``(B, I, J)`` 的矩阵积。

    Note:
        外积会显式占用 ``B*I*K*J`` 的临时内存；对于超长序列建议使用
        :func:`manual_linear` 自带的分块逻辑（``chunk_size``），
        否则大矩阵下容易触发 OOM / 训练极慢（详见 ``DEBUG_JOURNAL.md``）。
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"manual_bmm 只支持 3D 输入，收到 a.ndim={a.ndim}, b.ndim={b.ndim}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"batch 维不一致: {a.shape[0]} != {b.shape[0]}")
    if a.shape[2] != b.shape[1]:
        raise ValueError(f"收缩维不一致: a.shape[2]={a.shape[2]} != b.shape[1]={b.shape[1]}")
    if mode == "outer":
        # (B, I, K, J) = 外积，随后 sum over K
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(dim=-2)
    if mode == "einsum":
        # 同一个求和公式，交给底层优化内核收缩
        return torch.einsum("bik,bkj->bij", a, b)
    raise ValueError(f"未知 matmul 模式: {mode} (可选 outer/einsum)")


def manual_linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    chunk_size: int = 1024,
    mode: str = "einsum",
) -> Tensor:
    """全连接投影，内部使用手写矩阵乘法（分块以避免 OOM）。

    Args:
        x: 输入特征，形状 ``(..., in_features)``。
        weight: 权重，形状 ``(out_features, in_features)``。
        bias: 偏置，形状 ``(out_features,)``；可为 ``None``。
        chunk_size: 把展开后的 token 序列切成该大小的块，逐块做手写 matmul，
            用于控制外积展开的峰值显存。
        mode: 传给 :func:`manual_bmm` 的收缩模式。

    Returns:
        形状 ``(..., out_features)`` 的投影结果。
    """
    if weight.ndim != 2:
        raise ValueError(f"weight 必须是 2D，收到 shape={tuple(weight.shape)}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"特征维不匹配: x.shape[-1]={x.shape[-1]} != weight.shape[1]={weight.shape[1]}"
        )

    leading = x.shape[:-1]
    flat = x.reshape(-1, weight.shape[1])  # (T, in_features)
    n_tokens = flat.shape[0]
    out_chunks = []
    for start in range(0, n_tokens, chunk_size):
        chunk = flat[start : start + chunk_size]  # (C, in)
        # 手动 batched matmul：把 chunk 看作 batch=1
        prod = manual_bmm(
            chunk.unsqueeze(0), weight.t().unsqueeze(0), mode=mode
        ).squeeze(0)
        if bias is not None:
            prod = prod + bias
        out_chunks.append(prod)
    out = torch.cat(out_chunks, dim=0)
    return out.reshape(*leading, weight.shape[0])


def _xavier_uniform_(t: Tensor) -> Tensor:
    """对权重做 Xavier 均匀初始化（等价于 ``nn.init.xavier_uniform_``）。"""
    nn.init.xavier_uniform_(t)
    return t


def _constant_zero_(t: Tensor) -> Tensor:
    """把参数清零（等价于 ``nn.init.constant_(t, 0)``）。"""
    nn.init.constant_(t, 0.0)
    return t


__all__ = ["matmul_naive", "manual_bmm", "manual_linear"]
