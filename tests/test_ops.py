"""算子正确性单测：手写实现 vs PyTorch 参考实现。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.ops.deformable_attention import MSDeformAttn
from minimal_detr.ops.matmul import manual_bmm, manual_linear, matmul_naive
from minimal_detr.ops.sampling import grid_sample_manual
from minimal_detr.utils import giou, hungarian


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (1, 7, 7, 7), (4, 16, 8, 8)])
def test_manual_bmm_matches_torch(shape: tuple[int, ...]) -> None:
    """手写 batched matmul 与 torch.matmul 一致。"""
    b, i, k, j = shape
    a = torch.randn(b, i, k)
    c = torch.randn(b, k, j)
    ref = torch.matmul(a, c)
    for mode in ("outer", "einsum"):
        out = manual_bmm(a, c, mode=mode)
        assert torch.allclose(out, ref, atol=1e-5), (
            f"mode={mode} max err = {(out - ref).abs().max()}"
        )


def test_manual_linear_matches_nn_linear() -> None:
    """手写线性投影与 nn.Linear（同权重）一致。"""
    in_f, out_f = 8, 16
    x = torch.randn(2, 5, in_f)
    w = torch.randn(out_f, in_f)
    b = torch.randn(out_f)
    out = manual_linear(x, w, b)
    ref = F.linear(x, w, b)
    assert torch.allclose(out, ref, atol=1e-5)
    # 无偏置
    out2 = manual_linear(x, w)
    ref2 = F.linear(x, w)
    assert torch.allclose(out2, ref2, atol=1e-5)


@pytest.mark.parametrize("align_corners", [False, True])
def test_grid_sample_matches_torch(align_corners: bool) -> None:
    """手写双线性采样与 F.grid_sample（bilinear + zeros padding）一致。"""
    torch.manual_seed(0)
    features = torch.randn(2, 3, 5, 7)
    grid = torch.rand(2, 100, 1, 2) * 2.4 - 1.2  # 包含越界点
    out = grid_sample_manual(features, grid.reshape(2, 100, 2), align_corners=align_corners)
    ref = F.grid_sample(
        features, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    ).permute(0, 2, 3, 1).reshape(2, 100, 3)
    assert torch.allclose(out, ref, atol=1e-5), f"max err = {(out - ref).abs().max()}"


def test_ms_deform_attn_matches_reference() -> None:
    """MSDeformAttn（全手写）与「torch.matmul + F.grid_sample」参考实现一致。"""
    torch.manual_seed(1)
    n, lq, d_model, n_heads, n_levels, n_points = 2, 11, 8, 2, 2, 3
    attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
    query = torch.randn(n, lq, d_model)
    shapes = [(7, 7), (4, 4)]
    values = [torch.randn(n, d_model, h, w) for h, w in shapes]
    ref_pts = torch.rand(n, lq, 2)

    out = attn(query, values, ref_pts)

    # ---- 参考实现（使用官方 API，仅用于验证）----
    def ref_ms_deform_attn() -> torch.Tensor:
        d_v = attn.d_value
        q = F.linear(query, attn.q_proj_w, attn.q_proj_b).view(n, lq, n_heads, d_v)
        offsets = F.linear(query, attn.offset_proj_w, attn.offset_proj_b).view(
            n, lq, n_heads, n_levels, n_points, 2
        )
        attn_w = F.linear(query, attn.attn_proj_w, attn.attn_proj_b).view(
            n, lq, n_heads, n_levels * n_points
        )
        attn_w = attn_w.softmax(-1).view(n, lq, n_heads, n_levels, n_points)
        sampled = []
        for lvl, feat in enumerate(values):
            h_l, w_l = feat.shape[-2:]
            v = F.linear(feat.permute(0, 2, 3, 1), attn.value_proj_w, attn.value_proj_b)
            v = v.view(n, h_l, w_l, n_heads, d_v)
            scale = torch.tensor([w_l, h_l], dtype=query.dtype)
            loc_norm = ref_pts.unsqueeze(2).unsqueeze(3) + attn.offsets_scale * offsets[:, :, :, lvl] / scale
            loc_grid = loc_norm * 2.0 - 1.0
            s_total = lq * n_heads * n_points
            grid = loc_grid.reshape(n, s_total, 2)
            sampled_heads = []
            for head in range(n_heads):
                v_head = v[..., head, :].permute(0, 3, 1, 2)
                s = F.grid_sample(
                    v_head, grid.reshape(n, s_total, 1, 2),
                    mode="bilinear", padding_mode="zeros", align_corners=False,
                )
                sampled_heads.append(s.permute(0, 2, 3, 1).reshape(n, s_total, d_v))
            s = torch.stack(sampled_heads, dim=2)  # (n, S, nH, dv)
            s = s.view(n, lq, n_heads, n_points, n_heads, d_v)
            head_idx = torch.arange(n_heads)
            sampled.append(s[:, :, head_idx, :, head_idx, :].permute(1, 2, 0, 3, 4))
        stacked = torch.stack(sampled, dim=3)
        out_ref = (stacked * attn_w.unsqueeze(-1)).sum(dim=(3, 4)).reshape(n, lq, n_heads * d_v)
        return F.linear(out_ref, attn.out_proj_w, attn.out_proj_b)

    ref = ref_ms_deform_attn()
    assert torch.allclose(out, ref, atol=1e-5), f"max err = {(out - ref).abs().max()}"


def test_ms_deform_attn_modes_agree() -> None:
    """matmul 收缩模式与采样模式的不同组合产出相同结果。"""
    torch.manual_seed(3)
    n, lq, d_model, n_heads, n_levels, n_points = 2, 11, 8, 2, 2, 3
    query = torch.randn(n, lq, d_model)
    values = [torch.randn(n, d_model, 7, 7), torch.randn(n, d_model, 4, 4)]
    ref_pts = torch.rand(n, lq, 2)
    refs = {
        "einsum_manual": MSDeformAttn(d_model, n_levels, n_heads, n_points,
                                      matmul_mode="einsum", sampling_mode="manual"),
        "einsum_native": MSDeformAttn(d_model, n_levels, n_heads, n_points,
                                      matmul_mode="einsum", sampling_mode="native"),
        "outer_manual": MSDeformAttn(d_model, n_levels, n_heads, n_points,
                                     matmul_mode="outer", sampling_mode="manual"),
    }
    refs["einsum_native"].load_state_dict(refs["einsum_manual"].state_dict())
    refs["outer_manual"].load_state_dict(refs["einsum_manual"].state_dict())
    outs = {k: m(query, values, ref_pts) for k, m in refs.items()}
    assert torch.allclose(outs["einsum_manual"], outs["einsum_native"], atol=1e-5)
    assert torch.allclose(outs["einsum_manual"], outs["outer_manual"], atol=1e-5)


@pytest.mark.parametrize("rows,cols", [(5, 3), (3, 5), (6, 6), (1, 4)])
def test_hungarian_matches_scipy(rows: int, cols: int) -> None:
    """自研匈牙利算法与 scipy.linear_sum_assignment 最优解一致。"""
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(0)
    for _ in range(5):
        cost = rng.uniform(0, 10, (rows, cols))
        r1, c1 = hungarian(cost)
        r2, c2 = linear_sum_assignment(cost)
        assert np.array_equal(np.sort(r1), np.sort(r2))
        assert np.array_equal(np.sort(c1), np.sort(c2))
        assert np.isclose(cost[r1, c1].sum(), cost[r2, c2].sum(), atol=1e-8)


def test_giou_bounds() -> None:
    """GIoU 的边界行为：完全重合=1，完全分离=-1。"""
    a = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    assert torch.allclose(giou(a, a), torch.ones(1, 1), atol=1e-5)
    far = torch.tensor([[0.05, 0.05, 0.1, 0.1]])
    g = giou(a, far)
    assert g.item() < -0.85
