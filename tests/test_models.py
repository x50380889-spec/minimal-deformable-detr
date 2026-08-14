"""模型/损失/蒸馏冒烟测试：形状、梯度流、数值稳定。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config
from minimal_detr.distill import DistillCriterion
from minimal_detr.models.detector import build_detector
from minimal_detr.models.losses import DenseCriterion, HungarianMatcher, SetCriterion


def _fake_batch(num_classes: int, batch: int = 2, m: int = 4) -> dict[str, torch.Tensor]:
    return {
        "images": torch.rand(batch, 1, 64, 64),
        "boxes": torch.rand(batch, m, 4),
        "labels": torch.randint(0, num_classes, (batch, m)),
        "valid": torch.tensor([[True] * m, [True] * (m - 1) + [False]]),
    }


@pytest.mark.parametrize("backbone", ["simple_cnn", "mobilenet_v3_small"])
def test_detector_forward_and_loss(backbone: str) -> None:
    """检测器前向形状正确且 dense 损失可反传（无 NaN）。"""
    torch.manual_seed(0)
    model = build_detector(
        num_classes=4,
        backbone_name=backbone,
        d_model=32,
        n_heads=2,
        n_points=2,
        enc_layers=1,
    )
    batch = _fake_batch(4, batch=2, m=3)
    out = model(batch["images"])
    assert len(out["pred_logits"]) == len(model.strides)
    n_levels = len(model.strides)
    assert out["pred_logits"][0].shape[1] == 5
    assert out["pred_boxes"][0].shape[1] == 4

    criterion = DenseCriterion(4)
    losses = criterion(out, batch["boxes"], batch["labels"], batch["valid"])
    assert torch.isfinite(losses["loss"])
    assert losses["num_pos"] > 0
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_dense_targets_small_object_has_positive() -> None:
    """小目标（小于 stride）也必须有一个正样本邻域（中心采样分配）。"""
    from minimal_detr.models.losses import build_dense_targets

    boxes = torch.tensor([[[0.5, 0.5, 0.03, 0.03]]])  # 3% 大小，远小于 stride 8
    labels = torch.tensor([[1]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    cls_t, _, _, pos = build_dense_targets(
        (4, 4), (8, 16, 32), 0, 128, boxes, labels, valid, num_classes=4
    )
    assert pos.sum().item() == 9  # 3x3 中心邻域
    assert (cls_t[pos] == 1).all()


def test_set_criterion_still_works() -> None:
    """DETR 原版集合预测损失保留可用（对照实现）。"""
    model = build_detector(
        num_classes=4, backbone_name="simple_cnn", d_model=32, n_heads=2,
        n_points=2, enc_layers=1,
    )
    batch = _fake_batch(4, batch=2, m=3)
    # 构造集合预测格式输出（仅测试 SetCriterion 本身，前向仍用密集输出）
    dummy = {
        "pred_logits": torch.randn(2, 15, 5),
        "pred_boxes": torch.rand(2, 15, 4),
    }
    criterion = SetCriterion(4, HungarianMatcher())
    losses = criterion(dummy, batch["boxes"], batch["labels"], batch["valid"])
    assert torch.isfinite(losses["loss"])
    assert model is not None


def test_distill_criterion_forward() -> None:
    """蒸馏损失前向：task loss + KD 分量均可计算。"""
    torch.manual_seed(0)
    cfg = Config()
    teacher = build_detector(
        num_classes=4, backbone_name="simple_cnn", d_model=32, n_heads=2,
        n_points=2, enc_layers=1,
    )
    student = build_detector(
        num_classes=4, backbone_name="mobilenet_v3_small", d_model=32, n_heads=2,
        n_points=2, enc_layers=1,
    )
    task_criterion = DenseCriterion(4)
    dc = DistillCriterion(
        num_classes=4,
        student_d_model=32,
        teacher_d_model=32,
        cfg=cfg.distill,
        task_criterion=task_criterion,
        image_size=64,
    )
    batch = _fake_batch(4, batch=2, m=3)
    with torch.no_grad():
        teacher_out = teacher(batch["images"])
    student_out = student(batch["images"])
    losses = dc(student_out, teacher_out, batch["boxes"], batch["labels"], batch["valid"])
    assert all(torch.isfinite(v) for v in losses.values())
    losses["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in student.parameters())
