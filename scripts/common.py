"""Shared CLI helpers for the train/eval scripts."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config, StudentConfig, TeacherConfig
from minimal_detr.models.backbones import MobileNetV3Small
from minimal_detr.models.detector import MinimalDeformableDETR, build_detector
from minimal_detr.models.losses import DenseCriterion
from minimal_detr.utils import resolve_device, set_seed


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """添加各脚本共用的命令行参数。"""
    parser.add_argument("--config", type=str, default="configs/defect.json", help="配置文件路径")
    parser.add_argument("--device", type=str, default=None, help="设备（默认 auto）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epoch 数")
    parser.add_argument("--out", type=str, default=None, help="输出权重路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")


def prepare(cfg: Config, args: argparse.Namespace) -> torch.device:
    """设置种子与设备。"""
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)
    device = resolve_device(args.device or cfg.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"[env] device={device} seed={seed}")
    return device


def build_teacher(cfg: Config) -> MinimalDeformableDETR:
    """按配置构造教师模型（极简 Deformable DETR）。"""
    t: TeacherConfig = cfg.teacher
    return build_detector(
        num_classes=cfg.num_classes,
        backbone_name=t.backbone,
        d_model=t.d_model,
        n_heads=t.n_heads,
        n_points=t.n_points,
        enc_layers=t.enc_layers,
        matmul_mode=t.matmul_mode,
        sampling_mode=t.sampling_mode,
    )


def build_student(cfg: Config) -> MinimalDeformableDETR:
    """按配置构造学生模型（MobileNetV3-Small）。"""
    s: StudentConfig = cfg.student
    return build_detector(
        num_classes=cfg.num_classes,
        backbone_name=s.backbone,
        d_model=s.d_model,
        n_heads=s.n_heads,
        n_points=s.n_points,
        enc_layers=s.enc_layers,
        width_mult=s.width_mult,
        matmul_mode=s.matmul_mode,
        sampling_mode=s.sampling_mode,
    )


def make_criterion(cfg: Config) -> DenseCriterion:
    """构造密集头损失（教师/学生通用）。"""
    return DenseCriterion(
        num_classes=cfg.num_classes,
        image_size=cfg.data.image_size,
        weight_class=5.0,
        weight_bbox=5.0,
        weight_giou=1.0,
        weight_ctr=1.0,
        focal_alpha=0.8,
        focal_gamma=0.0,
    )


def make_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float, milestones: tuple[int, ...]
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.MultiStepLR]:
    """构造 AdamW + MultiStepLR。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(milestones), gamma=0.1
    )
    return optimizer, scheduler


def model_args_of(model: MinimalDeformableDETR) -> dict[str, Any]:
    """记录重建模型所需的超参（随 checkpoint 保存）。"""
    backbone_name = (
        "mobilenet_v3_small" if isinstance(model.backbone, MobileNetV3Small) else "simple_cnn"
    )
    return {
        "num_classes": model.num_classes,
        "backbone_name": backbone_name,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_points": model.n_points,
        "enc_layers": model.enc_layers,
        "width_mult": getattr(model.backbone, "width_mult", 1.0),
        "matmul_mode": model.matmul_mode,
        "sampling_mode": model.sampling_mode,
    }


def rebuild_model(model_args: dict[str, Any]) -> MinimalDeformableDETR:
    """按 checkpoint 里的超参重建模型。"""
    args = dict(model_args)
    args["backbone_name"] = args.get("backbone_name", "simple_cnn")
    return build_detector(**args)


def save_checkpoint(path: str | Path, model: torch.nn.Module, model_args: dict[str, Any],
                    epoch: int, map50: float) -> None:
    """保存 checkpoint（含重建所需超参）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_args": model_args,
            "epoch": epoch,
            "map50": map50,
        },
        path,
    )
