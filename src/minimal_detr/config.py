"""Typed configuration objects (JSON-friendly, merged over sane defaults)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def _merge_into(dc: T, data: dict[str, Any]) -> T:
    """把 ``data`` 中的字段递归覆盖到 dataclass 默认值上。"""
    field_map = {f.name: f for f in fields(dc)}
    for key, value in data.items():
        if key not in field_map:
            raise KeyError(f"未知配置字段: {key} (期望: {sorted(field_map)})")
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            setattr(dc, key, _merge_into(current, value))
        else:
            setattr(dc, key, value)
    return dc


@dataclass
class DataConfig:
    """数据与数据加载配置。"""

    train_size: int = 600
    val_size: int = 200
    image_size: int = 128
    max_objects: int = 5
    batch_size: int = 8
    num_workers: int = 0
    seed: int = 42


@dataclass
class TeacherConfig:
    """教师网络（极简 Deformable DETR）训练配置。"""

    backbone: str = "simple_cnn"
    channels: tuple[int, ...] = (32, 64, 128, 256)
    d_model: int = 128
    n_heads: int = 4
    n_points: int = 4
    enc_layers: int = 2
    dec_layers: int = 2
    num_queries: int = 20
    matmul_mode: str = "einsum"
    sampling_mode: str = "native"
    epochs: int = 20
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lr_milestones: tuple[int, ...] = (8, 15)


@dataclass
class StudentConfig:
    """学生网络（MobileNetV3-Small）训练配置。"""

    backbone: str = "mobilenet_v3_small"
    width_mult: float = 1.0
    d_model: int = 96
    n_heads: int = 3
    n_points: int = 4
    enc_layers: int = 2
    dec_layers: int = 1
    num_queries: int = 15
    matmul_mode: str = "einsum"
    sampling_mode: str = "native"
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_milestones: tuple[int, ...] = (8, 16)


@dataclass
class DistillConfig:
    """知识蒸馏训练配置。"""

    teacher_ckpt: str = "outputs/teacher.pt"
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_milestones: tuple[int, ...] = (8, 16)
    temperature: float = 4.0
    kd_cls_w: float = 0.5
    kd_box_w: float = 0.5
    kd_giou_w: float = 0.2
    kd_feat_w: float = 0.2
    task_cls_w: float = 1.0
    task_box_w: float = 2.0
    task_giou_w: float = 1.0


@dataclass
class Config:
    """全局配置，由 ``configs/defect.json`` 覆盖默认值。"""

    num_classes: int = 4
    seed: int = 42
    device: str = "auto"
    image_size: int = 128
    root: str = "."
    output_dir: str = "outputs"
    log_dir: str = "training_logs"
    tensorboard_dir: str = "training_logs/tensorboard"
    scalars_dir: str = "training_logs/scalars"
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        """从 JSON 文件加载配置，未出现的字段使用默认值。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls()
        return _merge_into(cfg, data)

    def save(self, path: str | Path) -> None:
        """把当前配置序列化为 JSON，便于复现。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


__all__ = ["Config", "DataConfig", "TeacherConfig", "StudentConfig", "DistillConfig"]
