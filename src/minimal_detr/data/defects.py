"""合成「工业质检小目标」数据集。

场景设定：灰度工业相机画面中出现小尺寸表面缺陷（划痕 / 裂纹 / 污渍 / 凹坑），
目标框边长默认 16~40 像素（相对 128x128 图像占比约 12%~31%，仍属小目标范畴，
且可通过 ``min_size / max_size`` 参数调节到更极端的 3~14 像素）。
用于复现工业质检中典型的「小目标漏检」难题。

数据完全程序化生成（不依赖外部标注），每个样本固定随机种子，
保证训练/验证切分完全可复现。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

DEFECT_NAMES = ("scratch", "crack", "stain", "dent")
_DEFECT_KINDS = (0, 1, 2, 3)  # scratch / crack / stain / dent


def _draw_scratch(canvas: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """画一条划痕（亮色细线，区别于暗色裂纹/污渍）。"""
    h, w = canvas.shape
    cx = rng.uniform(12, w - 12)
    cy = rng.uniform(12, h - 12)
    length = rng.uniform(14, 30)
    angle = rng.uniform(0, math.pi)
    width = rng.integers(1, 3)
    dx, dy = math.cos(angle), math.sin(angle)
    for t in np.linspace(-length / 2, length / 2, max(2, int(length))):
        x = int(round(cx + dx * t))
        y = int(round(cy + dy * t))
        if 0 <= x < w and 0 <= y < h:
            shade = rng.uniform(55, 110)  # 提高对比度
            y0 = max(0, y - width // 2)
            y1 = min(h, y + width // 2 + 1)
            canvas[y0:y1, x] = np.clip(canvas[y0:y1, x] + shade, 0, 255)
    bbox_w = length * abs(dx) + width
    bbox_h = length * abs(dy) + width
    return cx, cy, bbox_w, bbox_h


def _draw_crack(canvas: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """画一条曲折裂纹（暗色折线 + 细小分叉）。"""
    h, w = canvas.shape
    cx = rng.uniform(15, w - 15)
    cy = rng.uniform(15, h - 15)
    n_seg = rng.integers(3, 6)
    pts: list[tuple[int, int]] = []
    x, y = cx, cy
    for _ in range(n_seg):
        x += rng.uniform(-10, 10)
        y += rng.uniform(-10, 10)
        pts.append((int(x), int(y)))
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        steps = max(2, int(math.hypot(x1 - x0, y1 - y0)))
        for t in np.linspace(0, 1, steps):
            xx = int(round(x0 + (x1 - x0) * t + rng.uniform(-0.5, 0.5)))
            yy = int(round(y0 + (y1 - y0) * t + rng.uniform(-0.5, 0.5)))
            if 0 <= xx < w and 0 <= yy < h:
                canvas[yy, xx] = np.clip(canvas[yy, xx] - rng.uniform(70, 120), 0, 255)
                # 细小分叉（裂纹区别于普通暗线）
                if rng.random() < 0.25:
                    bx = int(round(xx + rng.uniform(-2, 2)))
                    by = int(round(yy + rng.uniform(-2, 2)))
                    if 0 <= bx < w and 0 <= by < h:
                        canvas[by, bx] = np.clip(canvas[by, bx] - 50, 0, 255)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return float(np.mean(xs)), float(np.mean(ys)), float(max(xs) - min(xs) + 3), float(max(ys) - min(ys) + 3)


def _draw_stain(canvas: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """画一团高斯模糊污渍（椭圆软边缘）。"""
    h, w = canvas.shape
    cx = rng.uniform(12, w - 12)
    cy = rng.uniform(12, h - 12)
    rx = rng.uniform(5.0, 15.0)
    ry = rng.uniform(5.0, 15.0)
    peak = rng.uniform(45, 90)
    y0, y1 = max(0, int(cy - ry * 2.2)), min(h, int(cy + ry * 2.2) + 1)
    x0, x1 = max(0, int(cx - rx * 2.2)), min(w, int(cx + rx * 2.2) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = np.clip(region - peak * g, 0, 255)
    return cx, cy, rx * 2, ry * 2


def _draw_dent(canvas: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """画一个凹坑（中心暗 + 边缘高光）。"""
    h, w = canvas.shape
    cx = rng.uniform(12, w - 12)
    cy = rng.uniform(12, h - 12)
    r = rng.uniform(5.0, 15.0)
    y0, y1 = max(0, int(cy - r * 2.2)), min(h, int(cy + r * 2.2) + 1)
    x0, x1 = max(0, int(cx - r * 2.2)), min(w, int(cx + r * 2.2) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt(((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r + 1e-6))
    shade = 55.0 * np.clip(1.0 - d, 0, 1) - 45.0 * np.clip(d - 0.8, 0, 0.4) / 0.4
    canvas[y0:y1, x0:x1] = np.clip(canvas[y0:y1, x0:x1] + shade, 0, 255)
    return cx, cy, r * 2, r * 2


_DRAWERS = {0: _draw_scratch, 1: _draw_crack, 2: _draw_stain, 3: _draw_dent}


class DefectDataset(Dataset):
    """程序化生成的工业缺陷数据集。

    Args:
        num_samples: 样本数量。
        image_size: 图像边长（正方形灰度图）。
        max_objects: 每张图最多目标数。
        seed_offset: 种子偏移（train/val 用不同偏移以生成不同样本）。
    """

    def __init__(
        self,
        num_samples: int = 600,
        image_size: int = 128,
        max_objects: int = 5,
        seed_offset: int = 0,
        cache: bool = True,
        min_size: int = 16,
        max_size: int = 40,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_objects = max_objects
        self.seed_offset = seed_offset
        self.cache = cache
        self.min_size = min_size
        self.max_size = max_size
        self._cache: dict[int, tuple[Tensor, dict[str, Tensor]]] = {}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        """生成一张样本。

        Returns:
            ``(image, target)``；``image`` 为 ``(1, H, W)`` 灰度图，
            ``target`` 含 ``boxes``（归一化 cxcywh）、``labels``。
        """
        if self.cache and index in self._cache:
            return self._cache[index]
        rng = np.random.default_rng(1000 + self.seed_offset * 977 + index * 7919)
        size = self.image_size
        # 平滑背景：低频渐变 + 高斯噪声（模拟工业相机材质表面）
        base = 105.0 + 18.0 * np.sin(np.linspace(0, 2 * np.pi, size)).reshape(1, -1)
        base = base + 12.0 * np.cos(np.linspace(0, 3.1, size)).reshape(-1, 1)
        img = base + rng.normal(0, 2.5, (size, size))  # 降低背景噪声，减少纹理误检
        img = np.clip(img, 0, 255)

        boxes: list[list[float]] = []
        labels: list[int] = []
        n_objs = int(rng.integers(1, self.max_objects + 1))
        attempts = 0
        while len(boxes) < n_objs and attempts < 40:
            attempts += 1
            kind = int(rng.choice(_DEFECT_KINDS))
            cx, cy, bw, bh = _DRAWERS[kind](img, rng)
            # 小目标约束：边长 min_size~max_size 像素，且离图边缘留 4 像素
            bw = float(np.clip(bw, self.min_size, self.max_size))
            bh = float(np.clip(bh, self.min_size, self.max_size))
            cx = float(np.clip(cx, 4 + bw / 2, size - 4 - bw / 2))
            cy = float(np.clip(cy, 4 + bh / 2, size - 4 - bh / 2))
            if len(boxes) > 0:
                arr = np.asarray(boxes)
                dist = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
                if np.any(dist < (bw + 4) / 2 + 2):
                    continue
            boxes.append([cx, cy, bw, bh])
            labels.append(kind)

        img = img.astype(np.float32) / 255.0
        image = torch.from_numpy(img).unsqueeze(0)
        box_t = torch.as_tensor(np.asarray(boxes, dtype=np.float32), dtype=torch.float32)
        box_t[:, 0] /= size
        box_t[:, 1] /= size
        box_t[:, 2] /= size
        box_t[:, 3] /= size
        target = {"boxes": box_t, "labels": torch.as_tensor(labels, dtype=torch.long)}
        if self.cache:
            self._cache[index] = (image, target)
        return image, target


def pad_collate(batch: list[tuple[Tensor, dict[str, Tensor]]]) -> dict[str, Tensor]:
    """把变长目标 padding 成 batch（带 valid_mask 标记真实目标）。"""
    images = torch.stack([b[0] for b in batch], dim=0)
    max_objs = max(b[1]["boxes"].shape[0] for b in batch)
    num = len(batch)
    device = images.device
    boxes = torch.zeros(num, max_objs, 4, dtype=images.dtype)
    labels = torch.zeros(num, max_objs, dtype=torch.long)
    valid = torch.zeros(num, max_objs, dtype=torch.bool)
    for i, (_, target) in enumerate(batch):
        m = target["boxes"].shape[0]
        boxes[i, :m] = target["boxes"]
        labels[i, :m] = target["labels"]
        valid[i, :m] = True
    return {"images": images, "boxes": boxes, "labels": labels, "valid": valid}


def build_dataloaders(
    train_size: int,
    val_size: int,
    image_size: int,
    max_objects: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    min_size: int = 16,
    max_size: int = 40,
) -> tuple[DataLoader, DataLoader]:
    """构建训练/验证 DataLoader。

    Args:
        train_size / val_size: 训练/验证样本数。
        image_size: 图像边长。
        max_objects: 每图最大目标数。
        batch_size / num_workers: DataLoader 参数。
        seed: 种子（train/val 各用不同偏移，避免重复样本）。
        min_size / max_size: 缺陷目标框的像素尺寸范围（小目标难度旋钮）。

    Returns:
        ``(train_loader, val_loader)``。
    """
    train_ds = DefectDataset(
        train_size, image_size, max_objects, seed_offset=seed,
        min_size=min_size, max_size=max_size,
    )
    val_ds = DefectDataset(
        val_size, image_size, max_objects, seed_offset=seed + 1000,
        min_size=min_size, max_size=max_size,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=pad_collate
    )
    return train_loader, val_loader


__all__ = ["DefectDataset", "DEFECT_NAMES", "build_dataloaders", "pad_collate"]
