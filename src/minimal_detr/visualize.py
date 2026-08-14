"""Visualization helpers: dataset samples, prediction grids, comparison charts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from .data.defects import DEFECT_NAMES


def draw_dataset_samples(
    dataset,
    indices: Sequence[int],
    save_path: str | Path,
    cols: int = 3,
) -> None:
    """把数据集中若干样本画成网格（含真值框）。"""
    n = len(indices)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.ravel()
    for ax, idx in zip(axes, indices):
        img, target = dataset[idx]
        ax.imshow(img.squeeze(0), cmap="gray", vmin=0, vmax=1)
        size = img.shape[-1]
        for box, label in zip(target["boxes"], target["labels"]):
            cx, cy, w, h = box * size
            rect = plt.Rectangle(
                (cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor="lime", linewidth=1.2
            )
            ax.add_patch(rect)
            ax.text(
                cx - w / 2, cy - h / 2 - 2, DEFECT_NAMES[int(label)],
                color="lime", fontsize=7, va="bottom",
            )
        ax.set_title(f"sample #{idx}")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def draw_predictions_grid(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    save_path: str | Path,
    num_samples: int = 6,
    conf_threshold: float = 0.3,
) -> None:
    """在若干验证图上画预测框与真值框。"""
    from .engine import predict

    model.eval()
    rows = (num_samples + 2) // 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = axes.ravel()
    drawn = 0
    for batch in loader:
        images = batch["images"]
        for b in range(images.shape[0]):
            if drawn >= num_samples:
                break
            ax = axes[drawn]
            img = images[b]
            size = img.shape[-1]
            ax.imshow(img.squeeze(0).cpu(), cmap="gray", vmin=0, vmax=1)
            m = int(batch["valid"][b].sum().item())
            for i in range(m):
                cx, cy, w, h = batch["boxes"][b, i] * size
                ax.add_patch(
                    plt.Rectangle(
                        (cx - w / 2, cy - h / 2), w, h,
                        fill=False, edgecolor="lime", linewidth=1.2,
                    )
                )
            res = predict(model, images[b : b + 1].to(device), conf_threshold=conf_threshold)
            for box, label, score in zip(res["boxes"], res["labels"], res["scores"]):
                x1, y1, x2, y2 = box.tolist()
                ax.add_patch(
                    plt.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        fill=False, edgecolor="red", linewidth=1.2,
                    )
                )
                ax.text(x1, y1 - 2, f"{DEFECT_NAMES[int(label)]} {float(score):.2f}",
                        color="red", fontsize=7, va="bottom")
            ax.set_title(f"green=GT, red=pred (sample #{drawn})")
            ax.axis("off")
            drawn += 1
        if drawn >= num_samples:
            break
    for ax in axes[drawn:]:
        ax.axis("off")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


__all__ = ["draw_dataset_samples", "draw_predictions_grid"]
