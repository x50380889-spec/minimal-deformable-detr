"""Training / evaluation / benchmarking loops with optional TensorBoard logging."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import Config
from .models.losses import decode_dense_boxes
from .utils import AverageMeter, JsonlWriter, box_cxcywh_to_xyxy, box_iou, nms


def make_tb_writer(log_dir: str | Path):
    """创建 TensorBoard SummaryWriter（优先 torch，回退 tensorboardX）；都未安装时返回 None。"""
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        try:
            from tensorboardX import SummaryWriter

            return SummaryWriter(logdir=str(log_dir))
        except ImportError:
            return None


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Any = None,
    scalars: Optional[JsonlWriter] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_steps: Optional[int] = None,
) -> dict[str, float]:
    """训练一个 epoch，返回平均损失。

    Args:
        model / criterion / loader / optimizer / device: 常规训练组件。
        epoch: 当前 epoch（用于日志）。
        writer: TensorBoard writer（可为 None）。
        scalars: JSONL 标量记录器（可为 None）。
        scaler: AMP GradScaler（CPU 时为 None）。
        max_steps: 单 epoch 最多迭代步（调试用）。
    """
    model.train()
    meters = {name: AverageMeter() for name in ("loss", "loss_class", "loss_bbox", "loss_giou")}
    global_step = (epoch - 1) * len(loader)
    use_amp = scaler is not None and torch.cuda.is_available()

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = batch["images"].to(device)
        gt_boxes = batch["boxes"].to(device)
        gt_labels = batch["labels"].to(device)
        valid = batch["valid"].to(device)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                losses = criterion(outputs, gt_boxes, gt_labels, valid)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            losses = criterion(outputs, gt_boxes, gt_labels, valid)
            losses["loss"].backward()
            optimizer.step()

        for name, meter in meters.items():
            meter.update(float(losses[name].detach().cpu()), images.shape[0])

        if step % 10 == 0:
            values = {name: meter.val for name, meter in meters.items()}
            values["lr"] = float(optimizer.param_groups[0]["lr"])
            if writer is not None:
                writer.add_scalars("train", values, global_step + step)
            if scalars is not None:
                scalars.write(global_step + step, {"train/" + k: v for k, v in values.items()})

    return {name: meter.avg for name, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    conf_threshold: float = 0.01,
) -> dict[str, Any]:
    """在验证集上评估：返回 mAP@0.5、各类 AP、验证 loss。

    Args:
        model: 检测模型。
        criterion: 可选（计算验证 loss）。
        loader: 验证 DataLoader。
        device / num_classes: 设备与前景类别数。
        conf_threshold: 预测置信度阈值（mAP 计算时更低阈值更严谨）。
    """
    model.eval()
    loss_meter = AverageMeter()
    all_preds: list[dict[str, list[Tensor]]] = []
    all_targets: list[dict[str, Tensor]] = []

    for batch in loader:
        images = batch["images"].to(device)
        gt_boxes = batch["boxes"].to(device)
        gt_labels = batch["labels"].to(device)
        valid = batch["valid"].to(device)
        if criterion is not None:
            outputs = model(images)
            losses = criterion(outputs, gt_boxes, gt_labels, valid)
            loss_meter.update(float(losses["loss"].detach().cpu()), images.shape[0])
        else:
            outputs = model(images)

        n_batch = images.shape[0]
        preds_per_image: list[dict[str, list[Tensor]]] = [
            {"boxes": [], "scores": [], "labels": []} for _ in range(n_batch)
        ]
        for lvl in range(len(outputs["pred_logits"])):
            cls_l: Tensor = outputs["pred_logits"][lvl].softmax(1)  # type: ignore[assignment]
            ctr_l = torch.sigmoid(outputs["centerness"][lvl]).squeeze(1)  # type: ignore[index]
            scores_l, labels_l = cls_l[:, :num_classes].max(dim=1)
            scores_l = scores_l * ctr_l
            raw_l = outputs["pred_boxes"][lvl]  # type: ignore[index]
            boxes_l = box_cxcywh_to_xyxy(
                decode_dense_boxes(raw_l.permute(0, 2, 3, 1))
            )  # (N, H, W, 4) 归一化 xyxy
            keep = scores_l > conf_threshold
            for b in range(n_batch):
                k = keep[b]
                preds_per_image[b]["boxes"].append(boxes_l[b][k].detach().cpu())
                preds_per_image[b]["scores"].append(scores_l[b][k].detach().cpu())
                preds_per_image[b]["labels"].append(labels_l[b][k].detach().cpu())

        for b in range(images.shape[0]):
            boxes_c = torch.cat(preds_per_image[b]["boxes"], dim=0)
            scores_c = torch.cat(preds_per_image[b]["scores"], dim=0)
            labels_c = torch.cat(preds_per_image[b]["labels"], dim=0)
            # 密集头标准后处理：逐类 NMS（与 predict() 口径一致）
            keep_boxes, keep_scores, keep_labels = [], [], []
            for cls in range(num_classes):
                cls_mask = labels_c == cls
                if cls_mask.any():
                    idx = nms(boxes_c[cls_mask], scores_c[cls_mask], iou_threshold=0.5)
                    keep_boxes.append(boxes_c[cls_mask][idx])
                    keep_scores.append(scores_c[cls_mask][idx])
                    keep_labels.append(labels_c[cls_mask][idx])
            all_preds.append(
                {
                    "boxes": torch.cat(keep_boxes, dim=0) if keep_boxes else torch.zeros(0, 4),
                    "scores": torch.cat(keep_scores, dim=0) if keep_scores else torch.zeros(0),
                    "labels": torch.cat(keep_labels, dim=0) if keep_labels else torch.zeros(0, dtype=torch.long),
                }
            )
            m = int(valid[b].sum().item())
            gt = gt_boxes[b, :m]
            all_targets.append(
                {
                    "boxes": box_cxcywh_to_xyxy(gt).detach().cpu(),
                    "labels": gt_labels[b, :m].detach().cpu(),
                }
            )

    ap_per_class = compute_ap_per_class(all_preds, all_targets, num_classes)
    map50 = float(np.mean(ap_per_class)) if len(ap_per_class) else 0.0
    return {
        "map50": map50,
        "ap_per_class": [float(x) for x in ap_per_class],
        "loss": loss_meter.avg,
        "num_preds": sum(len(p["scores"]) for p in all_preds),
    }


def compute_ap_per_class(
    preds: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> list[float]:
    """按类计算 AP（101 点插值，IoU 匹配阈值 0.5）。

    Args:
        preds: 每张图的预测（boxes xyxy / scores / labels）。
        targets: 每张图的真值。
        num_classes: 前景类别数。
        iou_threshold: 匹配 IoU 阈值。
    """
    aps: list[float] = []
    for cls in range(num_classes):
        score_tp: list[tuple[float, int]] = []
        n_gt = 0
        for pred, target in zip(preds, targets):
            gt_mask = target["labels"] == cls
            gt_boxes = target["boxes"][gt_mask]
            n_gt += gt_boxes.shape[0]
            matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
            cls_mask = pred["labels"] == cls
            cls_boxes = pred["boxes"][cls_mask]
            cls_scores = pred["scores"][cls_mask]
            order = cls_scores.argsort(descending=True)
            for i in order:
                if gt_boxes.shape[0] == 0:
                    score_tp.append((float(cls_scores[i]), 0))
                    continue
                ious = box_iou(cls_boxes[i : i + 1], gt_boxes)[0]
                best = int(ious.argmax())
                if ious[best] >= iou_threshold and not matched[best]:
                    matched[best] = True
                    score_tp.append((float(cls_scores[i]), 1))
                else:
                    score_tp.append((float(cls_scores[i]), 0))

        if n_gt == 0:
            aps.append(0.0)
            continue
        score_tp.sort(key=lambda x: x[0], reverse=True)
        tp = np.cumsum([t for _, t in score_tp])
        fp = np.cumsum([1 - t for _, t in score_tp])
        recall = tp / max(n_gt, 1)
        precision = tp / np.maximum(tp + fp, 1e-9)
        # 101 点插值
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            idx = recall >= t
            ap += float(precision[idx].max()) if idx.any() else 0.0
        aps.append(ap / 101.0)
    return aps


@torch.inference_mode()
def benchmark_fps(
    model: nn.Module,
    device: torch.device,
    image_size: int = 128,
    batch_size: int = 1,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, float]:
    """端到端推理速度基准。

    Args:
        model: 检测模型。
        device: 设备。
        image_size: 输入边长。
        batch_size: 测速 batch（默认 1，贴合端侧单帧）。
        warmup / iters: 预热与计时次数。

    Returns:
        ``{"fps": ..., "ms": ..., "ms_std": ...}``（每帧）。
    """
    model.eval()
    x = torch.randn(batch_size, 1, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = float(np.mean(times)) / batch_size
    ms_std = float(np.std(times)) / batch_size
    return {"fps": 1000.0 / ms if ms > 0 else 0.0, "ms": ms, "ms_std": ms_std}


def predict(
    model: nn.Module,
    image: Tensor,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.5,
) -> dict[str, Tensor]:
    """单图推理（含 NMS），返回 xyxy 像素坐标结果。

    Args:
        model: 检测模型。
        image: ``(1, 1, H, W)`` 归一化灰度图。
        conf_threshold: 置信度阈值。
        iou_threshold: NMS 阈值。

    Returns:
        ``{"boxes": (M,4) xyxy 像素, "scores": (M,), "labels": (M,)}``。
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        outputs = model(image.to(device))
    h, w = image.shape[-2:]
    out_boxes, out_scores, out_labels = [], [], []
    for lvl in range(len(outputs["pred_logits"])):
        cls_l = outputs["pred_logits"][lvl].softmax(1)[0]  # (C+1, H, W)
        ctr_l = torch.sigmoid(outputs["centerness"][lvl])[0, 0]  # (H, W)
        scores, labels = cls_l[:-1].max(dim=0)  # (H, W)
        scores = scores * ctr_l
        raw_l = outputs["pred_boxes"][lvl][0]  # (4, H, W)
        boxes = box_cxcywh_to_xyxy(
            decode_dense_boxes(raw_l.permute(1, 2, 0))
        )  # (H, W, 4) 归一化
        keep = scores > conf_threshold
        boxes_f = boxes[keep]
        scores_f = scores[keep]
        labels_f = labels[keep]
        if not boxes_f.numel():
            continue
        idx = nms(boxes_f, scores_f, iou_threshold)
        out_boxes.append(boxes_f[idx] * torch.tensor([w, h, w, h], device=device))
        out_scores.append(scores_f[idx])
        out_labels.append(labels_f[idx])
    if out_boxes:
        return {
            "boxes": torch.cat(out_boxes).cpu(),
            "scores": torch.cat(out_scores).cpu(),
            "labels": torch.cat(out_labels).cpu(),
        }
    return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}


__all__ = ["make_tb_writer", "train_one_epoch", "evaluate", "compute_ap_per_class", "benchmark_fps", "predict"]
