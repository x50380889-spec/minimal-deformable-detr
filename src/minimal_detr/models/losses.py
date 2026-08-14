"""Set-prediction losses: Hungarian matching, focal classification, GIoU boxes."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils import box_cxcywh_to_xyxy, box_iou, giou, hungarian


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    num_classes: int | None = None,
) -> Tensor:
    """多分类 Focal Loss（DETR 风格）。

    Args:
        logits: ``(N, C)`` 未归一化 logits。
        targets: ``(N,)`` 类别索引。
        alpha: 前景类的 alpha 权重；背景类使用 ``1 - alpha``。
            默认 0.75 加重前景监督，缓解集合预测里背景 query 占多数导致的
            「全预测为背景」收敛陷阱。
        gamma: focal 参数。
        num_classes: 前景类别数；用于区分背景（索引 == num_classes）。

    Returns:
        逐样本 loss，形状 ``(N,)``。
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    p = torch.exp(-ce)
    alpha_t = torch.full_like(ce, 1.0 - alpha)
    if num_classes is not None:
        alpha_t = torch.where(targets == num_classes, 1.0 - alpha, alpha)
    return alpha_t * ((1.0 - p) ** gamma) * ce


def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """逐类 sigmoid focal loss（RetinaNet / FCOS 风格，多标签）。

    Args:
        inputs: ``(N, C)`` 未归一化 logits。
        targets: ``(N, C)`` 0/1 目标。
        alpha: 正样本权重；负样本权重为 ``1 - alpha``。
        gamma: 易样本降权系数。

    Returns:
        ``(N,)`` 逐样本 loss（跨类求和）。
    """
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * ((1.0 - p_t) ** gamma) * ce).sum(dim=-1)


class HungarianMatcher(nn.Module):
    """匈牙利匹配器：把预测 query 匹配到真实目标（DETR 的 bipartite matching）。

    代价 = 分类代价 + bbox L1 代价 + GIoU 代价。
    """

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: Tensor,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        valid_mask: Tensor,
    ) -> list[tuple[Tensor, Tensor]]:
        """逐样本匹配。

        Args:
            pred_logits: ``(B, Q, C+1)``。
            pred_boxes: ``(B, Q, 4)`` 归一化 cxcywh。
            gt_boxes: ``(B, M, 4)``（已 padding）。
            gt_labels: ``(B, M)``。
            valid_mask: ``(B, M)`` 真实目标掩码。

        Returns:
            每个样本的 ``(src_indices, tgt_indices)`` 列表。
        """
        batch = pred_logits.shape[0]
        matches: list[tuple[Tensor, Tensor]] = []
        probs = pred_logits.softmax(-1)
        for b in range(batch):
            n_gt = int(valid_mask[b].sum().item())
            if n_gt == 0:
                matches.append(
                    (torch.empty(0, dtype=torch.long, device=pred_logits.device),
                     torch.empty(0, dtype=torch.long, device=pred_logits.device))
                )
                continue
            gt_b = gt_boxes[b, :n_gt]
            labels_b = gt_labels[b, :n_gt]
            cost_class = -probs[b][:, labels_b]  # (Q, M)
            cost_bbox = torch.cdist(pred_boxes[b], gt_b, p=1.0)
            cost_giou = -giou(
                pred_boxes[b].unsqueeze(0), gt_b.unsqueeze(1)
            )  # (M, Q) -> 转置
            cost_giou = cost_giou.transpose(0, 1)
            cost = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            rows, cols = hungarian(cost.detach().cpu().numpy())
            matches.append(
                (
                    torch.as_tensor(rows, dtype=torch.long, device=pred_logits.device),
                    torch.as_tensor(cols, dtype=torch.long, device=pred_logits.device),
                )
            )
        return matches


class SetCriterion(nn.Module):
    """集合预测训练损失：focal 分类 + L1 回归 + GIoU。"""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: dict[str, Tensor | list[Tensor]],
        gt_boxes: Tensor,
        gt_labels: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """前向。

        Args:
            outputs: ``pred_logits`` / ``pred_boxes``。
            gt_boxes: ``(B, M, 4)`` 归一化 cxcywh。
            gt_labels: ``(B, M)``。
            valid_mask: ``(B, M)``。
        """
        pred_logits: Tensor = outputs["pred_logits"]  # type: ignore[assignment]
        pred_boxes: Tensor = outputs["pred_boxes"]  # type: ignore[assignment]
        batch, num_queries = pred_logits.shape[:2]
        device = pred_logits.device

        matches = self.matcher(pred_logits, pred_boxes, gt_boxes, gt_labels, valid_mask)

        # ---- 分类：匹配到的 query 用真实类别，其余为背景 ----
        class_targets = torch.full(
            (batch, num_queries), self.num_classes, dtype=torch.long, device=device
        )
        matched_src: list[Tensor] = []
        matched_tgt: list[Tensor] = []
        for b, (src_idx, tgt_idx) in enumerate(matches):
            if src_idx.numel() > 0:
                class_targets[b, src_idx] = gt_labels[b, tgt_idx]
                matched_src.append(torch.stack([torch.full_like(src_idx, b), src_idx], dim=1))
                matched_tgt.append(torch.stack([torch.full_like(tgt_idx, b), tgt_idx], dim=1))
        loss_class = focal_loss(
            pred_logits.reshape(-1, self.num_classes + 1),
            class_targets.reshape(-1),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            num_classes=self.num_classes,
        ).mean()

        # ---- 回归：只对匹配对计算 L1 + GIoU ----
        if matched_src:
            src_idx = torch.cat(matched_src, dim=0)
            tgt_idx = torch.cat(matched_tgt, dim=0)
            pred_box = pred_boxes[src_idx[:, 0], src_idx[:, 1]]
            gt_box = gt_boxes[tgt_idx[:, 0], tgt_idx[:, 1]]
            loss_bbox = F.l1_loss(pred_box, gt_box)
            loss_giou = (1.0 - giou(pred_box, gt_box)).mean()
        else:
            loss_bbox = pred_boxes.sum() * 0.0
            loss_giou = pred_boxes.sum() * 0.0

        total = (
            self.weight_class * loss_class
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )
        return {
            "loss_class": loss_class,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss": total,
        }


# ---------------------------------------------------------------------------
# 密集解码头（YOLO/FCOS 风格）损失
# ---------------------------------------------------------------------------
def build_dense_targets(
    spatial: tuple[int, int],
    strides: tuple[int, ...],
    level_idx: int,
    image_size: int,
    gt_boxes: Tensor,
    gt_labels: Tensor,
    valid_mask: Tensor,
    num_classes: int,
    radius: float = 1.5,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """为单个尺度构建密集监督目标。

    正样本定义：**目标中心周围半径 ``radius`` 个网格单元内的所有单元**
    （FCOS 中心采样思想），且**每个目标只分配给一个尺度**：
    最粗的满足 ``stride <= 目标像素尺寸`` 的尺度；若所有尺度都太粗，
    则分配给最细尺度。避免小目标在粗尺度上产生大量无意义正样本。

    Args:
        spatial: ``(H, W)`` 该尺度网格尺寸。
        strides: 所有尺度的 stride 列表。
        level_idx: 当前尺度在 ``strides`` 中的下标。
        image_size: 输入图像边长（用于把归一化框换算为像素尺寸）。
        gt_boxes: ``(N, M, 4)`` 归一化 cxcywh。
        gt_labels: ``(N, M)``。
        valid_mask: ``(N, M)``。
        num_classes: 前景类别数（背景索引 == num_classes）。
        radius: 正样本邻域半径（网格单元数，默认 1.5 覆盖 3x3 邻域）。

    Returns:
        ``(cls_target, box_target, ctr_target, pos_mask)``：
        ``cls_target (N, H, W)``、``box_target (N, H, W, 4)``、
        ``ctr_target (N, H, W)``（FCOS center-ness）、``pos_mask (N, H, W)``。
    """
    h_l, w_l = spatial
    stride = strides[level_idx]
    n = gt_boxes.shape[0]
    device = gt_boxes.device
    cls_target = torch.full((n, h_l, w_l), num_classes, dtype=torch.long, device=device)
    box_target = torch.zeros(n, h_l, w_l, 4, dtype=gt_boxes.dtype, device=device)
    ctr_target = torch.zeros(n, h_l, w_l, dtype=gt_boxes.dtype, device=device)
    pos_mask = torch.zeros(n, h_l, w_l, dtype=torch.bool, device=device)

    for b in range(n):
        m = int(valid_mask[b].sum().item())
        for k in range(m):
            cx, cy, bw, bh = gt_boxes[b, k].unbind()
            # 单尺度分配：目标像素尺寸归属到最粗的 stride <= 尺寸 的尺度
            pixel = max(float(bw), float(bh)) * image_size
            assigned = -1
            for l, s in enumerate(strides):
                if pixel >= s:
                    assigned = l
            if assigned == -1:
                assigned = 0  # 所有尺度都太粗：交给最细尺度
            if assigned != level_idx:
                continue
            j0 = int((cx * w_l).floor().clamp(0, w_l - 1).item())
            i0 = int((cy * h_l).floor().clamp(0, h_l - 1).item())
            r = int(radius)
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    i, j = i0 + di, j0 + dj
                    if not (0 <= i < h_l and 0 <= j < w_l):
                        continue
                    if pos_mask[b, i, j]:
                        # 同一单元冲突：保留面积更小的目标（小目标更需要监督）
                        old_w, old_h = box_target[b, i, j, 2], box_target[b, i, j, 3]
                        if bw * bh >= old_w * old_h:
                            continue
                    cls_target[b, i, j] = gt_labels[b, k]
                    box_target[b, i, j] = gt_boxes[b, k]
                    # FCOS center-ness：min(l,r)*min(t,b) / max(l,r)*max(t,b) 开方
                    cell_x = (j + 0.5) / w_l
                    cell_y = (i + 0.5) / h_l
                    l = cell_x - (cx - bw / 2)
                    r_dist = (cx + bw / 2) - cell_x
                    t = cell_y - (cy - bh / 2)
                    btm = (cy + bh / 2) - cell_y
                    num = torch.minimum(l, r_dist) * torch.minimum(t, btm)
                    den = torch.maximum(l, r_dist) * torch.maximum(t, btm)
                    ctr = torch.sqrt((num / den.clamp(min=1e-7)).clamp(min=0.0, max=1.0))
                    ctr_target[b, i, j] = ctr
                    pos_mask[b, i, j] = True
    return cls_target, box_target, ctr_target, pos_mask


def decode_dense_boxes(raw: Tensor) -> Tensor:
    """把回归头原始输出解码为归一化 cxcywh（cx/cy sigmoid，w/h softplus）。"""
    cx = torch.sigmoid(raw[..., 0])
    cy = torch.sigmoid(raw[..., 1])
    w = torch.nn.functional.softplus(raw[..., 2]).clamp(max=1.0)
    h = torch.nn.functional.softplus(raw[..., 3]).clamp(max=1.0)
    return torch.stack([cx, cy, w, h], dim=-1)


class DenseCriterion(nn.Module):
    """密集头训练损失：focal 分类 + L1/GIoU 回归 + center-ness（逐尺度）。"""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 128,
        weight_class: float = 2.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        weight_ctr: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.weight_ctr = weight_ctr
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: dict[str, object],
        gt_boxes: Tensor,
        gt_labels: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """前向。

        Args:
            outputs: ``pred_logits`` / ``pred_boxes`` / ``centerness``（逐尺度列表）与 ``strides``。
            gt_boxes / gt_labels / valid_mask: 与 :class:`SetCriterion` 相同格式。
        """
        cls_list: list[Tensor] = outputs["pred_logits"]  # type: ignore[assignment]
        reg_list: list[Tensor] = outputs["pred_boxes"]  # type: ignore[assignment]
        ctr_list: list[Tensor] = outputs["centerness"]  # type: ignore[assignment]
        strides: tuple[int, ...] = outputs["strides"]  # type: ignore[assignment]
        loss_class = reg_list[0].sum() * 0.0
        loss_bbox = reg_list[0].sum() * 0.0
        loss_giou = reg_list[0].sum() * 0.0
        loss_ctr = reg_list[0].sum() * 0.0
        num_pos = 0

        for lvl in range(len(cls_list)):
            cls_l = cls_list[lvl]  # (N, C+1, H, W)
            reg_l = reg_list[lvl]  # (N, 4, H, W)
            ctr_l = ctr_list[lvl]  # (N, 1, H, W)
            n, c1, h_l, w_l = cls_l.shape
            cls_target, box_target, ctr_target, pos_mask = build_dense_targets(
                (h_l, w_l), strides, lvl, self.image_size,
                gt_boxes, gt_labels, valid_mask, self.num_classes,
            )
            num_pos += int(pos_mask.sum().item())

            loss_class = loss_class + focal_loss(
                cls_l.permute(0, 2, 3, 1).reshape(-1, c1),
                cls_target.reshape(-1),
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                num_classes=self.num_classes,
            ).mean()

            raw = reg_l.permute(0, 2, 3, 1)  # (N, H, W, 4)
            pred_box = decode_dense_boxes(raw)
            if pos_mask.any():
                pb = pred_box[pos_mask]
                tb = box_target[pos_mask]
                loss_bbox = loss_bbox + F.l1_loss(pb, tb)
                loss_giou = loss_giou + (1.0 - giou(pb, tb)).mean()
                ctr_logit = ctr_l.permute(0, 2, 3, 1).reshape(n, h_l, w_l)[pos_mask]
                loss_ctr = loss_ctr + F.binary_cross_entropy_with_logits(
                    ctr_logit, ctr_target[pos_mask]
                )

        n_levels = max(len(cls_list), 1)
        loss_class = loss_class / n_levels
        loss_bbox = loss_bbox / n_levels
        loss_giou = loss_giou / n_levels
        loss_ctr = loss_ctr / n_levels
        total = (
            self.weight_class * loss_class
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
            + self.weight_ctr * loss_ctr
        )
        return {
            "loss": total,
            "loss_class": loss_class,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
            "loss_ctr": loss_ctr,
            "num_pos": float(num_pos),
        }


__all__ = [
    "HungarianMatcher",
    "SetCriterion",
    "DenseCriterion",
    "build_dense_targets",
    "decode_dense_boxes",
    "focal_loss",
]
