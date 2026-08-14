"""Knowledge distillation for the dense-head detector.

蒸馏设计（对应 README 的「推理逻辑迁移」）：

1. **逐像素 logits KL**：教师/学生在同一空间网格上对齐类别分布
   （含背景类，temperature-scaled），不依赖匈牙利匹配，天然逐像素对应；
2. **box 对齐**：在教师的正样本网格单元上对齐预测框（L1 + GIoU）；
3. **特征对齐**：可变形编码器逐尺度输出经 1x1 投影后做 L2，
   迁移多尺度空间表征；
4. 同时保留学生自身的 GT 监督（task loss）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import DistillConfig
from .models.losses import (
    DenseCriterion,
    build_dense_targets,
    decode_dense_boxes,
)
from .utils import giou


class DistillCriterion(nn.Module):
    """组合式蒸馏损失（逐像素 logits KL + box 对齐 + 特征对齐 + 学生 task loss）。

    Args:
        num_classes: 前景类别数。
        student_d_model: 学生编码器特征维数。
        teacher_d_model: 教师编码器特征维数。
        cfg: :class:`DistillConfig`。
        task_criterion: 学生自身的密集头损失（DenseCriterion）。
    """

    def __init__(
        self,
        num_classes: int,
        student_d_model: int,
        teacher_d_model: int,
        cfg: DistillConfig,
        task_criterion: DenseCriterion,
        image_size: int = 128,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.task_criterion = task_criterion
        self.image_size = image_size
        # 学生特征 -> 教师特征维度的 1x1 投影器（每个学生尺度一个，特征蒸馏专用）
        self.feat_projector = nn.Conv2d(student_d_model, teacher_d_model, 1)

    def _align_levels(
        self,
        s_list: list[Tensor],
        t_list: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        """按空间尺寸把学生的每个尺度与教师对应尺度对齐。"""
        pairs: list[tuple[Tensor, Tensor]] = []
        for s in s_list:
            t = next((f for f in t_list if f.shape[-2:] == s.shape[-2:]), None)
            if t is None:
                raise ValueError(
                    f"教师特征层空间尺寸 {[tuple(f.shape[-2:]) for f in t_list]} "
                    f"与学生 {tuple(s.shape[-2:])} 无法对齐，无法蒸馏"
                )
            pairs.append((s, t))
        return pairs

    def forward(
        self,
        student_out: dict[str, object],
        teacher_out: dict[str, object],
        gt_boxes: Tensor,
        gt_labels: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        """前向。

        Args:
            student_out / teacher_out: 两个模型的输出 dict。
            gt_boxes / gt_labels / valid_mask: 与 :class:`DenseCriterion` 相同的标注格式。

        Returns:
            包含 ``loss`` 及各分量的 dict。
        """
        cfg = self.cfg
        s_cls: list[Tensor] = student_out["pred_logits"]  # type: ignore[assignment]
        t_cls: list[Tensor] = [x.detach() for x in teacher_out["pred_logits"]]  # type: ignore[assignment]
        s_reg: list[Tensor] = student_out["pred_boxes"]  # type: ignore[assignment]
        t_reg: list[Tensor] = [x.detach() for x in teacher_out["pred_boxes"]]  # type: ignore[assignment]
        s_feats: list[Tensor] = student_out["encoder_features"]  # type: ignore[assignment]
        t_feats: list[Tensor] = [x.detach() for x in teacher_out["encoder_features"]]  # type: ignore[assignment]
        s_strides: tuple[int, ...] = student_out["strides"]  # type: ignore[assignment]

        # ---- 1) 学生自身 task loss（GT 监督）----
        task = self.task_criterion(student_out, gt_boxes, gt_labels, valid_mask)

        # ---- 2) 特征对齐：每个学生尺度对到教师同尺寸尺度做 L2 ----
        loss_feat = s_feats[0].sum() * 0.0
        for s_map, t_map in self._align_levels(s_feats, t_feats):
            loss_feat = loss_feat + F.mse_loss(self.feat_projector(s_map), t_map)
        loss_feat = loss_feat / max(len(s_feats), 1)

        # ---- 3) 逐像素 logits KL + 正样本 box 对齐 ----
        kl_terms: list[Tensor] = []
        l1_terms: list[Tensor] = []
        giou_terms: list[Tensor] = []
        temp = cfg.temperature
        cls_pairs = self._align_levels(s_cls, t_cls)
        reg_pairs = self._align_levels(s_reg, t_reg)
        for lvl_s, ((s_logits, t_logits), (s_raw, t_raw)) in enumerate(
            zip(cls_pairs, reg_pairs)
        ):
            n, c1, h_l, w_l = s_logits.shape
            # 分类 KL：所有网格单元（含背景类）
            s_prob = F.log_softmax(s_logits.permute(0, 2, 3, 1).reshape(-1, c1) / temp, dim=-1)
            t_prob = F.softmax(t_logits.permute(0, 2, 3, 1).reshape(-1, c1) / temp, dim=-1)
            kl_terms.append(F.kl_div(s_prob, t_prob, reduction="batchmean") * temp * temp)

            # box 对齐：只在教师正样本单元上计算
            _, box_target, _, pos_mask = build_dense_targets(
                (h_l, w_l), s_strides, lvl_s, self.image_size,
                gt_boxes, gt_labels, valid_mask, self.num_classes,
            )
            del box_target
            if pos_mask.any():
                s_box = decode_dense_boxes(s_raw.permute(0, 2, 3, 1))[pos_mask]
                t_box = decode_dense_boxes(t_raw.permute(0, 2, 3, 1))[pos_mask]
                l1_terms.append(F.l1_loss(s_box, t_box))
                giou_terms.append((1.0 - giou(s_box, t_box)).mean())

        n_align = max(len(kl_terms), 1)
        loss_kd_cls = torch.stack(kl_terms).mean()
        if l1_terms:
            loss_kd_box = torch.stack(l1_terms).mean()
            loss_kd_giou = torch.stack(giou_terms).mean()
        else:
            zero = s_cls[0].sum() * 0.0
            loss_kd_box = zero
            loss_kd_giou = zero

        # ---- 组合 ----
        loss = (
            task["loss"]
            + cfg.kd_cls_w * loss_kd_cls
            + cfg.kd_box_w * loss_kd_box
            + cfg.kd_giou_w * loss_kd_giou
            + cfg.kd_feat_w * loss_feat
        )
        return {
            "loss": loss,
            "task_loss": task["loss"],
            "loss_class": task["loss_class"],
            "loss_bbox": task["loss_bbox"],
            "loss_giou": task["loss_giou"],
            "loss_ctr": task["loss_ctr"],
            "kd_cls": loss_kd_cls,
            "kd_box": loss_kd_box,
            "kd_giou": loss_kd_giou,
            "kd_feat": loss_feat,
        }


__all__ = ["DistillCriterion"]
