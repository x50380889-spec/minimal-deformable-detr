"""知识蒸馏：用训练好的教师（Deformable DETR）蒸馏 MobileNet 学生。

损失 = 学生 task loss + logits KL + box L1/GIoU + 编码器特征 L2。

用法::

    python scripts/distill.py --config configs/defect.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config
from minimal_detr.data.defects import build_dataloaders
from minimal_detr.distill import DistillCriterion
from minimal_detr.engine import evaluate, make_tb_writer
from minimal_detr.utils import AverageMeter, JsonlWriter

from common import (
    add_common_args,
    build_student,
    make_criterion,
    make_optimizer,
    model_args_of,
    prepare,
    rebuild_model,
    save_checkpoint,
)


def distill_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    criterion: DistillCriterion,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer,
    scalars: JsonlWriter,
    max_steps: int | None = None,
) -> dict[str, float]:
    """蒸馏训练一个 epoch（教师冻结、不反传）。"""
    student.train()
    teacher.eval()
    keys = ("loss", "task_loss", "kd_cls", "kd_box", "kd_giou", "kd_feat")
    meters = {k: AverageMeter() for k in keys}
    global_step = (epoch - 1) * len(loader)
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = batch["images"].to(device)
        gt_boxes = batch["boxes"].to(device)
        gt_labels = batch["labels"].to(device)
        valid = batch["valid"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_out = teacher(images)
        student_out = student(images)
        losses = criterion(student_out, teacher_out, gt_boxes, gt_labels, valid)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        optimizer.step()

        for k in keys:
            meters[k].update(float(losses[k].detach().cpu()), images.shape[0])
        if step % 10 == 0:
            values = {k: meters[k].val for k in keys}
            if writer is not None:
                writer.add_scalars("train", values, global_step + step)
            scalars.write(global_step + step, {"train/" + k: v for k, v in values.items()})
    return {k: meters[k].avg for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="知识蒸馏：教师 -> MobileNet 学生")
    add_common_args(parser)
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="教师权重路径")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 epoch 最多迭代步（冒烟测试用）")
    parser.add_argument("--init-student", type=str, default=None, help="学生初始化权重（如裸训练 checkpoint）")
    parser.add_argument("--lr", type=float, default=None, help="覆盖蒸馏学习率")
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    if args.epochs:
        cfg.distill.epochs = args.epochs
    device = prepare(cfg, args)

    teacher_ckpt_path = args.teacher_ckpt or cfg.distill.teacher_ckpt
    ckpt = torch.load(teacher_ckpt_path, map_location=device, weights_only=False)
    teacher = rebuild_model(ckpt["model_args"]).to(device)
    teacher.load_state_dict(ckpt["model"])
    teacher.eval()
    print(f"[teacher] loaded {teacher_ckpt_path} (mAP@50={ckpt.get('map50', float('nan')):.4f})")

    student = build_student(cfg).to(device)
    if args.init_student:
        init_ckpt = torch.load(args.init_student, map_location=device, weights_only=False)
        student.load_state_dict(init_ckpt["model"])
        print(
            f"[student] warm-start from {args.init_student} "
            f"(mAP@50={init_ckpt.get('map50', float('nan')):.4f})"
        )
    task_criterion = make_criterion(cfg)
    distill_criterion = DistillCriterion(
        num_classes=cfg.num_classes,
        student_d_model=cfg.student.d_model,
        teacher_d_model=cfg.teacher.d_model,
        cfg=cfg.distill,
        task_criterion=task_criterion,
        image_size=cfg.data.image_size,
    ).to(device)

    params = list(student.parameters()) + list(distill_criterion.feat_projector.parameters())
    distill_lr = args.lr if args.lr is not None else cfg.distill.lr
    optimizer = torch.optim.AdamW(params, lr=distill_lr, weight_decay=cfg.distill.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.distill.lr_milestones), gamma=0.1
    )

    train_loader, val_loader = build_dataloaders(
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        image_size=cfg.data.image_size,
        max_objects=cfg.data.max_objects,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.data.seed,
    )
    writer = make_tb_writer(Path(cfg.tensorboard_dir) / "distill")
    scalars = JsonlWriter(Path(cfg.scalars_dir) / "distill.jsonl")
    out_path = Path(args.out or (Path(cfg.output_dir) / "student_distilled.pt"))

    print(f"[student] params={sum(p.numel() for p in student.parameters()):,}")
    best_map = 0.0
    last_val = None
    for epoch in range(1, cfg.distill.epochs + 1):
        metrics = distill_one_epoch(
            student, teacher, distill_criterion, train_loader, optimizer, device,
            epoch, writer, scalars, max_steps=args.max_steps,
        )
        val = evaluate(student, task_criterion, val_loader, device, cfg.num_classes)
        last_val = val
        scheduler.step()
        print(
            f"[epoch {epoch:02d}/{cfg.distill.epochs}] loss={metrics['loss']:.4f} "
            f"kd_cls={metrics['kd_cls']:.4f} kd_box={metrics['kd_box']:.4f} "
            f"kd_feat={metrics['kd_feat']:.4f} val_mAP@50={val['map50']:.4f}"
        )
        values = {"epoch": epoch, "train/loss": metrics["loss"],
                  "kd_cls": metrics["kd_cls"], "kd_box": metrics["kd_box"],
                  "kd_feat": metrics["kd_feat"], "mAP@50": val["map50"]}
        if writer is not None:
            writer.add_scalars("epoch", values, epoch)
        scalars.write(epoch, {"epoch": epoch, "train/loss": metrics["loss"],
                              "train/kd_cls": metrics["kd_cls"], "train/kd_box": metrics["kd_box"],
                              "train/kd_feat": metrics["kd_feat"], "val/mAP@50": val["map50"]})
        if val["map50"] > best_map:
            best_map = val["map50"]
            save_checkpoint(out_path, student, model_args_of(student), epoch, best_map)
            print(f"  -> saved best distilled student (mAP={best_map:.4f})")

    final_path = out_path.with_name(out_path.stem + "_final.pt")
    final_map = last_val["map50"] if last_val is not None else 0.0
    save_checkpoint(final_path, student, model_args_of(student), cfg.distill.epochs, final_map)

    if writer is not None:
        writer.close()
    scalars.close()
    print(f"[done] best distilled student mAP@50 = {best_map:.4f} -> {out_path} (final -> {final_path})")


if __name__ == "__main__":
    main()
