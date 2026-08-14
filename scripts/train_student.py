"""训练学生模型基线（MobileNetV3-Small，无蒸馏，用于对照）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config
from minimal_detr.data.defects import build_dataloaders
from minimal_detr.engine import evaluate, make_tb_writer, train_one_epoch
from minimal_detr.utils import JsonlWriter

from common import (
    add_common_args,
    build_student,
    make_criterion,
    make_optimizer,
    model_args_of,
    prepare,
    save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练学生模型基线（无蒸馏）")
    add_common_args(parser)
    parser.add_argument("--max-steps", type=int, default=None, help="每个 epoch 最多迭代步（冒烟测试用）")
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    if args.epochs:
        cfg.student.epochs = args.epochs
    device = prepare(cfg, args)

    train_loader, val_loader = build_dataloaders(
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        image_size=cfg.data.image_size,
        max_objects=cfg.data.max_objects,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.data.seed,
    )
    model = build_student(cfg).to(device)
    criterion = make_criterion(cfg)
    optimizer, scheduler = make_optimizer(
        model, cfg.student.lr, cfg.student.weight_decay, cfg.student.lr_milestones
    )

    writer = make_tb_writer(Path(cfg.tensorboard_dir) / "student_scratch")
    scalars = JsonlWriter(Path(cfg.scalars_dir) / "student_scratch.jsonl")
    out_path = Path(args.out or (Path(cfg.output_dir) / "student_scratch.pt"))

    print(f"[model] student params={sum(p.numel() for p in model.parameters()):,}")
    best_map = 0.0
    last_val = None
    for epoch in range(1, cfg.student.epochs + 1):
        metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, writer, scalars,
            max_steps=args.max_steps,
        )
        val = evaluate(model, criterion, val_loader, device, cfg.num_classes)
        last_val = val
        scheduler.step()
        print(
            f"[epoch {epoch:02d}/{cfg.student.epochs}] "
            f"train_loss={metrics['loss']:.4f} "
            f"val_loss={val['loss']:.4f} mAP@50={val['map50']:.4f}"
        )
        values = {"epoch": epoch, "loss": metrics["loss"],
                  "val_loss": val["loss"], "mAP@50": val["map50"]}
        if writer is not None:
            writer.add_scalars("epoch", values, epoch)
        scalars.write(epoch, {"epoch": epoch, "train/loss": metrics["loss"],
                              "val/loss": val["loss"], "val/mAP@50": val["map50"]})
        if val["map50"] > best_map:
            best_map = val["map50"]
            save_checkpoint(out_path, model, model_args_of(model), epoch, best_map)

    final_path = out_path.with_name(out_path.stem + "_final.pt")
    final_map = last_val["map50"] if last_val is not None else 0.0
    save_checkpoint(final_path, model, model_args_of(model), cfg.student.epochs, final_map)

    if writer is not None:
        writer.close()
    scalars.close()
    print(f"[done] best student (scratch) mAP@50 = {best_map:.4f} -> {out_path} (final -> {final_path})")


if __name__ == "__main__":
    main()
