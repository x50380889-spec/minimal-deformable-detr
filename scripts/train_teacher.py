"""训练教师模型：极简 Deformable DETR（SimpleCNN 主干）。

用法::

    python scripts/train_teacher.py --config configs/defect.json
"""

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
from minimal_detr.visualize import draw_dataset_samples, draw_predictions_grid

from common import (
    add_common_args,
    build_teacher,
    make_criterion,
    make_optimizer,
    model_args_of,
    prepare,
    save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练教师模型（极简 Deformable DETR）")
    add_common_args(parser)
    parser.add_argument("--max-steps", type=int, default=None, help="每个 epoch 最多迭代步（冒烟测试用）")
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    if args.epochs:
        cfg.teacher.epochs = args.epochs
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
    model = build_teacher(cfg).to(device)
    criterion = make_criterion(cfg)
    optimizer, scheduler = make_optimizer(
        model, cfg.teacher.lr, cfg.teacher.weight_decay, cfg.teacher.lr_milestones
    )

    tb_dir = Path(cfg.tensorboard_dir) / "teacher"
    writer = make_tb_writer(tb_dir)
    scalars = JsonlWriter(Path(cfg.scalars_dir) / "teacher.jsonl")
    out_path = Path(args.out or (Path(cfg.output_dir) / "teacher.pt"))

    print(f"[model] teacher params={sum(p.numel() for p in model.parameters()):,}")
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    best_map = 0.0
    last_val = None
    for epoch in range(1, cfg.teacher.epochs + 1):
        metrics = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, writer, scalars,
            max_steps=args.max_steps,
        )
        val = evaluate(model, criterion, val_loader, device, cfg.num_classes)
        last_val = val
        scheduler.step()
        print(
            f"[epoch {epoch:02d}/{cfg.teacher.epochs}] "
            f"train_loss={metrics['loss']:.4f} "
            f"val_loss={val['loss']:.4f} mAP@50={val['map50']:.4f} "
            f"AP={[f'{x:.3f}' for x in val['ap_per_class']]}"
        )
        values = {
            "epoch": epoch,
            "loss": metrics["loss"],
            "val_loss": val["loss"],
            "mAP@50": val["map50"],
        }
        if writer is not None:
            writer.add_scalars("epoch", values, epoch)
        scalars.write(epoch, {"epoch": epoch, "train/loss": metrics["loss"],
                              "val/loss": val["loss"], "val/mAP@50": val["map50"]})

        if val["map50"] > best_map:
            best_map = val["map50"]
            save_checkpoint(out_path, model, model_args_of(model), epoch, best_map)
            print(f"  -> saved best checkpoint (mAP={best_map:.4f})")

    # 无论 mAP 是否提升都保存最终权重，避免「全程 mAP=0 无任何产物」
    final_path = out_path.with_name(out_path.stem + "_final.pt")
    final_map = last_val["map50"] if last_val is not None else 0.0
    save_checkpoint(final_path, model, model_args_of(model), cfg.teacher.epochs, final_map)

    # 可视化：数据集样本 + 教师预测
    sample_dir = Path(cfg.log_dir) / "samples"
    draw_dataset_samples(train_loader.dataset, [0, 7, 13, 21, 33, 47], sample_dir / "dataset.png")
    draw_predictions_grid(model, val_loader, device, sample_dir / "teacher_detections.png")
    if writer is not None:
        writer.close()
    scalars.close()
    print(f"[done] best teacher mAP@50 = {best_map:.4f} -> {out_path} (final -> {final_path})")


if __name__ == "__main__":
    main()
