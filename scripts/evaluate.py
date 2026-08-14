"""统一评估：mAP@50 + 端到端 FPS + 参数量，结果写入 training_logs/metrics.json。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config
from minimal_detr.data.defects import build_dataloaders
from minimal_detr.engine import benchmark_fps, evaluate
from minimal_detr.utils import count_parameters
from minimal_detr.visualize import draw_predictions_grid

from common import add_common_args, make_criterion, prepare, rebuild_model


def main() -> None:
    parser = argparse.ArgumentParser(description="评估模型：mAP / FPS / 参数量")
    add_common_args(parser)
    parser.add_argument("--fps-iters", type=int, default=30, help="FPS 计时次数")
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    device = prepare(cfg, args)
    _, val_loader = build_dataloaders(
        train_size=cfg.data.train_size,
        val_size=cfg.data.val_size,
        image_size=cfg.data.image_size,
        max_objects=cfg.data.max_objects,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        seed=cfg.data.seed,
    )
    criterion = make_criterion(cfg)

    candidates = [
        ("teacher", Path(cfg.output_dir) / "teacher.pt"),
        ("student_scratch", Path(cfg.output_dir) / "student_scratch.pt"),
        ("student_distilled", Path(cfg.output_dir) / "student_distilled.pt"),
    ]
    results: dict[str, dict] = {}
    for tag, ckpt_path in candidates:
        if not ckpt_path.exists():
            print(f"[skip] {tag}: 未找到 {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = rebuild_model(ckpt["model_args"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        val = evaluate(model, criterion, val_loader, device, cfg.num_classes)
        fps = benchmark_fps(model, device, image_size=cfg.image_size, iters=args.fps_iters)
        params = count_parameters(model)
        results[tag] = {
            "map50": round(val["map50"], 4),
            "ap_per_class": [round(x, 4) for x in val["ap_per_class"]],
            "fps": round(fps["fps"], 2),
            "ms": round(fps["ms"], 3),
            "params": params,
            "ckpt": str(ckpt_path),
        }
        print(
            f"[{tag}] mAP@50={val['map50']:.4f} AP={[f'{x:.3f}' for x in val['ap_per_class']]} "
            f"FPS={fps['fps']:.1f} ({fps['ms']:.1f} ms) params={params:,}"
        )
        draw_predictions_grid(
            model, val_loader, device,
            Path(cfg.log_dir) / "samples" / f"detections_{tag}.png",
        )

    # 摘要：教师 vs 蒸馏学生
    summary = {}
    if {"teacher", "student_distilled"} <= set(results):
        t, s = results["teacher"], results["student_distilled"]
        summary = {
            "speedup_x": round(s["fps"] / t["fps"], 2) if t["fps"] > 0 else 0.0,
            "map_delta_pts": round((s["map50"] - t["map50"]) * 100, 2),
            "param_reduction_x": round(t["params"] / max(s["params"], 1), 2),
        }
    out = {"models": results, "summary": summary}
    metrics_path = Path(cfg.log_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 评估结果 -> {metrics_path}")


if __name__ == "__main__":
    main()
