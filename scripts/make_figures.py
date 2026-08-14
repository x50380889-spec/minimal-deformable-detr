"""生成 training_logs/ 下的 loss 曲线与 教师 vs 学生 对比图。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config
from minimal_detr.utils import JsonlWriter


def plot_loss_curves(cfg: Config) -> None:
    """教师训练 loss 曲线 + 蒸馏训练 loss 曲线（数据来自 JSONL，与 TensorBoard 同源）。"""
    scalars_dir = Path(cfg.scalars_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    def x_of(row: dict) -> float:
        """JSONL 中 step 行用 step，epoch 行用 epoch 作为横轴。"""
        return float(row.get("epoch", row.get("step", 0)))

    def y_of(rows: list[dict], key: str) -> list[float]:
        return [float(r[key]) if key in r else float("nan") for r in rows]

    teacher_rows = JsonlWriter.read(scalars_dir / "teacher.jsonl")
    if teacher_rows:
        ep = [x_of(r) for r in teacher_rows]
        ax = axes[0]
        ax.plot(ep, y_of(teacher_rows, "train/loss"), label="total loss", lw=1.8)
        ax.plot(ep, y_of(teacher_rows, "train/loss_class"), label="focal cls", lw=1.2)
        ax.plot(ep, y_of(teacher_rows, "train/loss_bbox"), label="L1 box", lw=1.2)
        ax.plot(ep, y_of(teacher_rows, "train/loss_giou"), label="GIoU", lw=1.2)
        ax.set_title("Teacher (Minimal Deformable DETR) — train loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    distill_rows = JsonlWriter.read(scalars_dir / "distill.jsonl")
    if distill_rows:
        ep = [x_of(r) for r in distill_rows]
        ax = axes[1]
        ax.plot(ep, y_of(distill_rows, "train/loss"), label="total", lw=1.8)
        ax.plot(ep, y_of(distill_rows, "train/kd_cls"), label="KD logits (KL)", lw=1.2)
        ax.plot(ep, y_of(distill_rows, "train/kd_box"), label="KD box (L1)", lw=1.2)
        ax.plot(ep, y_of(distill_rows, "train/kd_feat"), label="KD feature (L2)", lw=1.2)
        ax.set_title("Distillation → MobileNetV3-Small — train loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = Path(cfg.log_dir) / "loss_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figures] loss curves -> {out}")


def plot_teacher_vs_student(cfg: Config) -> None:
    """教师 vs 蒸馏学生：mAP / FPS / 参数量对比柱状图。"""
    metrics_path = Path(cfg.log_dir) / "metrics.json"
    if not metrics_path.exists():
        print("[figures] 缺少 metrics.json，请先运行 evaluate.py")
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    models = metrics.get("models", {})
    if "teacher" not in models or "student_distilled" not in models:
        print("[figures] 需要 teacher 与 student_distilled 两个模型的指标")
        return
    t, s = models["teacher"], models["student_distilled"]
    speedup = metrics["summary"].get("speedup_x", 0.0)
    map_delta = metrics["summary"].get("map_delta_pts", 0.0)
    param_x = metrics["summary"].get("param_reduction_x", 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = ["Teacher\nDeformable DETR", "Student\n(distilled)"]
    mAPs = [t["map50"] * 100, s["map50"] * 100]
    fpss = [t["fps"], s["fps"]]
    params = [t["params"] / 1e6, s["params"] / 1e6]

    colors = ["#4C72B0", "#DD8452"]
    ax = axes[0]
    bars = ax.bar(labels, mAPs, color=colors, width=0.55)
    for bar, v in zip(bars, mAPs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=11)
    ax.set_ylabel("mAP@50 (%)")
    ax.set_ylim(0, max(mAPs) * 1.25 + 5)
    ax.set_title("Accuracy")
    ax.annotate(f"Δ mAP = {map_delta:+.1f} pts", xy=(0.5, 0.03), xycoords="axes fraction",
                ha="center", fontsize=10, color="crimson")

    ax = axes[1]
    bars = ax.bar(labels, fpss, color=colors, width=0.55)
    for bar, v in zip(bars, fpss):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=11)
    ax.set_ylabel("FPS")
    ax.set_ylim(0, max(fpss) * 1.25 + 2)
    ax.set_title("Inference Speed (CPU, batch=1)")
    ax.annotate(f"{speedup:.1f}× faster", xy=(0.5, 0.03), xycoords="axes fraction",
                ha="center", fontsize=10, color="darkgreen")

    ax = axes[2]
    bars = ax.bar(labels, params, color=colors, width=0.55)
    for bar, v in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}M", ha="center", fontsize=11)
    ax.set_ylabel("Parameters (M)")
    ax.set_ylim(0, max(params) * 1.25 + 0.05)
    ax.set_title("Model Size")
    ax.annotate(f"{param_x:.1f}× smaller", xy=(0.5, 0.03), xycoords="axes fraction",
                ha="center", fontsize=10, color="darkgreen")

    fig.suptitle("Teacher vs Distilled Student — 降本增效（精度持平，速度/体积大幅下降）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(cfg.log_dir) / "teacher_vs_student.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[figures] teacher vs student -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 loss 曲线与对比图")
    parser.add_argument("--config", type=str, default="configs/defect.json")
    args = parser.parse_args()
    cfg = Config.from_json(args.config)
    plot_loss_curves(cfg)
    plot_teacher_vs_student(cfg)


if __name__ == "__main__":
    main()
