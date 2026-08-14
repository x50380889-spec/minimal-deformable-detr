"""把 JSONL 标量日志导出为 TensorBoard 事件文件。

训练脚本在无 tensorboard 环境时只写 JSONL（结构化、零依赖）；
本脚本在训练完成后把这些标量回放为事件文件，便于 ``tensorboard --logdir`` 查看。

用法::

    python scripts/export_tb.py --config configs/defect.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minimal_detr.config import Config
from minimal_detr.engine import make_tb_writer
from minimal_detr.utils import JsonlWriter


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL -> TensorBoard 事件导出")
    parser.add_argument("--config", type=str, default="configs/defect.json")
    parser.add_argument("--runs", nargs="*", default=["teacher", "student_scratch", "distill"])
    args = parser.parse_args()

    cfg = Config.from_json(args.config)
    scalars_dir = Path(cfg.scalars_dir)
    tb_root = Path(cfg.tensorboard_dir)
    for run in args.runs:
        jsonl = scalars_dir / f"{run}.jsonl"
        if not jsonl.exists():
            print(f"[skip] {run}: 无 {jsonl}")
            continue
        writer = make_tb_writer(tb_root / run)
        if writer is None:
            print(f"[warn] tensorboard 未安装，{run} 只保留 JSONL")
            continue
        rows = JsonlWriter.read(jsonl)
        for row in rows:
            step = int(row.get("epoch", row.get("step", 0)))
            for key, value in row.items():
                if key in ("step", "epoch"):
                    continue
                # 去掉 train/ val/ 前缀，避免 TB 里出现 train/train/loss 这类嵌套
                flat = key.split("/")[-1]
                writer.add_scalar(flat, float(value), step)
        writer.close()
        print(f"[ok] {run}: {len(rows)} 条标量 -> {tb_root / run}")


if __name__ == "__main__":
    main()
