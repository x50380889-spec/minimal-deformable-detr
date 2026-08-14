"""Miscellaneous utilities: geometry, matching, seeding, metrics."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """把归一化 ``(cx, cy, w, h)`` 转成 ``(x1, y1, x2, y2)``。"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """计算两组 xyxy 框之间的 IoU 矩阵。

    Args:
        boxes1: ``(M, 4)``
        boxes2: ``(N, 4)``

    Returns:
        ``(M, N)`` IoU 矩阵。
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def giou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """计算两组 cxcywh 框之间的 GIoU（逐元素）。

    Args:
        boxes1: ``(..., 4)`` 归一化 cxcywh。
        boxes2: ``(..., 4)`` 归一化 cxcywh。

    Returns:
        形状同输入（去掉最后一维）的 GIoU 值，取值 ``[-1, 1]``。
    """
    b1 = box_cxcywh_to_xyxy(boxes1)
    b2 = box_cxcywh_to_xyxy(boxes2)
    lt = torch.min(b1[..., :2], b2[..., :2])
    rb = torch.max(b1[..., 2:], b2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]

    inter_lt = torch.max(b1[..., :2], b2[..., :2])
    inter_rb = torch.min(b1[..., 2:], b2[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / union.clamp(min=eps)
    return iou - (area_c - union) / area_c.clamp(min=eps)


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """经典 NMS（非极大值抑制），返回保留的索引。

    Args:
        boxes: ``(M, 4)`` xyxy。
        scores: ``(M,)``。
        iou_threshold: IoU 阈值。
    """
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ---------------------------------------------------------------------------
# 匈牙利匹配（Kuhn-Munkres，O(n^3)，最小化总代价）
# ---------------------------------------------------------------------------
def hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """求解最小化线性指派问题（匈牙利算法，Kuhn-Munkres，O(n^3)）。

    自研实现，不依赖 ``scipy.optimize.linear_sum_assignment``；
    支持非方阵（行数 >= 列数时，未匹配的行自然被忽略）。

    采用 emaxx 的 O(n^3) 版本：索引 1..n 为真实行列，索引 0 为虚拟哨兵。

    Args:
        cost: ``(n_rows, n_cols)`` 代价矩阵（非负即可，内部会处理负值）。

    Returns:
        ``(rows, cols)`` 两个等长数组，表示 ``rows[i]`` 匹配到 ``cols[i]``。
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # 补齐为方阵：行多补虚拟列，列多补虚拟行（大常数代价，避免抢占真实匹配）
    n = max(n_rows, n_cols)
    a = np.full((n, n), cost.max() + 1.0, dtype=np.float64)
    a[:n_rows, :n_cols] = cost
    a = a - a.min()  # 非负化（KM 要求）

    # 1-based 数组：索引 0 是虚拟哨兵，真实行列是 1..n
    u = np.zeros(n + 1, dtype=np.float64)  # 行势
    v = np.zeros(n + 1, dtype=np.float64)  # 列势
    p = np.zeros(n + 1, dtype=np.int64)  # p[j] = 匹配到第 j 列的行（0 表示未匹配）
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i  # 哨兵指向当前行
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            j1 = -1
            delta = np.inf
            for j in range(1, n + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:  # 到达未匹配列，增广路径完成
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # 提取匹配（跳过虚拟行/列）
    rows: list[int] = []
    cols: list[int] = []
    for j in range(1, n + 1):
        if p[j] != 0 and p[j] <= n_rows and j <= n_cols:
            rows.append(int(p[j]) - 1)
            cols.append(int(j) - 1)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


# ---------------------------------------------------------------------------
# 训练辅助
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """把配置里的设备字符串解析成 torch.device。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class AverageMeter:
    """滑动平均统计器（记录均值 / 最近值 / 计数）。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class JsonlWriter:
    """极简 JSONL 标量记录器（不依赖 TensorBoard 也能画曲线）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def write(self, step: int, values: dict[str, Any]) -> None:
        self._f.write(json.dumps({"step": step, **values}, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数量（单位：个）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "box_cxcywh_to_xyxy",
    "box_iou",
    "giou",
    "nms",
    "hungarian",
    "set_seed",
    "resolve_device",
    "AverageMeter",
    "JsonlWriter",
    "count_parameters",
]
