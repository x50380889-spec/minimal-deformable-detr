"""From-scratch operators used by the detector.

本包刻意不使用 mmcv / torchvision 的自定义算子：
- ``matmul``: 手写（batched）矩阵乘法与线性投影；
- ``sampling``: 手写双线性采样（等价于 ``F.grid_sample``）；
- ``deformable_attention``: 基于上述原语实现的 MSDeformAttn。
"""

from .deformable_attention import MSDeformAttn
from .matmul import manual_bmm, manual_linear, matmul_naive
from .sampling import grid_sample_manual

__all__ = [
    "MSDeformAttn",
    "manual_bmm",
    "manual_linear",
    "matmul_naive",
    "grid_sample_manual",
]
