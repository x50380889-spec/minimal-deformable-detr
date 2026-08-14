# Minimal Deformable DETR + Knowledge Distillation
# 用法：make train 一键复现（训练教师 -> 蒸馏学生 -> 评估 -> 出图）
# Windows 用户如果没有 make，可直接运行 scripts/ 下的等价命令（见 README）。

PYTHON ?= python
CONFIG ?= configs/defect.json

.PHONY: setup train teacher baseline distill eval export_tb figures test

setup:            ## 安装依赖（Windows 上安装 CPU 版 torch 即可）
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

train: teacher baseline distill eval export_tb figures   ## 完整复现：训练 -> 基线 -> 蒸馏 -> 评估 -> 图表

teacher:          ## 训练教师模型（极简 Deformable DETR）
	$(PYTHON) scripts/train_teacher.py --config $(CONFIG)

baseline:         ## 训练学生模型（MobileNetV3-Small，无蒸馏，用于对照）
	$(PYTHON) scripts/train_student.py --config $(CONFIG)

distill:          ## 用教师蒸馏学生模型
	$(PYTHON) scripts/distill.py --config $(CONFIG)

eval:             ## 评估 mAP / FPS / 参数量
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

export_tb:        ## 从 JSONL 导出 TensorBoard 事件文件
	$(PYTHON) scripts/export_tb.py --config $(CONFIG)

figures:          ## 生成 loss 曲线与 教师 vs 学生 对比图
	$(PYTHON) scripts/make_figures.py --config $(CONFIG)

test:             ## 运行算子正确性单测（手写 matmul/采样/注意力 vs 参考实现）
	$(PYTHON) -m pytest tests -q
