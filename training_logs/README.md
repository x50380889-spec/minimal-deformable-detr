# training_logs/ 内容说明

本目录存放可直接用于面试展示的训练产物（全部由 `make train` 生成并入库）：

| 文件 / 目录 | 内容 |
| --- | --- |
| `tensorboard/` | TensorBoard 事件文件（`teacher` / `distill` / `student_scratch` 三个 run） |
| `loss_curves.png` | 教师训练 loss 曲线 + 蒸馏训练 loss 曲线（与 TensorBoard 同源数据） |
| `teacher_vs_student.png` | **教师 vs 蒸馏学生对比图**：mAP / FPS / 参数量 |
| `metrics.json` | 评估汇总：mAP@50、各类 AP、FPS、参数量、速度提升与精度差 |
| `scalars/*.jsonl` | 结构化标量日志（不装 TensorBoard 也能画曲线） |
| `samples/` | 数据集样本、教师/学生检测可视化（绿框=真值，红框=预测） |

## 查看 TensorBoard

```bash
tensorboard --logdir training_logs/tensorboard
```

## 指标口径（面试时按这个口径讲）

- **mAP@50**：验证集 200 张（确定性生成），101 点插值，IoU 阈值 0.5；
- **FPS**：CPU 端到端推理（batch=1，128×128，含全部前后处理），
  预热后多次计时取平均——贴合端侧单帧实时性场景；
- **参数量**：仅统计可训练参数。
