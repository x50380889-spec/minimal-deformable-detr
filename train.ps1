<#
Windows 一键复现脚本（等价于 `make train`）。
在项目根目录执行：

    .\train.ps1

可选参数：
    .\train.ps1 -Config configs/defect.json

如果想跳过重训练、只基于已有权重重新评估出图：
    .\train.ps1 -EvalOnly
#>

param(
    [string]$Config = "configs/defect.json",
    [switch]$EvalOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "未找到 python，请先安装 Python 3.9+ 并加入 PATH"
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$StepArgs
    )
    Write-Host "`n==== $Name ====" -ForegroundColor Cyan
    & python @StepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "步骤失败：$Name"
    }
}

if (-not $EvalOnly) {
    Invoke-Step "训练教师模型（Teacher，约 40-50 分钟 CPU）" @("scripts/train_teacher.py", "--config", $Config)
    Invoke-Step "训练学生基线（Student baseline，约 15 分钟）" @("scripts/train_student.py", "--config", $Config)
    Invoke-Step "知识蒸馏（Distill，约 20 分钟）" @("scripts/distill.py", "--config", $Config)
}

Invoke-Step "评估 mAP / FPS / 参数量" @("scripts/evaluate.py", "--config", $Config)
Invoke-Step "导出 TensorBoard 事件" @("scripts/export_tb.py", "--config", $Config)
Invoke-Step "生成 loss 曲线与对比图" @("scripts/make_figures.py", "--config", $Config)

Write-Host "`n全部完成！" -ForegroundColor Green
Write-Host "  - 对比图:  training_logs/teacher_vs_student.png"
Write-Host "  - 曲线图:  training_logs/loss_curves.png"
Write-Host "  - 指标:    training_logs/metrics.json"
Write-Host "  - 查看 TensorBoard: tensorboard --logdir training_logs/tensorboard"
