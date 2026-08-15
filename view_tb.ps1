<#
启动 TensorBoard 并给出访问地址。

用法：
    .\view_tb.ps1              # 启动（若已在运行则跳过）并打印地址
    .\view_tb.ps1 -Open        # 启动后同时打开默认浏览器
#>

param(
    [int]$Port = 6006,
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "training_logs\tensorboard"

if (-not (Test-Path $LogDir)) {
    throw "未找到日志目录：$LogDir（请先运行 .\train.ps1 或分步训练脚本）"
}

function Test-TensorBoard {
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -UseBasicParsing -TimeoutSec 3
        return $resp.StatusCode -eq 200
    } catch {
        return $false
    }
}

if (Test-TensorBoard) {
    Write-Host "TensorBoard 已在运行：http://localhost:$Port/" -ForegroundColor Green
} else {
    Write-Host "正在启动 TensorBoard（日志目录：$LogDir）..." -ForegroundColor Cyan
    Start-Process -FilePath "tensorboard" -ArgumentList @(
        "--logdir", "`"$LogDir`"",
        "--host", "127.0.0.1",
        "--port", "$Port"
    ) -WindowStyle Hidden
    Start-Sleep -Seconds 8
    if (Test-TensorBoard) {
        Write-Host "已启动：http://localhost:$Port/" -ForegroundColor Green
    } else {
        Write-Warning "启动后仍未响应，请检查 tensorboard 是否已安装（python -m pip install tensorboard）"
    }
}

if ($Open) {
    Start-Process "http://localhost:$Port/"
} else {
    Write-Host "请在应用内浏览器打开或刷新 http://localhost:$Port/"
}
