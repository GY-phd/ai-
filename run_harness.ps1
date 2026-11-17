# Start the closed loop (drift -> diagnose -> experiment -> gate -> rollback).
#
# The harness package carries code and config only: the industrial model is
# trained at run time by rulkit, and the decision layer talks to the
# competition-specified model endpoint when one is configured.
#
#   .\run_harness.ps1
#   .\run_harness.ps1 -Data D:\datasets\cpu-b-rul -LlmMode policy -MaxExperiments 3

param(
    [string]$Data = "",
    [string]$Workspace = "runs\harness-b",
    [int]$BatchSize = 200,
    [int]$MaxExperiments = 5,
    [string]$LlmMode = "auto",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    $env:PYTHONIOENCODING = "utf-8"
    if ($Data) { $env:RUL_DATA_ROOT = (Resolve-Path $Data).Path }
    Write-Host "data root: $env:RUL_DATA_ROOT" -ForegroundColor DarkGray
    if (Test-Path $Workspace) {
        Write-Host "removing previous workspace $Workspace" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $Workspace
    }
    Write-Host "=== closed loop: $Workspace ===" -ForegroundColor Cyan
    & $Python -m harness.run --workspace $Workspace --batch-size $BatchSize `
        --max-experiments $MaxExperiments --llm-mode $LlmMode
    Write-Host "`nartefacts: $Workspace\run_report.json, ledger.jsonl, swanlab.jsonl" -ForegroundColor Green
}
finally {
    Pop-Location
}
