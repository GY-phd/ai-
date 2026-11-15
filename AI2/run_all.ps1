# B task (bearing RUL) - end-to-end workflow
#
# Usage:
#   .\run_all.ps1
#   .\run_all.ps1 -RunName round-2 -Epochs 600 -Hidden 64
#
# Steps: score baseline -> continue-train with distillation transfer -> score the
# new checkpoint -> row-by-row prediction -> re-score the prediction file.

param(
    [string]$Data = "data",
    [string]$RunName = "round-1",
    [int]$Epochs = 400,
    [int]$Hidden = 64,
    [double]$Lr = 0.003,
    [double]$WeightDecay = 0.01,
    [string]$FeatureTransform = "eng34",
    [int]$TransferEpochs = 80,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    $train = Join-Path $Data "public\train.jsonl"
    $val = Join-Path $Data "public\validation.jsonl"
    $baseline = Join-Path $Data "checkpoint.npz"
    $out = Join-Path "runs" $RunName
    if (Test-Path $out) { throw "output directory already exists: $out (each round needs a new directory)" }

    Write-Host "`n=== 1/5 score the official baseline (reference point) ===" -ForegroundColor Cyan
    & $Python -m rulkit.evaluate --checkpoint $baseline --data $val

    Write-Host "`n=== 2/5 continue training from the baseline checkpoint ===" -ForegroundColor Cyan
    & $Python -m rulkit.train --train $train --validation $val --from-baseline $baseline `
        --transfer --transfer-epochs $TransferEpochs --epochs $Epochs --hidden $Hidden `
        --learning-rate $Lr --weight-decay $WeightDecay --feature-transform $FeatureTransform `
        --output $out --stage-id $RunName

    Write-Host "`n=== 3/5 score the new checkpoint ===" -ForegroundColor Cyan
    & $Python -m rulkit.evaluate --checkpoint (Join-Path $out "checkpoint.npz") --data $val

    Write-Host "`n=== 4/5 predict row by row with the platform protocol ===" -ForegroundColor Cyan
    $requests = Join-Path $out "validation_requests.jsonl"
    $preds = Join-Path $out "validation_predictions.jsonl"
    & $Python (Join-Path $root "tools\make_requests.py") --data $val --output $requests
    & $Python -m rulkit.predict --checkpoint (Join-Path $out "checkpoint.npz") --input $requests --output $preds

    Write-Host "`n=== 5/5 re-score from the prediction file ===" -ForegroundColor Cyan
    & $Python -m rulkit.evaluate --pred $preds --data $val

    Write-Host "`ndone: $out" -ForegroundColor Green
}
finally {
    Pop-Location
}
