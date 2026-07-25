# EcoLoop end-to-end (Windows): baseline -> AI -> compare -> dashboard.
# Usage:  .\scripts\run_all.ps1 [-Smoke]
param([switch]$Smoke)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = ".\.venv\Scripts\python.exe"
$flag = @(); if ($Smoke) { $flag = @("--smoke") }

Write-Host "== 1/4 baseline =="
& $py -m ecoloop.pipeline.run_baseline @flag
Write-Host "== 2/4 AI closed-loop =="
& $py -m ecoloop.pipeline.run_ai @flag
Write-Host "== 3/4 compare =="
& $py -m ecoloop.pipeline.compare
Write-Host "== 4/4 dashboard (Ctrl-C to stop) =="
& ".\.venv\Scripts\streamlit.exe" run dashboard/app.py
