# EcoLoop setup (Windows PowerShell).
#   - verifies EnergyPlus is installed and discoverable
#   - creates a venv and pip-installs the pinned deps (+ editable package)
#   - pulls the default Ollama model
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$root = (Get-Location).Path
Write-Host "== EcoLoop setup =="
Write-Host "repo: $root"

# --- 1. EnergyPlus -----------------------------------------------------------
$eplus = $env:ENERGYPLUS_DIR
if (-not $eplus) {
  foreach ($c in @("C:\EnergyPlusV24-2-0","C:\EnergyPlusV24-1-0","C:\EnergyPlusV23-2-0")) {
    if (Test-Path $c) { $eplus = $c; break }
  }
}
if (-not $eplus -or -not (Test-Path $eplus)) {
  Write-Host "!! EnergyPlus not found. Install v24.x from https://energyplus.net/downloads"
  Write-Host "   then set it and re-run:  `$env:ENERGYPLUS_DIR='C:\EnergyPlusV24-2-0'; .\scripts\setup.ps1"
} else {
  Write-Host "-> EnergyPlus: $eplus"
  if (Test-Path (Join-Path $eplus "pyenergyplus")) {
    Write-Host "-> pyenergyplus present (config.py auto-adds it to sys.path at runtime)"
  } else {
    Write-Host "!! $eplus\pyenergyplus missing - is the install complete?"
  }
  & (Join-Path $eplus "energyplus.exe") --version 2>$null
  # Write the discovered path into config.yaml's install_dir if different.
  Write-Host "-> tip: set energyplus.install_dir in config\config.yaml to '$($eplus -replace '\\','/')'"
}

# --- 2. Python venv + deps ---------------------------------------------------
Write-Host "== Python venv =="
if (-not (Test-Path ".venv")) { python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.txt
& $py -m pip install -e .
Write-Host "-> deps installed into .venv"

# --- 3. Ollama model ---------------------------------------------------------
Write-Host "== Ollama =="
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
  try { ollama pull qwen2.5:3b-instruct } catch { Write-Host "!! model pull failed (is Ollama running?)" }
} else {
  Write-Host "!! ollama not found. Install from https://ollama.com/download, then:"
  Write-Host "   ollama pull qwen2.5:3b-instruct"
}

Write-Host "== Done. Next:  .\scripts\run_all.ps1 -Smoke =="
