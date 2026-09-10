$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonCommand = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pythonCommand) {
    & $pythonCommand.Source -3 -m venv .venv
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    & $pythonCommand.Source -m venv .venv
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install -e .

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $projectRoot ".env.example") `
        -Destination (Join-Path $projectRoot ".env")
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "1. Add your Groq key to .env."
Write-Host "2. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "3. Test with: python -m unittest discover -s tests -v"
Write-Host "4. Run with: robot-task `"Put the apple in the basket`""
