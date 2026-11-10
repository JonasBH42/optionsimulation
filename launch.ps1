$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
$env:TEMP = Join-Path $projectDirectory '.tmp'
$env:TMP = $env:TEMP
$env:PYTHONPATH = Join-Path $projectDirectory 'src'
& (Join-Path $projectDirectory '.venv\Scripts\Activate.ps1')
& (Join-Path $projectDirectory '.venv\Scripts\python.exe') -m qqq_sim @args
exit $LASTEXITCODE
