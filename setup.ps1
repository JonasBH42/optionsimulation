$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
if ([IO.Path]::GetFullPath($projectDirectory).TrimEnd('\') -ne 'D:\optionsimulation') {
    throw 'This project is configured to write only inside D:\optionsimulation.'
}
Set-Location -LiteralPath $projectDirectory
foreach ($folder in @('.tmp', '.cache\pip', '.cache\uv', '.tools', '.runtime')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $projectDirectory $folder) | Out-Null
}
$env:TEMP = Join-Path $projectDirectory '.tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectDirectory '.cache\pip'
$env:UV_CACHE_DIR = Join-Path $projectDirectory '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $projectDirectory '.runtime\python'
$uvExecutable = Join-Path $projectDirectory '.tools\bin\uv.exe'
if (-not (Test-Path -LiteralPath $uvExecutable)) {
    python -m pip install --target (Join-Path $projectDirectory '.tools') 'uv==0.13.7'
    if ($LASTEXITCODE -ne 0) { throw 'Project-local uv installation failed.' }
}
& $uvExecutable python install 3.12.16 --no-bin --no-registry
if ($LASTEXITCODE -ne 0) { throw 'Project-local Python installation failed.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $uvExecutable venv '.venv' --python 3.12.16 --managed-python --no-python-downloads
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& '.\.venv\Scripts\Activate.ps1'
& $uvExecutable pip sync --python '.venv\Scripts\python.exe' --require-hashes requirements.lock
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed.' }
& $uvExecutable pip install --python '.venv\Scripts\python.exe' --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& '.\.venv\Scripts\python.exe' -m qqq_sim doctor
Write-Host 'Ready. Run .\launch.ps1 or .\launch.ps1 --demo'
