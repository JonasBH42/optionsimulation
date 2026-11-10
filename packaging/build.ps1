$ErrorActionPreference = 'Stop'
$projectDirectory = 'D:\optionsimulation'
$env:TEMP = Join-Path $projectDirectory '.tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $projectDirectory '.cache\pip'
$env:NUITKA_CACHE_DIR = Join-Path $projectDirectory '.cache\nuitka'
$env:NUITKA_CACHE_DIR_DOWNLOADS = Join-Path $projectDirectory '.cache\nuitka\downloads'
$env:NUITKA_CACHE_DIR_CCACHE = Join-Path $projectDirectory '.cache\nuitka\ccache'
& (Join-Path $projectDirectory '.venv\Scripts\Activate.ps1')
& (Join-Path $projectDirectory '.venv\Scripts\python.exe') (Join-Path $projectDirectory 'tools\build_source_manifest.py')
if ($LASTEXITCODE -ne 0) { throw 'Source manifest generation failed.' }
Push-Location -LiteralPath (Join-Path $projectDirectory 'packaging')
try {
    # Native stderr warnings must not become terminating PowerShell errors.
    # Preserve all deployment output and use the actual process exit status.
    & (Join-Path $projectDirectory '.venv\Scripts\python.exe') -u -c 'import subprocess, sys; raise SystemExit(subprocess.call(sys.argv[1:], stderr=subprocess.STDOUT))' (Join-Path $projectDirectory '.venv\Scripts\pyside6-deploy.exe') -c 'pysidedeploy.spec' --no-install --force --keep-deployment-files
    if ($LASTEXITCODE -ne 0) { throw 'Windows deployment failed. Inspect the build output.' }
} finally {
    Pop-Location
}
