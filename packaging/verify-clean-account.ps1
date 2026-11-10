[CmdletBinding()]
param(
    [string]$Executable = 'D:\optionsimulation\dist\QQQSimulator.dist\QQQSimulator.exe',
    [string]$BuildManifest = 'D:\optionsimulation\runtime\results\release-manifest.json',
    [string]$FixturePack = 'D:\optionsimulation\runtime\model_packs\validation_fixture_v2',
    [string]$OutputRoot = ('D:\optionsimulation\runtime\results\clean-account\' + (Get-Date -Format 'yyyyMMdd-HHmmss')),
    [ValidateRange(15, 900)][int]$CommandTimeoutSeconds = 180,
    [switch]$RunSameAccountDiagnostics
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath('D:\optionsimulation').TrimEnd('\')
$utf8 = New-Object Text.UTF8Encoding($false)

function ProjectPath([string]$Value) {
    if (-not [IO.Path]::IsPathRooted($Value)) { $Value = Join-Path $projectRoot $Value }
    $resolved = [IO.Path]::GetFullPath($Value).TrimEnd('\')
    if ($resolved -ne $projectRoot -and -not $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'OUTPUT_OR_INPUT_OUTSIDE_PROJECT'
    }
    $candidate = $resolved
    while ($candidate -and $candidate.Length -ge $projectRoot.Length) {
        if (Test-Path -LiteralPath $candidate) {
            if ((Get-Item -LiteralPath $candidate -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'PROJECT_PATH_REPARSE_POINT_REFUSED'
            }
        }
        $candidate = [IO.Path]::GetDirectoryName($candidate)
    }
    return $resolved
}

function AccountIdentityHash {
    # Capture the SID only in memory. Never save the account name or SID.
    $account = @(& "$env:SystemRoot\System32\whoami.exe" /user /fo csv /nh | ConvertFrom-Csv -Header Account, Sid)
    if ($LASTEXITCODE -ne 0 -or $account.Count -ne 1 -or $account[0].Sid -notmatch '^S-[0-9]+(-[0-9]+)+$') {
        throw 'WINDOWS_ACCOUNT_IDENTITY_UNAVAILABLE'
    }
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($hasher.ComputeHash($utf8.GetBytes($account[0].Sid.ToUpperInvariant())))).Replace('-', '').ToLowerInvariant()
    } finally { $hasher.Dispose() }
}

function NativeArgument([string]$Value) {
    # Windows CommandLineToArgvW quoting, including trailing backslashes.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

$commands = New-Object 'Collections.Generic.List[object]'
$outputValidated = $false
$report = [ordered]@{
    schema = 'QQQ_NATIVE_CLEAN_ACCOUNT_VERIFICATION_V1'
    recorded_at_utc = [DateTime]::UtcNow.ToString('o')
    status = 'NOT_RUN'
    separate_clean_account_passed = $false
    production_probability_qualified = $false
    fixture_scope = 'SYNTHETIC_REPLAY_AND_UNQUALIFIED_MODEL_ABSTENTION'
    account_creation_attempted = $false
    system_changes_attempted = $false
    installation_or_download_attempted = $false
    development_paths_removed_from_children = $true
    qt_platform = 'windows'
}

function SaveReport {
    $report.commands = $commands.ToArray()
    [IO.File]::WriteAllText((Join-Path $OutputRoot 'result.json'), ($report | ConvertTo-Json -Depth 100), $utf8)
    [ordered]@{status=$report.status; reason=$report.reason; separate_clean_account_passed=$report.separate_clean_account_passed;
        output=(Join-Path $OutputRoot 'result.json'); commands=$commands.Count} | ConvertTo-Json
}

function InvokePortable([string]$Name, [string[]]$Arguments, [int]$ExpectedExit, [scriptblock]$Validate, [double]$Scale = 1, [double]$ExpectedDpr = 0) {
    Write-Host ("Running standalone check: " + $Name)
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $Executable
    $start.WorkingDirectory = $projectRoot
    $start.Arguments = (($Arguments | ForEach-Object { NativeArgument $_ }) -join ' ')
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    foreach ($environmentName in @('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','VIRTUAL_ENV_PROMPT','CONDA_PREFIX','QT_PLUGIN_PATH','QT_QPA_PLATFORM_PLUGIN_PATH','QT_SCREEN_SCALE_FACTORS','QT_ENABLE_HIGHDPI_SCALING','QT_FONT_DPI','QT_SCALE_FACTOR_ROUNDING_POLICY','QT_AUTO_SCREEN_SCALE_FACTOR','QT_SCALE_FACTOR')) {
        $start.EnvironmentVariables.Remove($environmentName)
    }
    $start.EnvironmentVariables['PATH'] = "$env:SystemRoot\System32;$env:SystemRoot"
    $start.EnvironmentVariables['TEMP'] = Join-Path $projectRoot '.tmp'
    $start.EnvironmentVariables['TMP'] = Join-Path $projectRoot '.tmp'
    $start.EnvironmentVariables['TMPDIR'] = Join-Path $projectRoot '.tmp'
    $start.EnvironmentVariables['QT_QPA_PLATFORM'] = 'windows'
    $start.EnvironmentVariables['QT_SCALE_FACTOR'] = $Scale.ToString([Globalization.CultureInfo]::InvariantCulture)
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $start
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    try {
        [void]$process.Start()
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($CommandTimeoutSeconds * 1000)) {
            $timedOut = $true
            # Limit termination to the process tree created by this command.
            $nativeErrorPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                & "$env:SystemRoot\System32\taskkill.exe" /PID $process.Id /T /F 2>&1 | Out-Null
            } finally { $ErrorActionPreference = $nativeErrorPreference }
            if (-not $process.WaitForExit(10000)) { $process.Kill(); $process.WaitForExit() }
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $exitCode = $process.ExitCode
    } finally { $clock.Stop(); $process.Dispose() }
    $stdoutPath = Join-Path $OutputRoot ($Name + '.stdout.json')
    $stderrPath = Join-Path $OutputRoot ($Name + '.stderr.txt')
    [IO.File]::WriteAllText($stdoutPath, $stdout, $utf8)
    [IO.File]::WriteAllText($stderrPath, $stderr, $utf8)
    $parsed = $null
    $assertionsPassed = $false
    try {
        $parsed = $stdout | ConvertFrom-Json
        $assertionsPassed = [bool](& $Validate $parsed)
    } catch { $assertionsPassed = $false }
    $record = [ordered]@{name=$Name; arguments=$Arguments; qt_scale_factor=$Scale; expected_effective_dpr=$(if ($ExpectedDpr -gt 0) { $ExpectedDpr } else { $null }); exit_code=$exitCode;
        expected_exit_code=$ExpectedExit; timed_out=$timedOut; seconds=$clock.Elapsed.TotalSeconds;
        structured_assertions_passed=$assertionsPassed; passed=(-not $timedOut -and $exitCode -eq $ExpectedExit -and $assertionsPassed);
        stdout=$stdoutPath; stderr=$stderrPath}
    $commands.Add($record)
    return $parsed
}

try {
    $OutputRoot = ProjectPath $OutputRoot
    $outputValidated = $true
    [void][IO.Directory]::CreateDirectory($OutputRoot)
    $Executable = ProjectPath $Executable
    $BuildManifest = ProjectPath $BuildManifest
    $FixturePack = ProjectPath $FixturePack
    $manifest = Get-Content -LiteralPath $BuildManifest -Raw | ConvertFrom-Json
    $report.current_account_identity_sha256 = AccountIdentityHash
    $report.build_account_identity_sha256 = $manifest.build_account_identity_sha256
    $report.expected_build_id = $manifest.build_id
    $report.build_manifest_sha256 = (Get-FileHash -LiteralPath $BuildManifest -Algorithm SHA256).Hash.ToLowerInvariant()
    $report.verifier_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($report.build_account_identity_sha256 -notmatch '^[0-9a-f]{64}$' -or $report.expected_build_id -notmatch '^[0-9a-f]{64}$') {
        throw 'BUILD_ACCOUNT_OR_BUILD_ID_EVIDENCE_MISSING'
    }
    $sameAccount = $report.current_account_identity_sha256 -eq $report.build_account_identity_sha256
    $report.account_is_distinct_from_build = -not $sameAccount
    if ($sameAccount -and -not $RunSameAccountDiagnostics) {
        $report.status = 'REFUSED_SAME_BUILD_ACCOUNT'
        $report.reason = 'The current account is the build account; this cannot satisfy the separate clean-account gate.'
        SaveReport
        exit 2
    }
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) { throw 'STANDALONE_EXECUTABLE_MISSING' }
    $fixture = Get-Content -LiteralPath (Join-Path $FixturePack 'manifest.json') -Raw | ConvertFrom-Json
    if ($fixture.pack_kind -ne 'VALIDATION_FIXTURE' -or $fixture.validation_status -ne 'UNQUALIFIED') {
        throw 'EXPLICIT_UNQUALIFIED_VALIDATION_FIXTURE_REQUIRED'
    }
    $job = Join-Path $FixturePack 'example_reference_job.json'
    if (-not (Test-Path -LiteralPath $job -PathType Leaf)) { throw 'FIXTURE_REFERENCE_JOB_MISSING' }
    $report.executable = $Executable
    $report.executable_sha256 = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash.ToLowerInvariant()
    $report.fixture_manifest_sha256 = (Get-FileHash -LiteralPath (Join-Path $FixturePack 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
    [void][IO.Directory]::CreateDirectory((ProjectPath (Join-Path $projectRoot '.tmp')))
    $doctor = InvokePortable 'doctor' @('doctor') 0 {param($r) $r.runtime_ready -eq $true -and $r.mode -eq 'STANDALONE' -and $r.build_id -eq $report.expected_build_id}
    $report.actual_build_id = $doctor.build_id
    if (-not $commands[$commands.Count-1].passed) { throw 'STANDALONE_DOCTOR_OR_BUILD_ID_FAILED' }
    $null = InvokePortable 'snapshot' @('--demo','snapshot','--at','2026-11-09T10:01:00-05:00','--output',(Join-Path $OutputRoot 'snapshot.json')) 0 {param($r) $r.status -eq 'OK' -and $r.contracts -gt 0}
    $replay = InvokePortable 'replay' @('--demo','replay','--at','2026-11-09T10:01:00-05:00','--minutes','5','--output',(Join-Path $OutputRoot 'replay.json')) 0 {param($r) $r.status -eq 'OK' -and -not [string]::IsNullOrWhiteSpace($r.session_id)}
    if ($replay.session_id) {
        $null = InvokePortable 'export' @('export',$replay.session_id,'--output',(Join-Path $OutputRoot 'export.json')) 0 {param($r) $r.status -eq 'OK'}
    } else { $commands.Add([ordered]@{name='export';passed=$false;reason='REPLAY_SESSION_ID_MISSING';executed=$false}) }
    $null = InvokePortable 'fast-query' @('fast-query',$job,'--pack',$FixturePack,'--output',(Join-Path $OutputRoot 'fast-query.json')) 2 {param($r) $r.status -eq 'UNSUPPORTED' -and $r.reason -eq 'PACK_NOT_QUALIFIED'}
    $null = InvokePortable 'validate-pack' @('validate-pack',$FixturePack,'--output',(Join-Path $OutputRoot 'validation.json')) 2 {param($r) $r.status -eq 'VALID_UNQUALIFIED_PACK' -and $r.online_import_boundary.passed -eq $true}
    $baseline = InvokePortable 'desktop-native-baseline' @('desktop-smoke','--pack',$FixturePack,'--output',(Join-Path $OutputRoot 'desktop-native-baseline')) 0 {param($r) $r.status -eq 'OK' -and $r.scale -gt 0 -and $r.window_geometry.fits_available_screen -eq $true} 1.0
    if (-not $commands[$commands.Count-1].passed) { throw 'NATIVE_DPI_BASELINE_FAILED' }
    $report.native_baseline_dpr = [double]$baseline.scale
    $report.effective_dpr_targets = @(1.0,1.5,2.0)
    foreach ($targetDpr in @(1.0, 1.5, 2.0)) {
        $name = 'desktop-' + [int]($targetDpr * 100)
        $processScale = $targetDpr / $report.native_baseline_dpr
        $validateScale = {param($r) $r.status -eq 'OK' -and $r.saved_and_restored -eq $true -and
            $r.window_geometry.fits_available_screen -eq $true -and [Math]::Abs($r.scale - $targetDpr) -lt 0.01 -and
            @($r.geometry_checks | Where-Object { -not $_.fits_available_screen }).Count -eq 0}.GetNewClosure()
        $null = InvokePortable $name @('desktop-smoke','--pack',$FixturePack,'--output',(Join-Path $OutputRoot $name)) 0 $validateScale $processScale $targetDpr
    }
    $passed = @($commands | Where-Object { -not $_.passed }).Count -eq 0
    $report.workflows_passed = $passed
    $report.separate_clean_account_passed = $passed -and -not $sameAccount
    if ($sameAccount) {
        $report.status = 'SAME_ACCOUNT_DIAGNOSTICS_ONLY'
        $report.reason = 'Actual command results are recorded, but the build account cannot satisfy the separate-account gate.'
    } elseif ($passed) {
        $report.status = 'PASS_SEPARATE_ACCOUNT_WORKFLOWS'
        $report.reason = 'Distinct account and the required standalone workflows passed; production numerical qualification is separate.'
    } else {
        $report.status = 'FAILED'
        $report.reason = 'One or more standalone command assertions failed; inspect the captured command output.'
    }
    SaveReport
    if ($report.separate_clean_account_passed) { exit 0 } else { exit 2 }
} catch {
    $report.status = 'FAILED'
    $report.reason = $_.Exception.Message
    if ($outputValidated -and (Test-Path -LiteralPath $OutputRoot -PathType Container)) { SaveReport }
    else { Write-Error 'Verification could not create a project-local report directory.' -ErrorAction Continue }
    exit 1
}
