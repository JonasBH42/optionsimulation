# Native standalone verification on a separate Windows account

`packaging/verify-clean-account.ps1` needs PowerShell and the already-built portable
distribution. It does not use Python, activate `.venv`, install software, create an
account, request elevation or download/build a model. Sign in to an existing
separate Windows account that can read the distribution and write the project.
Keep the complete distribution and an existing unqualified validation fixture.

```powershell
cd D:\optionsimulation
powershell -NoProfile -ExecutionPolicy Bypass -File .\packaging\verify-clean-account.ps1 `
  -Executable .\dist\QQQSimulator.dist\QQQSimulator.exe `
  -BuildManifest .\runtime\results\release-manifest.json `
  -FixturePack .\runtime\model_packs\validation_fixture_v2 `
  -OutputRoot .\runtime\results\clean-account\separate-account
```

The manifest must contain the actual build ID and `build_account_identity_sha256`
recorded by the build evidence collector. The script hashes the current account's
Windows SID in memory; neither the SID nor account name is stored. Matching hashes
produce `REFUSED_SAME_BUILD_ACCOUNT`, exit code 2, and no launched verification
commands. `-RunSameAccountDiagnostics` explicitly permits local diagnostics but
still never passes the separate-account gate.

For a distinct account, the script removes Python/virtual-environment variables
from each child and limits its PATH to Windows directories. It checks the actual
standalone doctor's build ID, snapshot, five-minute synthetic replay, saved-session
export, expected unqualified-pack rejection through spawned fast inference, pack
validation, and Windows Qt desktop smoke at effective device-pixel ratios 1, 1.5
and 2. A real native baseline run measures the monitor's existing DPI; each target
then uses `QT_SCALE_FACTOR = target / baseline`. The result must report the
requested effective ratio and a native window frame fully within the available
screen. Ticket, account, risk and closed-state screenshots are retained, with
the submit control explicitly scrolled into view and checked. Captured stdout,
stderr, exit codes, durations, timeouts and structured assertions are written under
the requested project-local output directory. Screenshots and session exports are
retained by the desktop smoke commands. No source or fixture build is implicit.

Exit 0 means these workflows passed on an account distinct from the recorded build
account. It does not qualify probability accuracy, historical broker parity or the
complete project. Same-account refusal was exercised locally; an actual separate
clean-account run remains an external release gate until its real report exists.
An invocation targeting the existing `C:\Windows` directory was also rejected
before any report write; the before/after check is recorded inside the project at
`runtime/results/clean-account/output-boundary.json`.
