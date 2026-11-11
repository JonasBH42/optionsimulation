# QQQ historical options simulator

A local PySide6 desktop application for causal QQQ replay, simulated options orders, exact cash accounting, and offline P50 research. It reads the QQQ archives without modifying them. All generated code, environments, caches, sessions, packs and exports stay inside `D:\optionsimulation`.

**The application is implemented, but the entire specification is not complete or scientifically qualified.** The strict production build has no accepted price cells and no production reference labels. The completed `research_frozen_v6` pack fails its accuracy gates and remains unqualified. A separate demanding 200,000-path reference calculation has a null headline: independent spatial convergence remained unresolved and its next refinement exceeded the declared memory budget.

The saved numerical evidence is in [final_risk_status.json](runtime/risk_validation/final_risk_status.json) and [risk_build_and_validation.md](docs/risk_build_and_validation.md). [release-status.json](docs/release-status.json) tracks the remaining gates. Missing inputs and unqualified models return explicit unavailable states; replay and simulated account workflows remain usable.

The fast-skip update passes all 577 tests, recorded in `runtime/results/fast-skip-tests.xml` with hashes in `runtime/results/fast-skip-test-evidence.json`. Its packaged verification is recorded in `dist/QQQSimulator.dist/evidence/fast-skip-verification.json`. The prior release's numerical, native/offscreen, and 100%, 150%, and 200% viewport reports remain historical evidence; a separate clean Windows account has not been tested.

## Run

The project uses its own 64-bit Python 3.12 environment. In PowerShell:

```powershell
cd D:\optionsimulation
.\.venv\Scripts\Activate.ps1
.\launch.ps1
```

`launch.cmd` also activates `.venv` and starts the application. Use `--demo` to explicitly open synthetic demonstration data:

```powershell
.\launch.ps1 --demo
```

The standalone Windows build is `dist\QQQSimulator.dist\QQQSimulator.exe`.
Keep the entire `QQQSimulator.dist` folder together. Development, testing and
offline Torch training use the project `.venv`; the standalone application
includes its own replay/inference runtime.

To run the portable application from the project directory:

```powershell
cd D:\optionsimulation
.\dist\QQQSimulator.dist\QQQSimulator.exe doctor
.\dist\QQQSimulator.dist\QQQSimulator.exe --demo
```

The standalone folder's `artifact-manifest.json` and `artifact-manifest.sha256`
identify its assembled files; `evidence/fast-skip-verification.json` records the
tested update and source/compiled session interoperability. Previous release
reports are under `evidence/historical-before-fast-skip`. Retain the supplied notices, licenses and corresponding dependency source
archives with that folder. This is a local project handoff: the complete editable
source, development environment, offline checkpoints and market-data paths remain
under or relative to `D:\optionsimulation`, rather than inside the executable.

For a fresh installation, run `setup.ps1`. It installs Python under `.runtime`, creates `.venv`, installs hash-locked dependencies, and does not change the system Python or register a new interpreter. If local script execution is disabled, run `powershell -ExecutionPolicy Bypass -File .\setup.ps1`; this does not persistently change policy.

Startup checks imports, timezone behavior, project-local writable directories,
disk reserve and schema compatibility. Missing market configuration opens setup
guidance; an absent or unqualified pack disables analytics without blocking
viewing or the explicit demo. Details are in [startup-checks.md](docs/startup-checks.md).

## Desktop workflow

1. Open the historical archives or select **Session → Open synthetic demonstration**. The toolbar shows both the completed candle and the New York decision time.
2. Advance by trading minutes or elapsed minutes. Chart aggregation is independent. Multi-minute advances process intervening execution/lifecycle events without rebuilding the display each minute, and stop at an unresolved required observation.
3. Select an option and use **Buy selected** or **Sell selected**. Edit signed quantities in the draft. Positive quantity is long. The signed limit is the minimum dollar premium cash **per ratio unit**: `+100` requires at least $100 credit; `-200` permits at most $200 debit. Fees are separate.
4. Set the finite target deadline and hard cutoff. **Check costs / risk** previews on a clone without changing the account. **Submit simulated order** reserves capital; fills require eligible later quotes.
5. Positions and orders provide close/roll/cancel actions. Keep the campaign ID through adjustments to preserve the original dollar target and cash history.
6. Sessions save after each accepted command and complete replay batch. Fast skips save after at most 64 trading minutes or approximately 250 ms of processing, plus session boundaries. Restore a saved SQLite file, branch from a committed event, or export a frozen JSON/TXT/CSV record. Quiet intermediate minutes are not separate saved branch points.

Sessions freeze source, dependency-lock and runtime identities alongside their
data/model revisions. Restore refuses missing or changed implementation identities
before rebuilding account state. The reviewed original executable and pre-update
source build have narrow copied upgrades: each original SQLite file is retained, its stored
history is verified, and continuation uses a new branch in a separate working copy.
Reopening the original selects the existing upgraded progress. Other incompatible
versions remain refused. Use `export-saved` to inspect an older session's
original stored state and analyses without replay or migration; see
`docs/runtime-identity.md` for the compatibility contract.

During Step or a jump, the status bar shows the requested time and the last saved
batch; Escape requests a stop after the current completed batch. The displayed chart and
diagnostics keep their previous completed snapshot until the action finishes.
The diagnostics `decision_at` and `state_version` identify that snapshot. Archive
warnings include source paths and row numbers when available.

Analysis requests and results retain their original snapshot, session, branch
and model revision. They do not advance the economic ledger. Shutdown archives
completed results and explicitly cancels unfinished requests; stale results stay
in the export without becoming current trade instructions. See
`docs/analytic-records.md` for the audit and recovery behavior.

The default account is the explicitly specified conservative cash-collateral profile with $5,000 initial cash, $5,000 capital cap, and $1,000 campaign/session loss caps. Fee rules are the fixed 2026-10-28 counterfactual simulation profile. Protected short spreads require the named simultaneous protective-delivery assumptions; these are simulation assumptions, not a broker guarantee.

## Data configuration

Copy `config.example.json` to a file inside the project and launch with `--config path.json`. Default read-only inputs are:

- `D:\QQQ\OHLCV_daily`
- `D:\QQQ\raw_theta`

Duplicate logical partitions are reported; the app does not silently select a copy or repair an archive. The real audit is in `docs/data-audit.json`. Quote viewing does not require an IV or a model pack. Trading needs verified contract metadata. Analytics additionally needs dated rate/carry/path-IV/fixing inputs and qualified pack coverage.

Read-only source inspection writes its reports only inside the project:

```powershell
.\launch.ps1 audit-data --output runtime/results/data-audit.json
.\launch.ps1 snapshot --at '2012-06-01T10:01:00-04:00' --output runtime/results/snapshot.json
```

Neither command repairs or changes the external archives. To use different input
roots, edit a project-local copy of `config.example.json` and prepend
`--config config.json` to the command. Missing or duplicate source observations
remain diagnosed instead of being fabricated.

`normalization_path` can point to a JSON mapping with `strike_divisor`, `iv_unit`, timestamp precision, and metadata evidence/scenario declaration. Do not mark uncertain real observations verified simply to enable a button. An explicit synthetic mapping is generated with the demo.

`parameters_path` points to a dated input record. The minimal explicit scenario format is:

```json
{
  "rate_cc": 0.04,
  "dividend_yield_cc": 0.01,
  "sigma_path": 0.25,
  "known_at": "2026-11-09T09:30:00-05:00",
  "fixing_at": "2026-11-13T16:00:00-05:00",
  "source": "EXPLICIT_SCENARIO"
}
```

Those numbers are an example scenario, not historical observations. Lifecycle sidecars can add `lifecycle_events`; see `docs/lifecycle-inputs.md`. Missing official fixing causes a strict stop unless an explicit proxy scenario is selected.

For a roll, add an `expirations` object keyed by ISO expiration date. Each entry must include its explicit `fixing_at` and may override dated model inputs. The held-position calculation keeps its own expiry; the proposed close/open changes use each contract's actual fixing. An absent fixing makes that expiry unavailable for analysis. Closing a campaign and re-entering under the same campaign ID preserves its original target and accumulated cash.

The managed forecast selects its path inputs from the admitted expiry nearest
the fixed hard cutoff and records that choice. The unchanged HOLD forecast keeps
its own inputs. Observed bid/ask half-spreads and actual last-trading times are
carried into the managed calculation; a new expiry beyond the campaign cutoff
is rejected.

When `sigma_path` is absent, the causal local-ATM estimator requires a separately qualified pricing pack and sufficient compatible, fresh quotes. It records the contributing contracts and weights. It never infers an IV from unknown units or launches an offline inverse-pricing solve during a GUI query.

`calendar_path` can point to an array (or an object with a `sessions` array) of explicit `SessionRecord` objects. Each record names its `local_date`, offset-aware `exchange_open`, `underlying_close`, `option_close`, optional `fixing_at` and exercise cutoff, status and source/profile identity. Unspecified dates use the visibly labeled calendar scenario. Calendar and normalization file hashes are frozen with the session.

## Headless commands

```powershell
.\launch.ps1 doctor
.\launch.ps1 audit-data --output runtime/results/data-audit.json
.\launch.ps1 snapshot --at '2012-06-01T10:01:00-04:00'
.\launch.ps1 --demo replay --at '2026-11-09T10:01:00-05:00' --minutes 5
.\launch.ps1 export SESSION_ID --output runtime/results/export.json
.\launch.ps1 export-saved --session SESSION_ID --output runtime/results/original-session.json
.\launch.ps1 build-model --fixture --output runtime/model_packs/validation-fixture
.\launch.ps1 build-model --build-config runtime/build-config.json --output runtime/model_packs/research
.\launch.ps1 validate-pack runtime/model_packs/validation-fixture
.\launch.ps1 fast-query runtime/model_packs/validation-fixture/example_reference_job.json --pack runtime/model_packs/validation-fixture
.\launch.ps1 reference runtime/model_packs/validation-fixture/example_reference_job.json
.\launch.ps1 --config config.json --at '2012-06-01T10:01:00-04:00' p50-reference
```

The reference CLI uses exactly three trade stages: expiration; repeated `p/c strike s/l` lines and `done`; then an absolute New York deadline. `undo`, `e`, and `q` support navigation. Input deficiencies identify the missing record instead of adding a hidden prompt.

With an empty `DEFAULT_SYMBOL`, a symbol is selected before those trade stages. The settings and exact position-specific overrides are in `src/qqq_sim/services/reference_wizard.py`; optional target percentages and date offsets share the same frozen calculation.

Model construction/reference commands are explicitly offline. Ordinary GUI analysis and replay never start Monte Carlo, PDE solves, training, or a rebuild. The NumPy fast runtime verifies pack files, input support and qualification before returning probabilities. Gross unchanged-position first hit and net executed-policy success remain distinct events.

## Resume the unfinished scientific build

Run the strict production configuration in the project `.venv`:

```powershell
cd D:\optionsimulation
.\.venv\Scripts\Activate.ps1
.\launch.ps1 build-model --build-config runtime/risk_validation/production_frozen_v6_config.json
```

This resumes the checked pipeline in `runtime/builds/production_frozen_v6` and
its compatible shared price cache. The saved configuration uses bounded work
blocks; increasing only `budget_seconds` preserves resume compatibility. A source
or semantic configuration change requires a fresh work directory. More runtime
does not guarantee convergence or qualified coverage, and accuracy gates must
remain unchanged.

The completed research run can be inspected without treating it as a production
model:

```powershell
.\launch.ps1 validate-pack runtime/model_packs/research_frozen_v6/55880437bd1f --output runtime/results/research-frozen-v6-validation.json
```

Validation is expected to report unqualified and exit nonzero. The remaining
work includes the general American price/Greek/inverse-IV basis, production
reference labels and HOLD/managed training, independent calibration/validation
and stress/control gates, strict reference convergence, and supported-domain
latency/memory qualification. See the numerical status document for measured
limits and the explicit expensive reference command.

## Verification and packaging

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q --junitxml runtime/results/tests.xml
.\packaging\build.ps1
python tools/refresh_portable.py
python tools/windows_smoke.py --platform offscreen
python tools/windows_smoke.py --platform windows
python tools/cross_runtime_smoke.py
python tools/collect_evidence.py
.\packaging\verify-clean-account.ps1 -RunSameAccountDiagnostics -OutputRoot D:\optionsimulation\runtime\results\clean-account\final-same-account
python tools/package_release.py
```

The native verifier deliberately exits 2 for same-account diagnostics, even when its workflows pass. It does not claim a clean-account pass. The final package includes captured fixture-publication and visual-review evidence from this build; rerun or update these records when changing the corresponding behavior.

The deployment settings were generated by the installed `pyside6-deploy --init` and are retained in `packaging/pysidedeploy.spec`. The standalone executable supports replay, inference, explicit reference, validation, and fixture publication; offline Torch training uses the development `.venv`. Raw archives remain external. The handoff includes the explicitly unqualified fixture and research packs. Display-scale images, test reports, numerical validation and build logs are stored under `runtime/results`.

`refresh_portable.py` copies the fresh compiler output and archives the previous
distribution inside the project, preventing obsolete files from surviving a
deployment merge. The smoke commands use the existing unqualified
`runtime/model_packs/validation_fixture_v2` fixture. `package_release.py` requires
matching successful smoke reports and test evidence before assembling notices,
dependency source archives and a SHA-256 inventory. Separate-account verification
is documented in `docs/clean-account-verification.md`.

Only evidence that matches the final build can close its packaging gates. A
successful bundle assembly does not qualify its probability models or certify
the complete specification. Keep the original project and saved numerical reports
available when handing off the portable folder.

The complete original specification remains unchanged in `qqq_simulator_5_part_implementation_plan_complete.txt`. `docs/requirements.json` maps its acceptance IDs to evidence and outstanding gates; conditional broker/SPX/Android adapters are not advertised as available.

Detailed numerical status and resumable build commands are in `docs/risk_build_and_validation.md`. `tools/reference_resource_benchmark.py` exercises the hashed disk-backed interval store and a spawned reference calculation. These measurements are separate from the required broad-domain probability accuracy and latency qualification.
