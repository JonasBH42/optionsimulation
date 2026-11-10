# Risk engine build and measured limits

No production price, HOLD, managed-policy or envelope model pack is qualified.
The application refuses unsupported fast probabilities. The source project
contains executable reference, training, validation and recovery machinery;
the specification's full scientific release is incomplete.

All commands run from `D:\optionsimulation` with the project `.venv`. Torch
training runs in that environment. The portable executable covers replay,
inference, explicit reference calculations, validation and fixture publication;
it intentionally excludes Torch training.

```powershell
.\.venv\Scripts\Activate.ps1
python -m qqq_sim build-model --build-config runtime/risk_validation/production_frozen_v6_config.json
python -m qqq_sim build-model --build-config runtime/risk_validation/research_frozen_v6_config.json
python -m qqq_sim validate-pack runtime/model_packs/research_frozen_v6/55880437bd1f
```

The final numerical implementation fingerprint is `eb746c4f40f424dbd578f2c1bd08b2dcc396833ec3e29ffb49db71d11b9ae7c5`. Repeating
the production command resumes the same checked source/configuration. Its
120-second setting exercises recovery after the earlier 600-second work blocks.
Increasing only `budget_seconds` is compatible. Source or semantic configuration
changes require a fresh pipeline work directory. The final configuration safely
shares the unchanged price-kernel cache at
`runtime/builds/production_frozen_v3/price_basis`.

The production design contains 32,768 base states and 1,000 separate mandatory
held-out cases. The final recovery run took 120.91
seconds and restored 3,818 saved PDE schedule
steps. It has 0 accepted price cells,
7 numerical exclusions, 1593 pending
cells, and 0 production reference labels.
No production training stage has been passed. Earlier 600-second v3/v4/v5 runs
performed actual PDE work and remain under their own immutable reports. The
first normalized law failed at 25,601 nodes with spatial sensitivity about
2.10e-7 against a 2.5e-11 component budget; the requested normalized price target
remains 1e-10. Runtime alone cannot predict completion or accepted coverage.
See [the current production report](../runtime/builds/production_frozen_v6/build_report.json).

The final research build completed in 123.11 seconds,
with five preliminary reference labels, 13 conditional states across eight
minute layers, 43 policy pairs and 58 exported numeric arrays. It exercised
synthetic exact cashflows, actual model fitting, independent rollouts, frozen
feature/label schemas, saved-array validation and atomic publication. The pack
is `runtime/model_packs/research_frozen_v6/55880437bd1f`. Integrity and online-import checks passed; scientific gates failed.
The one HOLD validation error was 0.993327; within-bin error plus reference
uncertainty was 0.983014; four-path policy rollout uncertainty was 0.244945.
These values do not support calibrated inference. Two requested reference
labels were refused for unresolved narrow closing islands, and one case lacked
an identifiable ATM anchor. See [the research report](../runtime/builds/research_frozen_v6/build_report.json).

The demanding five-minute 100/101 call-vertical reference completed 200,000 paths,
with 44,578 closing hits and Wilson interval 0.221071–0.224719. Monitoring passed
at 0.0625 seconds, but independent spatial convergence did not. On a rigorously
exit-bounded path domain, 129/257/513/1025/2049/4097/8193-node values were
approximately 0.580840/0.441515/0.306507/0.248088/0.230223/0.225995/0.224981.
The last change was 0.00101379, exceeding the unchanged 0.0005 gate. The next
12,801-node transition required an estimated 15,734,374,752 additional bytes and
was refused against 8 GiB. The 364.17-second report therefore has a null headline.
Observed process peak RSS was 4.59 GiB; sampled free system memory stayed above
14 GiB. Completed refinements and probability-preservation diagnostics are
retained in [the strict reference report](../runtime/risk_validation/nontrivial_path_domain_v5_reference.json).
That mathematical kernel is unchanged in final source; the subsequent change
only adds bounded Windows atomic-file sharing retries. To run this expensive
case again under the final IO source identity, use a fresh checkpoint name:

```powershell
python runtime/risk_validation/run_nontrivial_call.py --fine --budget 600 --memory-mb 8192 --name nontrivial_path_domain_v6
```

Recovery preserves completed price refinements, active PDE time steps, reference
shards, control labels, completed rollout paths, neural epochs, Adam moments,
best weights, shuffle RNG and scaler progress. Partial minibatches restart from
the preceding complete epoch. Numeric artifacts are hash checked; metadata has
an atomic commit point. Per-law OS locks, bounded memory caches, streaming
training batches and disk reserve checks prevent unbounded work from silently
changing the mathematical job. An observed Windows checkpoint-sharing failure
was recovered from the previous committed state; the final IO routine retries
only WinError 5/32/33 for six bounded attempts and preserves permanent failures.
Six focused IO tests and 82 combined risk/reference/cache/build regressions passed.

The price builder uses shared normalized PDE grids, independent time/space/
boundary/interpolation evidence, sampled financial inequalities, saved spatial
decision trees and independent Greek/cross-face checks. HOLD training includes
path-count-weighted likelihood and 32 existing-grid within-bin deadline samples;
loaded-array validation recomputes the within-bin gate. Managed training uses
full account/action/universe features, actual eligible-minute layers, exact
broker transitions, separate policy evaluation, stronger finite control search,
conditional stress reports and up to six calibration-only augmentation cycles.
The 1,000 held-out narrow/tangent/disjoint/near-expiry/low-budget designs were
materialized with actual synthetic broker cashflows; construction does not
qualify their probability labels.

Closing singleton/tangent regions and unresolved narrow islands abstain; an
endpoint-monitoring plateau is not a continuous-crossing proof. Moving
deterministic finite-island cases also abstain unless their time crossing is
resolved. The exactly static deterministic case remains available. The
independent solver retains 0/1 tail recursions, an analytic path-domain exit bound
at most 1e-8, per-transition finite/order/range checks with explicit 1e-9 roundoff
allowance, and no probability clipping.

Remaining release work includes the general American basis, the full production
label/training run, at least 200 independent calibration and validation cases per
admitted regime, all required accuracy/stress/control gates and full-size
performance qualification. Optional European-index parity/forward identification
is not implemented; QQQ uses explicit as-of rate/carry inputs. Assignment and
delayed-notice probability remains explicitly unavailable outside the declared
forecast lifecycle, with stressed risk results reported separately.

The complete requirement map and current measurements are in
[risk_coverage.json](risk_coverage.json) and
[final_risk_status.json](../runtime/risk_validation/final_risk_status.json).
