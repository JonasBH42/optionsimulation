# Saved-session implementation identity

Each committed state and normal export freezes `implementation_identity` with
schema `QQQ_SESSION_IMPLEMENTATION_V1`. The content digest covers the application
version, every `src/qqq_sim/**/*.py` source hash except generated `_build_info.py`,
the exact `requirements.lock` bytes, the installed versions of the seven declared
runtime dependencies, Python implementation/version/word size/byte order, and
the SQLite library version. Model, calendar, normalization, lifecycle and archive
revisions retain their separate existing bindings.

Live source runs derive the manifest from the loaded package directory; the
generated build's `DEVELOPMENT` marker does not substitute for source hashes.
Compiled runs verify the embedded build-manifest digest and any embedded online
audit source bytes, then select the same package-source subset. Package paths,
executable paths, source origin and packaging-generator files do not enter the
compatibility digest. Thus an otherwise identical `.venv` and standalone runtime
share an identity. The embedded check establishes internal content consistency,
not a publisher signature. Source/dependency inspection is cached once per
process; restart after changing installed code or dependencies.

Restore validates this identity before reconstructing account objects. It refuses
missing identities with `SESSION_IMPLEMENTATION_IDENTITY_MISSING`, malformed or
tampered identity records with `SESSION_IMPLEMENTATION_IDENTITY_INVALID`, and
different source/lock/runtime content with `SESSION_IMPLEMENTATION_REVISION_CHANGED`.
Branches preserve the exact frozen record, including its original provenance.
Reading, exporting and checking identity do not advance economic state versions.

The replay performance release has one explicit compatibility exception:
`QQQ_FAST_REPLAY_UPGRADE_V1`. Its two frozen predecessors are recorded in
`src/qqq_sim/services/_session_upgrade_policy.py`; these records are not inferred
from whatever source happens to be installed. They identify the original installed native
distribution and the source checkout immediately before the replay change.
Only these exact source manifests, their application version and dependency lock
are accepted, and the saved Python/SQLite/runtime
dependency versions must match the current process. General identity verification
remains strict; unknown versions and missing or damaged identities remain refused.

When either predecessor is restored, the original database stays unchanged as the
backup. A read-only SQLite transaction validates all branch/event/checkpoint and
modern analysis hashes, then SQLite backup captures the complete snapshot,
including committed WAL data. The staged working copy receives a new child branch
and an `IMPLEMENTATION_UPGRADED` event identifying both implementations and the
exact parent state. Its cash, positions, orders, campaign data and decision time
remain unchanged. Original events, identities and analyses remain historical;
new analysis requests use the new branch. Branch creation and the transition are
one transaction, and only a complete verified copy is published.

Working copies live under the saved configuration's
`runtime_root/session_upgrades/<snapshot-and-target-digest>/session.sqlite`.
Reopening the same original save reuses that copy and its subsequent progress.
Opening an unknown or damaged existing copy fails rather than overwriting it.
Selecting a legacy historical state likewise creates an explicit upgraded child;
it never relabels historical events. Worker recovery uses the actual database
path, which may differ from the original session-ID location.

## Export an older or incompatible session

```powershell
cd D:\optionsimulation
.\.venv\Scripts\Activate.ps1
.\launch.ps1 export-saved --session SESSION_ID --output runtime/results/original-session.json
# A project-local SQLite path can be used instead of SESSION_ID.
.\launch.ps1 export-saved --session runtime/sessions/SESSION_ID/session.sqlite --branch BRANCH_ID --output runtime/results/original-branch.json
```

`export-saved` opens known storage schema 1 with SQLite `mode=ro` and `query_only`
in a consistent read transaction. It validates stored branch/event/checkpoint
hashes, includes every original table row and schema declaration, preserves stored
JSON strings and analysis results, and adds an export digest. Its selected saved
state is the stored JSON state; it does not instantiate the broker, reprice a
position, load a current model or apply a migration. Legacy files without the
analysis-request table remain supported. Unknown storage schemas or corrupt hash
chains are refused. The database and output must be inside the project so SQLite
sidecar handling cannot write outside the authorized directory; copy an external
saved database and its required WAL files into the project first. Export may not
overwrite the database or its SQLite sidecars.

`tests/test_runtime_identity.py` exercises live/embedded digest equivalence,
embedded corruption, detached process caching, new-session restore/export/branch,
pre-reconstruction refusals, unchanged database bytes during legacy/different-build
raw exports, exact preserved analysis payloads, corrupt chains, CLI session-ID
routing and overwrite protection. These are compatibility and storage checks;
they do not qualify numerical probability models.

`tests/test_session_upgrade.py` covers both exact predecessors' admission, matching
runtime requirements, live-WAL copying, unchanged originals and economic state,
preserved historical records, reusable progress, atomic transition rollback,
failed publication cleanup and refusal to replace a corrupted working copy.
