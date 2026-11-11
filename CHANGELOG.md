# Change log

## Unreleased

Multi-minute Step, Jump and Next session use indexed causal quote replay, one destination display snapshot, and durable batches instead of a full refresh/save every minute. Orders, lifecycle events and missing-data stops retain their original execution barriers. Saved sessions from the reviewed original executable and pre-update source build continue through verified copied upgrades; original files and history are retained. Intermediate quiet minutes are no longer separate saved branch points.

Step replay reads newly available quote/trade windows and retains compact source indexes ahead of disposable observations. Forward chain summaries avoid rebuilding prior quote history; late or conflicting observations retain full causal validation. Advancing shows the requested time and committed progress, and diagnostics identify the displayed decision time and offending archive files/rows. Ambiguous OI and mismatched OHLCV sources remain unavailable.

## 0.1.0

Initial implementation of the supplied QQQ historical simulation specification: isolated Python environment, source audits and causal adapters, desktop replay, simulated execution and accounting, lifecycle scenarios, transactional recovery/branching/export, explicit numerical reference and model-pack research workflows. Production model qualification and distribution qualification are recorded separately in the release evidence.
