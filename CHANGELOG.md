# Change log

## Unreleased

Step replay reads newly available quote/trade windows and retains compact source indexes ahead of disposable observations. Forward chain summaries avoid rebuilding prior quote history; late or conflicting observations retain full causal validation. Advancing shows the requested time and committed progress, and diagnostics identify the displayed decision time and offending archive files/rows. Ambiguous OI and mismatched OHLCV sources remain unavailable.

## 0.1.0

Initial implementation of the supplied QQQ historical simulation specification: isolated Python environment, source audits and causal adapters, desktop replay, simulated execution and accounting, lifecycle scenarios, transactional recovery/branching/export, explicit numerical reference and model-pack research workflows. Production model qualification and distribution qualification are recorded separately in the release evidence.
