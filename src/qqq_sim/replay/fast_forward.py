"""Batched replay, retaining the original minute and quote decision barriers.

Only the display snapshot and physical saves are coalesced. Execution still
observes the same quote groups, and a failed minute never becomes visible.
"""
from copy import deepcopy
from itertools import groupby
from time import monotonic

from qqq_sim.broker.models import BrokerError
from qqq_sim.data.models import DataError
from qqq_sim.services.adapters import broker_quote


LIVE_ORDERS = ("WORKING", "PARTIALLY_FILLED", "DATA_BLOCKED")
MAX_BATCH_SLOTS = 64
BATCH_SECONDS = .25


class _OpeningBoundaryWindow:
    """Validate an unopened session without indexing its future execution data."""
    fast_quotes = True
    requires_snapshot_validation = False
    opening_boundary = True

    def __init__(self, data):
        self.data = data
        self.cancel = None

    def coverage_gap(self, start, end, required_contracts=None):
        return self.data.coverage_gap(start, end, required_contracts=required_contracts)

    def events_between(self, start, end, required_contracts=None):
        # The prior close was already replayed and its revisions remain bound.
        # At the new opening barrier even raw phase30 opening quotes are still
        # exclusive. Retain canonical new-session schema/source validation.
        return self.data.events_between(end, end, required_contracts=required_contracts)

    def terminal_quote_events(self, start, end):
        return ()


def _lifecycle_due(c, end):
    ledger = c.broker.ledger
    for record in c._lifecycle_registry.records:
        if record.available_at > end:
            break
        campaign = record.fields.get("campaign_id")
        if ((not campaign or campaign in ledger.campaigns)
                and "lifecycle-input:" + record.event_id not in ledger.event_ids):
            return True
    for contract, quantity in ledger.positions():
        session = c.data.calendar.session(contract.expiry)
        if (contract.fixing_at or session.fixing_at or session.underlying_close) <= end:
            return True
    return False


def run_fast_advance(c, target, command_id, report):
    """Return completed logical slots; ``report`` runs after durable commits.

    Snapshot/SQL failures restore the last durable batch. Data/execution failures
    discard just their incomplete minute and then publish the validated prefix.
    """
    durable = c._state()
    durable_snapshot = c.snapshot
    durable_version = c.version
    completed = durable_completed = batch_slots = 0
    batch_started = monotonic()
    batch_start = c.decision_at
    hydrated_at = c.decision_at
    window = getattr(c, "_initial_replay_window", None)
    if getattr(c, "_initial_replay_day", None) != c.decision_at.date():
        window = None
    window_day = c.decision_at.date() if window is not None else None
    if window is not None:
        window.cancel = c.cancel_requested.is_set

    def prepared(end):
        nonlocal window, window_day
        if window_day != end.date():
            session = c.data.calendar.session(end.date())
            horizon = max(end, min(target, session.underlying_close))
            try:
                previous = c.data.calendar.session(c.decision_at.date())
                if (end.date() > c.decision_at.date() and end <= session.exchange_open
                        and (previous.status != "OPEN" or c.decision_at >= previous.underlying_close)
                        and (previous.status != "OPEN" or getattr(window, "fast_quotes", False))):
                    window = _OpeningBoundaryWindow(c.data)
                else:
                    window = c.data.prepare_replay_window(c.decision_at, horizon,
                                                         cancel=c.cancel_requested.is_set)
            except InterruptedError:
                raise
            except Exception:
                # The ordinary reader discovers a malformed future observation
                # at its original minute, rather than failing at preparation.
                window = c.data
            window_day = end.date()
            c._initial_replay_window = None if getattr(window, "opening_boundary", False) else window
            c._initial_replay_day = window_day
        return window

    def hydrate(end):
        nonlocal hydrated_at
        if end <= hydrated_at:
            return
        reader = window if window is not None else c.data
        observer = getattr(reader, "cancel", None)
        if hasattr(reader, "cancel"):
            reader.cancel = None
        try:
            if hasattr(reader, "terminal_quote_events"):
                events = reader.terminal_quote_events(hydrated_at, end)
            else:
                events = tuple(e for e in c.data.events_between(hydrated_at, end) if e.kind == "quote")
        finally:
            if hasattr(reader, "cancel"):
                reader.cancel = observer
        # Catch-up is a cache merge, never an execution opportunity.
        for event in events:
            if event.kind == "quote":
                quote = broker_quote(event.payload, c.data.calendar)
                c.broker.quotes[quote.contract.key] = quote
        hydrated_at = end

    def flush():
        nonlocal durable, durable_snapshot, durable_version, durable_completed
        nonlocal batch_slots, batch_start, batch_started
        if not batch_slots:
            return
        hydrate(c.decision_at)
        state = c._commit(f"batch:{command_id}:{batch_start.isoformat()}:{c.decision_at.isoformat()}",
                  {"action": "ADVANCE_BATCH", "start": batch_start.isoformat(),
                   "at": c.decision_at.isoformat(), "completed_slots": batch_slots,
                   "requested": target.isoformat()})
        durable = state if state is not None else c._state()
        durable_snapshot, durable_version = c.snapshot, c.version
        durable_completed = completed
        batch_start, batch_slots, batch_started = c.decision_at, 0, monotonic()
        report(completed)

    def failed_save(error):
        c._rollback(durable)
        c.snapshot, c.version = durable_snapshot, durable_version
        if c.snapshot.asof != c.decision_at:
            c.snapshot = c.data.snapshot(c.decision_at, state_version=c.version)
        c.status, c.reason = "PARTIAL", error.as_dict()

    while c.decision_at < target:
        if c.cancel_requested.is_set():
            c.status, c.reason = "CANCELED", {"reason": "USER_CANCELED"}
            break
        start = c.decision_at
        end = min(target, c.data.calendar.advance_trading_minutes(start, 1))
        old_bound = dict(c.data.catalog._bound)
        old_days = set(c._bound_days)
        old_hydrated = hydrated_at
        old_snapshot = c.snapshot
        old_broker = None
        minute_completed = False
        try:
            crossing = start.date() != end.date()
            if crossing:
                # Finish the old session before its overnight lifecycle runs.
                flush()
                old_broker = deepcopy(c.broker, {id(c.broker.quotes): dict(c.broker.quotes)})
                hydrate(start)
                c._closed_period(end)
            reader = prepared(end)
            c._bind_day(end.date())
            orders = [o for o in c.broker.orders.values() if o.status in LIVE_ORDERS]
            required = [contract for contract, q in c.broker.ledger.positions()]
            required.extend(leg.contract for order in orders for leg in order.draft.legs)
            gap = reader.coverage_gap(start, end, required_contracts=required)
            if gap:
                raise DataError("DATA_GAP_BEFORE_TARGET", "Missing scheduled minute", at=gap[0], cause=gap[1])
            lifecycle = crossing or _lifecycle_due(c, end)
            initializing = any(campaign.target is None and campaign.status == "INITIALIZING"
                               for campaign in c.broker.ledger.campaigns.values())
            session = c.data.calendar.session(end.date())
            settling = any(not o.settled and o.due_date <= end.date()
                           for o in c.broker.ledger.obligations.values())
            full = (lifecycle or end >= session.underlying_close
                    or not getattr(reader, "fast_quotes", False))
            if old_broker is None and (orders or full or settling):
                # Immutable quote objects can be shared by a rollback clone.
                old_broker = deepcopy(c.broker, {id(c.broker.quotes): dict(c.broker.quotes)})
            if full:
                hydrate(start)
            if not crossing:
                c._closed_period(end)
            validate_snapshot = (not getattr(reader, "fast_quotes", False)
                                 or getattr(reader, "requires_snapshot_validation", False))
            snapshot = (c.data.snapshot(end, state_version=c.version + 1)
                        if end == target or validate_snapshot else None)
            if orders or full:
                selection = None if full else tuple(leg.contract for order in orders for leg in order.draft.legs)
                for at, grouped in groupby(reader.events_between(start, end, required_contracts=selection),
                                           key=lambda e: e.event_at):
                    c._process_lifecycle_until(at)
                    quotes = [broker_quote(event.payload, c.data.calendar)
                              for event in grouped if event.kind == "quote"]
                    if quotes:
                        c.broker.on_quotes(quotes, at)
                        initialized = initializing and any(
                            campaign.initialized_at == at and campaign.benchmark_id is None
                            for campaign in c.broker.ledger.campaigns.values())
                        if initialized and not full:
                            # Most working orders never fill. Read only their
                            # legs until a fill actually needs the full frozen
                            # initialization context. Re-execute that one minute
                            # from its private rollback clone with complete groups.
                            c.broker = old_broker
                            old_broker = deepcopy(c.broker, {id(c.broker.quotes): dict(c.broker.quotes)})
                            hydrated_at = old_hydrated
                            hydrate(start)
                            for full_at, full_group in groupby(reader.events_between(start, end), key=lambda e: e.event_at):
                                c._process_lifecycle_until(full_at)
                                full_quotes = [broker_quote(event.payload, c.data.calendar)
                                               for event in full_group if event.kind == "quote"]
                                if full_quotes:
                                    c.broker.on_quotes(full_quotes, full_at)
                                    c._freeze_new_benchmarks(full_at)
                            full = True
                            break
                        c._freeze_new_benchmarks(at)
                if full:
                    hydrated_at = end
            c._process_lifecycle_until(end)
            c.broker.settle_due(end)
            if end >= session.underlying_close:
                c.broker.expire_orders(end, session.underlying_close)
                c._freeze_new_benchmarks(end, snapshot)
            c.decision_at = end
            if snapshot is not None:
                c.snapshot = snapshot
            completed += 1
            batch_slots += 1
            minute_completed = True
            if (batch_slots >= MAX_BATCH_SLOTS or monotonic() - batch_started >= BATCH_SECONDS
                    or crossing or end >= session.underlying_close or end == target):
                flush()
        except InterruptedError:
            if minute_completed:
                completed -= 1
                batch_slots -= 1
            if old_broker is not None:
                c.broker = old_broker
            c.decision_at, c.snapshot, hydrated_at = start, old_snapshot, old_hydrated
            c.data.catalog._bound, c._bound_days = old_bound, old_days
            c.status, c.reason = "CANCELED", {"reason": "USER_CANCELED"}
            break
        except Exception as error:
            if isinstance(error, BrokerError) and error.code in ("SAVE_FAILED", "SAVED_COMMAND_STATE_CONFLICT"):
                failed_save(error)
                return durable_completed
            if minute_completed:
                completed -= 1
                batch_slots -= 1
            if old_broker is not None:
                c.broker = old_broker
            c.decision_at, c.snapshot, hydrated_at = start, old_snapshot, old_hydrated
            c.data.catalog._bound, c._bound_days = old_bound, old_days
            from qqq_sim.cli import error_record
            c.status, c.reason = "PARTIAL", error_record(error)
            break
    if c.cancel_requested.is_set() and c.status == "OK":
        c.status, c.reason = "CANCELED", {"reason": "USER_CANCELED"}
    try:
        flush()
        # Intermediate saves carry account state without constructing a chain.
        # Publish a matching display snapshot exactly once on a stopped command.
        if c.snapshot.asof != c.decision_at:
            c.snapshot = c.data.snapshot(c.decision_at, state_version=c.version)
    except Exception as error:
        if not isinstance(error, BrokerError):
            error = BrokerError("SAVE_FAILED", str(error))
        failed_save(error)
        return durable_completed
    return completed
