"""Single-writer causal replay with transactional valid-prefix publication."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from itertools import groupby
import json
from pathlib import Path
import threading
from time import monotonic
from uuid import uuid4

from qqq_sim.config import AppConfig, writable_path
from qqq_sim.serialization import primitive, digest, save_json
from qqq_sim.data.models import DataError, MarketSnapshot
from qqq_sim.broker.models import BrokerError, Contract, OrderLeg, OrderDraft
from qqq_sim.broker.service import Broker
from qqq_sim.services.adapters import snapshot_quotes, broker_quote, require_trade_metadata
from qqq_sim.services.runtime_identity import runtime_identity, verify_runtime_identity
from .calendar import local_datetime


@dataclass(frozen=True)
class StateBundle:
    session_id: str
    branch_id: str
    version: int
    decision_at: datetime
    snapshot: MarketSnapshot
    account: dict
    positions: tuple
    orders: tuple
    campaigns: tuple
    history: tuple
    status: str = "OK"
    reason: dict | None = None
    synthetic: bool = False
    stock_positions: tuple = ()
    stock_orders: tuple = ()
    config: dict | None = None
    pending_benchmarks: tuple = ()


class ReplayController:
    def __init__(self, config, data, broker, store, session_id, branch_id, decision_at, version=0):
        self.config, self.data, self.broker, self.store = config, data, broker, store
        self.session_id, self.branch_id = session_id, branch_id
        self.decision_at, self.version = local_datetime(decision_at), version
        self.snapshot = data.snapshot(self.decision_at, state_version=version)
        self.status, self.reason = "OK", None
        self.cancel_requested = threading.Event()
        self._lock = threading.RLock()
        self._bound_days = set()
        self._advance_ids = {}
        self.synthetic = "fixtures" in str(config.underlying_root).lower()
        from qqq_sim.services.lifecycle import LifecycleRegistry
        self._lifecycle_registry = LifecycleRegistry(config.parameters_path)
        self._pack_hash = self._read_pack_hash()
        self._reference_hashes = self._read_reference_hashes()
        self._implementation_identity = runtime_identity()
        self._initial_replay_window = None
        self._initial_replay_day = self.decision_at.date()
        session = data.calendar.session(self.decision_at.date())
        if session.status == "OPEN" and self.decision_at < session.underlying_close:
            try:
                # Build private source indexes during cold session loading. A
                # requested skip then reuses them instead of paying this cost.
                self._initial_replay_window = data.prepare_replay_window(self.decision_at, session.underlying_close)
            except Exception:
                pass  # Future source errors are discovered at their causal slot.

    @classmethod
    def create(cls, config, at=None):
        from qqq_sim.cli import make_data
        from qqq_sim.storage.events import EventStore
        data = make_data(config)
        catalog = data.catalog.scan()
        if at is None:
            underlying = {p.key.session_date for p in catalog.partitions if p.key.data_type == "underlying"}
            options = {p.key.session_date for p in catalog.partitions if p.key.data_type == "greeks"}
            available = sorted(underlying & options or underlying)
            if not available:
                raise DataError("DATA_ROOT_UNAVAILABLE", "No underlying partitions found; configure readable archive roots")
            at = data.calendar.session(available[0]).exchange_open + timedelta(minutes=31)
        at = local_datetime(at)
        start_day,end_day=date(2012,1,1),date(2030,12,31)
        settlement_holidays=set()
        for offset in range((end_day-start_day).days+1):
            day=start_day+timedelta(days=offset)
            if day.weekday()<5 and not data.calendar.is_settlement_day(day): settlement_holidays.add(day)
        broker = Broker(Decimal(config.initial_cash), strict=config.strict, operational_model=config.operational_model,
                        settlement_holidays=settlement_holidays,
                        settlement_day_overrides={day:record.settlement_business_day for day,record in data.calendar.records.items()})
        broker.bp_cap, broker.loss_cap, broker.session_loss_cap = map(Decimal, (config.bp_cap, config.loss_cap, config.session_loss_cap))
        if config.rule_mode == "strict_historical":
            raise BrokerError("BROKER_RULE_HISTORY_MISSING", "Supply complete dated profiles; the included profile is fixed-rule counterfactual")
        session_id = uuid4().hex
        path = writable_path(Path(config.runtime_root)/"sessions"/session_id/"session.sqlite", create_parent=True)
        store = EventStore(path)
        try:
            handle = store.create_session(asdict(config), session_id=session_id)
            obj = cls(config, data, broker, store, session_id, handle["branch_id"], at, handle["version"])
            obj._bind_day(at.date())
            obj._commit("initialize:"+session_id, {"action":"INITIALIZE"})
            return obj
        except Exception:
            store.connection.close()
            raise

    @classmethod
    def restore(cls, config, session_id, branch_id=None):
        from qqq_sim.storage.events import EventStore
        from qqq_sim.cli import make_data
        path = Path(session_id)
        if path.suffix != ".sqlite":
            path = Path(config.runtime_root)/"sessions"/session_id/"session.sqlite"
        from qqq_sim.services.session_upgrade import prepare_session_restore
        path, branch_id = prepare_session_restore(writable_path(path), branch_id)
        store = EventStore(path)
        try:
            return cls._restore_from_store(store,branch_id)
        except Exception:
            store.connection.close()
            raise

    @classmethod
    def _restore_from_store(cls,store,branch_id=None):
        from qqq_sim.cli import make_data
        rows = store.sessions()
        if len(rows) != 1:
            raise BrokerError("SESSION_IDENTITY_AMBIGUOUS")
        session = rows[0]
        saved_config = AppConfig(**json.loads(session["config_json"])).validate()
        branch_id = branch_id or session["active_branch"]
        restored = store.restore(branch_id)
        state = restored["state"]
        if not {'broker','decision_at'}<=state.keys(): raise BrokerError('SESSION_INITIALIZATION_INCOMPLETE')
        implementation_identity = verify_runtime_identity(state.get("implementation_identity"))
        if "config" in state:
            saved_config=AppConfig(**state["config"]).validate()
        obj = cls(saved_config, make_data(saved_config), Broker.from_dict(state["broker"]), store,
                  session["id"], branch_id, datetime.fromisoformat(state["decision_at"]), restored["version"])
        obj._implementation_identity = implementation_identity
        saved_lifecycle_hash=state.get("lifecycle_source_hash")
        if saved_lifecycle_hash and saved_lifecycle_hash!=obj._lifecycle_registry.source_hash:
            raise BrokerError("LIFECYCLE_SOURCE_REVISION_CHANGED")
        if state.get("pack_hash") != obj._pack_hash:
            raise BrokerError("PACK_REVISION_CHANGED")
        if state.get('reference_input_hashes',obj._reference_hashes)!=obj._reference_hashes:
            raise BrokerError('CALENDAR_OR_NORMALIZATION_REVISION_CHANGED')
        obj._restore_revision(state.get("data_revision", {}),verify_catalog=True)
        obj.data.catalog.verify_revision(deep=True)
        obj.snapshot = obj.data.snapshot(obj.decision_at, state_version=obj.version)
        obj._advance_ids = state.get("advance_ids", {})
        obj.status,obj.reason=state.get('status','OK'),state.get('reason')
        return obj

    @staticmethod
    def _verify_catalog_membership(data,revision):
        if revision.get('days'):
            report=data.catalog.scan()
            days={date.fromisoformat(value) for value in revision['days']}
            current={str(partition.path) for partition in report.partitions if partition.key.session_date in days
                     and partition.key.data_type in ('underlying','greeks','oi_before','ohlcv','trades')
                     and partition.key not in data.catalog._duplicates}
            frozen={str(item['path']) for item in revision.get('sources',[])}
            if current!=frozen:
                raise DataError('DATA_REVISION_CHANGED','Bound session partition membership changed while closed',
                                added_paths=sorted(current-frozen),removed_paths=sorted(frozen-current))

    def _restore_revision(self, revision,verify_catalog=False):
        if verify_catalog:
            self._verify_catalog_membership(self.data,revision)
        self.data.catalog._bound = {item["path"]: (item["size"],item["mtime_ns"],item["sha256"]) for item in revision.get("sources", [])}
        self._bound_days = {date.fromisoformat(s) for s in revision.get("days", [])}

    def _revision(self):
        records = [{"path": path, "size": v[0], "mtime_ns": v[1], "sha256": v[2]} for path,v in sorted(self.data.catalog._bound.items())]
        return {"sources": records, "days": sorted(d.isoformat() for d in self._bound_days), "hash": digest(records)}

    def _bind_day(self, day):
        if day in self._bound_days:
            self.data.catalog.verify_revision()
            return
        if self.data.catalog.report is None:
            self.data.catalog.scan()
        keys = list({p.key for p in self.data.catalog.report.partitions if p.key.session_date == day
                     and p.key.data_type in ("underlying","greeks","oi_before","ohlcv","trades")
                     and p.key not in self.data.catalog._duplicates})
        self.data.catalog.bind_revision(keys=keys)
        self._bound_days.add(day)

    def _state(self):
        return {"schema_version": 1, "decision_at": self.decision_at.isoformat(), "broker": self.broker.to_dict(),
                "data_revision": self._revision(), "advance_ids": dict(self._advance_ids),
                "status":self.status,"reason":self.reason,
                "lifecycle_source_hash": self._lifecycle_registry.source_hash,
                "config":asdict(self.config),"pack_hash":self._pack_hash,
                "reference_input_hashes":self._reference_hashes,
                "implementation_identity":primitive(self._implementation_identity),
                "barrier": "BAR_END_DECISION_BARRIER_V1", "config_hash": digest(self.config)}

    def _read_pack_hash(self):
        if not self.config.pack_path:
            return None
        from qqq_sim.data.catalog import file_hash
        path=Path(self.config.pack_path)/"manifest.json"
        if not path.is_file():
            return "PACK_MISSING"
        return file_hash(path)

    def _read_reference_hashes(self):
        from qqq_sim.data.catalog import file_hash
        result={}
        for field in ('calendar_path','normalization_path'):
            value=getattr(self.config,field,None)
            if value:
                path=Path(value)
                if not path.is_file(): raise BrokerError('INPUT_REFERENCE_MISSING',field=field,path=str(path))
                result[field]=file_hash(path)
        return result

    def _verify_reference_inputs(self):
        if self._read_reference_hashes()!=self._reference_hashes:
            raise BrokerError('CALENDAR_OR_NORMALIZATION_REVISION_CHANGED')

    def select_inputs(self, *, pack_path=None, parameters_path=None):
        """Explicit input changes fork a research branch and retain the prior basis."""
        values={key:value for key,value in (("pack_path",pack_path),("parameters_path",parameters_path)) if value is not None}
        new_config=replace(self.config,**values).validate()
        from qqq_sim.services.lifecycle import LifecycleRegistry
        registry=LifecycleRegistry(new_config.parameters_path)
        before=self._state(); old_branch,old_version=self.branch_id,self.version
        old_registry=self._lifecycle_registry
        handle=self.store.branch(self.branch_id,asdict(new_config),"Explicit analysis-input selection; original basis retained")
        self.branch_id,self.version=handle["branch_id"],handle["version"]
        self.config,self._lifecycle_registry=new_config,registry
        self._pack_hash=self._read_pack_hash()
        try:
            self._commit("inputs:"+uuid4().hex,{"action":"SELECT_INPUTS",**values})
        except Exception:
            self.config=AppConfig(**before['config']).validate()
            self._lifecycle_registry=old_registry
            self._pack_hash=before['pack_hash']
            self.branch_id,self.version=old_branch,old_version
            self._rollback(before)
            try:
                self.store.connection.execute('UPDATE sessions SET active_branch=? WHERE id=?',(old_branch,self.session_id))
            except Exception:
                pass  # The new empty branch still contains the prior valid state.
            raise
        return self.bundle()

    def _commit(self, command_id, payload):
        state = self._state()
        try:
            result = self.store.transact(self.branch_id, command_id, self.version, payload,
                                         self.broker.ledger.events, {"status":self.status}, state)
        except Exception as exc:
            # A publication/transport failure after COMMIT is not a rollback.
            # A durable idempotency record is authoritative even if its original
            # acknowledgment was lost. Before-COMMIT failure keeps the old state.
            try:
                row=self.store.connection.execute('SELECT result_json FROM commands WHERE branch_id=? AND command_id=?',
                                                   (self.branch_id,command_id)).fetchone()
                recovered=self.store.restore(self.branch_id) if row else None
            except Exception:
                recovered=None; row=None
            if row is None or recovered is None:
                raise BrokerError('SAVE_FAILED',str(exc),command_id=command_id) from exc
            if recovered['state']!=primitive(state):
                raise BrokerError('SAVED_COMMAND_STATE_CONFLICT',command_id=command_id) from exc
            result=json.loads(row['result_json'])
        self.version = result["version"]
        self.snapshot = replace(self.snapshot, state_version=self.version)
        return state

    def _expect(self, version):
        self._verify_reference_inputs()
        if version is not None and version != self.version:
            raise BrokerError("STATE_VERSION_CONFLICT", expected=version, actual=self.version)

    def _rollback(self, state):
        self.broker = Broker.from_dict(state["broker"])
        self.decision_at = datetime.fromisoformat(state["decision_at"])
        self._advance_ids=dict(state.get("advance_ids",{}))
        self._implementation_identity = verify_runtime_identity(state.get("implementation_identity"))
        self._restore_revision(state.get("data_revision",{}))

    def bundle(self):
        quotes = snapshot_quotes(self.snapshot, self.data.calendar)
        from qqq_sim.services.adapters import broker_contract
        current_keys={broker_contract(entry.contract,self.data.calendar).key for entry in self.snapshot.quotes if entry.status in ('OK','CURRENT')}
        marks = {key:q.midpoint for key,q in quotes.items() if q.valid and key in current_keys}
        account = primitive(self.broker.account_summary(marks, self.snapshot.spot))
        positions = tuple({"contract":primitive(c), "contract_key":c.key, "quantity":q,
                           "campaign_id":campaign_id,
                           "mark":str(marks[c.key]) if c.key in marks else None}
                          for campaign_id in sorted(self.broker.ledger.campaigns)
                          for c,q in self.broker.ledger.positions(campaign_id))
        return StateBundle(self.session_id, self.branch_id, self.version, self.decision_at, self.snapshot, account,
                           positions, tuple(primitive(o) for o in sorted(self.broker.orders.values(),key=lambda o:(o.submitted_at,o.submitted_sequence,o.order_id))),
                           tuple(primitive(self.broker.ledger.campaigns[cid]) for cid in sorted(self.broker.ledger.campaigns)),
                           tuple(self.broker.ledger.events), self.status, self.reason, self.synthetic,
                           tuple({"campaign_id":campaign_id,"symbol":"QQQ","quantity":self.broker.ledger.shares(campaign_id),
                                  "basis":str(sum((lot.basis for lot in self.broker.ledger.stock_lots if lot.campaign_id==campaign_id),Decimal(0))),
                                  "mark":self.snapshot.spot,"mark_basis":"HISTORICAL_UNDERLYING_CLOSE"}
                                 for campaign_id in sorted(self.broker.ledger.campaigns) if self.broker.ledger.shares(campaign_id)),
                           tuple(primitive(order) for order in sorted(self.broker.stock_orders.values(),key=lambda o:(o.submitted_at,o.submitted_sequence,o.order_id))),asdict(self.config),
                           tuple({'campaign_id':cid,'benchmark_id':campaign.benchmark_id} for cid,campaign in sorted(self.broker.ledger.campaigns.items())
                                 if campaign.benchmark_status=='PENDING' and campaign.benchmark_inputs is not None))

    def _freeze_new_benchmarks(self,at,snapshot=None):
        from qqq_sim.services.benchmarks import freeze_initial_benchmark
        for cid,campaign in self.broker.ledger.campaigns.items():
            if campaign.target is not None and campaign.initialized_at==at and campaign.benchmark_id is None:
                freeze_initial_benchmark(self,cid,at,snapshot)

    def prepare_benchmark(self,campaign_id):
        from qqq_sim.services.benchmarks import prepare_initial_benchmark
        result=prepare_initial_benchmark(self,campaign_id)
        return result if 'identity' in result else self.bundle()

    def record_benchmark(self,campaign_id,benchmark_id,result,expected_version=None,command_id=None):
        from qqq_sim.services.benchmarks import apply_benchmark_result
        with self._lock:
            campaign=self.broker.ledger.campaigns.get(campaign_id)
            if campaign and campaign.benchmark_id==benchmark_id and campaign.benchmark_result==primitive(result):
                return self.bundle()
            self._expect(expected_version)
            before=self._state()
            try:
                apply_benchmark_result(self.broker.ledger,campaign_id,benchmark_id,result,self.decision_at)
                self._commit(command_id or 'benchmark:'+benchmark_id,{'action':'RECORD_INITIAL_BENCHMARK','campaign_id':campaign_id,'benchmark_id':benchmark_id})
                return self.bundle()
            except Exception:
                self._rollback(before)
                raise

    def submit_stock_reduction(self,campaign_id,quantity,minimum_cash_per_share,expected_version=None,command_id=None,max_total_fees=Decimal(50)):
        with self._lock:
            if command_id and command_id in self.broker.commands:
                return self.bundle()
            self._expect(expected_version)
            command_id=command_id or uuid4().hex
            before=self._state()
            try:
                self.broker.submit_stock_reduction(campaign_id,int(quantity),Decimal(str(minimum_cash_per_share)),
                                                   self.decision_at,command_id,Decimal(str(max_total_fees)))
                self._commit(command_id,{"action":"STOCK_REDUCE","campaign_id":campaign_id,"quantity":quantity,
                                         "minimum_cash_per_share":str(minimum_cash_per_share),"max_total_fees":str(max_total_fees)})
                return self.bundle()
            except Exception:
                self._rollback(before)
                raise

    def submit(self, draft, expected_version=None, command_id=None, preview=False):
        with self._lock:
            if command_id and command_id in self.broker.commands and not preview:
                return self.bundle()
            self._expect(expected_version)
            require_trade_metadata(draft, self.snapshot, self.data.calendar)
            before = self._state()
            target = Broker.from_dict(self.broker.to_dict()) if preview else self.broker
            command_id = command_id or uuid4().hex
            try:
                order = target.submit(draft, self.decision_at, snapshot_quotes(self.snapshot,self.data.calendar), command_id)
                if preview:
                    return {"status":"OK", "order":primitive(order), "account_after_reservation":primitive(target.account_summary()),
                            "label":"SIMULATED CONDITIONAL LIMIT", "real_orders_sent":False}
                self._commit(command_id, {"action":"SUBMIT", "draft":primitive(draft)})
                self.status, self.reason = "OK", None
                return self.bundle()
            except Exception:
                if not preview:
                    self._rollback(before)
                raise

    def cancel(self, order_id, expected_version=None, command_id=None):
        with self._lock:
            if command_id and self.store.connection.execute("SELECT 1 FROM commands WHERE branch_id=? AND command_id=?",(self.branch_id,command_id)).fetchone():
                return self.bundle()
            self._expect(expected_version)
            before = self._state()
            try:
                self.broker.cancel(order_id, self.decision_at, command_id=command_id)
                self._freeze_new_benchmarks(self.decision_at,self.snapshot)
                self._commit(command_id or uuid4().hex, {"action":"CANCEL", "order_id":order_id})
                return self.bundle()
            except Exception:
                self._rollback(before)
                raise

    def _closed_period(self, next_instant):
        previous = self.data.calendar.session(self.decision_at.date())
        if next_instant.date() > self.decision_at.date():
            # Closed-period records are consumed by their knowledge clock before
            # the next market event. Missing expiry evidence is a typed stop.
            self._process_lifecycle_until(self.data.calendar.session(next_instant.date()).exchange_open)
            if self.broker.ledger.shares():
                financing=[record for record in self._lifecycle_registry.records if record.kind=="FINANCING"
                           and record.available_at<=next_instant and record.economic_at.date()>=self.decision_at.date()]
                if not financing:
                    raise BrokerError("LIFECYCLE_REFERENCE_REQUIRED", "Stock financing/dividend inputs are required across a closed period")
            self.broker.expire_orders(previous.underlying_close, previous.underlying_close)
            self._freeze_new_benchmarks(previous.underlying_close)

    def _process_lifecycle_until(self, instant):
        self._lifecycle_registry.process_until(self,instant)

    def advance(self, target=None, minutes=1, expected_version=None, command_id=None, progress=None):
        with self._lock:
            command_id = command_id or uuid4().hex
            if command_id in self._advance_ids:
                return self.bundle()
            self._expect(expected_version)
            if minutes < 0:
                raise BrokerError("REWIND_REQUIRES_BRANCH")
            target = local_datetime(target) if target else self.data.calendar.advance_trading_minutes(self.decision_at, minutes)
            if target < self.decision_at:
                raise BrokerError("REWIND_REQUIRES_BRANCH")
            self.cancel_requested.clear()
            self.status, self.reason = "OK", None
            started_at = monotonic()
            last_progress = started_at
            completed_slots = 0
            def report_progress(force=False, status="RUNNING"):
                nonlocal last_progress
                if progress is None:
                    return
                now = monotonic()
                if not force and now-last_progress < .25:
                    return
                last_progress = now
                try:
                    progress({"command_id":command_id,"decision_at":self.decision_at,"target":target,
                              "completed_slots":completed_slots,"state_version":self.version,
                              "elapsed_seconds":now-started_at,"status":status})
                except Exception:
                    pass  # Display notifications cannot interrupt a durable replay command.
            report_progress(force=True)
            fast = (getattr(self, "_fast_replay_enabled", True)
                    and target > self.data.calendar.advance_trading_minutes(self.decision_at, 1))
            if fast:
                from .fast_forward import run_fast_advance
                def batch_progress(count):
                    nonlocal completed_slots
                    completed_slots = count
                    report_progress()
                completed_slots = run_fast_advance(self, target, command_id, batch_progress)
            while not fast and self.decision_at < target:
                if self.cancel_requested.is_set():
                    self.status, self.reason = "CANCELED", {"reason":"USER_CANCELED"}
                    break
                next_slot = self.data.calendar.advance_trading_minutes(self.decision_at, 1)
                next_instant = min(target, next_slot)
                before = self._state()
                previous_snapshot = self.snapshot
                try:
                    self._closed_period(next_instant)
                    self._bind_day(next_instant.date())
                    required=[contract for contract,q in self.broker.ledger.positions()]
                    required.extend(leg.contract for order in self.broker.orders.values() if order.status in ("WORKING","PARTIALLY_FILLED","DATA_BLOCKED") for leg in order.draft.legs)
                    gap = self.data.coverage_gap(self.decision_at,next_instant,required_contracts=required)
                    if gap:
                        raise DataError("DATA_GAP_BEFORE_TARGET", "Missing scheduled minute", at=gap[0], cause=gap[1])
                    snapshot = self.data.snapshot(next_instant, state_version=self.version+1)
                    for at, grouped in groupby(self.data.events_between(self.decision_at,next_instant), key=lambda e:e.event_at):
                        self._process_lifecycle_until(at)
                        events = tuple(grouped)
                        quotes = [broker_quote(event.payload,self.data.calendar) for event in events if event.kind == "quote"]
                        if quotes:
                            self.broker.on_quotes(quotes, at)
                            self._freeze_new_benchmarks(at)
                    self._process_lifecycle_until(next_instant)
                    self.broker.settle_due(next_instant)
                    session = self.data.calendar.session(next_instant.date())
                    if next_instant >= session.underlying_close:
                        self.broker.expire_orders(next_instant, session.underlying_close)
                        self._freeze_new_benchmarks(next_instant,snapshot)
                    self.decision_at, self.snapshot = next_instant, snapshot
                    # Stable slot identity makes chunking independent of the toolbar step size.
                    self._commit(f"slot:{next_instant.isoformat()}", {"action":"ADVANCE_SLOT", "at":next_instant.isoformat()})
                    completed_slots += 1
                    report_progress()
                except Exception as exc:
                    self._rollback(before)
                    self.snapshot = previous_snapshot
                    self.status = "PARTIAL"
                    from qqq_sim.cli import error_record
                    self.reason = error_record(exc)
                    break
            before_command=self._state()
            self._advance_ids[command_id] = self.decision_at.isoformat()
            # Save idempotency even for a no-op/partial command without altering economic events.
            try:
                self._commit("advance-command:"+command_id, {"action":"ADVANCE", "requested":target.isoformat(), "actual":self.decision_at.isoformat(), "status":self.status})
            except BrokerError as error:
                self._rollback(before_command)
                self.status='PARTIAL'; self.reason=error.as_dict()
            report_progress(force=True, status=self.status)
            return self.bundle()

    def branch(self, sequence=None, reason="User research branch"):
        source_branch = self.branch_id
        if sequence is not None:
            # A copied upgrade retains the original prefix in ancestor branches.
            # Resolve its sequence there before constructing a new current branch.
            while True:
                source = self.store.connection.execute("SELECT parent_id,fork_sequence FROM branches WHERE id=?",
                                                       (source_branch,)).fetchone()
                if not source or not source['parent_id'] or sequence >= source['fork_sequence']:
                    break
                source_branch = source['parent_id']
        frozen=self.store.restore(source_branch,through_sequence=sequence)
        frozen_state=frozen['state']
        from qqq_sim.services.session_upgrade import is_upgradeable_identity, upgrade_legacy_branch
        legacy = is_upgradeable_identity(frozen_state.get("implementation_identity"))
        if not legacy:
            verify_runtime_identity(frozen_state.get("implementation_identity"))
        selected_config=AppConfig(**frozen_state.get('config',asdict(self.config))).validate()
        from qqq_sim.services.lifecycle import LifecycleRegistry
        registry=LifecycleRegistry(selected_config.parameters_path)
        if frozen_state.get('lifecycle_source_hash') and registry.source_hash!=frozen_state['lifecycle_source_hash']:
            raise BrokerError('LIFECYCLE_SOURCE_REVISION_CHANGED')
        previous_config=self.config
        self.config=selected_config
        try:
            if frozen_state.get('pack_hash')!=self._read_pack_hash(): raise BrokerError('PACK_REVISION_CHANGED')
            if frozen_state.get('reference_input_hashes',self._read_reference_hashes())!=self._read_reference_hashes():
                raise BrokerError('CALENDAR_OR_NORMALIZATION_REVISION_CHANGED')
        finally:
            self.config=previous_config
        candidate_data=self.data
        fields=('underlying_root','options_root','normalization_path','calendar_path')
        if any(getattr(previous_config,key)!=getattr(selected_config,key) for key in fields):
            from qqq_sim.cli import make_data
            candidate_data=make_data(selected_config)
        self._verify_catalog_membership(candidate_data,frozen_state.get('data_revision',{}))
        if legacy:
            restored = upgrade_legacy_branch(self.store, source_branch, sequence, reason)
            handle = {'branch_id': restored['branch_id']}
        else:
            handle = self.store.branch(source_branch, asdict(selected_config), reason, fork_sequence=sequence)
            restored = self.store.restore(handle["branch_id"])
        state = restored["state"]
        self.config=selected_config
        self._lifecycle_registry=registry
        self._pack_hash=state.get('pack_hash')
        self._reference_hashes=state.get('reference_input_hashes',self._read_reference_hashes())
        self.data=candidate_data
        self._initial_replay_window = None
        self._initial_replay_day = datetime.fromisoformat(state['decision_at']).date()
        self.branch_id, self.version = handle["branch_id"], restored["version"]
        self._rollback(state)
        self._restore_revision(state.get("data_revision",{}))
        self.snapshot = self.data.snapshot(self.decision_at,state_version=self.version)
        self.status, self.reason = "OK", None
        return self.bundle()

    def record_analysis(self,identity,result,request_hash=None,archive_path=None):
        from qqq_sim.services.analytic_records import record_analysis,record_archived_analysis
        from qqq_sim.services.analysis import AnalysisIdentity
        identity=identity if isinstance(identity,AnalysisIdentity) else AnalysisIdentity(**identity)
        with self._lock:
            # A restored backup may retain the same session/branch identifiers
            # while owning a different database and request history. Archive
            # into the exact store that froze the original request.
            different_archive=(archive_path is not None and writable_path(archive_path)!=self.store.path.resolve())
            if different_archive or identity.session_id!=self.session_id:
                if archive_path is None:
                    raise BrokerError('ANALYSIS_ARCHIVE_PATH_REQUIRED')
                return record_archived_analysis(archive_path,identity,result,request_hash)
            return record_analysis(self,identity,result,request_hash)

    def export_record(self):
        bundle = self.bundle()
        return {"schema_version":1, "status":self.status, "reason":self.reason,
                "session_id":self.session_id,"branch_id":self.branch_id,"state_version":self.version,
                "decision_at":self.decision_at,"bar_label":self.snapshot.candles[-1].label if self.snapshot.candles else None,
                "snapshot":self.snapshot,"account":bundle.account,"state":self._state(),
                "analyses":self.store.analysis_records(session_id=self.session_id),
                "implementation_identity":primitive(self._implementation_identity),
                "config":self.config,"real_orders_sent":False,"broker_parity":"UNVERIFIED",
                "calendar_basis":"STANDARD_US_CALENDAR_SCENARIO_V1", "synthetic":self.synthetic}

    def export(self, destination):
        path = save_json(destination,self.export_record())
        text_path = path.with_suffix(".txt")
        from qqq_sim.serialization import dumps
        text_path.write_text("QQQ HISTORICAL SIMULATOR — FROZEN SESSION\n"+dumps(self.export_record(),pretty=True),encoding="utf-8")
        import csv
        csv_path = path.with_suffix(".ledger.csv")
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=("sequence","event_id","event_type","at","event_hash"))
            writer.writeheader()
            for event in self.broker.ledger.events:
                writer.writerow({key:event.get(key) for key in writer.fieldnames})
        return path

    def dispatch(self, command):
        action = command["action"].lower()
        kwargs={"expected_version":command.get("expected_version",self.version),"command_id":command.get("command_id")}
        if action=="advance":
            return self.advance(datetime.fromisoformat(command["target"]) if command.get("target") else None,command.get("minutes",1),**kwargs)
        if action=="cancel":
            return self.cancel(command["order_id"],**kwargs)
        if action=="submit":
            raw=dict(command["draft"])
            raw["legs"]=tuple(OrderLeg(Contract.from_dict(x["contract"]),x["quantity"]) for x in raw["legs"])
            for key in ("deadline","hard_cutoff"):
                raw[key]=datetime.fromisoformat(raw[key])
            return self.submit(OrderDraft(**raw),**kwargs)
        if action=="stock_reduce":
            return self.submit_stock_reduction(command['campaign_id'],command['quantity'],command['minimum_cash_per_share'],
                                                max_total_fees=command.get('max_total_fees',50),**kwargs)
        if action=='prepare_benchmark':
            return self.prepare_benchmark(command['campaign_id'])
        if action=='record_benchmark':
            return self.record_benchmark(command['campaign_id'],command['benchmark_id'],command['result'],**kwargs)
        if action=='record_analysis':
            return self.record_analysis(command['identity'],command['result'],command.get('request_hash'),command.get('archive_path'))
        if action=="branch":
            return self.branch(command.get("sequence"),command.get("reason","Headless research branch"))
        if action=="export":
            return self.export(command["output"])
        raise BrokerError("UNKNOWN_COMMAND",action)

    def close(self):
        self.store.close()
