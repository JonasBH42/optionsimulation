"""Freeze canonical HOLD holdings separately from an exact proposed trade."""
from dataclasses import replace
from decimal import Decimal as D
import json
from pathlib import Path
from qqq_sim.broker.models import BrokerError,OrderLeg
from qqq_sim.broker.service import Broker
from qqq_sim.broker.fees import FeeEngine
from qqq_sim.broker.margin import MarginEngine
from qqq_sim.broker.risk_checks import certify_campaign
from qqq_sim.serialization import digest,primitive
from qqq_sim.services.analysis import AnalysisIdentity
from qqq_sim.services.adapters import risk_job,broker_contract,snapshot_quotes,parameters_for_expiry,risk_change_legs,probability_snapshot_quality


def cached_analysis_pack(controller):
    """Verify artifact bytes on load, then reuse this immutable pack revision."""
    from qqq_sim.risk.contracts import RiskError
    from qqq_sim.risk.pack_loader import ModelPack
    c = controller
    if not c.config.pack_path:
        return None
    if c._read_pack_hash() != c._pack_hash:
        raise RiskError("PACK_REVISION_CHANGED")
    identity = (str(c.config.pack_path), c._pack_hash)
    cached = getattr(c, "_analysis_pack_cache", None)
    if cached is None or cached[0] != identity:
        pack = ModelPack.load(c.config.pack_path)
        tracked = tuple((path, path.stat().st_size, path.stat().st_mtime_ns) for path in Path(c.config.pack_path).glob("*") if path.is_file())
        c._analysis_pack_cache = identity, pack, tracked
    else:
        for path, size, mtime in cached[2]:
            if not path.is_file() or path.stat().st_size != size or path.stat().st_mtime_ns != mtime:
                raise RiskError("PACK_REVISION_CHANGED", artifact=str(path))
    return c._analysis_pack_cache[1]


def _prepare_analysis(controller,draft,selection_hash,*,current_position=False):
    from qqq_sim.risk.contracts import ReferenceJob,RiskLeg,utc_us,stable_hash,canonical
    from qqq_sim.risk.policy import make_action,next_opportunity
    c=controller; config=c.config
    c._lifecycle_registry.verify()
    parameters=json.loads(Path(config.parameters_path).read_text(encoding='utf-8-sig')) if config.parameters_path else None
    campaign=c.broker.ledger.campaigns.get(draft.campaign_id)
    if current_position and campaign is None:
        raise BrokerError('CURRENT_CAMPAIGN_REQUIRED')
    context=None; canonical_legs=None
    if campaign:
        if campaign.target is None:
            raise BrokerError('CAMPAIGN_INITIALIZING' if campaign.status=='INITIALIZING' else 'ORIGINAL_TARGET_UNAVAILABLE')
        canonical_legs=tuple(OrderLeg(contract,q) for contract,q in c.broker.ledger.positions(campaign.campaign_id))
        if not canonical_legs and campaign.target_realized:
            return {'status':'OK','reason':'CAMPAIGN_SUCCEEDED',
                    'managed_probability':1.0,'hold_probability':None,
                    'original_target_usd':str(campaign.target),'known_net_realized_usd':str(campaign.net_cash)}
        context={'target_usd':str(campaign.target),'entry_cash_usd':str(campaign.trade_cash+campaign.income),
                 'deadline':campaign.deadline}
    pack = cached_analysis_pack(c) if config.pack_path else None
    job=risk_job(c.snapshot,draft,parameters,c.data.calendar,context,current_position=canonical_legs,pack=pack)
    identity=AnalysisIdentity(c.session_id,c.branch_id,c.version,c.snapshot.snapshot_id,selection_hash,
                              digest(primitive(campaign)),digest(job),c._pack_hash,digest(config.rule_profile))
    quotes={broker_contract(entry.contract,c.data.calendar).key:entry.latest_event for entry in c.snapshot.quotes}
    if campaign:
        managed={'net_cash_usd':str(campaign.net_cash),'gross_cash_usd':str(campaign.trade_cash+campaign.income),
                 'target_usd':str(campaign.target),'remaining_expense_budget_usd':str(campaign.expense_remaining),
                 'hard_cutoff_utc_us':utc_us(campaign.hard_cutoff),'p0_hold':campaign.p0_hold,'p0_policy':campaign.p0_policy,
                 'benchmark_status':campaign.benchmark_status,'operational_model':campaign.operational_model,
                 'benchmark_policy_id':campaign.benchmark_policy_id,'benchmark_deadline_utc_us':campaign.benchmark_deadline_utc_us,
                 'benchmark_hard_cutoff_utc_us':campaign.benchmark_hard_cutoff_utc_us,'p0_policy_range':campaign.p0_policy_range,
                 'initial_deadline_utc_us':utc_us(campaign.deadline),'active_deadline_utc_us':utc_us(campaign.deadline),
                 'initial_fill_utc_us':None if campaign.initialized_at is None else utc_us(campaign.initialized_at),
                 'economically_completed':campaign.target_realized,
                 'protection_instruction_recorded':campaign.standing_protection,
                 'unresolved_stock_obligation':bool(c.broker.ledger.shares(campaign.campaign_id)),
                 'campaign_id':campaign.campaign_id,
                 'open_contract_quantities':{contract.key:q for contract,q in c.broker.ledger.positions(campaign.campaign_id)}}
    else:
        engine=FeeEngine(); fees=D(0)
        for leg in draft.legs:
            quote=quotes[leg.contract.key]
            fees+=sum(engine.assess_option('DRAFT',leg.contract,leg.quantity,quote.ask if leg.quantity>0 else quote.bid,
                                           abs(leg.quantity),c.decision_at.date()).values(),D(0))
        managed={'net_cash_usd':str(D(job['entry_cash_usd'])-fees),'gross_cash_usd':job['entry_cash_usd'],
                 'target_usd':job['target_usd'],'remaining_expense_budget_usd':str(draft.expense_budget),
                 'hard_cutoff_utc_us':utc_us(draft.hard_cutoff),'benchmark_status':'INDICATIVE_DRAFT_NOT_INITIAL_FILLED_BENCHMARK',
                 'operational_model':draft.operational_model,'protection_instruction_recorded':draft.standing_protection,
                 'campaign_id':draft.campaign_id}
        managed.update(initial_deadline_utc_us=job['deadline_utc_us'],active_deadline_utc_us=job['deadline_utc_us'])
    other_positions=[]; other_lower=D(0)
    for cid,other in c.broker.ledger.campaigns.items():
        if campaign and cid==campaign.campaign_id: continue
        cert=certify_campaign(c.broker.ledger,cid,c.broker.loss_cap)
        if cert.lower_bound is None: raise BrokerError('OTHER_CAMPAIGN_RISK_CERTIFICATE_UNAVAILABLE',campaign_id=cid)
        other_lower+=cert.lower_bound
        other_positions.extend(c.broker.ledger.positions(cid))
    margin=MarginEngine().evaluate(other_positions)
    if margin.collateral is None: raise BrokerError('OTHER_CAMPAIGN_COLLATERAL_UNAVAILABLE')
    reserved=sum((o.reservation for o in c.broker.orders.values() if o.status in ('WORKING','PARTIALLY_FILLED','DATA_BLOCKED')),D(0))
    available=c.broker.ledger.economic_cash-margin.collateral-reserved
    if campaign is None: available+=D(managed['net_cash_usd'])
    limits={'loss_cap_usd':config.loss_cap,'session_loss_cap_usd':config.session_loss_cap,
            'available_cash_usd':str(available),'other_campaign_net_usd':str(other_lower),
            'other_campaign_bound_basis':'CONSERVATIVE_FINAL_NET_LOWER_BOUNDS',
            'project_bp_cap_usd':config.bp_cap,'broker_state':c.broker.to_dict(),'campaign_id':draft.campaign_id}
    from qqq_sim.risk.account_features import bind_account_features
    managed=bind_account_features(managed,limits)
    universe=[]
    universe_exclusions={}
    for entry in c.snapshot.quotes:
        quote=entry.latest_event
        if quote.valid and quote.iv is not None and entry.contract.metadata_verified:
            from qqq_sim.risk.contracts import RiskError
            from datetime import datetime
            try:
                scoped=parameters_for_expiry(parameters,entry.contract.expiration)
                known=datetime.fromisoformat(scoped['known_at'])
                if known.tzinfo is None or known>c.snapshot.asof:
                    raise RiskError('FUTURE_PARAMETER')
                probability_snapshot_quality(c.snapshot,[entry])
                fixing=utc_us(datetime.fromisoformat(scoped['fixing_at']))
                from qqq_sim.services.adapters import last_trading_for_expiry
                contract=broker_contract(entry.contract,c.data.calendar)
                last_trading=utc_us(last_trading_for_expiry(scoped,entry.contract.expiration,c.data.calendar,contract))
            except RiskError as exc:
                universe_exclusions[exc.code]=universe_exclusions.get(exc.code,0)+1
                continue
            universe.append({'contract_id':contract.key,'right':entry.contract.right,
                             'strike':str(entry.contract.strike),'quantity':1,'fixing_utc_us':fixing,
                             'last_trading_utc_us':last_trading,'exercise_style':entry.contract.exercise_style,
                             'multiplier':entry.contract.multiplier,
                             'deliverable_id':'100_QQQ_SHARES' if entry.contract.deliverable_id=='STANDARD_100_SHARES' else entry.contract.deliverable_id,
                             'half_spread':float((quote.ask-quote.bid)/2),
                             'sigma':quote.iv,'bid':float(quote.bid),'ask':float(quote.ask),'known_at_utc_us':utc_us(quote.available_at)})
    from qqq_sim.risk.universe_admission import admit_known_universe
    admitted,universe_admission=admit_known_universe(ReferenceJob.from_dict(job),universe,managed['hard_cutoff_utc_us'])
    universe=canonical(admitted)
    from qqq_sim.services.managed_anchor import managed_job_from_known_anchor
    managed_job,managed_anchor=managed_job_from_known_anchor(job,admitted,managed['hard_cutoff_utc_us'],
                                                            c.snapshot,parameters,c.data.calendar,pack)
    limits['managed_job']=managed_job
    limits['managed_anchor']=managed_anchor
    identity=replace(identity,objective_hash=digest({'hold_job':job,'managed_job':managed_job,'managed_anchor':managed_anchor}))
    proposal=None
    # Only an explicit view mode denotes unchanged holdings. A proposal adding
    # an identical spread still needs its own full poststate/budget checks.
    if campaign and not current_position:
        changes=risk_change_legs(c.snapshot,draft,parameters,c.data.calendar,context,pack=pack)
        family='FULL_CLOSE' if draft.intent=='CLOSE' else 'ROLL_EXPIRATION' if draft.intent=='ROLL' else 'HEDGE'
        proposal=make_action(ReferenceJob.from_dict(job),family,changes,next_opportunity(ReferenceJob.from_dict(job)),managed)
        proposal.update(limit_cost_usd=str(-draft.limit_cash*draft.units),max_total_fees_usd=str(draft.max_total_fees),
                        order_type=draft.order_type,target_usd=str(campaign.target))
        try:
            trial=Broker.from_dict(c.broker.to_dict())
            accepted=trial.submit(draft,c.decision_at,snapshot_quotes(c.snapshot,c.data.calendar),'proposal-preview:'+digest(draft))
            proposal['exact_broker_preview']={'status':'OK','reservation_usd':str(accepted.reservation)}
        except BrokerError as error:
            proposal['exact_broker_preview']=error.as_dict()
        proposal['id']=stable_hash(proposal)
    return {'identity':identity,'job':job,'pack_path':config.pack_path,'campaign':managed,
            'universe':universe,'universe_exclusions':universe_exclusions,'universe_admission':universe_admission,
            'managed_anchor':managed_anchor,'limits':limits,'proposal':proposal}


def prepare_analysis(controller,draft,selection_hash,*,current_position=False,register=True):
    """Freeze input provenance before dispatch; benchmark transactions opt out."""
    from .analytic_records import register_analysis_request,archive_unavailable
    with controller._lock:
        try:
            prepared=_prepare_analysis(controller,draft,selection_hash,current_position=current_position)
        except Exception as exc:
            if register:
                result=exc.as_dict() if hasattr(exc,'as_dict') else exc.to_dict() if hasattr(exc,'to_dict') else {'status':'UNAVAILABLE','reason':getattr(exc,'code','ANALYSIS_INPUTS_UNAVAILABLE'),'message':str(exc)}
                result.setdefault('hold_probability',None)
                result.setdefault('managed_probability',None)
                exc.analysis_record=archive_unavailable(controller,draft,selection_hash,result,current_position=current_position)
            raise
        if not register:
            return prepared
        if 'identity' not in prepared:
            return archive_unavailable(controller,draft,selection_hash,prepared,current_position=current_position)
        return register_analysis_request(controller,prepared,draft=draft,current_position=current_position)
