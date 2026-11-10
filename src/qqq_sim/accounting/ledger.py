"""Cash buckets, signed FIFO basis, campaign targets and immutable postings."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from ..broker.models import D, Contract, Fill, BrokerError, cents, money, decimal, encode, fingerprint


@dataclass
class Lot:
    lot_id: str
    contract: Contract
    campaign_id: str
    quantity: int
    basis: object
    opening_fees: object = D(0)


@dataclass
class StockLot:
    lot_id: str
    campaign_id: str
    quantity: int
    basis: object
    opening_fees: object = D(0)


@dataclass
class Campaign:
    campaign_id: str
    deadline: datetime
    hard_cutoff: datetime
    operational_model: str
    standing_protection: bool
    expense_remaining: object = D(100)
    target_fraction: object = D('.5')
    status: str = 'INITIALIZING'
    original_max_profit: object = None
    target: object = None
    original_entry_cash: object = None
    original_snapshot: str | None = None
    p0_hold: float | None = None
    p0_policy: float | None = None
    benchmark_status: str = 'UNKNOWN'
    benchmark_reason: str = 'INITIAL_FILL_ANALYSIS_UNAVAILABLE'
    initialized_at: datetime | None = None
    initial_fill_basis: dict | None = None
    benchmark_id: str | None = None
    benchmark_inputs: dict | None = None
    benchmark_result: dict | None = None
    benchmark_policy_id: str | None = None
    benchmark_deadline_utc_us: int | None = None
    benchmark_hard_cutoff_utc_us: int | None = None
    p0_policy_range: list | None = None
    trade_cash: object = D(0)
    income: object = D(0)
    expenses: object = D(0)
    realized_gross: object = D(0)
    realized_fees: object = D(0)
    target_realized: bool = False
    realized_at: datetime | None = None

    @property
    def net_cash(self): return self.trade_cash+self.income-self.expenses


@dataclass
class Obligation:
    obligation_id: str
    campaign_id: str | None
    amount_cents: int
    due_date: date
    category: str
    settled: bool = False


class Ledger:
    def __init__(self, initial_cash=D(5000)):
        self.initial_cash_cents=cents(initial_cash)
        self.settled_cash_cents=self.initial_cash_cents
        self.lots: list[Lot]=[]
        self.stock_lots: list[StockLot]=[]
        self.campaigns: dict[str,Campaign]={}
        self.obligations: dict[str,Obligation]={}
        self.events: list[dict]=[]
        self.event_ids: set[str]=set()
        self.external_contributions=D(0)

    @property
    def economic_cash(self):
        return D(self.settled_cash_cents+sum(o.amount_cents for o in self.obligations.values() if not o.settled))/100

    @property
    def settled_cash(self): return D(self.settled_cash_cents)/100

    def positions(self, campaign_id=None):
        positions={}; contracts={}
        for lot in self.lots:
            if campaign_id is None or lot.campaign_id==campaign_id:
                contracts[lot.contract.key]=lot.contract
                positions[lot.contract.key]=positions.get(lot.contract.key,0)+lot.quantity
        return [(contracts[k],q) for k,q in sorted(positions.items()) if q]

    def shares(self,campaign_id=None):
        return sum(x.quantity for x in self.stock_lots if campaign_id is None or x.campaign_id==campaign_id)

    def append_event(self,event_id,kind,at,payload,postings=()):
        if event_id in self.event_ids: return False
        self.event_ids.add(event_id)
        previous=self.events[-1]['event_hash'] if self.events else ''
        event={'event_id':event_id,'event_type':kind,'at':at.isoformat(),'payload':encode(payload),
               'postings':encode(list(postings)),'previous_hash':previous,'sequence':len(self.events)+1}
        event['event_hash']=fingerprint(event); self.events.append(event)
        return True

    def create_obligation(self,obligation_id,campaign_id,amount,due_date,category):
        if obligation_id in self.obligations: return
        self.obligations[obligation_id]=Obligation(obligation_id,campaign_id,cents(amount),due_date,category)

    def settle(self,obligation_id,at):
        o=self.obligations[obligation_id]
        if o.settled: return False
        if at.date()<o.due_date: raise BrokerError('SETTLEMENT_NOT_DUE')
        self.settled_cash_cents+=o.amount_cents; o.settled=True
        self.append_event('settle:'+obligation_id,'SETTLEMENT',at,{'obligation_id':obligation_id},
                          [{'amount_cents':o.amount_cents,'bucket':'settled','category':'transfer'},
                           {'amount_cents':-o.amount_cents,'bucket':'unsettled','category':'transfer'}])
        return True

    def opening_quantity(self,campaign_id,contract,delta):
        opposing=[x for x in self.lots if x.contract.key==contract.key and x.quantity*delta<0]
        if any(x.campaign_id!=campaign_id for x in opposing): raise BrokerError('AMBIGUOUS_CROSS_CAMPAIGN_OFFSET')
        closing=min(abs(delta),sum(abs(x.quantity) for x in opposing))
        return abs(delta)-closing

    def apply_fill(self,fill: Fill,due_date):
        if fill.fill_id in self.event_ids: return False
        campaign=self.campaigns[fill.campaign_id]
        remaining=fill.quantity; closing_count=0; total=abs(remaining)
        self.opening_quantity(fill.campaign_id,fill.contract,remaining)
        from ..broker.fees import allocate_cents
        allocation_weights=[]; unallocated_quantity=total
        for lot in self.lots:
            if lot.contract.key==fill.contract.key and lot.campaign_id==fill.campaign_id and lot.quantity*remaining<0:
                n=min(abs(lot.quantity),unallocated_quantity)
                if n: allocation_weights.append((lot.lot_id,n)); unallocated_quantity-=n
                if not unallocated_quantity: break
        if unallocated_quantity: allocation_weights.append(('opening:'+fill.fill_id,unallocated_quantity))
        fee_allocations=allocate_cents(fill.fees,allocation_weights)
        for lot in self.lots:
            if lot.contract.key!=fill.contract.key or lot.campaign_id!=fill.campaign_id or lot.quantity*remaining>=0: continue
            n=min(abs(lot.quantity),abs(remaining)); sign=1 if lot.quantity>0 else -1
            fraction=D(n)/abs(lot.quantity); removed_basis=lot.basis*fraction
            removed_fees=money(lot.opening_fees*fraction)
            closing_cash=D(sign*n*fill.contract.multiplier)*fill.price
            campaign.realized_gross+=closing_cash-removed_basis
            campaign.realized_fees+=removed_fees+fee_allocations[lot.lot_id]
            lot.quantity-=sign*n; lot.basis-=removed_basis; lot.opening_fees-=removed_fees
            remaining+=sign*n; closing_count+=n
            if not remaining: break
        self.lots=[x for x in self.lots if x.quantity]
        if remaining:
            opening_fee=fee_allocations['opening:'+fill.fill_id]
            self.lots.append(Lot(fill.fill_id,fill.contract,fill.campaign_id,remaining,
                                 D(remaining*fill.contract.multiplier)*fill.price,opening_fee))
        premium=-D(fill.quantity*fill.contract.multiplier)*fill.price
        campaign.trade_cash+=premium; campaign.expenses+=fill.fees
        if campaign.status!='INITIALIZING': campaign.expense_remaining-=fill.fees
        self.create_obligation(fill.fill_id+':premium',fill.campaign_id,premium,due_date,'OPTION_PREMIUM')
        self.settled_cash_cents-=cents(fill.fees)
        self.append_event(fill.fill_id,'FILL',fill.event_at,fill,
                          [{'campaign_id':fill.campaign_id,'amount_cents':cents(premium),'bucket':'unsettled','category':'OPTION_PREMIUM','obligation_id':fill.fill_id+':premium','due_date':due_date.isoformat()},
                           {'campaign_id':fill.campaign_id,'amount_cents':-cents(fill.fees),'bucket':'settled','category':'FEES'}])
        self._check_realization(campaign,fill.event_at)
        return True

    def _check_realization(self,campaign,at):
        if campaign.status=='INITIALIZING': return
        if not self.positions(campaign.campaign_id) and not self.shares(campaign.campaign_id):
            campaign.status='FLAT'
            # All trade basis has crystallized; delivery fees and transferred
            # option opening fees must reconcile to the final campaign cash.
            campaign.realized_gross=campaign.trade_cash
            campaign.realized_fees=campaign.expenses
            if campaign.target is not None and campaign.net_cash>=campaign.target and at<=campaign.deadline:
                campaign.target_realized=True; campaign.status='SUCCEEDED'; campaign.realized_at=at

    def initialize_target(self,campaign_id,snapshot_id=None,at=None):
        from ..broker.margin import payoff_extrema
        c=self.campaigns[campaign_id]
        if c.target is not None: return
        extrema=payoff_extrema(self.positions(campaign_id))
        if extrema.maximum_unbounded:
            c.status='ACTIVE'; c.benchmark_reason='UNBOUNDED_MAX_PROFIT_NO_P50_TARGET'; return
        maximum=c.trade_cash+extrema.maximum
        if maximum<=0:
            c.status='ACTIVE'; c.benchmark_reason='NONPOSITIVE_MAX_PROFIT_NO_P50_TARGET'; return
        c.original_max_profit=maximum; c.target=maximum*c.target_fraction
        c.original_entry_cash=c.trade_cash; c.original_snapshot=snapshot_id; c.status='ACTIVE'
        c.initialized_at=at
        c.initial_fill_basis=encode({'initialized_at':at,'target_usd':c.target,'original_max_profit_usd':maximum,
            'trade_cash_usd':c.trade_cash,'recognized_expenses_usd':c.expenses,'net_cash_usd':c.net_cash,
            'deadline':c.deadline,'hard_cutoff':c.hard_cutoff,'positions':self.positions(campaign_id),
            'fill_event_ids':[event['event_id'] for event in self.events if event['event_type']=='FILL' and event['payload'].get('campaign_id')==campaign_id]})

    def accrue_income(self,event_id,campaign_id,amount,due_date,at,category='DIVIDEND'):
        if event_id in self.event_ids: return False
        amount=money(amount)
        if campaign_id: self.campaigns[campaign_id].income+=amount
        self.create_obligation(event_id,campaign_id,amount,due_date,category)
        self.append_event(event_id,'INCOME_ACCRUAL',at,{'campaign_id':campaign_id,'amount':amount,'due_date':due_date,'category':category},
                          [{'campaign_id':campaign_id,'amount_cents':cents(amount),'bucket':'unsettled','category':category,'obligation_id':event_id,'due_date':due_date.isoformat()}])
        return True

    def accrue_expense(self,event_id,campaign_id,amount,due_date,at,category='INTEREST'):
        if event_id in self.event_ids: return False
        amount=money(amount)
        if campaign_id:
            c=self.campaigns[campaign_id]; c.expenses+=amount; c.expense_remaining-=amount
        self.create_obligation(event_id,campaign_id,-amount,due_date,category)
        self.append_event(event_id,'EXPENSE_ACCRUAL',at,{'campaign_id':campaign_id,'amount':amount,'due_date':due_date,'category':category})
        return True

    def equity(self,marks,stock_mark=None):
        known=self.economic_cash; missing=[]
        for contract,q in self.positions():
            value=marks.get(contract.key)
            if value is None: missing.append(contract.key)
            else: known+=q*contract.multiplier*decimal(value)
        if self.shares():
            if stock_mark is None: missing.append('QQQ_STOCK')
            else: known+=self.shares()*decimal(stock_mark)
        return {'equity':None if missing else known,'known_subtotal':known,'missing':missing,
                'status':'INCOMPLETE' if missing else 'OK'}

    def pnl(self,campaign_id,marks,stock_mark=None):
        c=self.campaigns[campaign_id]; value=D(0); missing=[]
        for contract,q in self.positions(campaign_id):
            if marks.get(contract.key) is None: missing.append(contract.key)
            else: value+=q*contract.multiplier*decimal(marks[contract.key])
        if self.shares(campaign_id):
            if stock_mark is None: missing.append('QQQ_STOCK')
            else: value+=self.shares(campaign_id)*decimal(stock_mark)
        return {'gross_realized':c.realized_gross,'net_realized':c.realized_gross-c.realized_fees,
                'gross_mark':None if missing else c.trade_cash+c.income+value,
                'net_mark':None if missing else c.net_cash+value,'missing':missing}

    def to_dict(self):
        return encode({'schema_version':1,'initial_cash_cents':self.initial_cash_cents,
                       'settled_cash_cents':self.settled_cash_cents,'lots':self.lots,'stock_lots':self.stock_lots,
                       'campaigns':self.campaigns,'obligations':self.obligations,'events':self.events,
                       'external_contributions':self.external_contributions})

    @classmethod
    def from_dict(cls,data):
        obj=cls(D(data['initial_cash_cents'])/100); obj.settled_cash_cents=data['settled_cash_cents']
        for raw in data['lots']:
            x=dict(raw); x['contract']=Contract.from_dict(x['contract'])
            for k in ('basis','opening_fees'): x[k]=D(x[k])
            obj.lots.append(Lot(**x))
        for raw in data.get('stock_lots',[]):
            x=dict(raw); x['basis']=D(x['basis']); x['opening_fees']=D(x.get('opening_fees','0')); obj.stock_lots.append(StockLot(**x))
        for key,raw in data['campaigns'].items():
            x=dict(raw)
            for k in ('deadline','hard_cutoff','realized_at','initialized_at'):
                if x.get(k): x[k]=datetime.fromisoformat(x[k])
            for k in ('expense_remaining','target_fraction','original_max_profit','target','original_entry_cash',
                      'trade_cash','income','expenses','realized_gross','realized_fees'):
                if x.get(k) is not None: x[k]=D(x[k])
            obj.campaigns[key]=Campaign(**x)
        for key,raw in data['obligations'].items():
            x=dict(raw); x['due_date']=date.fromisoformat(x['due_date']); obj.obligations[key]=Obligation(**x)
        obj.events=list(data['events']); obj.event_ids={x['event_id'] for x in obj.events}
        obj.external_contributions=D(data.get('external_contributions','0'))
        return obj

