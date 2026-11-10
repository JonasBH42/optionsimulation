"""Causal local lifecycle sidecars, separate economic and knowledge clocks.

The optional parameters JSON accepts ``lifecycle_events``. Each event has id,
kind, economic_at and known_at. This is explicit runtime evidence/scenario data,
never an inferred assignment, official close, dividend or financing rate.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,date
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from qqq_sim.broker.models import BrokerError,Contract,D,fingerprint

PHASES={'OFFICIAL_FIXING':10,'EXERCISE_INSTRUCTION':20,'ASSIGNMENT':30,'EXERCISE':30,
        'PAIRED_DELIVERY':30,'STOCK_QUOTE':35,'DIVIDEND':40,'RISK_REVIEW':45,'EXPIRY':50,'EXPENSE':60,'FINANCING':60}


@dataclass(frozen=True)
class LifecycleRecord:
    event_id: str
    kind: str
    economic_at: datetime
    known_at: datetime
    fields: dict

    @property
    def available_at(self): return max(self.economic_at,self.known_at)


class LifecycleRegistry:
    def __init__(self,path=None):
        self.path=Path(path) if path else None
        raw=self.path.read_bytes() if self.path else b'{}'
        self.source_hash=sha256(raw).hexdigest()
        values=json.loads(raw.decode('utf-8-sig'))
        events=values.get('lifecycle_events',[]) if isinstance(values,dict) else []
        records=[]; ids=set()
        for item in events:
            fields=dict(item); event_id=str(fields.pop('id')); kind=str(fields.pop('kind')).upper()
            if event_id in ids: raise BrokerError('DUPLICATE_LIFECYCLE_EVENT_ID',event_id=event_id)
            ids.add(event_id)
            if kind not in PHASES: raise BrokerError('UNKNOWN_LIFECYCLE_EVENT_KIND',kind=kind)
            economic=datetime.fromisoformat(fields.pop('economic_at'))
            known=datetime.fromisoformat(fields.pop('known_at'))
            if economic.tzinfo is None or known.tzinfo is None: raise BrokerError('NAIVE_LIFECYCLE_TIMESTAMP')
            records.append(LifecycleRecord(event_id,kind,economic,known,fields))
        self.records=tuple(sorted(records,key=lambda r:(r.available_at,PHASES[r.kind],r.event_id)))

    def verify(self):
        raw=self.path.read_bytes() if self.path else b'{}'
        if sha256(raw).hexdigest()!=self.source_hash: raise BrokerError('LIFECYCLE_SOURCE_REVISION_CHANGED')

    def fixing(self,expiry,instant):
        matches=[r for r in self.records if r.kind=='OFFICIAL_FIXING' and r.available_at<=instant and
                 date.fromisoformat(r.fields.get('expiry',r.economic_at.date().isoformat()))==expiry]
        if not matches: return None
        newest=max(r.known_at for r in matches)
        matches=[r for r in matches if r.known_at==newest]
        if len({str(r.fields.get('price')) for r in matches})>1: raise BrokerError('CONFLICTING_OFFICIAL_FIXING')
        return matches[0]

    @staticmethod
    def _contract(controller,fields,key='contract'):
        value=fields.get(key)
        if isinstance(value,dict): return Contract.from_dict(value)
        contract_key=fields.get(key+'_key',value)
        contracts={c.key:c for c,q in controller.broker.ledger.positions()}
        if contract_key not in contracts: raise BrokerError('LIFECYCLE_CONTRACT_NOT_HELD',contract_key=contract_key)
        return contracts[contract_key]

    @staticmethod
    def _due(controller,record):
        value=record.fields.get('due_date')
        return date.fromisoformat(value) if value else controller.data.calendar.stock_settlement_date(record.economic_at.date())

    def process_until(self,controller,instant):
        self.verify(); broker=controller.broker; ledger=broker.ledger
        for record in self.records:
            if record.available_at>instant: break
            marker='lifecycle-input:'+record.event_id
            if marker in ledger.event_ids: continue
            f=record.fields; campaign_id=f.get('campaign_id')
            if campaign_id and campaign_id not in ledger.campaigns:
                # A scenario about an unrelated/not-yet-opened campaign has no
                # local economic effect and is not guessed into existence.
                continue
            kind=record.kind; processed_at=record.available_at
            if kind=='OFFICIAL_FIXING':
                if D(str(f['price']))<=0: raise BrokerError('INVALID_OFFICIAL_FIXING')
            elif kind in ('ASSIGNMENT','EXERCISE'):
                contract=self._contract(controller,f)
                quantity=int(f['quantity'])
                positions=dict((c.key,q) for c,q in ledger.positions(campaign_id))
                if kind=='ASSIGNMENT' and positions.get(contract.key,0)>=0: raise BrokerError('ASSIGNMENT_REQUIRES_SHORT')
                if kind=='EXERCISE' and positions.get(contract.key,0)<=0: raise BrokerError('EXERCISE_REQUIRES_LONG')
                broker.lifecycle.deliver('lifecycle-delivery:'+record.event_id,campaign_id,contract,quantity,
                                          record.economic_at,self._due(controller,record),f.get('batch_id',record.event_id),
                                          known_at=record.known_at,processed_at=processed_at)
                if record.known_at>record.economic_at:
                    ledger.campaigns[campaign_id].operational_model='DELAYED_NOTICE_EXPLICIT_SCENARIO'
                self._revise_dividends(controller,campaign_id,record)
            elif kind=='PAIRED_DELIVERY':
                if record.known_at>record.economic_at: raise BrokerError('PAIRED_DELIVERY_NOTICE_ASSUMPTION_FAILED')
                short=self._contract(controller,f,'short'); long=self._contract(controller,f,'long')
                broker.lifecycle.paired_delivery(record.event_id,campaign_id,short,long,int(f['quantity']),
                                                  record.economic_at,self._due(controller,record))
            elif kind=='STOCK_QUOTE':
                if f.get('symbol','QQQ')!='QQQ': raise BrokerError('UNSUPPORTED_STOCK_QUOTE_SYMBOL')
                if (record.available_at-record.economic_at).total_seconds()<=60:
                    broker.on_stock_quote(f['bid'],f['ask'],record.available_at,f.get('quote_id',record.event_id),f.get('bid_size'),f.get('ask_size'))
            elif kind=='RISK_REVIEW':
                self._risk_review(controller,record)
            elif kind=='DIVIDEND':
                if f.get('dividend_type','STANDARD_CASH')!='STANDARD_CASH': raise BrokerError('SPECIAL_DIVIDEND_UNSUPPORTED')
                if record.known_at>record.economic_at: raise BrokerError('DIVIDEND_DECLARATION_NOT_KNOWN_AT_EX_DATE')
                selected=[campaign_id] if campaign_id else sorted(ledger.campaigns)
                for cid in selected: self._dividend_entitlement(controller,record,cid,record.event_id)
            elif kind=='EXERCISE_INSTRUCTION':
                contract=self._contract(controller,f)
                q=dict((c.key,q) for c,q in ledger.positions(campaign_id))[contract.key]
                session=controller.data.calendar.session(contract.expiry)
                broker.lifecycle.validate_instruction(contract,q,f['instruction'],record.economic_at,session.underlying_close,
                    datetime.fromisoformat(f['cutoff']) if f.get('cutoff') else None)
            elif kind=='EXPIRY':
                self._expire(controller,record.economic_at,instant,record)
            elif kind=='EXPENSE':
                ledger.accrue_expense('lifecycle-expense:'+record.event_id,campaign_id,D(str(f['amount'])),
                                     self._due(controller,record),record.economic_at,f.get('category','EXPLICIT_EXPENSE'))
            elif kind=='FINANCING':
                # A supplied amount/rate is an explicitly named scenario. Never
                # synthesize a missing borrow rate from an option quote archive.
                if not f.get('due_date'): raise BrokerError('FINANCING_PAYMENT_DATE_REQUIRED')
                broker.financing.accrue(ledger,campaign_id,record.economic_at,date.fromisoformat(f['due_date']),
                                        stock_mark=f.get('stock_mark'),borrow_rate=f.get('borrow_rate'))
            ledger.append_event(marker,'LIFECYCLE_INPUT_CONSUMED',processed_at,
                                {'input_id':record.event_id,'kind':kind,'economic_at':record.economic_at,
                                 'known_at':record.known_at,'source_record_hash':fingerprint(record),
                                 'scenario_basis':f.get('basis','EXPLICIT_LOCAL_RECORD')})
        # A fixing is an observation; consumption of expiration follows that
        # observation at the same instant, before a later quote could fill.
        for expiry in sorted({c.expiry for c,q in ledger.positions()}):
            contracts=[c for c,q in ledger.positions() if c.expiry==expiry]
            session=controller.data.calendar.session(expiry)
            fixing_at=max((c.fixing_at or session.fixing_at or session.underlying_close for c in contracts))
            if fixing_at<=instant:
                self._expire(controller,fixing_at,instant)

    def _instructions(self,instant):
        result={}
        for r in self.records:
            if r.kind=='EXERCISE_INSTRUCTION' and r.available_at<=instant:
                key=r.fields.get('contract_key')
                if not key and isinstance(r.fields.get('contract'),dict): key=Contract.from_dict(r.fields['contract']).key
                result[key]=r.fields['instruction']
        return result

    def _risk_review(self,controller,record):
        from qqq_sim.broker.models import OrderDraft,OrderLeg
        from qqq_sim.broker.risk_checks import certify_campaign
        broker=controller.broker; fields=record.fields
        selected=[fields['campaign_id']] if fields.get('campaign_id') else sorted(broker.ledger.campaigns)
        for campaign_id in selected:
            positions=broker.ledger.positions(campaign_id)
            if not positions: continue
            campaign=broker.ledger.campaigns[campaign_id]
            scenarios=broker.lifecycle.adverse_assignments(campaign_id)
            certificate=certify_campaign(broker.ledger,campaign_id,broker.loss_cap)
            safe=(certificate.status=='VERIFIED_WITHIN_DECLARED_MODEL' and certificate.delivery_bound is not None
                  and broker.ledger.economic_cash>=certificate.delivery_bound+certificate.mandatory_delivery_fees)
            if safe: continue
            details={'campaign_id':campaign_id,'supported_assignment_scenario_count':len(scenarios),
                     'certificate_status':certificate.status,'reason':certificate.reason}
            if not fields.get('automatic',False):
                raise BrokerError('EXPIRY_RISK_REVIEW_REQUIRES_INSTRUCTION',**details)
            first_fills=[datetime.fromisoformat(event['at']) for event in broker.ledger.events if event['event_type']=='FILL'
                         and event['payload'].get('campaign_id')==campaign_id]
            if first_fills and record.known_at>min(first_fills): raise BrokerError('RISK_DESK_POLICY_NOT_PRESELECTED')
            if 'minimum_package_cash' not in fields: raise BrokerError('RISK_REVIEW_BOUNDED_LIMIT_REQUIRED')
            draft=OrderDraft(campaign_id,tuple(OrderLeg(contract,-quantity) for contract,quantity in positions),
                             D(str(fields['minimum_package_cash'])),campaign.deadline,campaign.hard_cutoff,
                             intent='CLOSE',max_total_fees=D(str(fields.get('max_total_fees',50))),
                             operational_model=campaign.operational_model,standing_protection=campaign.standing_protection)
            broker.submit(draft,record.available_at,broker.quotes,'risk-review:'+record.event_id+':'+campaign_id)

    def _expire(self,controller,economic_at,instant,record=None):
        contracts=[c for c,q in controller.broker.ledger.positions() if c.expiry==economic_at.date()]
        if not contracts: return
        fixing=self.fixing(economic_at.date(),instant)
        scenario=record.fields.get('scenario','EXPIRY_BASELINE_V1') if record else 'EXPIRY_BASELINE_V1'
        price=D(str(fixing.fields['price'])) if fixing else None
        proxy=record.fields.get('proxy_close') if record else None
        if price is None and (scenario!='CLOSE_PROXY_V1' or proxy is None):
            raise BrokerError('SETTLEMENT_REFERENCE_MISSING','Supply an as-of official fixing or explicit CLOSE_PROXY_V1 record',contracts=[c.key for c in contracts])
        controller.broker.lifecycle.expiry(economic_at,controller.data.calendar.stock_settlement_date(economic_at.date()),
                                            official_fixing=price,scenario=scenario,proxy_close=proxy,
                                            instructions=self._instructions(instant))

    def _dividend_entitlement(self,controller,record,campaign_id,revision_id,processed_at=None):
        ledger=controller.broker.ledger; f=record.fields
        shares=0
        for event in ledger.events:
            if event['event_type'] not in ('EXERCISE','ASSIGNMENT','STOCK_FILL'): continue
            payload=event['payload']
            if payload.get('campaign_id')==campaign_id and datetime.fromisoformat(event['at'])<=record.economic_at:
                shares+=int(payload.get('stock_quantity_delta',0))
        previous=sum(int(e['payload']['share_delta']) for e in ledger.events
                     if e['event_type']=='DIVIDEND_ENTITLEMENT' and e['payload']['input_id']==record.event_id and e['payload']['campaign_id']==campaign_id)
        delta=shares-previous
        event_id=f'dividend:{record.event_id}:{campaign_id}:{revision_id}'
        if event_id in ledger.event_ids: return
        processed_at=processed_at or max(record.economic_at,record.known_at)
        ledger.accrue_income(event_id,campaign_id,delta*D(str(f['per_share'])),date.fromisoformat(f['pay_date']),
                             processed_at,'DIVIDEND')
        ledger.append_event(event_id+':entitlement','DIVIDEND_ENTITLEMENT',processed_at,
                            {'input_id':record.event_id,'campaign_id':campaign_id,'share_delta':delta,'entitled_shares':shares,
                             'per_share':str(f['per_share']),'economic_at':record.economic_at,'known_at':processed_at,'revision_id':revision_id})

    def _revise_dividends(self,controller,campaign_id,assignment):
        for dividend in self.records:
            if dividend.kind=='DIVIDEND' and assignment.economic_at<=dividend.economic_at<assignment.available_at:
                if 'lifecycle-input:'+dividend.event_id in controller.broker.ledger.event_ids:
                    self._dividend_entitlement(controller,dividend,campaign_id,assignment.event_id,assignment.available_at)

