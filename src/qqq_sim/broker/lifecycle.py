"""Physical QQQ delivery, explicit lifecycle scenarios and dividend entitlements."""
from __future__ import annotations
from datetime import timedelta
from itertools import product
from ..accounting.ledger import StockLot
from .models import D,BrokerError,cents,money,decimal


class LifecycleEngine:
    def __init__(self,ledger,fee_engine): self.ledger=ledger; self.fees=fee_engine

    def deliver(self,event_id,campaign_id,contract,quantity,at,due_date,batch_id=None,known_at=None,processed_at=None):
        """Quantity is a positive number of existing option contracts to exercise/assign."""
        if event_id in self.ledger.event_ids: return False
        if quantity<=0 or not isinstance(quantity,int): raise BrokerError('INVALID_DELIVERY_QUANTITY')
        if known_at is not None and known_at>(processed_at or at): raise BrokerError('ASSIGNMENT_NOTICE_NOT_YET_KNOWN')
        lots=[x for x in self.ledger.lots if x.campaign_id==campaign_id and x.contract.key==contract.key]
        if not lots or quantity>sum(abs(x.quantity) for x in lots): raise BrokerError('DELIVERY_EXCEEDS_POSITION')
        sign=1 if lots[0].quantity>0 else -1
        kind='EXERCISE' if sign>0 else 'ASSIGNMENT'
        removed_basis=D(0); removed_opening_fees=D(0); remaining=quantity
        for lot in lots:
            n=min(remaining,abs(lot.quantity)); frac=D(n)/abs(lot.quantity)
            removed_basis+=lot.basis*frac; lot.basis*=1-frac
            removed_opening_fees+=lot.opening_fees*frac
            lot.opening_fees*=1-frac; lot.quantity-=sign*n; remaining-=n
            if not remaining: break
        self.ledger.lots=[x for x in self.ledger.lots if x.quantity]
        shares=quantity*contract.multiplier*sign*(1 if contract.right=='C' else -1)
        strike_cash=-shares*contract.strike
        stock_basis=-strike_cash+removed_basis
        c=self.ledger.campaigns[campaign_id]
        self._stock_transfer(event_id,campaign_id,shares,stock_basis,removed_opening_fees)
        fee=self.fees.assess_delivery(contract,at.date(),batch_id or event_id,kind)
        c.trade_cash+=strike_cash; c.expenses+=fee; c.expense_remaining-=fee
        self.ledger.settled_cash_cents-=cents(fee)
        self.ledger.create_obligation(event_id+':delivery',campaign_id,strike_cash,due_date,'PHYSICAL_DELIVERY')
        self.ledger.append_event(event_id,kind,at,{'campaign_id':campaign_id,'contract':contract,'quantity':quantity,
                                               'stock_quantity_delta':shares,'strike_cash':strike_cash,'transferred_basis':stock_basis,
                                               'fee':fee,'batch_id':batch_id,'known_at':known_at or at,
                                               'processed_at':processed_at or at,'outcome_basis':'SIMULATED_SCENARIO'},
                                 [{'campaign_id':campaign_id,'amount_cents':cents(strike_cash),'bucket':'unsettled','category':'PHYSICAL_DELIVERY',
                                   'obligation_id':event_id+':delivery','due_date':due_date.isoformat()},
                                  {'campaign_id':campaign_id,'amount_cents':-cents(fee),'bucket':'settled','category':'DELIVERY_FEE'}])
        self.ledger._check_realization(c,at)
        return True

    def _stock_transfer(self,event_id,campaign_id,quantity,basis,opening_fees=D(0)):
        c=self.ledger.campaigns[campaign_id]; remaining=quantity; residual_basis=basis
        residual_fees=opening_fees
        for lot in self.ledger.stock_lots:
            if lot.campaign_id!=campaign_id or lot.quantity*remaining>=0: continue
            n=min(abs(lot.quantity),abs(remaining)); sign=1 if lot.quantity>0 else -1
            removed=lot.basis*D(n)/abs(lot.quantity)
            closed_fees=lot.opening_fees*D(n)/abs(lot.quantity)
            transfer=residual_basis*D(n)/abs(remaining)
            transferred_fees=residual_fees*D(n)/abs(remaining)
            c.realized_gross+=-transfer-removed
            c.realized_fees+=closed_fees+transferred_fees
            lot.basis-=removed; lot.opening_fees-=closed_fees; lot.quantity-=sign*n
            residual_basis-=transfer; residual_fees-=transferred_fees; remaining+=sign*n
            if not remaining: break
        self.ledger.stock_lots=[x for x in self.ledger.stock_lots if x.quantity]
        if remaining: self.ledger.stock_lots.append(StockLot(event_id,campaign_id,remaining,residual_basis,residual_fees))

    def paired_delivery(self,batch_id,campaign_id,short,long,quantity,at,due_date):
        c=self.ledger.campaigns[campaign_id]
        if c.operational_model!='SIMULTANEOUS_DELIVERY_HEDGE_V1' or not c.standing_protection:
            raise BrokerError('PAIRED_DELIVERY_NOT_AUTHORIZED')
        if short.partition!=long.partition or short.right!=long.right: raise BrokerError('INCOMPATIBLE_PROTECTION')
        # The containing replay transaction commits the two transfers atomically.
        import copy
        old=copy.deepcopy(self.ledger.to_dict()); fee_groups=copy.deepcopy(self.fees.to_dict())
        try:
            self.deliver(batch_id+':short',campaign_id,short,quantity,at,due_date,batch_id)
            self.deliver(batch_id+':long',campaign_id,long,quantity,at,due_date,batch_id)
            if self.ledger.shares(campaign_id): raise BrokerError('PAIRED_DELIVERY_LEFT_STOCK')
        except Exception:
            restored=self.ledger.from_dict(old); self.ledger.__dict__.update(restored.__dict__); self.fees.load(fee_groups)
            raise

    def expiry(self,at,due_date,official_fixing=None,scenario='EXPIRY_BASELINE_V1',proxy_close=None,instructions=None):
        instructions=instructions or {}
        if official_fixing is None:
            if scenario!='CLOSE_PROXY_V1' or proxy_close is None: raise BrokerError('SETTLEMENT_REFERENCE_MISSING')
            fixing=decimal(proxy_close); basis='RTH_PROXY'
        else: fixing=decimal(official_fixing); basis='OFFICIAL_FIXING'
        deliveries=[]
        for campaign_id in sorted(self.ledger.campaigns):
            campaign=self.ledger.campaigns[campaign_id]
            if campaign.operational_model=='SIMULTANEOUS_DELIVERY_HEDGE_V1' and campaign.standing_protection:
                from .risk_checks import certify_campaign
                certificate=certify_campaign(self.ledger,campaign_id)
                contracts={contract.key:contract for contract,q in self.ledger.positions(campaign_id)}
                for short_key,long_key,quantity in certificate.pairs:
                    short,long=contracts[short_key],contracts[long_key]
                    if short.expiry==at.date() and short.intrinsic(fixing)>=D('.01'):
                        if instructions.get(long_key)=='DNE':
                            raise BrokerError('PROTECTIVE_EXERCISE_REQUIRED_BY_SELECTED_MODEL')
                        self.paired_delivery(f'expiry-paired:{campaign_id}:{short_key}:{long_key}',
                                             campaign_id,short,long,quantity,at,due_date)
                        deliveries.extend([short_key,long_key])
            for contract,q in list(self.ledger.positions(campaign_id)):
                if contract.expiry!=at.date(): continue
                if contract.fixing_at is not None and at<contract.fixing_at:
                    raise BrokerError('ECONOMIC_FIXING_NOT_REACHED')
                instruction=instructions.get(contract.key)
                if q<0 and instruction=='DNE': raise BrokerError('SHORT_DNE_FORBIDDEN')
                do_exercise=contract.intrinsic(fixing)>=D('.01')
                if q>0 and instruction=='DNE': do_exercise=False
                if q>0 and instruction=='EXERCISE': do_exercise=True
                if do_exercise:
                    if q>0:
                        delivered_shares=q*contract.multiplier*(1 if contract.right=='C' else -1)
                        delivery_cash=-delivered_shares*contract.strike
                        if self.ledger.economic_cash+delivery_cash<D(0):
                            raise BrokerError('AUTO_EXERCISE_FUNDING_REVIEW_REQUIRED')
                        if delivered_shares<0 and self.ledger.shares(campaign_id)+delivered_shares<0:
                            raise BrokerError('AUTO_EXERCISE_SHORT_STOCK_REVIEW_REQUIRED')
                    self.deliver(f'expiry:{campaign_id}:{contract.key}',campaign_id,contract,abs(q),at,due_date)
                    deliveries.append(contract.key)
                else:
                    event_id=f'expire-worthless:{campaign_id}:{contract.key}'
                    if event_id in self.ledger.event_ids: continue
                    c=self.ledger.campaigns[campaign_id]
                    removed=[x for x in self.ledger.lots if x.campaign_id==campaign_id and x.contract.key==contract.key]
                    c.realized_gross-=sum((x.basis for x in removed),D(0)); c.realized_fees+=sum((x.opening_fees for x in removed),D(0))
                    self.ledger.lots=[x for x in self.ledger.lots if x not in removed]
                    self.ledger.append_event(event_id,'EXPIRED_WORTHLESS',at,{'contract':contract,'quantity':q,'fixing':fixing,'fixing_basis':basis})
                    self.ledger._check_realization(c,at)
        return {'fixing':fixing,'fixing_basis':basis,'outcomes':'SIMULATED','delivered':deliveries}

    def dividend(self,event_id,campaign_id,per_share,ex_at,pay_date,known_at):
        if known_at>ex_at: raise BrokerError('DIVIDEND_NOT_KNOWN_ASOF')
        return self.ledger.accrue_income(event_id,campaign_id,self.ledger.shares(campaign_id)*decimal(per_share),pay_date,ex_at)

    @staticmethod
    def validate_instruction(contract,position_quantity,instruction,requested_at,underlying_close,exact_cutoff=None):
        cutoff=exact_cutoff or underlying_close-timedelta(hours=1)
        if requested_at>cutoff: raise BrokerError('EXERCISE_INSTRUCTION_CUTOFF_PASSED')
        if position_quantity<=0: raise BrokerError('SHORT_DNE_FORBIDDEN')
        if instruction not in ('DNE','EXERCISE'): raise BrokerError('UNKNOWN_EXERCISE_INSTRUCTION')
        return {'instruction':instruction,'contract':contract.key,'cutoff':cutoff.isoformat(),'basis':'SIMULATED_CUTOFF' if exact_cutoff is None else 'EXPLICIT_CUTOFF'}

    def adverse_assignments(self,campaign_id,max_scenarios=10000):
        shorts=[(c,-q) for c,q in self.ledger.positions(campaign_id) if q<0]
        count=1
        for c,n in shorts: count*=n+1
        if count>max_scenarios: raise BrokerError('LIFECYCLE_SCENARIO_BUDGET_EXCEEDED')
        return [{'assignments':[(c.key,n) for (c,_),n in zip(shorts,counts)],'basis':'FINITE_ADVERSE_SCENARIO'}
                for counts in product(*(range(n+1) for c,n in shorts))]

