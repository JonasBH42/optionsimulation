"""Serialized local execution facade. Callers persist each completed event batch."""
from __future__ import annotations
import copy
from dataclasses import replace
from datetime import date,datetime,timedelta
from decimal import ROUND_CEILING,ROUND_FLOOR
from zoneinfo import ZoneInfo
from .models import (D,Contract,Quote,OrderLeg,OrderDraft,Order,Fill,BrokerError,StockReductionOrder,cents,
                     decimal,money,encode,fingerprint,require_aware)
from .fees import FeeEngine,FeeProfile
from .margin import MarginEngine,WorkBudget
from .risk_checks import certify_campaign,certify_session
from .settlement import settlement_date
from .lifecycle import LifecycleEngine
from ..accounting.ledger import Ledger,Campaign

NY=ZoneInfo('America/New_York')


class Broker:
    def __init__(self,initial_cash=D(5000),strict=True,
                 operational_model='SIMULTANEOUS_DELIVERY_HEDGE_V1',
                 account_type='MARGIN',fee_profile=None,settlement_holidays=(),settlement_day_overrides=None):
        self.ledger=Ledger(initial_cash); self.fees=FeeEngine(fee_profile)
        from ..accounting.financing import FinancingAccruals
        self.financing=FinancingAccruals()
        self.margin=MarginEngine(); self.orders={}; self.stock_orders={}; self.quotes={}; self.consumed={}
        self.commands={}; self.strict=strict; self.operational_model=operational_model
        self.account_type=account_type; self.settlement_holidays=set(settlement_holidays)
        self.settlement_day_overrides=dict(settlement_day_overrides or {})
        self.sequence=0; self.bp_cap=D(5000); self.loss_cap=D(1000)
        self.session_loss_cap=D(1000)

    @property
    def lifecycle(self): return LifecycleEngine(self.ledger,self.fees)

    def _new_campaign(self,draft):
        return Campaign(draft.campaign_id,draft.deadline,draft.hard_cutoff,draft.operational_model,
                        draft.standing_protection,draft.expense_budget,draft.target_fraction)

    @staticmethod
    def _quote_map(quotes):
        if isinstance(quotes,dict): return {q.contract.key:q for q in quotes.values()}
        return {q.contract.key:q for q in quotes}

    def submit(self,draft:OrderDraft,at:datetime,quotes=(),command_id=None,work_budget=None):
        require_aware(at)
        command_id=command_id or fingerprint({'draft':draft,'at':at,'sequence':self.sequence})
        if command_id in self.commands: return self.orders[self.commands[command_id]]
        existing_campaign=self.ledger.campaigns.get(draft.campaign_id)
        if existing_campaign is not None and existing_campaign.target_realized:
            raise BrokerError('CAMPAIGN_ALREADY_SUCCEEDED')
        if at>=draft.deadline or draft.deadline>draft.hard_cutoff: raise BrokerError('CAMPAIGN_HORIZON_INVALID')
        if self.strict and draft.order_type!='PROTECTED_LIMIT': raise BrokerError('UNVERIFIED_EXECUTION_BOUND')
        if draft.order_type=='PROTECTED_LIMIT' and draft.max_total_fees<=0: raise BrokerError('INVALID_FEE_CAP')
        supplied=self._quote_map(quotes)
        visible=dict(self.quotes); visible.update(supplied)
        held_expiries={contract.expiry for contract,quantity in self.ledger.positions(draft.campaign_id)}
        for leg in draft.legs:
            c=leg.contract
            if c.symbol!='QQQ' or c.multiplier!=100 or c.deliverable!='QQQ_STANDARD_100_SHARES': raise BrokerError('UNSUPPORTED_CONTRACT_METADATA')
            if c.expiry<at.astimezone(NY).date() or c.last_trade_at and at>=c.last_trade_at: raise BrokerError('CONTRACT_NOT_TRADABLE')
            if c.key not in visible or visible[c.key].event_at>at or not visible[c.key].valid: raise BrokerError('CURRENT_QUOTE_UNAVAILABLE',contract=c.key)
            opening=self.ledger.opening_quantity(draft.campaign_id,c,leg.quantity)
            if existing_campaign is not None and opening and c.expiry not in held_expiries:
                after_cutoff=(c.fixing_at>existing_campaign.hard_cutoff if c.fixing_at is not None
                              else c.expiry>existing_campaign.hard_cutoff.astimezone(NY).date())
                if after_cutoff:
                    raise BrokerError('ROLL_EXPIRY_EXCEEDS_HARD_CUTOFF',contract=c.key)
                if c.fixing_at is None and c.expiry==existing_campaign.hard_cutoff.astimezone(NY).date():
                    raise BrokerError('ROLL_EXPIRY_TIME_UNVERIFIED',contract=c.key)
            if draft.intent=='CLOSE' and opening: raise BrokerError('CLOSE_EXCEEDS_OWNED_QUANTITY')
            if draft.intent=='CLOSE':
                owned=abs(dict((x.key,q) for x,q in self.ledger.positions(draft.campaign_id)).get(c.key,0))
                reserved=sum(abs(l.quantity) for o in self.orders.values() if o.status in ('WORKING','PARTIALLY_FILLED')
                             and o.draft.campaign_id==draft.campaign_id and o.draft.intent=='CLOSE'
                             for l in o.draft.ratio if l.contract.key==c.key)*1
                # Per-ratio reservation quantities are expanded by remaining units.
                reserved=sum(abs(l.quantity)*o.remaining_units for o in self.orders.values() if o.status in ('WORKING','PARTIALLY_FILLED')
                             and o.draft.campaign_id==draft.campaign_id and o.draft.intent=='CLOSE'
                             for l in o.draft.ratio if l.contract.key==c.key)
                if abs(leg.quantity)>owned-reserved: raise BrokerError('CLOSE_QUANTITY_ALREADY_RESERVED')
        campaign=self.ledger.campaigns.get(draft.campaign_id)
        if campaign is None and len({leg.contract.partition for leg in draft.legs})>1:
            raise BrokerError('INITIAL_TARGET_MIXED_EXPIRY_UNSUPPORTED')
        if campaign:
            if campaign.status=='INITIALIZING' and any(o.draft.campaign_id==draft.campaign_id and o.status in ('WORKING','PARTIALLY_FILLED','DATA_BLOCKED') for o in self.orders.values()):
                raise BrokerError('CAMPAIGN_INITIALIZATION_IN_PROGRESS')
            if campaign.target_realized: raise BrokerError('CAMPAIGN_ALREADY_SUCCEEDED')
            if draft.deadline!=campaign.deadline or draft.hard_cutoff!=campaign.hard_cutoff:
                raise BrokerError('FROZEN_CAMPAIGN_HORIZON_CHANGED')
            if at>=campaign.hard_cutoff: raise BrokerError('CAMPAIGN_CUTOFF_REACHED')
        order_id=f'order:{fingerprint({"command":command_id,"sequence":self.sequence})[:24]}'
        order=Order(order_id,draft,at,self.sequence,
                    {l.contract.key:visible[l.contract.key].event_at.replace(second=0,microsecond=0).isoformat() for l in draft.legs},draft.units,
                    submission_quote_ids={l.contract.key:visible[l.contract.key].quote_id for l in draft.legs})
        reservation=self._reservation(order,visible,budget=work_budget)
        order.reservation=reservation
        self.orders[order_id]=order; self.commands[command_id]=order_id
        if campaign is None: self.ledger.campaigns[draft.campaign_id]=self._new_campaign(draft)
        self.quotes.update(supplied)
        self.sequence+=1
        self.ledger.append_event('accept:'+order_id,'ORDER_ACCEPTED',at,{'order':order,'reservation':reservation,
                                                                       'label':'Conditional simulated limit' if draft.order_type=='PROTECTED_LIMIT' else draft.order_type})
        return order

    def _reservation(self,order,quotes,budget=None):
        budget=budget or WorkBudget()
        before=self.margin.evaluate(self.ledger.positions(),budget)
        if before.status!='OK': raise BrokerError(before.reason or 'CAPITAL_UNAVAILABLE')
        old_cash=self.ledger.economic_cash; maximum=D(0)
        for units in range(1,order.remaining_units+1):
            trial=Ledger.from_dict(self.ledger.to_dict())
            if order.draft.campaign_id not in trial.campaigns: trial.campaigns[order.draft.campaign_id]=self._new_campaign(order.draft)
            intended_cash=order.draft.limit_cash*units
            fees=order.draft.max_total_fees if order.draft.order_type=='PROTECTED_LIMIT' else self._estimated_fee(order,quotes,units)
            self._preview_position(trial,order,quotes,units,intended_cash,fees)
            capital=self._check_poststate(trial,exclude_order=order.order_id,budget=budget,raising=True)
            increase=max(D(0),capital['collateral']-before.collateral-(trial.economic_cash-old_cash))
            maximum=max(maximum,increase)
        available=self.account_summary({})['available_option_bp']
        if maximum>available: raise BrokerError('INSUFFICIENT_AVAILABLE_CAPITAL',needed=str(maximum),available=str(available))
        if self.account_type=='CASH':
            debit=max(D(0),-order.draft.limit_cash*order.remaining_units)+order.draft.max_total_fees
            existing=sum((o.reservation for o in self.orders.values()),D(0))
            # Outstanding purchase payables have already claimed settled funding.
            # Unsettled sale proceeds cannot release that funding before settlement.
            payable=sum((D(o.amount_cents)/100 for o in self.ledger.obligations.values()
                         if not o.settled and o.amount_cents<0),D(0))
            if debit>self.ledger.settled_cash+payable-existing:
                raise BrokerError('SETTLED_CASH_ONLY_FUNDING_INSUFFICIENT')
        return maximum

    def _preview_position(self,trial,order,quotes,units,premium_cash,fees):
        """Use exact quantities; cash limit supplies conservative package economics."""
        prior=trial.economic_cash
        fee_engine=copy.deepcopy(self.fees)
        for index,leg in enumerate(order.draft.ratio):
            q=leg.quantity*units; quote=quotes[leg.contract.key]
            price=self._side(quote,q)
            components=fee_engine.assess_option(order.order_id,leg.contract,q,price,
                                                trial.opening_quantity(order.draft.campaign_id,leg.contract,q),order.submitted_at.astimezone(NY).date())
            fee=fees if index==0 else D(0)
            fill=Fill(f'preview:{order.order_id}:{units}:{index}',order.order_id,order.draft.campaign_id,leg.contract,q,price,
                      order.submitted_at,quote.quote_id,fee,components)
            trial.apply_fill(fill,order.submitted_at.date())
        actual=trial.economic_cash-prior
        adjustment=premium_cash-fees-actual
        trial.settled_cash_cents+=int(money(adjustment)*100)
        trial.campaigns[order.draft.campaign_id].trade_cash+=adjustment

    def _estimated_fee(self,order,quotes,units):
        engine=copy.deepcopy(self.fees); total=D(0)
        for leg in order.draft.ratio:
            q=leg.quantity*units
            total+=sum(engine.assess_option(order.order_id,leg.contract,q,self._side(quotes[leg.contract.key],q),
                                             self.ledger.opening_quantity(order.draft.campaign_id,leg.contract,q),order.submitted_at.date()).values())
        return total

    @staticmethod
    def _side(quote,quantity):
        price=quote.ask if quantity>0 else quote.bid
        return price.quantize(D('.01'),rounding=ROUND_CEILING if quantity>0 else ROUND_FLOOR)

    def _check_poststate(self,ledger,exclude_order=None,budget=None,raising=False):
        assessment=self.margin.evaluate(ledger.positions(),budget)
        if assessment.status!='OK':
            if raising: raise BrokerError(assessment.reason or 'MARGIN_UNAVAILABLE')
            return {'status':'FAILED','reason':assessment.reason}
        reserves=sum((o.reservation for oid,o in self.orders.items() if oid!=exclude_order and o.status in ('WORKING','PARTIALLY_FILLED')),D(0))
        available=ledger.economic_cash-assessment.collateral-reserves
        project=self._project_bp(ledger,assessment)+reserves
        reason=None
        if available<0: reason='INSUFFICIENT_AVAILABLE_CAPITAL'
        elif project>self.bp_cap: reason='PROJECT_BP_CAP_EXCEEDED'
        elif ledger.shares(): reason='UNSUPPORTED_STOCK_EXPOSURE'
        certificate=certify_session(ledger,self.session_loss_cap)
        if self.strict and certificate['status']!='VERIFIED_WITHIN_DECLARED_MODEL': reason=certificate.get('reason') or 'RISK_CERTIFICATE_UNAVAILABLE'
        for cid,c in certificate['campaigns'].items():
            if self.strict and (c.loss_bound is None or c.loss_bound>self.loss_cap): reason=c.reason or 'CAMPAIGN_LOSS_CAP_EXCEEDED'
        if reason and raising: raise BrokerError(reason)
        return {'status':'OK' if reason is None else 'FAILED','reason':reason,'collateral':assessment.collateral,
                'available':available,'project_bp_used':project,'risk_certificate':certificate}

    @staticmethod
    def _project_bp(ledger,assessment):
        total=D(0)
        for component in assessment.components:
            basis=D(0)
            for key,qty in component.legs:
                matching=[lot for lot in ledger.lots if lot.contract.key==key]
                allqty=sum(abs(lot.quantity) for lot in matching)
                if allqty:
                    basis+=sum((lot.basis+lot.opening_fees for lot in matching),D(0))*D(abs(qty))/allqty
            total+=max(D(0),component.collateral+basis)
        return total

    def on_quotes(self,quotes,at:datetime):
        require_aware(at); updates=self._quote_map(quotes)
        if any(q.event_at>at for q in updates.values()): raise BrokerError('FUTURE_QUOTE_EVENT')
        changed={key for key,q in updates.items() if key not in self.quotes or self.quotes[key].quote_id!=q.quote_id}
        self.quotes.update(updates); fills=[]
        for order in sorted(self.orders.values(),key=lambda o:(o.submitted_at,o.submitted_sequence,o.order_id)):
            if order.status not in ('WORKING','PARTIALLY_FILLED','DATA_BLOCKED'): continue
            selected={l.contract.key for l in order.draft.ratio}
            if not selected.intersection(changed): continue
            eligible=True; selected_quotes=[]
            for leg in order.draft.ratio:
                quote=self.quotes.get(leg.contract.key)
                if quote is None or not quote.valid: eligible=False; break
                if quote.contract.last_trade_at and at>=quote.contract.last_trade_at: eligible=False; break
                if quote.contract.expiry<at.astimezone(NY).date(): eligible=False; break
                if quote.precision=='minute':
                    old=datetime.fromisoformat(order.submission_quote_minutes[leg.contract.key])
                    if quote.event_at.replace(second=0,microsecond=0)<=old: eligible=False; break
                    # event_at is the earliest possible instant in the recorded
                    # minute; availability is handled by the causal event source.
                    age=(at-quote.event_at).total_seconds()
                else:
                    if quote.event_at<order.submitted_at: eligible=False; break
                    if quote.event_at==order.submitted_at:
                        # Raw-event phase30 at D follows a phase26 command at
                        # the exclusive completed-bar barrier. Quote identity
                        # proves it was not already seen at submission; there
                        # is no invented epsilon or shifted market timestamp.
                        if not order.submission_quote_ids or quote.quote_id==order.submission_quote_ids.get(leg.contract.key):
                            eligible=False; break
                    age=(at-quote.event_at).total_seconds()
                if age>60 or age<0: eligible=False; break
                selected_quotes.append(quote)
            if not eligible: continue
            times=[q.event_at for q in selected_quotes]
            latest_possible=[q.event_at+timedelta(minutes=1) if q.precision=='minute' else q.event_at for q in selected_quotes]
            if (max(latest_possible)-min(times)).total_seconds()>60: continue
            units=order.remaining_units; size_unverified=False
            signature=[]
            for leg,quote in zip(order.draft.ratio,selected_quotes):
                side='ask' if leg.quantity>0 else 'bid'; size=quote.ask_size if leg.quantity>0 else quote.bid_size
                used=self.consumed.get(quote.quote_id+'|'+side,0)
                residual=None if size is None else max(0,size-used)
                signature.append((quote.quote_id,side,quote.event_at.isoformat(),residual))
                if size is None: size_unverified=True
                else: units=min(units,residual//abs(leg.quantity))
            signature_hash=fingerprint(signature)
            if signature_hash==order.last_signature: continue
            order.last_signature=signature_hash; order.last_attempt_at=at
            if size_unverified and order.draft.strict_depth:
                order.observations.append({'at':at.isoformat(),'reason':'INSUFFICIENT_QUOTE_DEPTH'}); continue
            if units<=0: continue
            package_cash=-sum((leg.quantity*units*leg.contract.multiplier*self._side(q,leg.quantity) for leg,q in zip(order.draft.ratio,selected_quotes)),D(0))
            if order.draft.order_type!='MARKET' and package_cash<order.draft.limit_cash*units:
                order.observations.append({'at':at.isoformat(),'reason':'LIMIT_NOT_MET'}); continue
            new_fees=copy.deepcopy(self.fees); trial=Ledger.from_dict(self.ledger.to_dict()); proposed=[]
            for index,(leg,quote) in enumerate(zip(order.draft.ratio,selected_quotes)):
                quantity=leg.quantity*units; price=self._side(quote,quantity)
                components=new_fees.assess_option(order.order_id,leg.contract,quantity,price,
                                                  trial.opening_quantity(order.draft.campaign_id,leg.contract,quantity),at.astimezone(NY).date())
                fee=sum(components.values(),D(0))
                fill=Fill(f'fill:{order.order_id}:{order.filled_units}:{index}:{quote.quote_id}',order.order_id,order.draft.campaign_id,
                          leg.contract,quantity,price,at,quote.quote_id,fee,components,
                          'SIMULATED_FROM_QUOTES;SIZE_UNVERIFIED_SIMULATION' if size_unverified else 'SIMULATED_FROM_QUOTES;OBSERVED_SIZE')
                due=settlement_date(at.astimezone(NY).date(),'OPTION',self.fees.profile.rule_asof,self.settlement_holidays,self.settlement_day_overrides)
                trial.apply_fill(fill,due); proposed.append(fill)
            total_fees=sum((fill.fees for fill in proposed),D(0))
            cap=self._check_poststate(trial,exclude_order=order.order_id)
            if order.draft.order_type=='PROTECTED_LIMIT':
                reason=None
                if order.fees+total_fees>order.draft.max_total_fees: reason='ORDER_TOTAL_FEE_CAP_EXCEEDED'
                elif cap['status']!='OK': reason=cap['reason']
                if reason:
                    order.observations.append({'at':at.isoformat(),'reason':'CONDITIONAL_FILL_NOT_ELIGIBLE','detail':reason}); continue
            old_remaining=order.remaining_units
            order.filled_units+=units; order.remaining_units-=units; order.fees+=total_fees
            order.reservation=money(order.reservation*D(order.remaining_units)/old_remaining)
            order.status='FILLED' if order.remaining_units==0 else 'PARTIALLY_FILLED'
            self.ledger=trial; self.fees=new_fees
            for leg,quote in zip(order.draft.ratio,selected_quotes):
                side='ask' if leg.quantity>0 else 'bid'; key=quote.quote_id+'|'+side
                self.consumed[key]=self.consumed.get(key,0)+abs(leg.quantity)*units
            if not order.remaining_units and self.ledger.campaigns[order.draft.campaign_id].status=='INITIALIZING':
                self.ledger.initialize_target(order.draft.campaign_id,fingerprint(selected_quotes),at)
            self.sequence+=1; fills.extend(proposed)
        return fills

    def cancel(self,order_id,at=None,command_id=None):
        if order_id in self.stock_orders:
            order=self.stock_orders[order_id]
            if order.status=='WORKING':
                order.status='CANCELED'; self.sequence+=1
                self.ledger.append_event('cancel:'+order_id,'STOCK_ORDER_CANCELED',at or order.submitted_at,{'order_id':order_id})
            if command_id: self.commands[command_id]=order_id
            return order
        order=self.orders[order_id]
        if command_id and command_id in self.commands: return order
        if order.status not in ('FILLED','CANCELED','EXPIRED','REJECTED'):
            order.status='CANCELED'; order.reservation=D(0); self.sequence+=1
            at=at or order.submitted_at
            self.ledger.append_event('cancel:'+order_id,'ORDER_CANCELED',at,{'order_id':order_id,'filled_units':order.filled_units})
            campaign=self.ledger.campaigns[order.draft.campaign_id]
            if campaign.status=='INITIALIZING' and order.filled_units: self.ledger.initialize_target(campaign.campaign_id,at=at)
            elif campaign.status=='INITIALIZING' and not self.ledger.positions(campaign.campaign_id):
                campaign.status='CANCELED'; campaign.expense_remaining=D(0)
        if command_id: self.commands[command_id]=order_id
        return order

    def expire_orders(self,at,session_close=None):
        expired=[]
        for order in sorted(self.orders.values(),key=lambda o:(o.submitted_at,o.submitted_sequence,o.order_id)):
            if order.status not in ('WORKING','PARTIALLY_FILLED','DATA_BLOCKED'): continue
            end=(order.draft.tif=='DAY' and session_close is not None and at>=session_close)
            cutoff=at>=order.draft.hard_cutoff or any(l.contract.last_trade_at and at>=l.contract.last_trade_at for l in order.draft.legs)
            if end or cutoff:
                self.cancel(order.order_id,at); order.status='EXPIRED'; expired.append(order.order_id)
        return expired

    def settle_due(self,at):
        ordered=sorted(self.ledger.obligations.items(),key=lambda item:(item[1].due_date,item[0]))
        return [key for key,o in ordered if not o.settled and o.due_date<=at.date() and self.ledger.settle(key,at)]

    def submit_stock_reduction(self,campaign_id,quantity,minimum_cash_per_share,at,command_id,max_total_fees=D(50)):
        """Explicit bounded reduction of assignment-created stock; never a new short."""
        require_aware(at)
        if command_id in self.commands:
            return self.stock_orders[self.commands[command_id]]
        shares=self.ledger.shares(campaign_id)
        reserved=sum(abs(o.remaining) for o in self.stock_orders.values() if o.campaign_id==campaign_id and o.status=='WORKING')
        if not isinstance(quantity,int) or quantity*shares>=0 or abs(quantity)>abs(shares)-reserved:
            raise BrokerError('STOCK_ORDER_MUST_REDUCE_EXISTING_EXPOSURE')
        limit=decimal(minimum_cash_per_share)
        if limit%D('.01'): raise BrokerError('INVALID_STOCK_TICK')
        order_id='stock-order:'+fingerprint({'command':command_id,'sequence':self.sequence})[:24]
        order=StockReductionOrder(order_id,campaign_id,quantity,quantity,limit,at,decimal(max_total_fees),submitted_sequence=self.sequence)
        self.stock_orders[order_id]=order; self.commands[command_id]=order_id; self.sequence+=1
        self.ledger.append_event('accept:'+order_id,'STOCK_REDUCTION_ACCEPTED',at,order)
        return order

    def on_stock_quote(self,bid,ask,at,quote_id,bid_size=None,ask_size=None):
        """Only an explicit later stock quote can execute stock; bars are not quotes."""
        require_aware(at); bid,ask=decimal(bid),decimal(ask)
        if bid<0 or ask<=0 or bid>ask: return []
        fills=[]
        for order in sorted(self.stock_orders.values(),key=lambda o:(o.submitted_at,o.submitted_sequence,o.order_id)):
            if order.status!='WORKING' or at<=order.submitted_at or quote_id==order.last_quote_id: continue
            order.last_quote_id=quote_id
            sign=1 if order.remaining>0 else -1
            price=ask if sign>0 else bid; side='stock-ask' if sign>0 else 'stock-bid'
            if -sign*price<order.minimum_cash_per_share: continue
            size=ask_size if sign>0 else bid_size
            available=abs(order.remaining) if size is None else max(0,size-self.consumed.get(quote_id+'|'+side,0))
            quantity=sign*min(abs(order.remaining),available)
            if not quantity: continue
            fees=copy.deepcopy(self.fees)
            components=fees.assess_stock(order.order_id,quantity,price,at.astimezone(NY).date())
            total=sum(components.values(),D(0))
            if order.fees+total>order.max_total_fees: continue
            event_id=f'stock-fill:{order.order_id}:{quote_id}'
            self.lifecycle._stock_transfer(event_id,order.campaign_id,quantity,quantity*price)
            c=self.ledger.campaigns[order.campaign_id]
            cash=-quantity*price; c.trade_cash+=cash; c.expenses+=total; c.expense_remaining-=total
            self.ledger.settled_cash_cents-=cents(total)
            due=settlement_date(at.astimezone(NY).date(),'STOCK',self.fees.profile.rule_asof,self.settlement_holidays,self.settlement_day_overrides)
            self.ledger.create_obligation(event_id,order.campaign_id,cash,due,'STOCK_TRADE')
            payload={'campaign_id':order.campaign_id,'quantity':quantity,'stock_quantity_delta':quantity,'price':price,
                     'fees':total,'fee_components':components,'quote_id':quote_id,'assumption':'SIMULATED_FROM_EXPLICIT_STOCK_QUOTE'}
            self.ledger.append_event(event_id,'STOCK_FILL',at,payload,
                                     [{'campaign_id':order.campaign_id,'amount_cents':cents(cash),'bucket':'unsettled','category':'STOCK_TRADE','obligation_id':event_id,'due_date':due.isoformat()},
                                      {'campaign_id':order.campaign_id,'amount_cents':-cents(total),'bucket':'settled','category':'STOCK_FEES'}])
            self.ledger._check_realization(c,at)
            self.fees=fees; order.fees+=total; order.remaining-=quantity
            order.status='FILLED' if not order.remaining else 'WORKING'
            self.consumed[quote_id+'|'+side]=self.consumed.get(quote_id+'|'+side,0)+abs(quantity)
            self.sequence+=1; fills.append(payload)
        return fills

    def account_summary(self,marks=None,stock_mark=None):
        marks=marks if marks is not None else {k:q.midpoint for k,q in self.quotes.items()}
        assessment=self.margin.evaluate(self.ledger.positions())
        reserves=sum((o.reservation for o in self.orders.values() if o.status in ('WORKING','PARTIALLY_FILLED','DATA_BLOCKED')),D(0))
        equity=self.ledger.equity(marks,stock_mark)
        collateral=assessment.collateral
        free=None if collateral is None else self.ledger.economic_cash-collateral-reserves
        return {'profile':'OPTIONS_COLLATERAL_QQQ_V1','rule_mode':self.fees.profile.mode,'rule_asof':self.fees.profile.rule_asof.isoformat(),
                'qualification':'Conservative simulation profile; no account-specific broker parity',
                'economic_cash':self.ledger.economic_cash,'settled_cash':self.ledger.settled_cash,
                'collateral':collateral,'working_order_reserve':reserves,
                'available_option_bp':max(D(0),free) if free is not None else None,
                'deficit':max(D(0),-free) if free is not None else None,
                'project_bp_used':self._project_bp(self.ledger,assessment)+reserves if collateral is not None else None,
                'equity':equity['equity'],'equity_known_subtotal':equity['known_subtotal'],'missing_marks':equity['missing'],
                'account_economic_pnl':None if equity['equity'] is None else equity['equity']-D(self.ledger.initial_cash_cents)/100-self.ledger.external_contributions,
                'stock_bp':None,'stock_bp_reason':'NOT_APPLICABLE_PROFILE','sma':None,'sma_reason':'NOT_APPLICABLE_PROFILE',
                'capital_status':assessment.status,'capital_reason':assessment.reason,
                'fees':sum((c.expenses for c in self.ledger.campaigns.values()),D(0))}

    def to_dict(self):
        return encode({'schema_version':1,'ledger':self.ledger.to_dict(),'fee_groups':self.fees.to_dict(),
                       'financing_accruals':self.financing.to_dict(),
                       'fee_profile':self.fees.profile,'orders':self.orders,'stock_orders':self.stock_orders,'quotes':self.quotes,'consumed':self.consumed,
                       'commands':self.commands,'strict':self.strict,'operational_model':self.operational_model,
                       'account_type':self.account_type,'settlement_holidays':sorted(self.settlement_holidays),
                       'settlement_day_overrides':{day.isoformat():value for day,value in self.settlement_day_overrides.items()},
                       'sequence':self.sequence,'bp_cap':self.bp_cap,'loss_cap':self.loss_cap,'session_loss_cap':self.session_loss_cap})

    @classmethod
    def from_dict(cls,data):
        fp=dict(data.get('fee_profile',{}))
        if fp:
            fp['rule_asof']=date.fromisoformat(fp['rule_asof'])
            for key in fp:
                if key not in ('id','rule_asof','mode'): fp[key]=D(fp[key])
        obj=cls(strict=data['strict'],operational_model=data['operational_model'],account_type=data['account_type'],fee_profile=FeeProfile(**fp),
                settlement_holidays=[date.fromisoformat(d) for d in data.get('settlement_holidays',[])],
                settlement_day_overrides={date.fromisoformat(d):value for d,value in data.get('settlement_day_overrides',{}).items()})
        obj.ledger=Ledger.from_dict(data['ledger']); obj.fees.load(data['fee_groups'])
        from ..accounting.financing import FinancingAccruals
        obj.financing=FinancingAccruals.from_dict(data.get('financing_accruals',{}))
        for key,raw in data['orders'].items():
            x=dict(raw); draft=dict(x['draft']); draft['legs']=tuple(OrderLeg(Contract.from_dict(l['contract']),l['quantity']) for l in draft['legs'])
            for name in ('deadline','hard_cutoff'): draft[name]=datetime.fromisoformat(draft[name])
            x['draft']=OrderDraft(**draft); x['submitted_at']=datetime.fromisoformat(x['submitted_at'])
            if x.get('last_attempt_at'): x['last_attempt_at']=datetime.fromisoformat(x['last_attempt_at'])
            for name in ('fees','reservation'): x[name]=D(x[name])
            obj.orders[key]=Order(**x)
        for key,raw in data['quotes'].items():
            x=dict(raw); x['contract']=Contract.from_dict(x['contract']); x['event_at']=datetime.fromisoformat(x['event_at']); obj.quotes[key]=Quote(**x)
        for key,raw in data.get('stock_orders',{}).items():
            x=dict(raw); x['submitted_at']=datetime.fromisoformat(x['submitted_at'])
            for name in ('minimum_cash_per_share','max_total_fees','fees'): x[name]=D(x[name])
            obj.stock_orders[key]=StockReductionOrder(**x)
        obj.consumed=dict(data['consumed']); obj.commands=dict(data['commands']); obj.sequence=data['sequence']
        for key in ('bp_cap','loss_cap','session_loss_cap'): setattr(obj,key,D(data[key]))
        return obj

