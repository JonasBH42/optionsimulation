"""Bounded, deterministic cash-collateral component allocation (APP-C06)."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations, product
from .models import Contract, BrokerError, D, decimal


@dataclass(frozen=True)
class PayoffExtrema:
    minimum: object
    maximum: object
    minimum_unbounded: bool = False
    maximum_unbounded: bool = False


def payoff_extrema(positions) -> PayoffExtrema:
    """Exact piecewise-linear terminal payoff on S >= 0; separate infinity flags."""
    positions=[(c,int(q)) for c,q in positions if q]
    expiries={c.partition for c,q in positions}
    if len(expiries)>1: raise BrokerError('MIXED_EXPIRY_PAYOFF_UNSUPPORTED')
    knots=sorted({D(0)}|{c.strike for c,q in positions})
    values=[sum((q*c.multiplier*c.intrinsic(s) for c,q in positions),D(0)) for s in knots]
    tail=sum((q*c.multiplier for c,q in positions if c.right=='C'),0)
    return PayoffExtrema(None if tail<0 else min(values),None if tail>0 else max(values),tail<0,tail>0)


@dataclass
class WorkBudget:
    state_limit: int = 10_000
    request_limit: int = 100_000
    request_nodes: int = 0
    state_nodes: int = 0

    def begin_state(self): self.state_nodes=0
    def consume(self):
        self.state_nodes+=1; self.request_nodes+=1
        if self.state_nodes>self.state_limit or self.request_nodes>self.request_limit:
            raise BrokerError('MARGIN_GROUPING_BUDGET_EXCEEDED')


@dataclass(frozen=True)
class Component:
    name: str
    legs: tuple[tuple[str,int],...]
    collateral: object


@dataclass(frozen=True)
class MarginAssessment:
    status: str
    collateral: object
    components: tuple[Component,...] = ()
    reason: str | None = None
    grouping_nodes: int = 0
    profile: str = 'OPTIONS_COLLATERAL_QQQ_V1'


class MarginEngine:
    def __init__(self): self.cache={}

    def evaluate(self, positions, budget=None):
        budget=budget or WorkBudget(); budget.begin_state()
        inventory={}; contracts={}
        for c,q in positions:
            inventory[c.key]=inventory.get(c.key,0)+int(q); contracts[c.key]=c
        inventory={k:q for k,q in inventory.items() if q}
        if len(inventory)>16 or any(abs(q)>20 for q in inventory.values()):
            return MarginAssessment('INDETERMINATE',None,reason='MARGIN_DIMENSION_LIMIT')
        cache_key=tuple(sorted(inventory.items()))
        if cache_key in self.cache:
            try: budget.consume()
            except BrokerError as error: return MarginAssessment('INDETERMINATE',None,reason=error.code)
            return self.cache[cache_key]
        partitions={}
        for k,q in inventory.items(): partitions.setdefault(contracts[k].partition,[]).append((contracts[k],q))
        result=[]
        try:
            for partition in sorted(partitions, key=str):
                result.extend(self._partition(partitions[partition],budget))
        except BrokerError as error:
            return MarginAssessment('INDETERMINATE' if 'BUDGET' in error.code else 'UNSUPPORTED',
                                    None,reason=error.code,grouping_nodes=budget.state_nodes)
        assessment=MarginAssessment('OK',sum((c.collateral for c in result),D(0)),tuple(result),grouping_nodes=budget.state_nodes)
        self.cache[cache_key]=assessment
        return assessment

    def _partition(self, positions,budget):
        positions=sorted(positions,key=lambda x:x[0].key)
        contracts=[c for c,q in positions]; quantities=tuple(q for c,q in positions)
        n=len(contracts); components=[]
        if any(c.symbol!='QQQ' or c.multiplier!=100 or c.deliverable!='QQQ_STANDARD_100_SHARES' for c in contracts):
            raise BrokerError('UNSUPPORTED_DELIVERABLE')

        def add(name, entries):
            vector=tuple(entries.get(i,0) for i in range(n))
            if any(v and (v*quantities[i]<=0 or abs(v)>abs(quantities[i])) for i,v in enumerate(vector)): return
            extrema=payoff_extrema([(contracts[i],v) for i,v in enumerate(vector) if v])
            if extrema.minimum_unbounded: return
            collateral=max(D(0),-extrema.minimum)
            components.append((name,vector,collateral))

        for i,(c,q) in enumerate(positions):
            if q>0: add('LONG_OPTION',{i:1})
            elif c.right=='P': add('CASH_SECURED_PUT',{i:-1})
        for i,j in combinations(range(n),2):
            if contracts[i].right==contracts[j].right and quantities[i]*quantities[j]<0 and contracts[i].strike!=contracts[j].strike:
                add('VERTICAL',{i:1 if quantities[i]>0 else -1,j:1 if quantities[j]>0 else -1})
        for indices in combinations(range(n),3):
            ids=sorted(indices,key=lambda i:contracts[i].strike)
            a,b,c=ids
            if len({contracts[i].right for i in ids})==1 and contracts[b].strike-contracts[a].strike==contracts[c].strike-contracts[b].strike:
                for sign in (1,-1): add('BUTTERFLY',{a:sign,b:-2*sign,c:sign})
        puts=[i for i,c in enumerate(contracts) if c.right=='P']
        calls=[i for i,c in enumerate(contracts) if c.right=='C']
        for lp,sp in product(puts,puts):
            if contracts[lp].strike>=contracts[sp].strike: continue
            for sc,lc in product(calls,calls):
                if contracts[sp].strike<=contracts[sc].strike<contracts[lc].strike:
                    add('IRON_BUTTERFLY' if contracts[sp].strike==contracts[sc].strike else 'IRON_CONDOR', {lp:1,sp:-1,sc:-1,lc:1})
        components=sorted(set(components),key=lambda c:(c[0],c[1]))
        # Exact single-template recognition is one operation.
        direct=[]
        for name,vector,collateral in components:
            ratios={quantities[i]//v for i,v in enumerate(vector) if v and quantities[i]%v==0}
            if len(ratios)==1:
                units=next(iter(ratios))
                if units>0 and all(vector[i]*units==quantities[i] for i in range(n)):
                    direct.append((collateral*units,name,vector,units))
        if direct:
            best=min(direct,key=lambda c:(c[0],c[1],c[2])); budget.consume()
            # Payoff lower bound proves optimality for a whole supported template.
            return [Component(best[1],tuple((contracts[i].key,v*best[3]) for i,v in enumerate(best[2]) if v),best[0])]

        memo={}
        def solve(remaining):
            budget.consume()
            if not any(remaining): return D(0),()
            if remaining in memo: return memo[remaining]
            first=next(i for i,q in enumerate(remaining) if q)
            candidates=[]
            for name,vector,collateral in components:
                if not vector[first]: continue
                if any(v and (remaining[i]*v<=0 or abs(v)>abs(remaining[i])) for i,v in enumerate(vector)): continue
                max_units=min(abs(remaining[i]//v) for i,v in enumerate(vector) if v)
                for units in range(1,max_units+1):
                    rest=tuple(q-units*vector[i] for i,q in enumerate(remaining))
                    sub=solve(rest)
                    if sub is not None:
                        part=(name,tuple((contracts[i].key,v*units) for i,v in enumerate(vector) if v),collateral*units)
                        candidates.append((collateral*units+sub[0],tuple(sorted((part,)+sub[1],key=str))))
            best=min(candidates,key=lambda x:(x[0],str(x[1]))) if candidates else None
            memo[remaining]=best
            return best
        solution=solve(quantities)
        if solution is None: raise BrokerError('UNSUPPORTED_MARGIN_STRUCTURE')
        return [Component(*item) for item in solution[1]]


def naked_component(contract, quantity, premium, spot):
    """Inspection only; does not confer permission or available buying power."""
    p=decimal(premium); s=decimal(spot); n=abs(quantity); m=contract.multiplier
    otm=max(D(0),contract.strike-s if contract.right=='C' else s-contract.strike)
    second=D('.10')*(s if contract.right=='C' else contract.strike)
    return {'requirement':n*max(m*(p+max(D('.20')*s-otm,second)),D(250)),
            'status':'INSPECTION_ONLY','permission':False}

