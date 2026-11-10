"""Cumulative decimal charges: APP-C05 fixed simulation profile."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP
from .models import Contract, D, decimal, money, PROFILE_ID, fingerprint


@dataclass(frozen=True)
class FeeProfile:
    id: str = PROFILE_ID
    rule_asof: date = date(2026, 10, 28)
    mode: str = 'fixed_rules_counterfactual'
    opening_commission: Decimal = D('1')
    opening_cap: Decimal = D('10')
    clearing: Decimal = D('.10')
    orf: Decimal = D('.02')
    option_taf: Decimal = D('.00329')
    stock_taf: Decimal = D('.000195')
    stock_taf_cap: Decimal = D('9.79')
    stock_clearing: Decimal = D('.0008')
    section31: Decimal = D('0.00002060')
    exercise_assignment: Decimal = D('5')

    @property
    def hash(self): return fingerprint(self)

    def taf_rate(self, stock=False, economic_date=None):
        d = self.rule_asof if self.mode == 'fixed_rules_counterfactual' else economic_date
        if d is None: raise ValueError('BROKER_RULE_HISTORY_MISSING')
        return D(0) if date(2026,10,1) <= d <= date(2026,12,31) else (self.stock_taf if stock else self.option_taf)


class FeeEngine:
    def __init__(self, profile: FeeProfile | None = None):
        self.profile = profile or FeeProfile()
        self.groups: dict[str, dict[str, Decimal]] = {}

    def _accrue(self, key, amount, ceiling=False, cap=None):
        key = '|'.join(map(str, key))
        group = self.groups.setdefault(key, {'raw': D(0), 'posted': D(0)})
        group['raw'] += decimal(amount)
        total = min(group['raw'], cap) if cap is not None else group['raw']
        rounded = total.quantize(D('.01'), rounding=ROUND_CEILING if ceiling else ROUND_HALF_UP)
        delta = rounded-group['posted']
        group['posted'] = rounded
        return delta

    def assess_option(self, order_id, contract: Contract, quantity: int, price,
                      opening_quantity: int, trade_date: date):
        p = self.profile; n = abs(quantity); price = decimal(price)
        base = (order_id, contract.key)
        daily = base + (trade_date.isoformat(),)
        result = {
            'commission': self._accrue(base+('opening',), abs(opening_quantity)*p.opening_commission, cap=p.opening_cap),
            'clearing': self._accrue(daily+('clearing',), n*p.clearing),
            'orf': self._accrue(daily+('orf',), n*p.orf),
            'taf': self._accrue(daily+('taf',), n*p.taf_rate(economic_date=trade_date) if quantity < 0 else 0, ceiling=True),
            'section31': self._accrue(daily+('section31',), n*contract.multiplier*price*p.section31 if quantity < 0 else 0, ceiling=True),
        }
        return result

    def assess_stock(self, order_id, quantity, price, trade_date):
        p=self.profile; n=abs(quantity); key=(order_id,'QQQ_STOCK',trade_date.isoformat())
        return {
            'clearing': self._accrue(key+('clearing',), n*p.stock_clearing),
            'taf': self._accrue(key+('taf',), n*p.taf_rate(True,trade_date) if quantity < 0 else 0, True, p.stock_taf_cap),
            'section31': self._accrue(key+('section31',), n*decimal(price)*p.section31 if quantity < 0 else 0, True),
        }

    def assess_delivery(self, contract, economic_date, batch_id, kind):
        key=('delivery', economic_date.isoformat(), contract.key, batch_id, kind)
        # Fixed group charge, independent of quantity; subsequent fragments add zero.
        text='|'.join(map(str,key))
        return D(0) if text in self.groups else self._accrue(key,self.profile.exercise_assignment)

    def to_dict(self):
        return {key:{k:str(v) for k,v in values.items()} for key,values in self.groups.items()}

    def load(self, groups):
        self.groups={key:{k:D(v) for k,v in values.items()} for key,values in groups.items()}


def allocate_cents(amount, weighted_ids):
    """Largest remainder, stable IDs; exact cents sum, including signed refunds."""
    total_cents=int(money(amount)*100); sign=1 if total_cents >= 0 else -1
    total_cents=abs(total_cents)
    entries=sorted((str(k),abs(decimal(w))) for k,w in weighted_ids)
    denominator=sum((w for _,w in entries),D(0))
    if not denominator:
        if total_cents: raise ValueError('Cannot allocate nonzero fees to zero quantity')
        return {k:D(0) for k,_ in entries}
    raw={k:D(total_cents)*w/denominator for k,w in entries}
    floor={k:int(v) for k,v in raw.items()}
    remainder=total_cents-sum(floor.values())
    for k in sorted(raw,key=lambda k:(-(raw[k]-floor[k]),k))[:remainder]: floor[k]+=1
    return {k:D(sign*n)/100 for k,n in floor.items()}

