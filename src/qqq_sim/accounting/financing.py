"""Explicit ACT/360 scenario accruals with cumulative cent recognition."""
from datetime import date
from decimal import ROUND_HALF_UP
from ..broker.models import D,BrokerError
from ..broker.settlement import daily_interest,daily_borrow


class FinancingAccruals:
    def __init__(self): self.groups={}; self.days=set(); self.debit_days=set()

    def accrue(self,ledger,campaign_id,at,payment_date,stock_mark=None,borrow_rate=None):
        day=at.date(); key=(campaign_id,day.isoformat())
        if key in self.days: return D(0)
        # Account debit interest is one charge, even when several campaigns
        # have independent stock-borrow accrual records on the same date.
        debit=daily_interest(ledger.settled_cash) if day.isoformat() not in self.debit_days else D(0)
        shares=ledger.shares(campaign_id)
        borrow=daily_borrow(shares,stock_mark,borrow_rate) if shares<0 else D(0)
        month=f'{day.year:04d}-{day.month:02d}'
        total=D(0)
        for kind,raw in [('DEBIT_INTEREST',debit),('STOCK_BORROW',borrow)]:
            group=self.groups.setdefault((campaign_id,month,kind),{'raw':D(0),'recognized':D(0)})
            group['raw']+=raw
            rounded=group['raw'].quantize(D('.01'),rounding=ROUND_HALF_UP)
            delta=rounded-group['recognized']; group['recognized']=rounded
            if delta:
                ledger.accrue_expense(f'{kind}:{campaign_id}:{day}',campaign_id,delta,payment_date,at,kind)
                total+=delta
        self.days.add(key)
        self.debit_days.add(day.isoformat())
        return total

    def to_dict(self):
        return {'groups':[{'key':list(key),'raw':str(value['raw']),'recognized':str(value['recognized'])}
                          for key,value in sorted(self.groups.items(),key=lambda item:str(item[0]))],
                'days':[list(key) for key in sorted(self.days,key=str)],'debit_days':sorted(self.debit_days)}

    @classmethod
    def from_dict(cls,record):
        obj=cls()
        obj.groups={tuple(item['key']):{'raw':D(item['raw']),'recognized':D(item['recognized'])} for item in record.get('groups',[])}
        obj.days={tuple(item) for item in record.get('days',[])}
        obj.debit_days=set(record.get('debit_days',[]))
        return obj

