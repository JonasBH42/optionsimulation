"""Explicit scenario settlement calendars and financing accruals."""
from datetime import date,timedelta
from .models import D,decimal,BrokerError


def stock_settlement_lag(rule_date):
    if rule_date>=date(2024,5,28): return 1
    if rule_date>=date(2017,9,5): return 2
    return 3


def settlement_date(trade_date,kind='OPTION',rule_asof=date(2026,10,28),holidays=frozenset(),overrides=None):
    from qqq_sim.replay.calendar import CalendarService
    calendar=CalendarService()
    overrides=overrides or {}
    def eligible(day):
        if day in overrides: return bool(overrides[day])
        return calendar.is_settlement_day(day) and day not in holidays
    lag=1 if kind=='OPTION' else stock_settlement_lag(rule_asof)
    current=trade_date
    for _ in range(lag):
        current+=timedelta(days=1)
        while not eligible(current): current+=timedelta(days=1)
    return current


def financing_rate(debit):
    debit=decimal(debit)
    for ceiling,rate in [(25000,'.11'),(50000,'.105'),(100000,'.10'),(250000,'.095'),
                          (500000,'.09'),(1000000,'.085')]:
        if debit<ceiling: return D(rate)
    return D('.08')


def daily_interest(settled_cash):
    debit=max(D(0),-decimal(settled_cash))
    return debit*financing_rate(debit)/360


def daily_borrow(short_shares,stock_mark,annual_rate):
    if annual_rate is None: raise BrokerError('BORROW_RATE_MISSING')
    if stock_mark is None: raise BrokerError('STOCK_MARK_MISSING')
    return abs(short_shares)*decimal(stock_mark)*decimal(annual_rate)/360

