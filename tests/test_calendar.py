from datetime import date, datetime, timedelta, timezone
import pytest
from qqq_sim.replay.calendar import CalendarService, CalendarError, NY, add_calendar_days, add_elapsed_seconds, easter, local_datetime
from qqq_sim.replay.ordering import before_decision_barrier, merge_events
from qqq_sim.data.models import SourceEvent


def test_regular_early_close_and_closed_slots():
    calendar = CalendarService()
    normal = calendar.minute_labels(date(2026, 11, 9))
    assert len(normal) == 390 and normal[0].strftime("%H:%M") == "09:30" and normal[-1].strftime("%H:%M") == "15:59"
    early = calendar.minute_labels(date(2026, 11, 27))
    assert len(early) == 210 and early[-1].strftime("%H:%M") == "12:59"
    assert calendar.minute_labels(date(2026, 11, 26)) == ()
    assert calendar.minute_labels(date(2012, 10, 29)) == ()
    assert calendar.minute_labels(date(2025, 1, 9)) == ()
    assert calendar.session(date(2026, 11, 9)).scenario_flag


def test_new_year_saturday_does_not_close_previous_friday():
    assert CalendarService().session(date(2021, 12, 31)).status == "OPEN"


def test_dst_rejects_ambiguous_and_nonexistent_input():
    with pytest.raises(CalendarError, match="AMBIGUOUS"):
        local_datetime(datetime(2026, 11, 1, 1, 30))
    with pytest.raises(CalendarError, match="NONEXISTENT"):
        local_datetime(datetime(2026, 3, 8, 2, 30))
    assert local_datetime(datetime.fromisoformat("2026-11-01T01:30:00-05:00")).fold == 1


def test_clock_duration_differs_from_calendar_day_over_dst():
    start = datetime(2026, 3, 7, 12, tzinfo=NY)
    assert add_calendar_days(start, 1).hour == 12
    assert add_elapsed_seconds(start, 86400).hour == 13


def test_step_counts_scheduled_slots_and_crosses_holiday():
    calendar = CalendarService()
    start = datetime(2026, 11, 25, 15, 59, tzinfo=NY)
    target = calendar.advance_trading_minutes(start, 2)
    assert target == datetime(2026, 11, 27, 9, 31, tzinfo=NY)
    assert calendar.advance_trading_minutes(datetime(2026, 11, 9, 9, 30, tzinfo=NY), 1).minute == 31


def test_settlement_banking_and_stock_historical_lag():
    calendar = CalendarService()
    assert calendar.settlement_date(date(2026, 11, 10)) == date(2026, 11, 12)
    assert calendar.stock_settlement_date(date(2017, 9, 1), historical=True) == date(2017, 9, 7)
    assert calendar.stock_settlement_date(date(2024, 5, 24), historical=True) == date(2024, 5, 29)
    assert calendar.stock_settlement_date(date(2024, 5, 28), historical=True) == date(2024, 5, 29)
    assert easter(2026) == date(2026, 4, 5)


def test_barrier_precedes_same_timestamp_raw_events():
    decision = datetime(2026, 11, 9, 10, 1, tzinfo=NY)
    candle = SourceEvent(decision, 20, "a", "candle", None)
    quote = SourceEvent(decision, 30, "b", "quote", None)
    assert list(merge_events([quote], [candle])) == [candle, quote]
    assert before_decision_barrier(candle, decision)
    assert not before_decision_barrier(quote, decision)
