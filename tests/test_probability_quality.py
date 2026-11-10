from dataclasses import replace
from datetime import datetime, timedelta
import pytest
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import write_synthetic_dataset, synthetic_mappings
from qqq_sim.replay.calendar import NY
from qqq_sim.risk.contracts import RiskError
from qqq_sim.services.adapters import probability_snapshot_quality


@pytest.fixture
def snapshot(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / "data", minutes=4)
    data = HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())
    return data.snapshot(datetime(2026, 11, 9, 9, 34, tzinfo=NY))


def changed(entry, **fields):
    return replace(entry, latest_event=replace(entry.latest_event, **fields))


def test_historical_quality_requires_current_completed_authoritative_candle(snapshot):
    assert probability_snapshot_quality(snapshot, snapshot.quotes[:2])["profile"] == "REPLAY_QQQ_MINUTE_V1"
    with pytest.raises(RiskError, match="current completed QQQ candle"):
        probability_snapshot_quality(replace(snapshot, asof=snapshot.asof+timedelta(minutes=1)), snapshot.quotes[:2])


def test_probability_quote_age_is_300_seconds_and_not_execution_60(snapshot):
    earlier = snapshot.asof - timedelta(seconds=300)
    entry = changed(snapshot.quotes[0], event_at=earlier, available_at=earlier)
    assert probability_snapshot_quality(snapshot, [entry])["quote_ages_seconds"] == [300]
    with pytest.raises(RiskError, match="STALE_PROBABILITY_QUOTE"):
        probability_snapshot_quality(snapshot, [changed(entry, event_at=earlier-timedelta(microseconds=1))])


def test_independent_iv_timestamp_cannot_hide_behind_recent_quote(snapshot):
    entry = changed(snapshot.quotes[0], raw_fields=(("iv_timestamp", (snapshot.asof-timedelta(seconds=301)).isoformat()),))
    with pytest.raises(RiskError, match="STALE_PROBABILITY_IV"):
        probability_snapshot_quality(snapshot, [entry])
    entry = changed(entry, raw_fields=(("iv_timestamp", (snapshot.asof+timedelta(seconds=1)).isoformat()),))
    with pytest.raises(RiskError, match="later than t0"):
        probability_snapshot_quality(snapshot, [entry])


def test_minute_interval_uses_worst_age_once_and_worst_dispersion(snapshot):
    stamp = snapshot.asof-timedelta(minutes=1)
    first = changed(snapshot.quotes[0], event_at=stamp, available_at=snapshot.asof, timestamp_precision="minute")
    second = changed(snapshot.quotes[1], event_at=stamp, available_at=snapshot.asof, timestamp_precision="minute")
    result = probability_snapshot_quality(snapshot, [first, second])
    assert result["quote_ages_seconds"] == [60, 60] and result["dispersion_seconds"] == 60
    with pytest.raises(RiskError, match="PROBABILITY_QUOTE_DISPERSION"):
        probability_snapshot_quality(snapshot, [first, changed(second, event_at=stamp-timedelta(minutes=1))])
