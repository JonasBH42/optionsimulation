"""Explicitly requested synthetic observations; never a production fallback."""
from __future__ import annotations
import csv
import gzip
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from qqq_sim.config import writable_path
from qqq_sim.replay.calendar import CalendarService
from .models import NormalizationMapping


def synthetic_mappings():
    return {"options": NormalizationMapping(strike_divisor=Decimal(1), iv_unit="decimal", metadata_verified=True, timestamp_precision="subminute")}


def write_synthetic_dataset(root, day: date = date(2026, 11, 9), minutes: int | None = None, missing_minutes=()) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = writable_path(root)
    underlying, options = root / "OHLCV_daily", root / "raw_theta"
    underlying.mkdir(parents=True, exist_ok=True)
    for kind in ("greeks", "ohlcv", "oi_before", "oi", "trades"):
        (options / kind).mkdir(parents=True, exist_ok=True)
    calendar = CalendarService()
    labels = calendar.minute_labels(day)
    if minutes is not None:
        labels = labels[:minutes]
    expiry = day + timedelta(days=(4 - day.weekday()) % 7)
    candles = []
    for index, label in enumerate(labels):
        if index in missing_minutes:
            continue
        spot = 600.0 + index * 0.01
        candles.append({"regular_time": label.replace(tzinfo=None), "open": spot, "high": spot + 0.10,
                        "low": spot - 0.10, "close": spot + 0.02, "volume": 1000 + index})
    pq.write_table(pa.Table.from_pylist(candles), underlying / f"QQQ_1m_{day.isoformat()}.parquet")
    quotes, bars, prior, eod, trades = [], [], [], [], []
    for strike in (590, 595, 600, 605, 610):
        for right in ("C", "P"):
            base = {"symbol": "QQQ", "expiration": expiry.isoformat(), "strike": str(strike), "right": right}
            prior.append({**base, "open_interest": 1000 + strike})
            eod.append({**base, "open_interest": 9999999, "known_at": datetime.combine(day + timedelta(days=1), datetime.min.time()).isoformat()})
            for index, label in enumerate(labels):
                if index in missing_minutes:
                    continue
                spot = 600.0 + index * 0.01
                intrinsic = max(spot - strike, 0) if right == "C" else max(strike - spot, 0)
                time_value = max(0.10, 2.0 - abs(strike - spot) * 0.15 - index * 0.002)
                bid = round(intrinsic + time_value, 2)
                ask = round(bid + 0.10, 2)
                stamp = label + timedelta(seconds=30)
                quotes.append({**base, "timestamp": stamp.replace(tzinfo=None).isoformat(), "bid": bid, "ask": ask,
                               "implied_vol": 0.25, "delta": 0.5 if right == "C" else -0.5, "gamma": 0.02,
                               "vega": 0.10, "theta": -0.05, "underlying_price": 999999.0,
                               "sequence": index, "bid_size": 20, "ask_size": 20})
                bars.append({**base, "timestamp": label.replace(tzinfo=None).isoformat(), "open": bid, "high": ask, "low": bid,
                             "close": bid, "volume": 10})
                trades.append({**base, "timestamp": stamp.replace(tzinfo=None).isoformat(), "price": bid, "size": 10,
                               "sequence": index, "condition": "REGULAR"})
    def write(kind, prefix, rows):
        target = options / kind / f"{prefix}_{day:%Y%m%d}_exp={expiry.isoformat()}.csv.gz"
        with gzip.open(target, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for kind, prefix, rows in (("greeks", "greeks", quotes), ("ohlcv", "ohlcv", bars), ("trades", "trades", trades), ("oi_before", "oi_prev", prior), ("oi", "oi_eod", eod)):
        if rows:
            write(kind, prefix, rows)
    manifest = {"mode": "SYNTHETIC_FIXTURE", "underlying_root": str(underlying), "options_root": str(options),
                "day": day.isoformat(), "expiry": expiry.isoformat(), "calendar": "STANDARD_US_CALENDAR_SCENARIO_V1",
                "mapping": {"strike_divisor": "1", "iv_unit": "decimal", "metadata_verified": True, "timestamp_precision": "subminute"},
                "rate": 0.04, "carry": 0.01, "scenario_input": True}
    (root / "SYNTHETIC_FIXTURE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
