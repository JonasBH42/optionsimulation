#!/usr/bin/env python3
r"""QQQ whole-chain 4-calendar-DTE lifecycle research/backtest, specification v1.

This is a standalone offline research engine, independent of qqq-simulator.
Dependencies: numpy, pandas, pyarrow, scipy, pandas_market_calendars, psutil, threadpoolctl.
Optional atlas plots: matplotlib. Python >= 3.11. No downloads or broker access.

Quick start (PowerShell):
  python .\qqq_4dte_backtest.py init --config .\qqq_4dte_config.json
  python .\qqq_4dte_backtest.py audit --config .\qqq_4dte_config.json
  # Document verified clock conventions in the config before build-panel/run.
  python .\qqq_4dte_backtest.py run --config .\qqq_4dte_config.json
  python .\qqq_4dte_backtest.py train --config .\qqq_4dte_config.json
  # freeze-release and evaluate-final are separate, explicit test-once stages.
  python .\qqq_4dte_backtest.py self-test

Dates select EXPIRATIONS, not source sessions: E-4 and all intervening sessions
are loaded even when they precede the requested expiration date range. Unbounded
dates discover the archive. run is resumable and never opens the final test.

Tables retain every observed strike/right, including invalid quotes and late or
disappearing contracts. Missing slots are a sparse panel + a scheduled grid + a
contract inventory, never fabricated prices. Current-minute quotes and prior-asof
diagnostics are separate. Unknown product terms, IV conventions, terminal prices
and timestamp semantics disable only calculations that require them.

The supplied plan is a research specification, not a trading rule. Midpoint
changes are theoretical marks. This engine deliberately has no strategy search,
order routing, automatic acquisition, or account-level exercise simulation.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import html
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

# Bound inner libraries; partitions are processed sequentially to bound RAM.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

try:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import pandas_market_calendars as mcal
    import psutil
    from threadpoolctl import threadpool_limits
    from scipy.optimize import Bounds, LinearConstraint, brentq, minimize
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: " + str(exc) + "\nInstall: python -m pip install "
        "numpy pandas pyarrow scipy pandas_market_calendars psutil threadpoolctl matplotlib"
    ) from exc

VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
NY = ZoneInfo("America/New_York")
ONE_MINUTE = pd.Timedelta(minutes=1)
SOURCE_NAMES = {
    "underlying": "QQQ_OHLCV_data_contract_prompt.txt",
    "options": "QQQ_options_raw_theta_data_contract_prompt.txt",
    "plan": "QQQ_4DTE_whole_chain_lifecycle_research_plan.txt",
}
OPTION_KINDS = ("greeks", "ohlcv", "trades", "oi_before", "oi")
SOURCE_RE = re.compile(r"(?P<session>\d{8}).*?exp=(?P<expiration>\d{4}-\d{2}-\d{2})", re.I)
UNDERLYING_RE = re.compile(r"QQQ_1m_(\d{4}-\d{2}-\d{2})(?:[^\d].*)?\.parquet$", re.I)
CANONICAL_UNDERLYING_RE = re.compile(r"QQQ_1m_(\d{4}-\d{2}-\d{2})\.parquet$", re.I)

ALIASES = {
    "timestamp": ("timestamp", "datetime", "regular_time", "time", "quote_time", "date_time", "date"),
    "ms_of_day": ("ms_of_day", "msofday", "milliseconds_of_day"),
    "symbol": ("symbol", "underlying_symbol"),
    "root": ("root", "option_root"),
    "expiration": ("expiration", "expiration_date", "expiry", "exp"),
    "strike": ("strike", "strike_price", "strikeprice", "strike_milli", "strike_millis"),
    "right": ("right", "option_type", "type", "put_call", "cp"),
    "bid": ("bid", "bid_price"), "ask": ("ask", "ask_price"),
    "price": ("price", "last", "last_price", "trade_price"),
    "implied_vol": ("implied_vol", "implied_volatility", "impliedvolatility", "iv"),
    "underlying_price": ("underlying_price", "underlying_px", "underlying"),
    "underlying_timestamp": ("underlying_timestamp", "underlying_time"),
    "open_interest": ("open_interest", "openinterest", "oi"),
    "sequence": ("sequence", "sequence_number", "seq"),
    "size": ("size", "trade_size"), "bid_size": ("bid_size", "bidsize"),
    "ask_size": ("ask_size", "asksize"),
    "multiplier": ("multiplier", "contract_multiplier"),
    "deliverable_id": ("deliverable_id", "deliverable"),
    **{x: (x,) for x in ("open", "high", "low", "close", "volume", "delta", "gamma", "theta", "vega", "iv_error")},
}
REQUIRED = {
    "underlying": {"timestamp", "open", "high", "low", "close", "volume"},
    "greeks": {"strike", "right", "bid", "ask"},
    "ohlcv": {"strike", "right", "open", "high", "low", "close", "volume"},
    "trades": {"strike", "right", "price", "size"},
    "oi_before": {"strike", "right", "open_interest"},
    "oi": {"strike", "right", "open_interest"},
}
MONEYNESS_EDGES = np.array([-np.inf, -.15, -.075, -.03, -.01, .01, .03, .075, .15, np.inf])
MONEYNESS_NAMES = ("x<-0.15", "[-0.15,-0.075)", "[-0.075,-0.03)", "[-0.03,-0.01)", "[-0.01,0.01)", "[0.01,0.03)", "[0.03,0.075)", "[0.075,0.15)", "x>=0.15")


class ResearchError(RuntimeError):
    def __init__(self, code: str, message: str, **evidence: Any):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.evidence = code, message, evidence


def require(condition: bool, code: str, message: str, **evidence: Any) -> None:
    if not condition:
        raise ResearchError(code, message, **evidence)


def json_default(value: Any) -> Any:
    if isinstance(value, (Path, Decimal, date, datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(type(value).__name__)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(clean_json(value), default=json_default, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(clean_json(value), default=json_default, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    if path.suffix == ".csv":
        frame.to_csv(tmp, index=False)
    else:
        frame.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ny_timestamp(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    return t.tz_localize(NY, ambiguous="raise", nonexistent="raise") if t.tzinfo is None else t.tz_convert(NY)


def ny_times(values: pd.Series, session: str | None = None, ms: bool = False) -> pd.Series:
    if ms:
        base = ny_timestamp(session)
        numbers = pd.to_numeric(values, errors="coerce")
        return base + pd.to_timedelta(numbers, unit="ms")
    text = values.astype("string").str.strip()
    if session:
        time_only = text.str.match(r"^\d\d:\d\d(?::\d\d(?:\.\d+)?)?$").fillna(False)
        text = text.where(~time_only, session + "T" + text)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="In a future version of pandas, parsing datetimes with mixed time zones", category=FutureWarning)
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_convert(NY)
        if pd.api.types.is_datetime64_dtype(parsed):
            return parsed.dt.tz_localize(NY, ambiguous="NaT", nonexistent="NaT")
    except (ValueError, TypeError):
        pass
    # Mixed offsets/naive values are normalized individually, never guessed UTC.
    def parse_one(value: Any) -> pd.Timestamp:
        try:
            return ny_timestamp(value)
        except (ValueError, TypeError):
            return pd.NaT
    return pd.to_datetime(text.map(parse_one), utc=True).dt.tz_convert(NY)


def actual_seconds(start: Any, end: Any) -> float:
    return (ny_timestamp(end).value - ny_timestamp(start).value) / 1e9


def exact_strike(raw: Any, divisor: Any) -> str:
    try:
        value = Decimal(str(raw)) / Decimal(str(divisor))
        require(value.is_finite() and value > 0, "INVALID_STRIKE", "Strike must be positive", raw=str(raw))
        return format(value.normalize(), "f")
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise ResearchError("INVALID_STRIKE", str(raw)) from exc


def resolve_strike_scale(raw: pd.Series, spot: pd.Series, configured: Any = None, alias: str = "strike") -> tuple[float, str]:
    if configured is not None:
        divisor = float(configured)
        require(math.isfinite(divisor) and divisor > 0, "AMBIGUOUS_STRIKE_SCALE", "Invalid declared divisor")
        return divisor, "EXPLICIT_CONFIG"
    if alias.lower() in ("strike_milli", "strike_millis"):
        return 1000., "EXPLICIT_MILLI_ALIAS"
    strikes = pd.to_numeric(raw, errors="coerce")
    spots = pd.to_numeric(spot, errors="coerce")
    strikes = strikes[np.isfinite(strikes) & (strikes > 0)]
    spots = spots[np.isfinite(spots) & (spots > 0)]
    require(len(strikes) > 0 and len(spots) > 0, "AMBIGUOUS_STRIKE_SCALE", "No authoritative spot/strike evidence; declare divisor")
    ratio = float(strikes.median() / spots.median())
    plausible = lambda candidates: [d for d in candidates if .25 <= ratio / d <= 4]
    matches = plausible((1., 1000.))
    if not matches:
        broader = plausible((.001, .01, .1, 10., 100., 10000., 100000., 1000000.))
        # An isolated distant strike is not evidence of a new unit convention.
        # Broader automatic scales require cross-sectional support across spot.
        matches = [d for d in broader if strikes.min() / d <= spots.median() <= strikes.max() / d]
    require(len(matches) == 1, "AMBIGUOUS_STRIKE_SCALE", "Scale candidates are not uniquely supported", candidates=matches, median_raw_to_spot=ratio)
    return matches[0], "AUTHORITATIVE_SPOT_PLAUSIBILITY"


def normalize_iv(values: pd.Series, unit: str | None) -> tuple[pd.Series, str]:
    raw = pd.to_numeric(values, errors="coerce")
    if unit not in ("DECIMAL", "PERCENT"):
        return pd.Series(np.nan, index=values.index, dtype=float), "AMBIGUOUS_IV_SCALE"
    normalized = raw / (100. if unit == "PERCENT" else 1.)
    normalized = normalized.where(np.isfinite(normalized) & (normalized >= 0))
    return normalized, "VERIFIED_" + unit


def classify_quotes(bid: pd.Series, ask: pd.Series) -> pd.Series:
    finite = np.isfinite(bid) & np.isfinite(ask)
    valid = finite & (bid >= 0) & (ask > 0) & (ask >= bid)
    state = pd.Series("INVALID_NUMERIC", index=bid.index, dtype="object")
    state.loc[finite & (bid > ask)] = "CROSSED_MARKET"
    state.loc[valid] = "VALID_TWO_SIDED"
    state.loc[valid & bid.eq(0)] = "VALID_ZERO_BID"
    state.loc[valid & bid.eq(ask)] = "LOCKED_MARKET"
    return state


VALID_QUOTES = {"VALID_TWO_SIDED", "VALID_ZERO_BID", "LOCKED_MARKET"}


def select_intraday(frame: pd.DataFrame, selector: str, dataset_type: str) -> pd.DataFrame:
    """Preserve the contracts' legacy all-session selector, including the IV view."""
    if dataset_type in ("oi_before", "oi") or (selector == "00:00:00" and dataset_type in ("greeks", "iv", "ohlcv")):
        return frame.copy()
    require("source_time_ny" in frame, "INVALID_SOURCE_SCHEMA", "Selector requires parsed source times")
    clock = datetime.strptime(selector, "%H:%M:%S").time()
    return frame.loc[frame.source_time_ny.dt.time.eq(clock)].copy()


def default_config(contract_directory: Path | None = None) -> dict:
    source = contract_directory or Path(r"D:\strategy calculation")
    return {
        "schema_version": SCHEMA_VERSION, "symbol": "QQQ", "mode": "LOCAL_READ_ONLY",
        "run_id": "qqq_4dte_v1", "seed": 41729,
        "source_policy": {"version": "CANONICAL_SOURCE_CONTRACTS_V1", "root_override_evidence": None},
        "source_contracts": {k: str(source / v) for k, v in SOURCE_NAMES.items()},
        "expected_source_contract_hashes": {},
        "paths": {"underlying_root": r"D:\QQQ\OHLCV_daily", "options_root": r"D:\QQQ\raw_theta", "research_root": str(Path(__file__).resolve().parent / "qqq_4dte_research")},
        "date_range": {"start": None, "end": None},
        "cohort_definition": "CALENDAR_DATE_4",
        "clocks": {"timezone": "America/New_York", "calendar": "NASDAQ", "calendar_file": None,
                   "underlying_bar_semantics": "START_LABELED", "underlying_bar_evidence": "User verified 2026-11-10: 09:30 summarizes 09:30-09:31 and is available at 09:31.",
                   "quote_timestamp_semantics": "SNAPSHOT_RECORD_TIME", "quote_timestamp_evidence": "SOURCE-OPTIONS as-of/revealed-minute contract and inspected one-minute Greeks records; source timestamp is used only as the record clock, not a verified original update clock.",
                   "option_bar_semantics": "REQUIRES_EVIDENCE", "option_bar_evidence": None},
        "normalization": {"strike_divisor": None, "vendor_iv_unit": None, "vendor_iv_evidence": None},
        "underlying_asof_max_age_seconds": 0, "prior_asof_max_age_seconds": None,
        "retain_prior_asof_view": True,
        "reference": {"product_rules_file": None, "contract_metadata_file": None,
                      "terminal_reference_file": None, "listed_universe_complete_expirations": []},
        "labels": {"horizons_minutes": [1, 5, 15, 30, 60], "anchor_spacing_minutes": 5,
                   "sampled_hit_targets_price_units": [0.10], "named_horizons": True},
        "shape": {"enabled": True, "anchor_spacing_minutes": 5, "numeric_scale_floor": .01,
                  "max_interpolation_log_gap": .03, "midpoint_penalty": .05, "smooth_penalty": 0.},
        "forecast": {"sealed_test_fraction": .2, "initial_development_fraction": .6, "validation_blocks": 4,
                     "neighbor_candidates": [50, 100, 200], "ridge_candidates": [.1, 1., 10.],
                     "min_expirations": 30, "min_calendar_blocks": 20, "block_sessions": 5,
                     "bootstrap_repetitions": 300, "max_scenarios": 200, "max_train_anchors": 50000,
                     "factor_candidates": [0, 2, 4, 6, 8, 12], "factor_iterations": 20,
                     "release_tolerances": {"interval90_coverage_error_max": .05, "standardized_bias_max": .25,
                                            "missing_label_fraction_max": .25, "alpha": .05, "tail_probability_error_max": .03,
                                            "joint_tail_calibration_error_max": .02, "min_tail_events": 20,
                                            "max_target_crps_increase_standardized": .02,
                                            "basis": "PROSPECTIVE_ENGINEERING_TOLERANCES_NOT_EMPIRICALLY_CERTIFIED"}},
        "resources": {"memory_budget_gib": 20., "underlying_cache_sessions": 8, "inner_numeric_threads": 1},
        "execution_research": {"enabled": False},
    }


def merge_config(base: dict, update: dict) -> dict:
    result = dict(base)
    for key, value in update.items():
        result[key] = merge_config(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return result


def validate_config(config: dict) -> None:
    require(config["symbol"] == "QQQ" and config["mode"] == "LOCAL_READ_ONLY", "INVALID_CONFIG", "Only QQQ local read-only research is implemented")
    require(config["schema_version"] == SCHEMA_VERSION, "INVALID_CONFIG", "Unsupported schema_version")
    require(config["cohort_definition"] in ("CALENDAR_DATE_4", "TRADING_SESSION_OFFSET_4"), "INVALID_CONFIG", "Unknown cohort definition")
    require(config["clocks"]["timezone"] == "America/New_York", "INVALID_CONFIG", "NY wall clock is binding")
    require(not config["execution_research"]["enabled"], "EXECUTION_MODEL_UNVERIFIED", "Action/policy layer is deliberately disabled")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", config["run_id"]) is not None and config["run_id"] not in (".", ".."), "INVALID_CONFIG", "Unsafe run_id")
    require(0 < config["forecast"]["sealed_test_fraction"] < .5, "INVALID_CONFIG", "Invalid holdout fraction")
    require(all(isinstance(h, int) and h > 0 for h in config["labels"]["horizons_minutes"]), "INVALID_CONFIG", "Horizons must be positive integer minutes")
    require(config["resources"]["memory_budget_gib"] > 0, "INVALID_CONFIG", "Invalid memory budget")
    require(isinstance(config["resources"]["inner_numeric_threads"], int) and config["resources"]["inner_numeric_threads"] >= 1, "INVALID_CONFIG", "Numeric thread count must be a positive integer")
    for key, canonical in (("underlying_root", r"D:\QQQ\OHLCV_daily"), ("options_root", r"D:\QQQ\raw_theta")):
        if Path(config["paths"][key]).resolve() != Path(canonical).resolve():
            policy = config.get("source_policy", {})
            require(policy.get("version") != "CANONICAL_SOURCE_CONTRACTS_V1" and bool(policy.get("root_override_evidence")),
                    "SOURCE_POLICY_CONFLICT", "Noncanonical source roots need an explicit versioned policy and evidence", field=key)
    for key in ("underlying_asof_max_age_seconds", "prior_asof_max_age_seconds"):
        value = config[key]
        require(value is None or value >= 0, "INVALID_CONFIG", "Negative asof age", field=key)
    for key in ("anchor_spacing_minutes",):
        require(config["labels"][key] > 0, "INVALID_CONFIG", "Invalid anchor spacing")
    out = Path(config["paths"]["research_root"]).resolve()
    for name in ("underlying_root", "options_root"):
        source = Path(config["paths"][name]).resolve()
        require(not out.is_relative_to(source) and not source.is_relative_to(out), "IMMUTABLE_SOURCE_VIOLATION", "Research output overlaps a source directory")


class Run:
    """Immutable semantic identity + append-only diagnostics; no raw writes."""
    def __init__(self, config: dict, dry_run: bool = False):
        validate_config(config)
        self.started = time.perf_counter()
        self.config, self.dry_run = config, dry_run
        self.root = Path(config["paths"]["research_root"]).resolve()
        self.path = self.root / "runs" / config["run_id"]
        self.code_hash = sha256_file(Path(__file__))
        self.contracts = {}
        for name, path in config["source_contracts"].items():
            p = Path(path)
            require(p.is_file(), "REQUIRED_SOURCE_CONTRACT_MISSING", str(p))
            # Read all bytes, not only a remembered fingerprint.
            content = p.read_bytes()
            content.decode("utf-8-sig")
            self.contracts[name] = {"path": str(p.resolve()), "sha256": hashlib.sha256(content).hexdigest()}
        require(all(k in self.contracts for k in SOURCE_NAMES), "REQUIRED_SOURCE_CONTRACT_MISSING", "All three supplied documents are required")
        for name, expected in config["expected_source_contract_hashes"].items():
            require(self.contracts[name]["sha256"] == expected, "SOURCE_CONTRACT_CHANGED_REVIEW_REQUIRED", name)
        self.reference_hashes = {}
        for key, value in {**config["reference"], "calendar_file": config["clocks"]["calendar_file"]}.items():
            if key.endswith("_file") and value:
                p = Path(value)
                require(p.is_file(), "REFERENCE_FILE_MISSING", str(p))
                self.reference_hashes[key] = {**file_identity(p), "sha256": sha256_file(p)}
        self.semantic_id = digest({"code": self.code_hash, "config": config, "contracts": self.contracts, "references": self.reference_hashes})
        self.issues: list[dict] = []
        if dry_run:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock_stream = (self.path / ".run.lock").open("a+b")
        if self._lock_stream.tell() == 0:
            self._lock_stream.write(b"0"); self._lock_stream.flush()
        self._lock_stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_stream.close()
            raise ResearchError("RUN_ALREADY_ACTIVE", "Another command is writing this run_id; choose a different run or let it finish") from exc
        self._numeric_limit = threadpool_limits(limits=int(config["resources"]["inner_numeric_threads"]))
        manifest = self.path / "run_manifest.json"
        if manifest.exists():
            previous_manifest = read_json(manifest)
            require(previous_manifest["source_contracts"] == self.contracts, "SOURCE_CONTRACT_CHANGED_REVIEW_REQUIRED", "Source handling documents changed; review and version policy before reusing this study")
            require(previous_manifest["semantic_id"] == self.semantic_id, "RUN_ID_CONFLICT", "Code/config/reference changed: select a new run_id; do not reuse learned artifacts")
        else:
            packages = {}
            for package in ("numpy", "pandas", "pyarrow", "scipy", "pandas_market_calendars", "psutil", "threadpoolctl", "matplotlib"):
                with contextlib.suppress(importlib.metadata.PackageNotFoundError):
                    packages[package] = importlib.metadata.version(package)
            atomic_json(manifest, {"schema_version": SCHEMA_VERSION, "engine_version": VERSION,
                        "data_origin": "TEST_FIXTURE_ONLY" if config["source_policy"]["version"].startswith("TEST_FIXTURE_ONLY") else "ACTUAL_ARCHIVE",
                        "semantic_id": self.semantic_id, "code_hash": self.code_hash,
                        "config": config, "source_contracts": self.contracts, "reference_hashes": self.reference_hashes,
                        "python": sys.version, "platform": platform.platform(), "packages": packages,
                        "resources": {"logical_cpu_count": os.cpu_count(), "physical_ram_gib": psutil.virtual_memory().total / 2**30},
                        "release_labels": {"data_population": "OBSERVED_UNIVERSE_ONLY", "prediction_status": "DESCRIPTIVE_ONLY", "path_status": "NOT_IMPLEMENTED", "action_status": "DISABLED"}})
            for name, info in self.contracts.items():
                dest = self.path / "provenance" / SOURCE_NAMES[name]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(Path(info["path"]).read_bytes())
        atomic_json(self.path / "resolved_config.json", config)

    def event(self, stage: str, code: str, message: str, scope: str = "run", **evidence: Any) -> None:
        issue = {"stage": stage, "code": code, "scope": scope, "message": message, "evidence": clean_json(evidence)}
        self.issues.append(issue)
        if not self.dry_run:
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical_json({"recorded_at": datetime.now(NY).isoformat(), **issue}) + "\n")

    def memory_check(self) -> float:
        rss = psutil.Process().memory_info().rss / 2**30
        require(rss <= self.config["resources"]["memory_budget_gib"], "RESOURCE_LIMIT", "Memory budget exceeded; task remains incomplete", rss_gib=rss)
        return rss

    def close(self) -> None:
        stream = getattr(self, "_lock_stream", None)
        if stream is not None and not stream.closed:
            stream.close()  # OS locks are released even after an interrupted process.
        controller = getattr(self, "_numeric_limit", None)
        if controller is not None:
            controller.restore_original_limits()
            self._numeric_limit = None

    def __del__(self):
        with contextlib.suppress(Exception): self.close()


class Catalog:
    def __init__(self, run: Run):
        self.run = run
        rows = []
        paths = run.config["paths"]
        for kind, root in [("underlying", Path(paths["underlying_root"]))] + [(k, Path(paths["options_root"]) / k) for k in OPTION_KINDS]:
            if not root.is_dir():
                run.event("audit", "MISSING_SOURCE_FOLDER", str(root), kind)
                continue
            for path in sorted(root.iterdir()):
                if not path.is_file():
                    continue
                match = UNDERLYING_RE.fullmatch(path.name) if kind == "underlying" else SOURCE_RE.search(path.name)
                if not match or (kind != "underlying" and not path.name.lower().endswith(".csv.gz")):
                    continue
                try:
                    session = date.fromisoformat(match.group(1)) if kind == "underlying" else datetime.strptime(match.group("session"), "%Y%m%d").date()
                    expiration = None if kind == "underlying" else date.fromisoformat(match.group("expiration"))
                except ValueError:
                    run.event("audit", "INVALID_PARTITION_NAME", path.name, kind)
                    continue
                canonical = kind != "underlying" or CANONICAL_UNDERLYING_RE.fullmatch(path.name) is not None
                identity = file_identity(path)
                rows.append({"partition_id": digest({"kind": kind, **identity}), "dataset_type": kind,
                             "session_date": session.isoformat(), "expiration_date": expiration.isoformat() if expiration else "",
                             **identity, "fingerprint": digest(identity), "sha256": None, "schema_id": None,
                             "source_contract_hash": run.contracts["underlying" if kind == "underlying" else "options"]["sha256"],
                             "validation_status": "METADATA_ONLY" if canonical else "NONCANONICAL_FILENAME", "issue_codes": "" if canonical else "NONCANONICAL_FILENAME"})
        self.frame = pd.DataFrame(rows, columns=["partition_id", "dataset_type", "session_date", "expiration_date", "path", "bytes", "mtime_ns", "fingerprint", "sha256", "schema_id", "source_contract_hash", "validation_status", "issue_codes"])
        self.groups: dict[tuple, list[int]] = defaultdict(list)
        for row in self.frame.itertuples(index=True):
            self.groups[(row.dataset_type, row.session_date, row.expiration_date)].append(row.Index)
        duplicates = {key: indices for key, indices in self.groups.items() if len(indices) > 1}
        duplicate_indices = [index for indices in duplicates.values() for index in indices]
        if duplicate_indices:
            # Batch assignment/logging: thousands of per-file DataFrame writes and
            # Windows log opens otherwise dominate a metadata-only catalog scan.
            self.frame.loc[duplicate_indices, "validation_status"] = "DUPLICATE_LOGICAL_PARTITION"
            self.frame.loc[duplicate_indices, "issue_codes"] = "DUPLICATE_LOGICAL_PARTITION"
            for kind in sorted({key[0] for key in duplicates}):
                keys = [key for key in duplicates if key[0] == kind]
                run.event("audit", "DUPLICATE_LOGICAL_PARTITION", "All duplicate units quarantined; complete file/key details are in source_catalog.parquet", kind,
                          logical_partition_count=len(keys), examples=[{"key": key, "files": self.frame.loc[duplicates[key], "path"].tolist()} for key in keys[:3]])
        previous_path = run.path / "source_catalog.parquet"
        if previous_path.exists():
            previous = pd.read_parquet(previous_path)
            previous = previous.drop_duplicates("partition_id").set_index("partition_id")
            for column in ("sha256", "schema_id", "validation_status", "issue_codes"):
                if column in previous:
                    restored = self.frame.partition_id.map(previous[column])
                    allowed = self.frame.validation_status.eq("METADATA_ONLY") & restored.notna()
                    if column == "validation_status":
                        allowed &= ~restored.isin(["DUPLICATE_LOGICAL_PARTITION", "NONCANONICAL_FILENAME"])
                    self.frame.loc[allowed, column] = restored.loc[allowed]
        self.row_dicts = self.frame.to_dict("index")
        self.partition_indices = {row["partition_id"]: index for index, row in self.row_dicts.items()}
        self.identity = digest(self.frame[["dataset_type", "session_date", "expiration_date", "fingerprint"]].to_dict("records"))

    def get(self, kind: str, session: str, expiration: str = "") -> dict | None:
        indices = self.groups.get((kind, str(session), str(expiration)), [])
        if len(indices) != 1:
            return None
        row = self.row_dicts[indices[0]]
        return dict(row) if row["validation_status"] not in ("DUPLICATE_LOGICAL_PARTITION", "NONCANONICAL_FILENAME") else None

    def status(self, kind: str, session: str, expiration: str = "") -> str:
        indices = self.groups.get((kind, str(session), str(expiration)), [])
        if not indices:
            return "MISSING_SESSION_PARTITION"
        return "DUPLICATE_LOGICAL_PARTITION" if len(indices) > 1 else str(self.row_dicts[indices[0]]["validation_status"])

    def inspect(self, row: dict, full_hash: bool = False) -> dict:
        path = Path(row["path"])
        if row["dataset_type"] == "underlying":
            headers = pq.ParquetFile(path).schema_arrow.names
        else:
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
                headers = next(csv.reader(stream), [])
        mapping = map_columns(headers, row["dataset_type"])
        index = self.partition_indices[row["partition_id"]]
        self.frame.loc[index, "schema_id"] = digest(headers)
        self.frame.loc[index, "validation_status"] = "SCHEMA_VALIDATED"
        if full_hash:
            self.frame.loc[index, "sha256"] = sha256_file(path)
        self.row_dicts[index].update(self.frame.loc[index].to_dict())
        return mapping


def map_columns(headers: Iterable[str], kind: str) -> dict[str, str]:
    normalized = {str(h).strip().lower(): str(h) for h in headers}
    mapping = {}
    for canonical, aliases in ALIASES.items():
        # regular_time has authority for underlying even when datetime is present.
        choices = ("regular_time", "datetime", "timestamp", "time", "date") if kind == "underlying" and canonical == "timestamp" else aliases
        for alias in choices:
            if alias in normalized:
                mapping[canonical] = normalized[alias]
                break
    missing = REQUIRED[kind] - mapping.keys()
    require(not missing, "INVALID_SOURCE_SCHEMA", "Required columns absent", kind=kind, missing=sorted(missing), headers=list(headers))
    if kind not in ("oi_before", "oi"):
        require("timestamp" in mapping or "ms_of_day" in mapping, "INVALID_SOURCE_SCHEMA", "No recognized timestamp", kind=kind)
    return mapping


class Calendar:
    def __init__(self, config: dict, start: date, end: date):
        custom = config["clocks"]["calendar_file"]
        if custom:
            data = read_json(Path(custom))
            require(data.get("version") and data.get("provenance"), "CALENDAR_UNVERIFIED", "Custom calendar requires version and provenance")
            self.version = data["version"]
            self.provenance = data["provenance"]
            records = data["sessions"]
            schedule = pd.DataFrame(records)
            schedule["session_date"] = schedule["session_date"].astype(str)
            schedule["open"] = schedule.regular_open_ny.map(ny_timestamp)
            schedule["close"] = schedule.regular_close_ny.map(ny_timestamp)
        else:
            library = mcal.get_calendar(config["clocks"]["calendar"])
            self.version = "pandas_market_calendars:" + importlib.metadata.version("pandas_market_calendars") + ":" + config["clocks"]["calendar"]
            self.provenance = {"library_class": type(library).__name__, "rule_basis": "VERSIONED_LIBRARY_HISTORICAL_SCHEDULE", "product_rules": "SEPARATE_NOT_INFERRED"}
            source = library.schedule(start_date=start, end_date=end, tz="America/New_York", interruptions=True)
            schedule = pd.DataFrame({"session_date": source.index.strftime("%Y-%m-%d"), "open": source.market_open.array, "close": source.market_close.array})
        require(not schedule.session_date.duplicated().any(), "CALENDAR_UNVERIFIED", "Duplicate calendar sessions")
        self.sessions = {row.session_date: {"open": ny_timestamp(row.open), "close": ny_timestamp(row.close)} for row in schedule.itertuples(index=False)}
        for day, session in self.sessions.items():
            require(session["open"] < session["close"], "CALENDAR_UNVERIFIED", day)
        self.days = sorted(self.sessions)

    def between(self, start: str | date, end: str | date) -> list[str]:
        return [d for d in self.days if str(start) <= d <= str(end)]

    def grid(self, session: str) -> pd.DatetimeIndex:
        item = self.sessions[session]
        return pd.date_range(item["open"] + ONE_MINUTE, item["close"], freq="min")

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{"session_date": day, "regular_open_ny": x["open"], "regular_close_ny": x["close"], "early_close": x["close"].hour < 16,
                              "calendar_version": self.version, "provenance": canonical_json(self.provenance)} for day, x in sorted(self.sessions.items())])


class References:
    def __init__(self, run: Run):
        self.run = run
        config = run.config["reference"]
        self.rules = read_json(Path(config["product_rules_file"])) if config["product_rules_file"] else {"profiles": [], "boundaries": []}
        self.metadata = pd.read_csv(config["contract_metadata_file"], dtype={"strike": "string"}) if config["contract_metadata_file"] else pd.DataFrame()
        self.terminals = pd.read_csv(config["terminal_reference_file"]) if config["terminal_reference_file"] else pd.DataFrame()
        self.complete_expirations = set(config["listed_universe_complete_expirations"])
        self._terminal_cache = {}
        if not self.metadata.empty:
            needed = {"root", "expiration", "strike", "right", "deliverable_id", "multiplier", "listing_effective_time", "listing_known_at", "source"}
            require(needed <= set(self.metadata), "INVALID_SOURCE_SCHEMA", "Incomplete contract reference schema", missing=sorted(needed - set(self.metadata)))
            self.metadata["exact_strike"] = self.metadata.strike.map(lambda x: exact_strike(x, 1))
            self.metadata["right"] = self.metadata.right.str.upper().replace({"CALL": "C", "PUT": "P"})
            for column in ("listing_effective_time", "listing_known_at"):
                self.metadata[column] = ny_times(self.metadata[column])
            keys = ["root", "expiration", "exact_strike", "right", "deliverable_id", "listing_effective_time"]
            require(not self.metadata.duplicated(keys).any(), "DUPLICATE_CONTRACT_REFERENCE", "Contract reference keys must be unique")
        for expiration in self.complete_expirations:
            rows = self.metadata[self.metadata.expiration.astype(str).eq(expiration)] if not self.metadata.empty else pd.DataFrame()
            require(len(rows) > 0 and rows.listing_effective_time.notna().all() and rows.listing_known_at.notna().all(),
                    "UNVERIFIED_LISTED_UNIVERSE", "Completeness assertion needs actual point-in-time listing intervals", expiration=expiration)

    def profile(self, root: str, session: str) -> dict | None:
        matches = [p for p in self.rules.get("profiles", []) if root in p.get("roots", []) and p["effective_from"] <= session <= p["effective_to"]]
        require(len(matches) <= 1, "AMBIGUOUS_PRODUCT_RULES", "Overlapping profiles", root=root, session=session)
        if not matches:
            return None
        item = matches[0]
        require(item.get("evidence") and item.get("standard_unadjusted") is True and item.get("multiplier", 0) > 0 and item.get("deliverable_id") and item.get("exercise_style") == "AMERICAN" and item.get("settlement_type") == "PHYSICAL", "UNSUPPORTED_ADJUSTED_DELIVERABLE", "Invalid standard-product evidence or unverified exercise/settlement terms")
        return item

    def boundary(self, expiration: str) -> dict | None:
        matches = [r for r in self.rules.get("boundaries", []) if r["expiration"] == expiration]
        require(len(matches) <= 1, "AMBIGUOUS_PRODUCT_RULES", "Duplicate expiry boundary")
        if not matches:
            return None
        item = matches[0]
        require(item.get("evidence") and item.get("economic_expiry_ny") and item.get("official_last_option_trading_time_ny"), "TERMINAL_REFERENCE_MISSING", "Unverified product boundary")
        return item

    def terminal(self, expiration: str) -> dict | None:
        if expiration not in self._terminal_cache:
            try:
                self._terminal_cache[expiration] = self._validated_terminal(expiration)
            except (ResearchError, ValueError, TypeError) as exc:
                self.run.event("terminal_reference", "TERMINAL_REFERENCE_MISSING", "Dependent terminal labels blocked; raw descriptive study continues", expiration, detail=str(exc))
                self._terminal_cache[expiration] = None
        return self._terminal_cache[expiration]

    def _validated_terminal(self, expiration: str) -> dict | None:
        if self.terminals.empty:
            return None
        rows = self.terminals[self.terminals.expiration.astype(str).eq(expiration)]
        require(len(rows) <= 1, "AMBIGUOUS_TERMINAL_REFERENCE", expiration)
        if rows.empty:
            return None
        row = rows.iloc[0].to_dict()
        require(row.get("source") and row.get("evidence") and row.get("reference_time_ny") and row.get("known_at_ny") and np.isfinite(float(row["price"])) and float(row["price"]) > 0, "TERMINAL_REFERENCE_MISSING", "Terminal evidence schema/price invalid")
        boundary = self.boundary(expiration)
        require(boundary is not None and boundary.get("settlement_reference_time_ny"), "TERMINAL_REFERENCE_MISSING", "Terminal price needs a verified reference-time rule")
        require(ny_timestamp(row["reference_time_ny"]) == ny_timestamp(boundary["settlement_reference_time_ny"]), "TERMINAL_REFERENCE_MISSING", "Terminal price time disagrees with product reference rule")
        require(ny_timestamp(row["known_at_ny"]) >= ny_timestamp(row["reference_time_ny"]), "TERMINAL_REFERENCE_MISSING", "Reference cannot be published before its price time")
        return row


def cohort_grid(calendar: Calendar, cohort: dict, session: str) -> pd.DatetimeIndex:
    grid = calendar.grid(session)
    boundaries = [ny_timestamp(cohort[key]) for key in ("T_E", "official_last_option_trading_time") if key in cohort and pd.notna(cohort[key])]
    return grid[grid <= min(boundaries)] if boundaries else grid


def build_cohorts(run: Run, catalog: Catalog, calendar: Calendar, references: References) -> pd.DataFrame:
    all_expirations = set(catalog.frame.loc[catalog.frame.expiration_date.ne(""), "expiration_date"])
    if not references.metadata.empty:
        all_expirations.update(references.metadata.expiration.astype(str))
    rows = []
    start_limit, end_limit = run.config["date_range"]["start"], run.config["date_range"]["end"]
    archive_days = catalog.frame.loc[catalog.frame.dataset_type.eq("greeks"), "session_date"]
    archive_cutoff = archive_days.max() if len(archive_days) else ""
    for expiration in sorted(all_expirations):
        if (start_limit and expiration < start_limit) or (end_limit and expiration > end_limit):
            continue
        e_date = date.fromisoformat(expiration)
        possible_sessions = calendar.between(e_date - timedelta(days=30), e_date)
        last_session = possible_sessions[-1] if possible_sessions else None
        if run.config["cohort_definition"] == "CALENDAR_DATE_4":
            d0 = (e_date - timedelta(days=4)).isoformat()
        else:
            d0 = possible_sessions[-5] if len(possible_sessions) >= 5 else ""
        boundary = references.boundary(expiration)
        expected = calendar.between(d0, expiration) if d0 in calendar.sessions else []
        if boundary:
            boundary_clock = {"T_E": boundary["economic_expiry_ny"], "official_last_option_trading_time": boundary["official_last_option_trading_time_ny"]}
            expected = [day for day in expected if len(cohort_grid(calendar, boundary_clock, day))]
            if expected: last_session = expected[-1]
        observed = [d for d in expected if catalog.get("greeks", d, expiration)]
        if d0 not in calendar.sessions:
            status = "NO_EXACT_CALENDAR_4_SESSION" if run.config["cohort_definition"] == "CALENDAR_DATE_4" else "CALENDAR_UNVERIFIED"
        elif not catalog.get("greeks", d0, expiration) or not catalog.get("underlying", d0):
            status = "DATA_GAP_AT_START"
        elif not expected:
            status = "PRODUCT_BOUNDARY_PRECEDES_ENTRY"
        else:
            status = "ELIGIBLE_ARCHIVE_DISCOVERED"
        rows.append({"cohort_id": run.config["cohort_definition"] + ":" + expiration, "expiration_date": expiration,
                     "cohort_definition": run.config["cohort_definition"], "D0": d0,
                     "scheduled_t0": calendar.grid(d0)[0] if expected else pd.NaT,
                     "T_E": ny_timestamp(boundary["economic_expiry_ny"]) if boundary else pd.NaT,
                     "official_last_option_trading_time": ny_timestamp(boundary["official_last_option_trading_time_ny"]) if boundary else pd.NaT,
                     "exercise_instruction_boundary": boundary.get("exercise_instruction_boundary_ny") if boundary else None,
                     "boundary_status": "VERIFIED" if boundary else "TERMINAL_REFERENCE_MISSING",
                     "eligible_status": status, "expiration_universe_status": "ARCHIVE_DISCOVERED",
                     "initial_universe_status": "VERIFIED_LISTED" if expiration in references.complete_expirations else "UNVERIFIED_LISTED_UNIVERSE",
                     "listed_contract_count": None, "listed_quote_coverage": None,
                     "expected_sessions": canonical_json(expected), "observed_sessions": canonical_json(observed),
                     "last_scheduled_rth_session": last_session,
                     "right_censored": bool(last_session and archive_cutoff < last_session),
                     "terminal_status": "REFERENCE_PRESENT" if references.terminal(expiration) else "TERMINAL_REFERENCE_MISSING",
                     "expiry_weekday": e_date.strftime("%A"), "calendar_version": calendar.version})
    return pd.DataFrame(rows)


def plan_gaps(run: Run, catalog: Catalog, cohorts: pd.DataFrame, calendar: Calendar) -> pd.DataFrame:
    rows = []
    for cohort in cohorts.to_dict("records"):
        for session in json.loads(cohort["expected_sessions"]):
            for kind in ("underlying", *OPTION_KINDS):
                expiry_key = "" if kind == "underlying" else cohort["expiration_date"]
                status = catalog.status(kind, session, expiry_key)
                item = calendar.sessions[session]
                rows.append({"cohort_id": cohort["cohort_id"], "symbol": "QQQ", "dataset_type": kind,
                             "session_date": session, "expiration_date": cohort["expiration_date"],
                             "status": status, "rights": "C,P", "strikes": "ALL",
                             "expected_start_ny": item["open"], "expected_end_ny": item["close"],
                             "source_contract_hash": run.contracts["underlying" if kind == "underlying" else "options"]["sha256"],
                             "required_schema": canonical_json(sorted(REQUIRED[kind])),
                             "acquisition_status": "NOT_REQUESTED_LOCAL_READ_ONLY",
                             "remediation": "REVIEW_DUPLICATES" if status == "DUPLICATE_LOGICAL_PARTITION" else "REQUEST_SOURCE_DATA"})
    return pd.DataFrame(rows)


def require_causal_clocks(config: dict) -> None:
    clocks = config["clocks"]
    require(clocks["underlying_bar_semantics"] in ("START_LABELED", "END_LABELED", "ALREADY_COMPLETED") and bool(clocks["underlying_bar_evidence"]),
            "BLOCKED_TIMESTAMP_SEMANTICS", "Document underlying_bar_semantics and underlying_bar_evidence before causal research")
    require(clocks["quote_timestamp_semantics"] in ("EVENT_TIME", "SNAPSHOT_RECORD_TIME", "QUOTE_UPDATE_TIME") and bool(clocks["quote_timestamp_evidence"]),
            "BLOCKED_TIMESTAMP_SEMANTICS", "Document quote_timestamp_semantics and quote_timestamp_evidence")


def normalize_underlying(frame: pd.DataFrame, session: str, calendar: Calendar, semantics: str) -> tuple[pd.DataFrame, dict]:
    require(semantics in ("START_LABELED", "END_LABELED", "ALREADY_COMPLETED"), "BLOCKED_TIMESTAMP_SEMANTICS", semantics)
    frame = frame.copy()
    frame["source_record_ordinal"] = np.arange(len(frame), dtype=np.int64)
    frame["source_time_ny"] = ny_times(frame.timestamp, session)
    require(not frame.source_time_ny.dropna().duplicated().any(), "DUPLICATE_UNDERLYING_TIMESTAMP", "Reject duplicate timestamps; do not choose an arbitrary bar", session=session)
    start_labeled = semantics == "START_LABELED"
    frame["available_at_ny"] = frame.source_time_ny + ONE_MINUTE if start_labeled else frame.source_time_ny
    frame["bar_start_ny"] = frame.source_time_ny if start_labeled else frame.source_time_ny - ONE_MINUTE
    frame["bar_end_ny"] = frame.bar_start_ny + ONE_MINUTE
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    finite = np.isfinite(frame[["open", "high", "low", "close", "volume"]]).all(axis=1)
    geometry = (frame.high >= frame.low) & (frame.high >= frame[["open", "close"]].max(axis=1)) & (frame.low <= frame[["open", "close"]].min(axis=1))
    same_day = frame.source_time_ny.dt.strftime("%Y-%m-%d").eq(session)
    bounds = calendar.sessions[session]
    # The contracts exclude raw 16:00/EOD records even for end-labeled sources.
    canonical_raw_rth = (frame.source_time_ny >= bounds["open"]) & (frame.source_time_ny < bounds["close"])
    interval_rth = (frame.bar_start_ny >= bounds["open"]) & (frame.available_at_ny <= bounds["close"])
    valid = finite & geometry & same_day & canonical_raw_rth & interval_rth & (frame.volume > 0)
    diagnostics = {"source_rows": len(frame), "accepted_rows": int(valid.sum()), "invalid_numeric_or_geometry": int((~finite | ~geometry | (frame.volume < 0)).sum()),
                   "wrong_or_invalid_date": int((~same_day).sum()), "outside_rth": int((~canonical_raw_rth | ~interval_rth).sum()),
                   "zero_volume_excluded": int(frame.volume.eq(0).sum()), "bar_semantics": semantics}
    result = frame.loc[valid].sort_values("available_at_ny", kind="stable").reset_index(drop=True)
    result["session_date"] = session
    return result.drop(columns=["timestamp"]), diagnostics


def revealed_underlying_grid(frame: pd.DataFrame, grid: pd.DatetimeIndex, max_age: float | None) -> pd.DataFrame:
    left = pd.DataFrame({"decision_time_ny": grid})
    if frame.empty:
        result = left.assign(spot=np.nan, underlying_age_seconds=np.nan)
        result["underlying_source_time"] = pd.Series(pd.NaT, index=result.index, dtype=pd.DatetimeTZDtype(tz=NY, unit="ns"))
        result["underlying_status"] = "UNDERLYING_MISSING"
        for column in ("underlying_interval_seconds", "revealed_session_high", "revealed_session_low", "revealed_session_volume", "log_return_from_open", "open", "high", "low", "close", "volume"):
            result[column] = np.nan
        for lag in (1, 5, 30):
            result[f"underlying_log_return_{lag}m"] = np.nan
            result[f"underlying_rv_{lag}m"] = np.nan
        return result
    source = frame.sort_values("available_at_ny").copy()
    source["source_cumulative_volume"] = source.volume.cumsum()
    result = pd.merge_asof(left, source, left_on="decision_time_ny", right_on="available_at_ny", direction="backward")
    result["underlying_age_seconds"] = (result.decision_time_ny - result.available_at_ny).dt.total_seconds()
    result["underlying_source_time"] = result.source_time_ny
    result["spot"] = result.close
    if max_age is not None:
        result.loc[result.underlying_age_seconds > max_age, "spot"] = np.nan
    result["underlying_status"] = np.where(result.spot.notna(), "VALID_REVEALED_CLOSE", "UNDERLYING_MISSING")
    result["underlying_interval_seconds"] = result.decision_time_ny.diff().dt.total_seconds()
    for lag in (1, 5, 30):
        result[f"underlying_log_return_{lag}m"] = np.log(result.spot / result.spot.shift(lag))
        result[f"underlying_rv_{lag}m"] = np.log(result.spot / result.spot.shift(1)).pow(2).rolling(lag, min_periods=lag).sum()
    result["revealed_session_high"] = result.high.where(result.spot.notna()).cummax()
    result["revealed_session_low"] = result.low.where(result.spot.notna()).cummin()
    result["revealed_session_volume"] = result.source_cumulative_volume
    result["log_return_from_open"] = np.log(result.spot / result.open.iloc[0]) if pd.notna(result.open.iloc[0]) and result.open.iloc[0] > 0 else np.nan
    return result


class DataStore:
    def __init__(self, run: Run, catalog: Catalog, calendar: Calendar, references: References):
        self.run, self.catalog, self.calendar, self.references = run, catalog, calendar, references
        self.underlying_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def source_frame(self, partition: dict) -> tuple[pd.DataFrame, dict, str]:
        self.run.memory_check()
        path = Path(partition["path"])
        before = file_identity(path)
        require(before["bytes"] == partition["bytes"] and before["mtime_ns"] == partition["mtime_ns"], "SOURCE_CHANGED_DURING_RUN", str(path))
        mapping = self.catalog.inspect(partition)
        if partition["dataset_type"] == "underlying":
            frame = pd.read_parquet(path, columns=list(mapping.values()))
        else:
            string_columns = [mapping[k] for k in ("strike", "right", "symbol", "root", "expiration", "deliverable_id") if k in mapping]
            frame = pd.read_csv(path, usecols=list(mapping.values()), dtype={c: "string" for c in string_columns}, low_memory=False)
        source_hash = sha256_file(path)
        require(file_identity(path) == before, "SOURCE_CHANGED_DURING_RUN", str(path))
        index = self.catalog.partition_indices[partition["partition_id"]]
        self.catalog.frame.loc[index, "sha256"] = source_hash
        self.catalog.frame.loc[index, "validation_status"] = "READ_HASHED"
        self.catalog.row_dicts[index].update(self.catalog.frame.loc[index].to_dict())
        frame = frame.rename(columns={v: k for k, v in mapping.items()})
        frame["source_record_ordinal"] = np.arange(len(frame), dtype=np.int64)
        frame["source_partition_id"] = partition["partition_id"]
        frame["source_sha256"] = source_hash
        self.run.memory_check()
        return frame, mapping, source_hash

    def underlying(self, session: str) -> pd.DataFrame:
        if session in self.underlying_cache:
            self.underlying_cache.move_to_end(session)
            return self.underlying_cache[session].copy()
        partition = self.catalog.get("underlying", session)
        if not partition:
            self.run.event("underlying", "UNDERLYING_MISSING", "No unique canonical session", session)
            return pd.DataFrame()
        frame, _, source_hash = self.source_frame(partition)
        result, checks = normalize_underlying(frame, session, self.calendar, self.run.config["clocks"]["underlying_bar_semantics"])
        self.run.event("underlying", "VALIDATION_COUNTS", "Canonical validation", session, **checks)
        output = self.run.path / "underlying_observations" / (session + ".parquet")
        atomic_frame(output, result)
        self.underlying_cache[session] = result
        cap = self.run.config["resources"]["underlying_cache_sessions"]
        while len(self.underlying_cache) > cap:
            self.underlying_cache.popitem(last=False)
        return result.copy()

    def normalize_options(self, frame: pd.DataFrame, partition: dict, mapping: dict, spot_evidence: pd.Series) -> pd.DataFrame:
        frame = frame.copy()
        session, expiration = partition["session_date"], partition["expiration_date"]
        if "symbol" not in frame:
            frame["symbol"] = "QQQ"
            self.run.event("normalize", "SYMBOL_FROM_PARTITION_SCOPE", "Symbol absent; immutable QQQ-root partition scope used", partition["partition_id"])
        frame["root"] = frame["root"].fillna(frame.symbol) if "root" in frame else frame.symbol
        if "expiration" in frame:
            parsed = pd.to_datetime(frame.expiration, errors="coerce").dt.strftime("%Y-%m-%d")
            require(parsed.eq(expiration).all(), "PARTITION_EXPIRATION_MISMATCH", "Rows do not match filename expiration")
        frame["expiration_date"] = expiration
        frame["right"] = frame.right.str.strip().str.upper().replace({"CALL": "C", "PUT": "P"})
        require(frame.right.isin(["C", "P"]).all(), "INVALID_SOURCE_SCHEMA", "Unrecognized option right")
        require(frame.root.notna().all(), "INVALID_CONTRACT_IDENTITY", "Null root")
        require(frame.symbol.str.upper().str.startswith("QQQ").all(), "INVALID_CONTRACT_IDENTITY", "Non-QQQ source rows")
        divisor, scale_basis = resolve_strike_scale(frame.strike, spot_evidence, self.run.config["normalization"]["strike_divisor"], mapping["strike"])
        frame["raw_strike"] = frame.strike
        exact_map = {str(value): exact_strike(value, divisor) for value in frame.strike.unique()}
        frame["exact_strike"] = frame.strike.astype(str).map(exact_map)
        frame["strike"] = frame.exact_strike.astype(float)
        frame["strike_divisor"] = divisor
        frame["strike_scale_basis"] = scale_basis
        if "timestamp" in frame:
            frame["source_time_ny"] = ny_times(frame.timestamp, session)
        elif "ms_of_day" in frame:
            frame["source_time_ny"] = ny_times(frame.ms_of_day, session, ms=True)
        else:
            frame["source_time_ny"] = pd.Series(pd.NaT, index=frame.index, dtype=pd.DatetimeTZDtype(tz=NY, unit="ns"))
        frame["normalized_minute"] = frame.source_time_ny.dt.floor("min")
        # Quote/trade events at exactly a minute boundary belong to that new minute.
        frame["available_at_ny"] = frame.source_time_ny
        if partition["dataset_type"] == "ohlcv":
            convention = self.run.config["clocks"]["option_bar_semantics"]
            if convention == "START_LABELED" and self.run.config["clocks"]["option_bar_evidence"]:
                frame["available_at_ny"] = frame.source_time_ny + ONE_MINUTE
            elif convention not in ("END_LABELED", "ALREADY_COMPLETED") or not self.run.config["clocks"]["option_bar_evidence"]:
                frame["available_at_ny"] = pd.Series(pd.NaT, index=frame.index, dtype=pd.DatetimeTZDtype(tz=NY, unit="ns"))
        frame["first_snapshot_decision_ny"] = frame.available_at_ny if partition["dataset_type"] == "ohlcv" else frame.normalized_minute + ONE_MINUTE
        master_rows = []
        metadata = self.references.metadata
        unique = frame[["root", "exact_strike", "right"] + [c for c in ("multiplier", "deliverable_id") if c in frame]].drop_duplicates()
        for raw in unique.to_dict("records"):
            profile = self.references.profile(str(raw["root"]), session)
            matches = metadata[(metadata.root.astype(str) == str(raw["root"])) & (metadata.expiration.astype(str) == expiration) &
                               metadata.exact_strike.eq(raw["exact_strike"]) & metadata.right.eq(raw["right"])] if not metadata.empty else metadata
            if "effective_from" in matches:
                matches = matches[matches.effective_from.astype(str).le(session) & matches.effective_to.astype(str).ge(session)]
            require(len(matches) <= 1, "INVALID_CONTRACT_IDENTITY", "Multiple reference deliverables; source must identify the correct contract")
            ref = matches.iloc[0].to_dict() if len(matches) else {}
            multiplier = raw.get("multiplier", ref.get("multiplier", profile.get("multiplier") if profile else None))
            if multiplier is not None and pd.notna(multiplier):
                multiplier = float(multiplier)
                require(math.isfinite(multiplier) and multiplier > 0, "INVALID_CONTRACT_IDENTITY", "Invalid multiplier")
            else:
                multiplier = None
            deliverable = raw.get("deliverable_id", ref.get("deliverable_id", profile.get("deliverable_id") if profile else None))
            deliverable = str(deliverable) if deliverable is not None and pd.notna(deliverable) else None
            identity = {"symbol": "QQQ", "root": str(raw["root"]), "expiration_date": expiration, "right": raw["right"],
                        "exact_strike": raw["exact_strike"], "multiplier": multiplier, "deliverable_id": deliverable}
            standard = bool(profile and (deliverable == profile["deliverable_id"]) and multiplier == profile["multiplier"])
            master_rows.append({**raw, "contract_id": digest(identity), "resolved_multiplier": multiplier, "resolved_deliverable_id": deliverable,
                                "standard_supported": standard, "metadata_status": "VERIFIED_STANDARD_PROFILE" if standard else "UNSUPPORTED_ADJUSTED_OR_UNKNOWN_DELIVERABLE",
                                "listing_effective_time": ref.get("listing_effective_time", pd.NaT), "listing_known_at": ref.get("listing_known_at", pd.NaT),
                                "metadata_source": ref.get("source", canonical_json(profile["evidence"]) if profile else "UNKNOWN")})
        keys = ["root", "exact_strike", "right"] + [c for c in ("multiplier", "deliverable_id") if c in frame]
        frame = frame.merge(pd.DataFrame(master_rows), on=keys, how="left", validate="many_to_one")
        frame["multiplier"] = frame.pop("resolved_multiplier")
        frame["deliverable_id"] = frame.pop("resolved_deliverable_id")
        for column in ("listing_effective_time", "listing_known_at"):
            frame[column] = ny_times(frame[column])
        frame["schema_version"] = SCHEMA_VERSION
        if "implied_vol" in frame:
            frame["vendor_iv_raw"] = pd.to_numeric(frame.implied_vol, errors="coerce")
            unit = self.run.config["normalization"]["vendor_iv_unit"] if self.run.config["normalization"]["vendor_iv_evidence"] else None
            frame["vendor_iv_decimal"], iv_status = normalize_iv(frame.implied_vol, unit)
            frame["vendor_iv_status"] = iv_status
            frame = frame.drop(columns="implied_vol")
        for greek in ("delta", "gamma", "theta", "vega"):
            if greek in frame:
                frame = frame.rename(columns={greek: "vendor_" + greek})
        frame["vendor_greek_convention_status"] = "UNVERIFIED_DIAGNOSTIC_ONLY"
        if "underlying_price" in frame:
            frame = frame.rename(columns={"underlying_price": "vendor_underlying_price_diagnostic"})
        return frame.drop(columns=[c for c in ("timestamp", "expiration") if c in frame])

    def options(self, kind: str, session: str, expiration: str, spots: pd.Series) -> pd.DataFrame:
        partition = self.catalog.get(kind, session, expiration)
        if not partition:
            return pd.DataFrame()
        frame, mapping, _ = self.source_frame(partition)
        return self.normalize_options(frame, partition, mapping, spots)


def current_minute_quotes(observations: pd.DataFrame, grid: pd.DatetimeIndex, quote_clock: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = observations.copy()
    frame["bid"] = pd.to_numeric(frame.bid, errors="coerce")
    frame["ask"] = pd.to_numeric(frame.ask, errors="coerce")
    frame["quote_status"] = classify_quotes(frame.bid, frame.ask)
    frame["valid_quote"] = frame.quote_status.isin(VALID_QUOTES)
    frame["decision_time_ny"] = frame.normalized_minute + ONE_MINUTE
    allowed = frame.decision_time_ny.isin(grid) & frame.source_time_ny.notna() & (frame.source_time_ny < frame.decision_time_ny)
    rejected = frame.loc[~allowed].copy()
    accepted = frame.loc[allowed].copy()
    ordering = ["contract_id", "source_time_ny"] + (["sequence"] if "sequence" in accepted else []) + ["source_record_ordinal"]
    accepted = accepted.sort_values(ordering, kind="stable")
    group_keys = ["contract_id", "decision_time_ny"]
    last_any = accepted.drop_duplicates(group_keys, keep="last")
    last_valid = accepted[accepted.valid_quote].drop_duplicates(group_keys, keep="last")
    diagnostics = last_any[group_keys + ["quote_status", "source_time_ny"]].rename(columns={"quote_status": "last_record_status", "source_time_ny": "last_record_time_ny"})
    choose = pd.concat([last_valid, last_any[~pd.MultiIndex.from_frame(last_any[group_keys]).isin(pd.MultiIndex.from_frame(last_valid[group_keys]))]], ignore_index=True)
    choose = choose.merge(diagnostics, on=group_keys, how="left", validate="one_to_one")
    choose["raw_quote_time_ny"] = choose.source_time_ny
    choose["snapshot_record_age_seconds"] = (choose.decision_time_ny - choose.source_time_ny).dt.total_seconds()
    true_updates = quote_clock in ("EVENT_TIME", "QUOTE_UPDATE_TIME")
    choose["quote_age_seconds"] = choose.snapshot_record_age_seconds if true_updates else np.nan
    choose["quote_age_basis"] = "LAST_QUOTE_UPDATE" if true_updates else "QUOTE_UPDATE_AGE_UNKNOWN"
    choose["midpoint"] = ((choose.bid + choose.ask) / 2).where(choose.valid_quote)
    choose["spread_abs"] = (choose.ask - choose.bid).where(choose.valid_quote)
    choose["spread_relative"] = choose.spread_abs / choose.midpoint.where(choose.midpoint > 0)
    choose["zero_bid"] = choose.bid.eq(0) & choose.valid_quote
    for column in ("bid_size", "ask_size"):
        if column not in choose: choose[column] = np.nan
        choose[column] = pd.to_numeric(choose[column], errors="coerce")
    choose["quote_size_status"] = np.where(np.isfinite(choose.bid_size) & np.isfinite(choose.ask_size), "REPORTED_NOT_A_FILL_GUARANTEE", "UNKNOWN")
    choose["origin"] = "ACTUAL_CURRENT_COMPLETED_MINUTE"
    return choose.sort_values(["decision_time_ny", "contract_id"], kind="stable").reset_index(drop=True), rejected


def prior_asof_view(observations: pd.DataFrame, grid: pd.DatetimeIndex, max_age: float | None) -> pd.DataFrame:
    """Archival diagnostics only. No previous-session carry; no future backfill."""
    rows = []
    valid = observations[observations.valid_quote].sort_values(["contract_id", "source_time_ny", "source_record_ordinal"], kind="stable")
    for contract_id, group in valid.groupby("contract_id", sort=True):
        source_ns = group.source_time_ny.astype("int64").to_numpy()
        # strict < completed-bar boundary, including sub-minute event ordering
        positions = np.searchsorted(source_ns, grid.asi8, side="left") - 1
        found = positions >= 0
        if not found.any():
            continue
        chosen = group.iloc[positions[found]]
        decisions = grid[found]
        age = (decisions.asi8 - chosen.source_time_ny.astype("int64").to_numpy()) / 1e9
        stale = chosen.normalized_minute.array != (decisions - ONE_MINUTE).array
        result = pd.DataFrame({"contract_id": contract_id, "decision_time_ny": decisions, "raw_quote_time_ny": chosen.source_time_ny.array,
                               "snapshot_record_age_seconds": age, "bid": chosen.bid.to_numpy(), "ask": chosen.ask.to_numpy(),
                               "source_partition_id": chosen.source_partition_id.to_numpy(), "source_record_ordinal": chosen.source_record_ordinal.to_numpy(),
                               "status": np.where(stale, "STALE_ASOF", "CURRENT_MINUTE"), "origin": "ARCHIVAL_PRIOR_ASOF_NOT_EXECUTION"})
        if max_age is not None:
            result.loc[result.snapshot_record_age_seconds > max_age, ["bid", "ask"]] = np.nan
            result.loc[result.snapshot_record_age_seconds > max_age, "status"] = "MAXIMUM_AGE_EXCEEDED"
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def attach_spot_features(panel: pd.DataFrame, underlying_grid: pd.DataFrame, cohort: dict, calendar: Calendar, session: str, initial_spot: float | None) -> pd.DataFrame:
    columns = [c for c in underlying_grid if c not in ("source_partition_id", "source_record_ordinal", "source_sha256", "session_date", "schema_version")]
    result = panel.merge(underlying_grid[columns], on="decision_time_ny", how="left", validate="many_to_one", suffixes=("", "_underlying"))
    result["spot"] = result.spot.where(result.spot > 0)
    result["x_s"] = np.log(result.strike / result.spot)
    result["x_s0"] = np.log(result.strike / initial_spot) if initial_spot is not None and initial_spot > 0 else np.nan
    result["initial_spot"] = initial_spot
    intrinsic = np.where(result.right.eq("C"), np.maximum(result.spot - result.strike, 0), np.maximum(result.strike - result.spot, 0))
    result["intrinsic"] = np.where(result.standard_supported, intrinsic, np.nan)
    result["time_value_mid"] = result.midpoint - result.intrinsic
    result["p_norm"] = result.midpoint / result.spot
    result["time_value_norm"] = result.time_value_mid / result.spot
    result["spread_spot"] = result.spread_abs / result.spot
    result["midpoint_usd_per_contract"] = result.midpoint * result.multiplier
    result["forward_log_moneyness"] = np.nan
    result["forward_status"] = "FORWARD_INPUTS_MISSING"
    result["recomputed_iv"] = np.nan
    result["pricing_status"] = "PRICING_INPUTS_MISSING"
    result["cohort_id"] = cohort["cohort_id"]
    result["session_date"] = session
    result["calendar_date_dte"] = (date.fromisoformat(cohort["expiration_date"]) - date.fromisoformat(session)).days
    bounds = calendar.sessions[session]
    result["minutes_since_open"] = (result.decision_time_ny - bounds["open"]).dt.total_seconds() / 60
    result["minutes_to_session_close"] = (bounds["close"] - result.decision_time_ny).dt.total_seconds() / 60
    result["early_close"] = bounds["close"].hour < 16
    result["elapsed_calendar_seconds_since_t0"] = (result.decision_time_ny - ny_timestamp(cohort["scheduled_t0"])).dt.total_seconds()
    if pd.notna(cohort["T_E"]):
        result["actual_seconds_to_expiry"] = (ny_timestamp(cohort["T_E"]) - result.decision_time_ny).dt.total_seconds().clip(lower=0)
        result["tau_act365f"] = result.actual_seconds_to_expiry / (365 * 24 * 3600)
    else:
        result["actual_seconds_to_expiry"] = np.nan
        result["tau_act365f"] = np.nan
    result["coordinate_type"] = "SPOT_LOG_MONEYNESS"
    return result


def identity_columns() -> list[str]:
    return ["contract_id", "symbol", "root", "expiration_date", "right", "exact_strike", "strike", "multiplier", "deliverable_id", "standard_supported", "metadata_status", "listing_effective_time", "listing_known_at", "metadata_source"]


def contract_inventory(frames: list[pd.DataFrame], cohort: dict, initial_spot: float | None) -> pd.DataFrame:
    pieces = []
    for frame in frames:
        if frame.empty:
            continue
        columns = identity_columns() + ["source_time_ny", "available_at_ny", "first_snapshot_decision_ny", "source_partition_id"]
        pieces.append(frame[columns].copy())
    if not pieces:
        return pd.DataFrame(columns=identity_columns() + ["first_observed_time", "last_observed_time", "first_available_time", "observed_at_t0", "late_appearance", "x_s0", "cohort_id"])
    observations = pd.concat(pieces, ignore_index=True)
    # Unknown metadata stays unknown; never overwrite identity to make histories fit.
    identity = observations[identity_columns()].drop_duplicates("contract_id")
    times = observations.groupby("contract_id", sort=True).agg(first_observed_time=("source_time_ny", "min"), last_observed_time=("source_time_ny", "max"), first_available_time=("available_at_ny", "min"), first_scheduled_observed_decision=("first_snapshot_decision_ny", "min"))
    result = identity.merge(times, on="contract_id", how="left", validate="one_to_one")
    t0 = ny_timestamp(cohort["scheduled_t0"])
    result["observed_at_t0"] = result.first_scheduled_observed_decision.notna() & (result.first_scheduled_observed_decision <= t0)
    result["late_appearance"] = result.first_scheduled_observed_decision.isna() | (result.first_scheduled_observed_decision > t0)
    result["listing_status"] = np.where(result.listing_effective_time.notna(), "VERIFIED_LISTED", "OBSERVED_BUT_LISTING_TIME_UNKNOWN")
    result["x_s0"] = np.log(result.strike / initial_spot) if initial_spot is not None and initial_spot > 0 else np.nan
    result["cohort_id"] = cohort["cohort_id"]
    result["universe_basis"] = "RETROSPECTIVE_LIFECYCLE_OBSERVED_UNION"
    return result.sort_values(["right", "strike", "contract_id"], kind="stable")


def chain_snapshots(panel: pd.DataFrame, ugrid: pd.DataFrame, cohort: dict, calendar: Calendar, session: str, seen_count: int) -> pd.DataFrame:
    grid = cohort_grid(calendar, cohort, session)
    result = ugrid[[c for c in ugrid if c in ("decision_time_ny", "spot", "underlying_source_time", "underlying_age_seconds", "underlying_status", "underlying_log_return_1m", "underlying_log_return_5m", "underlying_log_return_30m", "underlying_rv_30m", "log_return_from_open", "overnight_log_return")]].copy()
    if result.empty:
        result = pd.DataFrame({"decision_time_ny": grid, "spot": np.nan})
    for column in ("observed_contracts", "valid_quotes", "zero_bids", "invalid_quotes", "call_valid", "put_valid"):
        result[column] = 0
    result["mean_spread_spot"] = np.nan
    result["mean_snapshot_record_age_seconds"] = np.nan
    for right in ("C", "P"):
        for i in range(len(MONEYNESS_NAMES)):
            result[f"cell_{right}_{i}"] = np.nan
            result[f"cell_count_{right}_{i}"] = 0
    result = result.set_index("decision_time_ny")
    if not panel.empty:
        groups = panel.groupby("decision_time_ny", sort=True)
        for decision, group in groups:
            if decision not in result.index:
                continue
            valid = group[group.valid_quote]
            result.loc[decision, ["observed_contracts", "valid_quotes", "zero_bids", "invalid_quotes", "call_valid", "put_valid"]] = [len(group), len(valid), int(valid.zero_bid.sum()), len(group) - len(valid), int(valid.right.eq("C").sum()), int(valid.right.eq("P").sum())]
            result.loc[decision, "mean_spread_spot"] = valid.spread_spot.mean()
            result.loc[decision, "mean_snapshot_record_age_seconds"] = valid.snapshot_record_age_seconds.mean()
            for right in ("C", "P"):
                part = valid[valid.right.eq(right) & valid.x_s.notna()].copy()
                bins = np.searchsorted(MONEYNESS_EDGES, part.x_s, side="right") - 1
                for cell, values in part.groupby(bins, sort=True):
                    if 0 <= cell < len(MONEYNESS_NAMES):
                        result.loc[decision, f"cell_{right}_{cell}"] = values.p_norm.mean()
                        result.loc[decision, f"cell_count_{right}_{cell}"] = len(values)
    result = result.reset_index()
    result["cohort_id"] = cohort["cohort_id"]
    result["expiration_date"] = cohort["expiration_date"]
    result["cohort_entry_date"] = cohort["D0"]
    result["session_date"] = session
    result["calendar_date_dte"] = (date.fromisoformat(cohort["expiration_date"]) - date.fromisoformat(session)).days
    result["minutes_since_open"] = (result.decision_time_ny - calendar.sessions[session]["open"]).dt.total_seconds() / 60
    result["minutes_to_session_close"] = (calendar.sessions[session]["close"] - result.decision_time_ny).dt.total_seconds() / 60
    future_sessions = calendar.between(session, cohort["last_scheduled_rth_session"])
    future_minutes = sum(len(cohort_grid(calendar, cohort, d)) for d in future_sessions[1:])
    current_study_close = cohort_grid(calendar, cohort, session)[-1]
    result["remaining_regular_session_minutes"] = (current_study_close - result.decision_time_ny).dt.total_seconds() / 60 + future_minutes
    result["remaining_session_count_including_current"] = len(future_sessions)
    result["weekday"] = date.fromisoformat(session).weekday()
    result["early_close"] = calendar.sessions[session]["close"].hour < 16
    all_days = calendar.days
    pos = all_days.index(session)
    result["next_closed_interval_seconds"] = actual_seconds(calendar.sessions[session]["close"], calendar.sessions[all_days[pos + 1]]["open"]) if pos + 1 < len(all_days) else np.nan
    result["actual_seconds_to_expiry"] = [(max(0., actual_seconds(t, cohort["T_E"])) if pd.notna(cohort["T_E"]) else np.nan) for t in result.decision_time_ny]
    result["observed_universe_seen_so_far"] = seen_count
    result["listed_quote_coverage"] = np.nan
    result["listed_coverage_status"] = "UNVERIFIED_LISTED_UNIVERSE"
    result["chain_id"] = [digest([cohort["cohort_id"], str(t)]) for t in result.decision_time_ny]
    result["origin"] = "ACTUAL_CURRENT_COMPLETED_MINUTE_AGGREGATE"
    if "overnight_log_return" not in result:
        result["overnight_log_return"] = np.nan
    return result


def join_prior_oi(panel: pd.DataFrame, oi: pd.DataFrame, session: str, calendar: Calendar) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = panel.copy()
    panel["prior_oi"] = np.nan
    panel["oi_status"] = "OI_NOT_AVAILABLE_ASOF"
    if oi.empty:
        return panel, oi
    oi = oi.copy()
    oi["open_interest"] = pd.to_numeric(oi.open_interest, errors="coerce")
    require(not oi.contract_id.duplicated().any(), "DUPLICATE_OI_CONTRACT", "Daily OI must be one unambiguous snapshot per identity")
    # Canonical oi_before is eligible at opening; a contradictory future source
    # timestamp overrides no clock and cannot be treated as already published.
    oi["known_at_ny"] = calendar.sessions[session]["open"]
    later = oi.source_time_ny.notna() & (oi.source_time_ny > oi.known_at_ny)
    oi.loc[later, "known_at_ny"] = oi.loc[later, "source_time_ny"]
    oi["source_type"] = "oi_before"
    oi["oi_effective_date"] = None
    oi["availability_evidence"] = "CANONICAL_OI_BEFORE_FOLDER_AT_SESSION_OPEN_WITH_FUTURE_SOURCE_TIMESTAMP_CHECK"
    values = oi[["contract_id", "open_interest", "known_at_ny"]]
    panel = panel.drop(columns="prior_oi").merge(values, on="contract_id", how="left", validate="many_to_one")
    available = (panel.known_at_ny <= panel.decision_time_ny) & np.isfinite(panel.open_interest) & (panel.open_interest >= 0)
    panel["prior_oi"] = panel.open_interest.where(available)
    panel["oi_status"] = np.where(available, "CANONICAL_OI_BEFORE_KNOWN", "OI_NOT_AVAILABLE_ASOF")
    return panel.drop(columns=["open_interest"]), oi


def build_lifecycle_panel(run: Run, catalog: Catalog, calendar: Calendar, cohorts: pd.DataFrame, references: References, resume: bool = True) -> dict:
    require_causal_clocks(run.config)
    store = DataStore(run, catalog, calendar, references)
    completed, blocked = [], []
    for cohort in cohorts.to_dict("records"):
        sessions = json.loads(cohort["expected_sessions"])
        if not sessions:
            continue
        expiration = cohort["expiration_date"]
        cohort_dir = run.path / "panels" / expiration
        warmup_sessions = {calendar.days[calendar.days.index(s) - 1] for s in sessions if calendar.days.index(s) > 0}
        relevant = catalog.frame[(catalog.frame.dataset_type.eq("underlying") & catalog.frame.session_date.isin(set(sessions) | warmup_sessions)) |
                                 (catalog.frame.expiration_date.eq(expiration) & catalog.frame.session_date.isin(sessions))]
        cohort_spec = {key: cohort.get(key) for key in ("cohort_id", "expiration_date", "cohort_definition", "D0", "scheduled_t0", "T_E", "official_last_option_trading_time", "expected_sessions", "last_scheduled_rth_session", "calendar_version", "boundary_status", "right_censored", "terminal_status")}
        cache_key = digest({"semantic_id": run.semantic_id, "sources": relevant.fingerprint.tolist(), "cohort": cohort_spec})
        checkpoint = cohort_dir / "complete.json"
        if resume and checkpoint.exists() and read_json(checkpoint).get("cache_key") == cache_key:
            # Verify every derivative too; a missing/corrupt file is not complete.
            info = read_json(checkpoint)
            if not info.get("blocked_units") and all((run.path / p).is_file() and sha256_file(run.path / p) == h for p, h in info["artifacts"].items()):
                completed.append(expiration)
                continue
        print(f"Building {expiration}: {len(sessions)} scheduled sessions", flush=True)
        inventory_pieces, chain_pieces = [], []
        initial_spot = None
        seen = set()
        artifact_paths = []
        unit_status = []
        for session in sessions:
            unit = f"{session}:{expiration}"
            try:
                underlying = store.underlying(session)
                grid = cohort_grid(calendar, cohort, session)
                source_cutoff = min([calendar.sessions[session]["close"]] + [ny_timestamp(cohort[k]) for k in ("T_E", "official_last_option_trading_time") if pd.notna(cohort.get(k))])
                ugrid = revealed_underlying_grid(underlying, grid, run.config["underlying_asof_max_age_seconds"])
                ugrid["overnight_log_return"] = np.nan
                pos = calendar.days.index(session)
                if pos > 0:
                    previous_session = calendar.days[pos - 1]
                    try:
                        previous = store.underlying(previous_session)
                        previous_close = previous.loc[previous.available_at_ny.eq(calendar.sessions[previous_session]["close"]), "close"] if len(previous) else pd.Series(dtype=float)
                        first_bar = underlying[underlying.bar_start_ny.eq(calendar.sessions[session]["open"])] if len(underlying) else pd.DataFrame()
                        if len(previous_close) and len(first_bar) and previous_close.iloc[0] > 0 and first_bar.open.iloc[0] > 0:
                            ugrid["overnight_log_return"] = math.log(first_bar.open.iloc[0] / previous_close.iloc[0])
                    except ResearchError as exc:
                        run.event("underlying_warmup", exc.code, exc.message, previous_session, **exc.evidence)
                if session == cohort["D0"] and len(ugrid) and pd.notna(ugrid.spot.iloc[0]):
                    initial_spot = float(ugrid.spot.iloc[0])
                spots = underlying.close if len(underlying) else pd.Series(dtype=float)
                quotes = store.options("greeks", session, expiration, spots)
                current = pd.DataFrame()
                if not quotes.empty:
                    current, rejected = current_minute_quotes(quotes, grid, run.config["clocks"]["quote_timestamp_semantics"])
                    # The full normalized event table remains recoverable, invalids included.
                    quotes["quote_status"] = classify_quotes(pd.to_numeric(quotes.bid, errors="coerce"), pd.to_numeric(quotes.ask, errors="coerce"))
                    quotes["valid_quote"] = quotes.quote_status.isin(VALID_QUOTES)
                    in_session = quotes.source_time_ny.notna() & quotes.source_time_ny.dt.strftime("%Y-%m-%d").eq(session) & (quotes.source_time_ny >= calendar.sessions[session]["open"]) & (quotes.source_time_ny < source_cutoff)
                    valid_times = quotes.loc[in_session]
                    inventory_pieces.append(valid_times[identity_columns() + ["source_time_ny", "available_at_ny", "first_snapshot_decision_ny", "source_partition_id"]].copy())
                    event_path = cohort_dir / "option_observations" / (session + ".parquet")
                    atomic_frame(event_path, quotes)
                    artifact_paths.append(event_path)
                    if len(rejected):
                        run.event("panel", "OUTSIDE_RTH_OR_INVALID_TIME", "Excluded from causal snapshots; retained in event archive", unit, count=len(rejected))
                    current = attach_spot_features(current, ugrid, cohort, calendar, session, initial_spot)
                    try:
                        oi = store.options("oi_before", session, expiration, spots)
                        current, oi_events = join_prior_oi(current, oi, session, calendar)
                    except ResearchError as exc:
                        run.event("oi", exc.code, exc.message, unit, **exc.evidence)
                        current, oi_events = join_prior_oi(current, pd.DataFrame(), session, calendar)
                    if not oi_events.empty:
                        oi_path = cohort_dir / "open_interest_observations" / (session + ".parquet")
                        atomic_frame(oi_path, oi_events)
                        artifact_paths.append(oi_path)
                    if run.config["retain_prior_asof_view"]:
                        asof = prior_asof_view(valid_times, grid, run.config["prior_asof_max_age_seconds"])
                        if not asof.empty:
                            asof_path = cohort_dir / "asof_diagnostics" / (session + ".parquet")
                            atomic_frame(asof_path, asof)
                            artifact_paths.append(asof_path)
                    minute_coverage = current.decision_time_ny.nunique() / max(1, len(grid))
                    if minute_coverage < .98:
                        run.event("panel", "PARTIAL_SESSION_DELIVERY", "Partition-level minute integrity warning; sparse contracts are retained", unit, union_minute_coverage=minute_coverage)
                # Trades and option OHLCV expand observed identity inventory even if
                # a contract has no Greek quote. Neither is substituted for bid/ask.
                for kind in ("trades", "ohlcv"):
                    try:
                        extra = store.options(kind, session, expiration, spots)
                        if not extra.empty:
                            time_mask = extra.source_time_ny.notna() & (extra.source_time_ny >= calendar.sessions[session]["open"]) & (extra.source_time_ny < source_cutoff)
                            extra = extra.loc[time_mask]
                            inventory_pieces.append(extra[identity_columns() + ["source_time_ny", "available_at_ny", "first_snapshot_decision_ny", "source_partition_id"]].copy())
                            if kind == "trades" and len(current):
                                extra["size"] = pd.to_numeric(extra["size"], errors="coerce")
                                extra["price"] = pd.to_numeric(extra["price"], errors="coerce")
                                extra = extra[np.isfinite(extra["size"]) & (extra["size"] >= 0) & np.isfinite(extra.price) & (extra.price >= 0)].copy()
                                volume = extra.assign(decision_time_ny=extra.normalized_minute + ONE_MINUTE).groupby(["contract_id", "decision_time_ny"], sort=True)["size"].sum().rename("trade_volume_current_minute").reset_index()
                                current = current.merge(volume, on=["contract_id", "decision_time_ny"], how="left", validate="one_to_one")
                                current["volume_source"] = "VALIDATED_TRADE_SIZE_SUM_CURRENT_COMPLETED_MINUTE"
                    except ResearchError as exc:
                        run.event(kind, exc.code, exc.message, unit, **exc.evidence)
                new_ids = set(current.contract_id) if len(current) else set()
                seen.update(new_ids)
                chains = chain_snapshots(current, ugrid, cohort, calendar, session, len(seen))
                chain_pieces.append(chains)
                if len(current):
                    output = cohort_dir / "lifecycle_observations" / (session + ".parquet")
                    atomic_frame(output, current)
                    artifact_paths.append(output)
                unit_status.append({"session_date": session, "status": "BUILT" if len(current) else "NO_VALID_CURRENT_OBSERVATION", "current_rows": len(current)})
            except ResearchError as exc:
                blocked.append(unit)
                run.event("panel", exc.code, exc.message, unit, **exc.evidence)
                unit_status.append({"session_date": session, "status": exc.code, "current_rows": 0})
                # A blocked unit still has every scheduled missing chain slot.
                missing_ugrid = revealed_underlying_grid(pd.DataFrame(), cohort_grid(calendar, cohort, session), 0)
                chain_pieces.append(chain_snapshots(pd.DataFrame(), missing_ugrid, cohort, calendar, session, len(seen)))
            run.memory_check()
        inventory = contract_inventory(inventory_pieces, cohort, initial_spot)
        # Explicit listed-but-unquoted identities enter only with PIT references.
        inventory = add_listed_inventory(inventory, references, cohort, initial_spot)
        master_path = cohort_dir / "contract_master.parquet"
        atomic_frame(master_path, inventory)
        artifact_paths.append(master_path)
        chains = pd.concat(chain_pieces, ignore_index=True).sort_values("decision_time_ny", kind="stable")
        chains["observed_bar_index"] = np.arange(len(chains))
        first_available = inventory.first_scheduled_observed_decision.dropna().sort_values() if len(inventory) else pd.Series(dtype="datetime64[ns, America/New_York]")
        chains["observed_universe_seen_so_far"] = np.searchsorted(pd.DatetimeIndex(first_available).asi8, pd.DatetimeIndex(chains.decision_time_ny).asi8, side="right")
        chains["retrospective_observed_contracts"] = int((inventory.universe_basis == "RETROSPECTIVE_LIFECYCLE_OBSERVED_UNION").sum()) if len(inventory) else 0
        denominator = chains.retrospective_observed_contracts.replace(0, np.nan)
        chains["quote_coverage_retrospective_observed_universe"] = chains.valid_quotes / denominator
        chains["missing_contract_slots_retrospective"] = chains.retrospective_observed_contracts - chains.valid_quotes
        if expiration in references.complete_expirations:
            listed = inventory[inventory.listing_known_at.notna() & inventory.listing_effective_time.notna()]
            if len(listed):
                known = pd.DatetimeIndex(listed.listing_known_at).asi8
                effective = pd.DatetimeIndex(listed.listing_effective_time).asi8
                thresholds = np.sort(np.maximum(known, effective))
                chains["listed_contract_count"] = np.searchsorted(thresholds, pd.DatetimeIndex(chains.decision_time_ny).asi8, side="right")
                numerators = []
                for path in sorted((cohort_dir / "lifecycle_observations").glob("*.parquet")):
                    observed = pd.read_parquet(path, columns=["decision_time_ny", "valid_quote", "listing_known_at", "listing_effective_time"])
                    visible = observed.valid_quote & observed.listing_known_at.notna() & observed.listing_effective_time.notna()
                    visible &= (observed.listing_known_at <= observed.decision_time_ny) & (observed.listing_effective_time <= observed.decision_time_ny)
                    numerators.append(observed.loc[visible].groupby("decision_time_ny").size())
                numerator = pd.concat(numerators).groupby(level=0).sum() if numerators else pd.Series(dtype=float)
                chains["listed_valid_quote_count"] = chains.decision_time_ny.map(numerator).fillna(0)
                chains["listed_quote_coverage"] = chains.listed_valid_quote_count / chains.listed_contract_count.replace(0, np.nan)
                chains["listed_coverage_status"] = "PIT_VERIFIED_LISTED_UNIVERSE"
        chains_path = cohort_dir / "chain_snapshots.parquet"
        atomic_frame(chains_path, chains)
        artifact_paths.append(chains_path)
        unit_path = cohort_dir / "unit_status.csv"
        atomic_frame(unit_path, pd.DataFrame(unit_status))
        artifact_paths.append(unit_path)
        grid_contract_path = cohort_dir / "sparse_grid_contract.json"
        atomic_json(grid_contract_path, {"schema_version": SCHEMA_VERSION, "cohort": cohort_spec,
                    "grid": "chain_snapshots.parquet decision_time_ny", "contracts": "contract_master.parquet",
                    "missingness": "Absent contract/time key in lifecycle_observations is missing, never zero. Listed intervals use known_at and effective_time.",
                    "late_appearance": "first_scheduled_observed_decision > scheduled_t0; unknown availability is explicitly late/unsupported. The union is retrospective, not an entry tradable universe.",
                    "initial_spot": initial_spot, "terminal_status": cohort["terminal_status"]})
        artifact_paths.append(grid_contract_path)
        atomic_json(checkpoint, {"cache_key": cache_key, "artifacts": {str(p.relative_to(run.path)): sha256_file(p) for p in artifact_paths},
                                 "units": unit_status, "status": "PARTIAL_WITH_BLOCKED_UNITS" if any(x.endswith(expiration) for x in blocked) else "COMPLETE_FOR_AVAILABLE_INPUTS", "blocked_units": [x for x in blocked if x.endswith(expiration)]})
        completed.append(expiration)
    atomic_frame(run.path / "source_catalog.parquet", catalog.frame)
    for index, cohort in cohorts.iterrows():
        directory = run.path / "panels" / cohort.expiration_date
        master_path = directory / "contract_master.parquet"
        if master_path.exists():
            master = pd.read_parquet(master_path)
            cohorts.loc[index, "initial_contract_count_observed"] = int(master.observed_at_t0.sum()) if len(master) else 0
            cohorts.loc[index, "lifecycle_observed_contract_count"] = int(master.universe_basis.eq("RETROSPECTIVE_LIFECYCLE_OBSERVED_UNION").sum()) if len(master) else 0
            cohorts.loc[index, "initial_contract_count_known"] = int(((master.listing_known_at <= cohort.scheduled_t0) & (master.listing_effective_time <= cohort.scheduled_t0)).sum()) if cohort.expiration_date in references.complete_expirations and len(master) else np.nan
    atomic_frame(run.path / "cohort_registry.parquet", cohorts)
    atomic_frame(run.path / "cohort_registry.csv", cohorts)
    result = {"completed_expirations": completed, "blocked_units": blocked, "status": "PARTIAL" if blocked else "BUILT_AVAILABLE_SCOPE"}
    atomic_json(run.path / "panel_build_status.json", result)
    return result


def add_listed_inventory(inventory: pd.DataFrame, references: References, cohort: dict, initial_spot: float | None) -> pd.DataFrame:
    metadata = references.metadata
    if metadata.empty:
        return inventory
    rows = []
    expiration = cohort["expiration_date"]
    for ref in metadata[metadata.expiration.astype(str).eq(expiration)].to_dict("records"):
        identity = {"symbol": "QQQ", "root": str(ref["root"]), "expiration_date": expiration, "right": ref["right"], "exact_strike": ref["exact_strike"],
                    "multiplier": float(ref["multiplier"]), "deliverable_id": str(ref["deliverable_id"])}
        contract_id = digest(identity)
        if len(inventory) and contract_id in set(inventory.contract_id):
            continue
        rows.append({**identity, "contract_id": contract_id, "strike": float(ref["exact_strike"]), "standard_supported": False,
                     "metadata_status": "LISTING_REFERENCE_ONLY_PRODUCT_PROFILE_REQUIRED", "listing_effective_time": ref["listing_effective_time"],
                     "listing_known_at": ref["listing_known_at"], "metadata_source": ref["source"], "first_observed_time": pd.NaT,
                     "last_observed_time": pd.NaT, "first_available_time": pd.NaT, "observed_at_t0": False, "late_appearance": False,
                     "first_scheduled_observed_decision": pd.NaT,
                     "listing_status": "LISTED_NO_OBSERVATION", "x_s0": math.log(float(ref["exact_strike"]) / initial_spot) if initial_spot else np.nan,
                     "cohort_id": cohort["cohort_id"], "universe_basis": "VERIFIED_LISTED_NO_OBSERVATION"})
    if not rows:
        return inventory
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries", category=FutureWarning)
        return pd.concat([inventory, pd.DataFrame(rows)], ignore_index=True)


def load_cohort_panel(directory: Path) -> pd.DataFrame:
    files = sorted((directory / "lifecycle_observations").glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).sort_values(["contract_id", "decision_time_ny"], kind="stable") if files else pd.DataFrame()


def scheduled_anchor_mask(times: pd.Series, calendar: Calendar, spacing: int, cohort: dict | None = None) -> np.ndarray:
    result = np.zeros(len(times), dtype=bool)
    for session in sorted(times.dt.strftime("%Y-%m-%d").unique()):
        positions = np.flatnonzero(times.dt.strftime("%Y-%m-%d").eq(session).to_numpy())
        grid = cohort_grid(calendar, cohort, session) if cohort is not None else calendar.grid(session)
        selected = set(grid[::spacing]) | {grid[0], grid[-1]}
        result[positions] = times.iloc[positions].isin(selected).to_numpy()
    return result


def endpoint_for(anchor: pd.Timestamp, horizon: int | str, calendar: Calendar, cohort: dict) -> tuple[pd.Timestamp | None, str]:
    session = anchor.strftime("%Y-%m-%d")
    grid = cohort_grid(calendar, cohort, session)
    if isinstance(horizon, (int, np.integer)):
        requested = anchor + int(horizon) * ONE_MINUTE
        return requested, "SCHEDULED" if requested <= grid[-1] else "HORIZON_OUTSIDE_SESSION"
    if horizon == "SESSION_FINAL_OBSERVATION":
        return grid[-1], "SCHEDULED"
    if horizon == "NEXT_SESSION_FIRST_SCHEDULED_COMPLETED_BAR":
        later = [d for d in calendar.days if d > session]
        if later and later[0] <= cohort["last_scheduled_rth_session"]:
            next_grid = cohort_grid(calendar, cohort, later[0])
            return (next_grid[0], "SCHEDULED") if len(next_grid) else (None, "HORIZON_OUTSIDE_LIFECYCLE")
        return None, "HORIZON_OUTSIDE_LIFECYCLE"
    if horizon == "VERIFIED_EXPIRY_PAYOFF":
        return (ny_timestamp(cohort["T_E"]), "TERMINAL_REFERENCE_REQUIRED") if pd.notna(cohort["T_E"]) else (None, "TERMINAL_REFERENCE_MISSING")
    raise ResearchError("INVALID_HORIZON", str(horizon))


def sampled_hit(path_values: np.ndarray, expected_slots: int, target: float) -> dict:
    finite = np.isfinite(path_values)
    hits = np.flatnonzero(finite & (path_values >= target))
    if len(hits):
        return {"status": "SUCCESS_OBSERVED", "first_hit_sample_index": int(hits[0]), "lower": 1., "upper": 1.}
    complete = len(path_values) == expected_slots and finite.all()
    return {"status": "FAILURE_OBSERVED" if complete else "UNRESOLVED", "first_hit_sample_index": None, "lower": 0., "upper": 0. if complete else 1.}


def generate_contract_labels(panel: pd.DataFrame, inventory: pd.DataFrame, cohort: dict, calendar: Calendar, config: dict, references: References) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    spacing = config["labels"]["anchor_spacing_minutes"]
    anchor_times = sorted({t for session in json.loads(cohort["expected_sessions"]) for t in cohort_grid(calendar, cohort, session)[::spacing]} |
                          {t for session in json.loads(cohort["expected_sessions"]) for t in (cohort_grid(calendar, cohort, session)[0], cohort_grid(calendar, cohort, session)[-1])})
    horizons: list[int | str] = list(config["labels"]["horizons_minutes"])
    if config["labels"]["named_horizons"]:
        horizons.extend(["SESSION_FINAL_OBSERVATION", "NEXT_SESSION_FIRST_SCHEDULED_COMPLETED_BAR", "VERIFIED_EXPIRY_PAYOFF"])
    terminal = references.terminal(cohort["expiration_date"])
    all_grid = pd.DatetimeIndex([t for session in json.loads(cohort["expected_sessions"]) for t in cohort_grid(calendar, cohort, session)])
    endpoint_cache = {}
    for anchor in anchor_times:
        for horizon in horizons:
            end, status = endpoint_for(anchor, horizon, calendar, cohort)
            left = int(all_grid.searchsorted(anchor, side="right"))
            right = int(all_grid.searchsorted(end, side="right")) if end is not None else left
            endpoint_cache[(anchor, str(horizon))] = (end, status, left, right)
    rows = []
    for contract_id, group in panel.groupby("contract_id", sort=True):
        group = group.set_index("decision_time_ny").sort_index()
        row_lookup = {row.Index: row for row in group.itertuples(index=True)}
        midpoint_grid = group.midpoint.reindex(all_grid).to_numpy(dtype=float)
        anchor_group = group.loc[group.index.isin(anchor_times)]
        for anchor, start in anchor_group.iterrows():
            for horizon in horizons:
                end, status, left, right = endpoint_cache[(anchor, str(horizon))]
                item = {"sample_id": digest([cohort["cohort_id"], contract_id, str(anchor), horizon]), "cohort_id": cohort["cohort_id"], "contract_id": contract_id,
                        "anchor_time": anchor, "horizon_id": str(horizon), "requested_end_time": end, "actual_end_time": pd.NaT,
                        "label_available_at": end, "right": start.right, "x_s0": start.x_s0, "x_s": start.x_s,
                        "requested_elapsed_calendar_seconds": actual_seconds(anchor, end) if end is not None else np.nan,
                        "label_basis": "FIXED_CONTRACT_MIDPOINT_PRICE_UNITS", "delta_mid": np.nan, "delta_normalized_price": np.nan,
                        "delta_time_value": np.nan, "delta_usd_per_contract": np.nan, "spread_change": np.nan,
                        "valid_quote_presence_at_end": None, "zero_bid_transition": None,
                        "quote_presence_basis": "CAPTURED_CURRENT_MINUTE_RECORD_NOT_MARKET_QUOTE_EXISTENCE",
                        "outcome_mask": False, "censoring_reason": status, "origin": "ACTUAL_FUTURE_LABEL_NEVER_FEATURE",
                        "anchor_source_partition_id": start.source_partition_id, "anchor_source_record_ordinal": int(start.source_record_ordinal),
                        "endpoint_source_partition_id": None, "endpoint_source_record_ordinal": None,
                        "sampled_grid_max_mid_change": np.nan, "sampled_grid_min_mid_change": np.nan,
                        "path_grid_complete": False, "sampled_hit_status": None, "first_hit_time_ny": pd.NaT}
                if horizon == "VERIFIED_EXPIRY_PAYOFF":
                    item["label_basis"] = "STANDARD_OPTION_VALUATION_PAYOFF_NOT_ACCOUNT_SETTLEMENT"
                    if status != "TERMINAL_REFERENCE_MISSING" and terminal and bool(start.standard_supported) and bool(start.valid_quote):
                        payoff = max(float(terminal["price"]) - start.strike, 0.) if start.right == "C" else max(start.strike - float(terminal["price"]), 0.)
                        item.update(delta_mid=payoff - start.midpoint, delta_normalized_price=(payoff - start.midpoint) / start.spot if start.spot > 0 else np.nan,
                                    delta_usd_per_contract=(payoff - start.midpoint) * start.multiplier,
                                    delta_time_value=-start.time_value_mid,
                                    actual_end_time=ny_timestamp(terminal["reference_time_ny"]), label_available_at=ny_timestamp(terminal["known_at_ny"]), outcome_mask=True,
                                    censoring_reason="VERIFIED_VALUATION_PAYOFF", endpoint_source_partition_id=str(terminal["source"]))
                elif status == "SCHEDULED":
                    target = row_lookup.get(end)
                    item["valid_quote_presence_at_end"] = bool(target.valid_quote) if target is not None else False
                    if bool(start.valid_quote) and target is not None and bool(target.valid_quote):
                        delta = target.midpoint - start.midpoint
                        item.update(delta_mid=delta, delta_normalized_price=delta / start.spot if start.spot > 0 else np.nan,
                                    delta_time_value=target.time_value_mid - start.time_value_mid,
                                    delta_usd_per_contract=delta * start.multiplier if pd.notna(start.multiplier) else np.nan,
                                    spread_change=target.spread_abs - start.spread_abs,
                                    zero_bid_transition=bool(target.zero_bid) and not bool(start.zero_bid),
                                    actual_end_time=end, outcome_mask=True, censoring_reason="OBSERVED_EXACT_ENDPOINT",
                                    endpoint_source_partition_id=target.source_partition_id, endpoint_source_record_ordinal=int(target.source_record_ordinal))
                    else:
                        item["censoring_reason"] = "LABEL_UNRESOLVED_MISSING_OR_INVALID_ENDPOINT"
                    if bool(start.valid_quote) and end is not None and end > anchor:
                        # Sampled RTH grid, never an implied continuous-time path.
                        expected = all_grid[left:right]
                        values = midpoint_grid[left:right] - start.midpoint
                        if np.isfinite(values).any():
                            item["sampled_grid_max_mid_change"] = float(np.nanmax(values))
                            item["sampled_grid_min_mid_change"] = float(np.nanmin(values))
                        item["path_grid_complete"] = bool(np.isfinite(values).all())
                        targets = config["labels"]["sampled_hit_targets_price_units"]
                        if targets:
                            # Every configured target has a separate, explicit field.
                            for target_index, target_price in enumerate(targets):
                                hit = sampled_hit(values, len(expected), float(target_price))
                                prefix = f"hit_target_{target_index}"
                                item[prefix + "_price_units"] = target_price
                                item[prefix + "_status"] = hit["status"]
                                item[prefix + "_lower_missingness_bound"] = hit["lower"]
                                item[prefix + "_upper_missingness_bound"] = hit["upper"]
                                hit_time = expected[hit["first_hit_sample_index"]] if hit["first_hit_sample_index"] is not None else pd.NaT
                                item[prefix + "_time_ny"] = hit_time
                                item[prefix + "_time_status"] = ("EXACT_ON_COMPLETE_OBSERVED_PREFIX" if hit["first_hit_sample_index"] is not None and np.isfinite(values[:hit["first_hit_sample_index"] + 1]).all() else "UNRESOLVED_OR_UPPER_BOUND_WITH_MISSING_PREFIX")
                                if target_index == 0:
                                    item["sampled_hit_status"] = hit["status"]
                                    item["first_hit_time_ny"] = hit_time
                if bool(cohort["right_censored"]) and not item["outcome_mask"] and item["censoring_reason"].startswith("LABEL_UNRESOLVED"):
                    item["censoring_reason"] = "COHORT_RIGHT_CENSORED"
                rows.append(item)
    return pd.DataFrame(rows)


def generate_labels(run: Run, cohorts: pd.DataFrame, calendar: Calendar, references: References) -> dict:
    result = {}
    for cohort in cohorts.to_dict("records"):
        directory = run.path / "panels" / cohort["expiration_date"]
        if not (directory / "complete.json").exists():
            continue
        checkpoint = directory / "labels_complete.json"
        source_key = digest({"panel": read_json(directory / "complete.json"), "labels": run.config["labels"], "code": run.code_hash})
        output = directory / "labels.parquet"
        if checkpoint.exists() and output.exists() and read_json(checkpoint).get("source_key") == source_key and read_json(checkpoint).get("sha256") == sha256_file(output):
            result[cohort["expiration_date"]] = "REUSED_VALIDATED_OUTPUT"
            continue
        panel = load_cohort_panel(directory)
        inventory = pd.read_parquet(directory / "contract_master.parquet")
        print(f"Labels {cohort['expiration_date']}", flush=True)
        labels = generate_contract_labels(panel, inventory, cohort, calendar, run.config, references)
        atomic_frame(output, labels)
        atomic_json(checkpoint, {"source_key": source_key, "sha256": sha256_file(output), "rows": len(labels)})
        result[cohort["expiration_date"]] = len(labels)
        run.memory_check()
    atomic_json(run.path / "labels_status.json", result)
    return result


def intrinsic_value(spot: float, strike: float, right: str) -> float:
    return max(spot - strike, 0.) if right == "C" else max(strike - spot, 0.)


def american_tree_price(spot: float, strike: float, tau: float, sigma: float, right: str, rate: float, dividend_yield: float, steps: int = 800) -> float:
    """Exercise-aware CRR benchmark with an EXPLICIT continuous-yield carry model.

    No default rate, dividend yield, multiplier, or cash-dividend substitution.
    Discrete-cash-dividend products require a different verified pricing adapter.
    """
    require(spot > 0 and strike > 0 and tau >= 0 and sigma >= 0 and right in ("C", "P") and steps >= 2, "INVALID_PRICING_INPUT", "Invalid American pricing inputs")
    if tau == 0:
        return intrinsic_value(spot, strike, right)
    if sigma < 1e-10:
        times = np.linspace(0., tau, steps + 1)
        future_spot = spot * np.exp((rate - dividend_yield) * times)
        values = np.maximum(future_spot - strike, 0.) if right == "C" else np.maximum(strike - future_spot, 0.)
        return float(np.max(np.exp(-rate * times) * values))
    dt = tau / steps
    up = math.exp(sigma * math.sqrt(dt))
    down = 1. / up
    probability = (math.exp((rate - dividend_yield) * dt) - down) / (up - down)
    require(0 <= probability <= 1, "PRICER_NOT_CONVERGED", "CRR probability outside [0,1]; refine steps")
    discount = math.exp(-rate * dt)
    nodes = spot * np.exp((2 * np.arange(steps + 1) - steps) * math.log(up))
    values = np.maximum(nodes - strike, 0.) if right == "C" else np.maximum(strike - nodes, 0.)
    for n in range(steps - 1, -1, -1):
        values = discount * (probability * values[1:] + (1 - probability) * values[:-1])
        nodes = spot * np.exp((2 * np.arange(n + 1) - n) * math.log(up))
        exercise = np.maximum(nodes - strike, 0.) if right == "C" else np.maximum(strike - nodes, 0.)
        values = np.maximum(values, exercise)
    return float(values[0])


def american_fd_price(spot: float, strike: float, tau: float, sigma: float, right: str, rate: float, dividend_yield: float,
                      space_steps: int = 500, time_steps: int = 500, domain_multiple: float = 4., tolerance: float = 1e-9) -> float:
    """Independent fully implicit obstacle solver, continuous-yield domain ONLY.

    Projected SOR handles exercise at every step. Domain/grid convergence must be
    established per supported input region; a single successful solve is insufficient.
    """
    if tau == 0 or sigma < 1e-10:
        return american_tree_price(spot, strike, tau, sigma, right, rate, dividend_yield, max(100, time_steps))
    require(spot > 0 and strike > 0 and tau > 0 and sigma > 0 and right in ("C", "P"), "INVALID_PRICING_INPUT", "Invalid FD inputs")
    maximum = domain_multiple * max(spot, strike)
    ds, dt = maximum / space_steps, tau / time_steps
    stock = np.linspace(0., maximum, space_steps + 1)
    payoff = np.maximum(stock - strike, 0.) if right == "C" else np.maximum(strike - stock, 0.)
    values = payoff.copy()
    i = np.arange(1, space_steps)
    diffusion = .5 * sigma**2 * i**2
    drift = (rate - dividend_yield) * i
    low, high = diffusion - .5 * drift, diffusion + .5 * drift
    bad = (low < 0) | (high < 0)
    low[bad] = diffusion[bad] + np.maximum(-drift[bad], 0.)
    high[bad] = diffusion[bad] + np.maximum(drift[bad], 0.)
    lower, upper = -dt * low, -dt * high
    diagonal = 1. + dt * (low + high + rate)
    for n in range(1, time_steps + 1):
        remaining = n * dt
        boundary_low = 0. if right == "C" else strike * max(1., math.exp(-rate * remaining))
        boundary_high = max(maximum - strike, maximum * math.exp(-dividend_yield * remaining) - strike * math.exp(-rate * remaining), 0.) if right == "C" else 0.
        rhs = values[1:-1].copy()
        rhs[0] -= lower[0] * boundary_low
        rhs[-1] -= upper[-1] * boundary_high
        interior = values[1:-1].copy()
        for iteration in range(10000):
            error = 0.
            for j in range(len(interior)):
                left = lower[j] * interior[j - 1] if j else 0.
                right_value = upper[j] * interior[j + 1] if j + 1 < len(interior) else 0.
                candidate = (rhs[j] - left - right_value) / diagonal[j]
                new = max(payoff[j + 1], interior[j] + 1.15 * (candidate - interior[j]))
                error = max(error, abs(new - interior[j]))
                interior[j] = new
            if error < tolerance:
                break
        else:
            raise ResearchError("PRICER_NOT_CONVERGED", "American complementarity iteration exhausted")
        values[0], values[-1], values[1:-1] = boundary_low, boundary_high, interior
    return float(np.interp(spot, stock, values))


def implied_vol_interval(pricer: Any, bid: float, ask: float, tau: float, maximum_sigma: float = 4., price_tolerance: float = 1e-5) -> dict:
    if tau <= 0:
        return {"status": "TERMINAL_NO_IV", "iv_mid": None, "iv_low": None, "iv_high": None}
    require(np.isfinite([bid, ask]).all() and bid >= 0 and ask >= bid and ask > 0, "INVALID_QUOTE", "Cannot invert invalid quotes")
    low_price, high_price = pricer(0.), pricer(maximum_sigma)
    if ask < low_price - price_tolerance or bid > high_price + price_tolerance:
        return {"status": "IV_NOT_IDENTIFIABLE_OUTSIDE_PRICER_DOMAIN", "iv_mid": None, "iv_low": None, "iv_high": None}
    if high_price - low_price <= price_tolerance:
        return {"status": "IV_NOT_IDENTIFIABLE_FLAT_EXERCISE_REGION", "iv_mid": None, "iv_low": 0., "iv_high": maximum_sigma}
    def inverse(price: float) -> float | None:
        if price <= low_price + price_tolerance:
            return 0.
        if price > high_price:
            return None
        return float(brentq(lambda sigma: pricer(sigma) - price, 1e-8, maximum_sigma, xtol=1e-7))
    lower, upper = inverse(bid), inverse(ask)
    midpoint = inverse((bid + ask) / 2)
    if midpoint is not None:
        bump = 1e-3
        vega = (pricer(min(maximum_sigma, midpoint + bump)) - pricer(max(0., midpoint - bump))) / max(1e-12, min(maximum_sigma, midpoint + bump) - max(0., midpoint - bump))
    else:
        vega = None
    return {"status": "IV_INTERVAL_SUPPORTED" if upper is not None and vega is not None and vega > price_tolerance else "IV_NOT_IDENTIFIABLE",
            "iv_mid": midpoint if vega is not None and vega > price_tolerance else None, "iv_low": lower, "iv_high": upper,
            "vega_price_units_per_vol_decimal": vega, "numerical_price_tolerance": price_tolerance, "maximum_sigma_domain": maximum_sigma}


def fit_constrained_shape(strikes: np.ndarray, bid: np.ndarray, ask: np.ndarray, spot: float, right: str, config: dict, discount_maximum: float | None = None) -> dict:
    """Interval-aware monotone/convex piecewise-linear baseline on actual strikes."""
    order = np.argsort(strikes, kind="stable")
    strikes, bid, ask = np.asarray(strikes, float)[order], np.asarray(bid, float)[order], np.asarray(ask, float)[order]
    require(len(strikes) >= 3 and np.all(np.diff(strikes) > 0), "SHAPE_SUPPORT_INSUFFICIENT", "Need three distinct supported strikes")
    require(spot > 0 and right in ("C", "P"), "SHAPE_SUPPORT_INSUFFICIENT", "Invalid shape domain")
    x, b, a = strikes / spot, bid / spot, ask / spot
    mid = (b + a) / 2
    scale = np.maximum((ask - bid) / 2, config["numeric_scale_floor"])
    weights = np.minimum(100., 1. / scale**2)
    weights /= weights.mean()
    intrinsic = np.maximum(1 - x, 0.) if right == "C" else np.maximum(x - 1, 0.)
    slopes = np.zeros((len(x) - 1, len(x)))
    for i, gap in enumerate(np.diff(x)):
        slopes[i, i], slopes[i, i + 1] = -1 / gap, 1 / gap
    convex = np.diff(slopes, axis=0)
    low = np.full(len(x) - 1, -np.inf if right == "C" else 0.)
    high = np.full(len(x) - 1, 0. if right == "C" else np.inf)
    if discount_maximum is not None:
        require(discount_maximum >= 1., "INVALID_PRICING_INPUT", "B includes D(t,t)=1")
        if right == "C": low[:] = -discount_maximum
        else: high[:] = discount_maximum
    linear = [LinearConstraint(slopes, low, high), LinearConstraint(convex, 0., np.inf)]
    lambda_mid, lambda_smooth = config["midpoint_penalty"], config["smooth_penalty"]
    def objective(prices: np.ndarray) -> float:
        interval_error = np.maximum(b - prices, 0.) + np.minimum(a - prices, 0.)
        return float(np.sum(weights * interval_error**2) + lambda_mid * np.sum(weights * (prices - mid)**2) + lambda_smooth * np.sum((convex @ prices)**2))
    def gradient(prices: np.ndarray) -> np.ndarray:
        interval_difference = prices - np.clip(prices, b, a)
        return 2 * weights * (interval_difference + lambda_mid * (prices - mid)) + 2 * lambda_smooth * convex.T @ (convex @ prices)
    fit = minimize(objective, np.maximum(mid, intrinsic), jac=gradient, method="SLSQP", bounds=Bounds(intrinsic, np.inf), constraints=linear, options={"maxiter": 300, "ftol": 1e-12})
    require(fit.success and np.min(convex @ fit.x) >= -1e-7 and np.all(slopes @ fit.x >= low - 1e-7) and np.all(slopes @ fit.x <= high + 1e-7), "SHAPE_FIT_FAILED", fit.message)
    prices = fit.x * spot
    slack = np.maximum(bid - prices, 0.) + np.maximum(prices - ask, 0.)
    return {"strikes": strikes, "fitted_prices": prices, "order": order, "slacks": slack, "weights": weights,
            "status": "INCONSISTENT_QUOTE_INTERVALS" if np.max(slack) > 1e-5 else "WITHIN_QUOTE_INTERVALS",
            "max_adjustment_price_units": float(np.max(np.abs(prices - (bid + ask) / 2))),
            "unsupported_gaps": np.diff(np.log(strikes)) > config["max_interpolation_log_gap"], "constraint_violation_count": 0}


def interpolate_supported_shape(fit: dict, strike: float) -> tuple[float, str]:
    k, prices = fit["strikes"], fit["fitted_prices"]
    if strike < k[0] or strike > k[-1]:
        return np.nan, "UNSUPPORTED_EXTRAPOLATION"
    exact = np.flatnonzero(k == strike)
    if len(exact):
        return float(prices[exact[0]]), "OBSERVED_STRIKE_FITTED_PRICE"
    position = np.searchsorted(k, strike) - 1
    if fit["unsupported_gaps"][position]:
        return np.nan, "UNSUPPORTED_LARGE_STRIKE_GAP"
    return float(np.interp(strike, k, prices)), "WITHIN_SUPPORTED_INTERPOLATION"


def moneyness_bin(values: pd.Series) -> pd.Series:
    # Edges are log coordinates, predeclared rather than learned from full history.
    result = pd.cut(values, MONEYNESS_EDGES, labels=MONEYNESS_NAMES, right=False).astype("object")
    return result.fillna("UNKNOWN_COORDINATE")


def calendar_block_ids(days: pd.Series, calendar: Calendar, block_sessions: int) -> pd.Series:
    positions = {day: i // block_sessions for i, day in enumerate(calendar.days)}
    return days.astype(str).map(positions)


def block_bootstrap_expiration_mean(block_table: pd.DataFrame, repetitions: int, seed: int) -> dict:
    """Resample calendar blocks with ALL co-occurring expirations kept together.

    Input columns: cohort_id, calendar_block, value_sum, value_count. Recompute
    within-expiration means before equally averaging the contributing expirations.
    Weight-concentration ESS is never called an independent sample count.
    """
    cohorts = sorted(block_table.cohort_id.unique())
    blocks = sorted(block_table.calendar_block.dropna().unique())
    if len(cohorts) < 2 or len(blocks) < 2:
        return {"lower": None, "upper": None, "status": "INSUFFICIENT_INDEPENDENT_SUPPORT", "calendar_blocks": len(blocks)}
    sums = block_table.pivot_table(index="cohort_id", columns="calendar_block", values="value_sum", aggfunc="sum", fill_value=0).reindex(index=cohorts, columns=blocks, fill_value=0).to_numpy(float)
    counts = block_table.pivot_table(index="cohort_id", columns="calendar_block", values="value_count", aggfunc="sum", fill_value=0).reindex(index=cohorts, columns=blocks, fill_value=0).to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        multiplicities = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks))
        numerator, denominator = sums @ multiplicities, counts @ multiplicities
        with np.errstate(invalid="ignore", divide="ignore"):
            means = numerator / denominator
        estimates.append(float(np.nanmean(means)) if np.isfinite(means).any() else np.nan)
    return {"lower": float(np.nanquantile(estimates, .025)), "upper": float(np.nanquantile(estimates, .975)),
            "status": "CALENDAR_BLOCK_BOOTSTRAP_95_PERCENT", "calendar_blocks": len(blocks), "expirations": len(cohorts), "repetitions": repetitions}


def describe(run: Run, cohorts: pd.DataFrame, calendar: Calendar, include_final: bool = False) -> dict:
    split_file = run.path / "split_manifest.json"
    cutoff = None if include_final or not split_file.exists() else ny_timestamp(read_json(split_file)["sealed_test_start"])
    estimands = {
        "population": "Every observed contract in scheduled exact cohort; observed-universe denominator is retrospective and is not a listing claim",
        "price_units": "Quoted option price units; USD only when actual multiplier is verified",
        "mean": "Equal mean of contributing expiration-level cell means, conditional on valid required observations",
        "quantiles": "Within-expiration observed quantiles; cross-expiration quantiles describe expiration cell means, not pooled quotes",
        "coverage": "Valid current-minute quotes / scheduled slots in retrospective observed universe; listed denominator remains separate",
        "uncertainty": "Calendar-session block bootstrap; all co-occurring contracts and expirations move together",
        "coordinates": {"fixed": "x_s0=log(K/S_t0)", "moving": "x_s=log(K/S_t)"},
        "cohort_definition": run.config["cohort_definition"], "decision_clock": run.config["clocks"],
        "sampled_hits": "One-minute observed midpoint grid; missing-data bounds, not confidence intervals or fills",
        "test_visibility": "FULL_ARCHIVE" if cutoff is None else "PRE_FINAL_TEST_ONLY",
    }
    atomic_json(run.path / "estimand_registry.json", estimands)
    summaries, coverage_rows, block_rows = [], [], []
    model_shape_rows = []
    for cohort in cohorts.to_dict("records"):
        directory = run.path / "panels" / cohort["expiration_date"]
        if not (directory / "complete.json").exists():
            continue
        panel = load_cohort_panel(directory)
        master = pd.read_parquet(directory / "contract_master.parquet")
        chains = pd.read_parquet(directory / "chain_snapshots.parquet")
        if cutoff is not None:
            panel = panel[panel.decision_time_ny < cutoff] if len(panel) else panel
            chains = chains[chains.decision_time_ny < cutoff]
        if chains.empty:
            continue
        for phase, part in chains.groupby("calendar_date_dte", sort=True):
            observed_master = master[master.universe_basis.eq("RETROSPECTIVE_LIFECYCLE_OBSERVED_UNION")]
            for right in ("C", "P"):
                identities = observed_master[observed_master.right.eq(right)].copy()
                identities["initial_bin"] = moneyness_bin(identities.x_s0)
                for initial_bin, ids in identities.groupby("initial_bin", sort=True):
                    source = panel[panel.calendar_date_dte.eq(phase) & panel.contract_id.isin(ids.contract_id)] if len(panel) else pd.DataFrame()
                    valid = int(source.valid_quote.sum()) if len(source) else 0
                    expected = len(ids) * len(part)
                    coverage_rows.append({"cohort_id": cohort["cohort_id"], "right": right, "initial_bin": initial_bin, "calendar_date_dte": phase,
                                          "contracts": len(ids), "scheduled_times": len(part), "expected_retrospective_slots": expected,
                                          "valid_slots": valid, "missing_or_invalid_slots": expected - valid,
                                          "quote_coverage_retrospective": valid / expected if expected else np.nan,
                                          "listed_coverage": np.nan, "denominator_basis": "RETROSPECTIVE_OBSERVED_UNIVERSE_NOT_LISTED"})
        if panel.empty:
            continue
        panel["initial_bin"] = moneyness_bin(panel.x_s0)
        panel["current_bin"] = moneyness_bin(panel.x_s)
        panel["calendar_block"] = calendar_block_ids(panel.session_date, calendar, run.config["forecast"]["block_sessions"])
        opening = panel[(panel.decision_time_ny == ny_timestamp(cohort["scheduled_t0"])) & panel.valid_quote].set_index("contract_id").midpoint
        panel["delta_mid_since_exact_t0"] = panel.midpoint - panel.contract_id.map(opening)
        for view, coordinate in (("FIXED_INITIAL_COORDINATE", "initial_bin"), ("MOVING_CURRENT_COORDINATE", "current_bin")):
            keys = ["right", coordinate, "calendar_date_dte"]
            for key, group in panel.groupby(keys, sort=True):
                valid = group[group.valid_quote]
                base = {"cohort_id": cohort["cohort_id"], "view": view, "right": key[0], "moneyness_bin": key[1], "calendar_date_dte": key[2],
                        "contracts": group.contract_id.nunique(), "decision_times": group.decision_time_ny.nunique(), "calendar_days": group.session_date.nunique(),
                        "observed_rows": len(group), "valid_rows": len(valid), "zero_bid_rows": int(valid.zero_bid.sum()),
                        "quote_age_basis": run.config["clocks"]["quote_timestamp_semantics"], "origin": "OBSERVED_DESCRIPTIVE_STATISTIC"}
                for metric in ("midpoint", "time_value_mid", "spread_abs", "spread_relative", "delta_mid_since_exact_t0"):
                    values = valid[metric].dropna()
                    base[metric + "_mean"] = values.mean()
                    for q, name in ((.1, "q10"), (.5, "q50"), (.9, "q90")):
                        base[metric + "_" + name] = values.quantile(q) if len(values) else np.nan
                summaries.append(base)
                for block, block_group in valid.groupby("calendar_block", sort=True):
                    values = block_group.delta_mid_since_exact_t0.dropna()
                    block_rows.append({"cohort_id": cohort["cohort_id"], "view": view, "right": key[0], "moneyness_bin": key[1], "calendar_date_dte": key[2],
                                       "calendar_block": block, "value_sum": values.sum(), "value_count": len(values)})
        if run.config["shape"]["enabled"]:
            anchor_mask = scheduled_anchor_mask(panel.decision_time_ny, calendar, run.config["shape"]["anchor_spacing_minutes"], cohort)
            supported = panel[anchor_mask & panel.valid_quote & panel.standard_supported & panel.spot.notna()]
            for key, group in supported.groupby(["decision_time_ny", "right", "root", "deliverable_id", "multiplier"], sort=True):
                if len(group) < 3:
                    continue
                try:
                    fit = fit_constrained_shape(group.strike.to_numpy(), group.bid.to_numpy(), group.ask.to_numpy(), float(group.spot.iloc[0]), key[1], run.config["shape"])
                    ordered = group.iloc[fit["order"]]
                    for i, row in enumerate(ordered.itertuples(index=False)):
                        model_shape_rows.append({"chain_id": digest([cohort["cohort_id"], str(key[0])]), "cohort_id": cohort["cohort_id"], "decision_time_ny": key[0],
                                                 "contract_id": row.contract_id, "strike": row.strike, "right": row.right, "fitted_price": fit["fitted_prices"][i],
                                                 "raw_midpoint": row.midpoint, "quote_interval_slack": fit["slacks"][i], "fit_status": fit["status"],
                                                 "weight": fit["weights"][i], "origin": "CONTEMPORANEOUS_CONSTRAINED_FIT_NOT_OBSERVATION"})
                except ResearchError as exc:
                    run.event("shape", exc.code, exc.message, str(key), **exc.evidence)
        run.memory_check()
    summary = pd.DataFrame(summaries)
    coverage = pd.DataFrame(coverage_rows)
    aggregate_rows = []
    blocks = pd.DataFrame(block_rows)
    if len(summary):
        group_keys = ["view", "right", "moneyness_bin", "calendar_date_dte"]
        for key, group in summary.groupby(group_keys, sort=True):
            row = dict(zip(group_keys, key))
            row.update(contributing_expirations=group.cohort_id.nunique(), contracts_summed_within_expiration=int(group.contracts.sum()),
                       valid_rows=int(group.valid_rows.sum()), weighting="EQUAL_EXPIRATION_MEANS", units="OPTION_QUOTED_PRICE_UNITS")
            for metric in ("midpoint", "time_value_mid", "spread_abs", "spread_relative", "delta_mid_since_exact_t0"):
                means = group[metric + "_mean"].dropna()
                row[metric + "_expiration_balanced_mean"] = means.mean()
                row[metric + "_expiration_mean_q10"] = means.quantile(.1) if len(means) else np.nan
                row[metric + "_expiration_mean_q50"] = means.quantile(.5) if len(means) else np.nan
                row[metric + "_expiration_mean_q90"] = means.quantile(.9) if len(means) else np.nan
            if len(blocks):
                selected = blocks
                for name, value in zip(group_keys, key): selected = selected[selected[name] == value]
                ci = block_bootstrap_expiration_mean(selected, run.config["forecast"]["bootstrap_repetitions"], run.config["seed"])
                row.update(delta_mid_mean_ci_low=ci["lower"], delta_mid_mean_ci_high=ci["upper"], uncertainty_status=ci["status"], calendar_blocks=ci["calendar_blocks"])
            aggregate_rows.append(row)
    atomic_frame(run.path / "description" / "within_expiration_cells.csv", summary)
    atomic_frame(run.path / "description" / "expiration_balanced_cells.csv", pd.DataFrame(aggregate_rows))
    atomic_frame(run.path / "description" / "fixed_initial_cell_coverage.csv", coverage)
    atomic_frame(run.path / "description" / "calendar_block_sufficient_statistics.parquet", blocks)
    atomic_frame(run.path / "description" / "constrained_shape_fits.parquet", pd.DataFrame(model_shape_rows))
    result = {"description_status": "RELEASED" if len(summary) else "BLOCKED_NO_CAUSAL_OBSERVATIONS", "scope": estimands["test_visibility"],
              "expiration_cells": len(summary), "coverage_cells": len(coverage), "shape_rows": len(model_shape_rows),
              "pricing_status": "PRICING_INPUTS_MISSING_CONTINUOUS_YIELD_SOLVERS_AVAILABLE_BUT_NOT_APPLIED_WITHOUT_INPUTS",
              "units_and_population": estimands}
    atomic_json(run.path / "description" / "status.json", result)
    return result


def freeze_splits(run: Run, catalog: Catalog, cohorts: pd.DataFrame, calendar: Calendar) -> dict:
    active = cohorts[cohorts.expected_sessions.ne("[]")]
    require(len(active) > 0, "INSUFFICIENT_INDEPENDENT_SUPPORT", "No schedulable cohorts")
    starts = [json.loads(value)[0] for value in active.expected_sessions]
    ends = [json.loads(value)[-1] for value in active.expected_sessions]
    days = calendar.between(min(starts), max(ends))
    require(len(days) >= 5, "INSUFFICIENT_INDEPENDENT_SUPPORT", "Need at least five calendar sessions to define a sealed split")
    config = run.config["forecast"]
    test_index = min(len(days) - 1, max(1, int(math.floor(len(days) * (1 - config["sealed_test_fraction"])))))
    test_start = calendar.sessions[days[test_index]]["open"]
    development = days[:test_index]
    initial = max(1, int(len(development) * config["initial_development_fraction"]))
    boundaries = np.linspace(initial, len(development), config["validation_blocks"] + 1).astype(int)
    def assignments(train_end: pd.Timestamp, validation_start: pd.Timestamp, validation_end: pd.Timestamp) -> dict:
        train, validation, purged = [], [], []
        for c in active.to_dict("records"):
            start = ny_timestamp(c["scheduled_t0"])
            finish = cohort_grid(calendar, c, json.loads(c["expected_sessions"])[-1])[-1]
            if start < train_end and finish < train_end:
                train.append(c["cohort_id"])
            elif validation_start <= start < validation_end and finish < validation_end:
                validation.append(c["cohort_id"])
            elif start < validation_end and finish >= train_end:
                purged.append(c["cohort_id"])
        require(not set(train) & set(validation), "SPLIT_LEAKAGE_DETECTED", "Shared lifecycle across partitions")
        return {"training_cohorts": train, "validation_cohorts": validation, "purged_boundary_cohorts": purged}
    folds = []
    for i, (start_index, end_index) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if start_index >= end_index or start_index >= len(development):
            continue
        val_start = calendar.sessions[development[start_index]]["open"]
        val_end = calendar.sessions[development[end_index]]["open"] if end_index < len(development) else test_start
        folds.append({"fold_id": f"development_{i + 1}", "training_cutoff": val_start, "validation_start": val_start, "validation_end": val_end,
                      **assignments(val_start, val_start, val_end)})
    final_end = calendar.sessions[days[-1]]["close"] + pd.Timedelta(nanoseconds=1)
    final = assignments(test_start, test_start, final_end)
    identity = {"calendar_version": calendar.version, "cohort_definition": run.config["cohort_definition"], "sessions": days,
                "cohorts": active[["cohort_id", "D0", "expected_sessions"]].to_dict("records"), "test_start": test_start, "horizons": run.config["labels"]}
    result = {"schema_version": SCHEMA_VERSION, "split_id": digest(identity), "catalog_identity": catalog.identity,
              "sealed_test_identity": digest({"symbol": "QQQ", "sessions": days[test_index:], "definition": run.config["cohort_definition"], "horizons": run.config["labels"]}),
              "sealed_test_start": test_start, "calendar_sessions": days, "folds": folds, "final": final,
              "protocol": "CHRONOLOGICAL_COMPLETE_LIFECYCLE_PURGE_AND_LABEL_AVAILABLE_AT_CUTOFF", "final_test_consumed": False}
    path = run.path / "split_manifest.json"
    if path.exists():
        old = read_json(path)
        require(old["split_id"] == result["split_id"] and old["catalog_identity"] == catalog.identity, "FROZEN_SPLIT_INPUT_CHANGED", "Use a new run_id after archive/scope changes")
        return old
    atomic_json(path, result)
    return read_json(path)


CELL_COLUMNS = [f"cell_{right}_{i}" for right in ("C", "P") for i in range(len(MONEYNESS_NAMES))]
CORE_FEATURES = ["calendar_date_dte", "minutes_since_open", "minutes_to_session_close", "remaining_regular_session_minutes", "weekday",
                 "underlying_log_return_1m", "underlying_log_return_5m", "underlying_log_return_30m", "underlying_rv_30m", "overnight_log_return", "mean_spread_spot", "valid_fraction_current"]
CORE_TARGETS = ["target_log_spot_return", "target_spread_spot_change", "target_valid_fraction_change"]


def chain_samples(run: Run, cohorts: pd.DataFrame, calendar: Calendar, cohort_ids: set[str], horizon: int) -> pd.DataFrame:
    rows = []
    for cohort in cohorts[cohorts.cohort_id.isin(cohort_ids)].to_dict("records"):
        path = run.path / "panels" / cohort["expiration_date"] / "chain_snapshots.parquet"
        if not path.exists():
            continue
        full = pd.read_parquet(path).sort_values("decision_time_ny", kind="stable")
        full["valid_fraction_current"] = full.valid_quotes / full.observed_contracts.replace(0, np.nan)
        anchors = full.loc[scheduled_anchor_mask(full.decision_time_ny, calendar, run.config["labels"]["anchor_spacing_minutes"], cohort)].copy()
        anchors["label_available_at"] = anchors.decision_time_ny + horizon * ONE_MINUTE
        end_columns = ["decision_time_ny", "spot", "mean_spread_spot", "valid_fraction_current", *CELL_COLUMNS]
        future = full[end_columns].rename(columns={c: "end_" + c for c in end_columns})
        anchors = anchors.merge(future, left_on="label_available_at", right_on="end_decision_time_ny", how="left", validate="one_to_one")
        same_session = anchors.label_available_at.dt.strftime("%Y-%m-%d").eq(anchors.session_date)
        within_close = np.array([end <= calendar.sessions[session]["close"] for end, session in zip(anchors.label_available_at, anchors.session_date)])
        anchors["label_clock_valid"] = same_session & within_close
        anchors["target_log_spot_return"] = np.log(anchors.end_spot / anchors.spot)
        anchors["target_spread_spot_change"] = anchors.end_mean_spread_spot - anchors.mean_spread_spot
        anchors["target_valid_fraction_change"] = anchors.end_valid_fraction_current - anchors.valid_fraction_current
        for cell in CELL_COLUMNS:
            anchors["target_delta_" + cell] = anchors["end_" + cell] - anchors[cell]
        targets = [c for c in anchors if c.startswith("target_")]
        anchors.loc[~anchors.label_clock_valid, targets] = np.nan
        anchors["horizon_minutes"] = horizon
        anchors["sample_id"] = [digest([cohort["cohort_id"], str(t), horizon, "CHAIN_JOINT"]) for t in anchors.decision_time_ny]
        anchors["calendar_block"] = calendar_block_ids(anchors.session_date, calendar, run.config["forecast"]["block_sessions"])
        rows.append(anchors)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


class MaskedFactors:
    """Training-only low-rank price-shape basis with explicit observation masks.

    EM reconstruction is an internal fit, never labeled observed. Encoding solves
    least squares on available cells only. Entirely absent columns remain outside
    the learned domain. This is a price-space fallback when time-value inputs are
    unverified; it never treats vendor IV/Greeks as model-verified measurements.
    """
    def fit(self, values: np.ndarray, rank: int, seed: int, iterations: int = 20) -> "MaskedFactors":
        values = np.asarray(values, float)
        self.columns = np.flatnonzero(np.sum(np.isfinite(values), axis=0) >= max(3, int(.1 * len(values))))
        require(len(self.columns) > 0, "SHAPE_SUPPORT_INSUFFICIENT", "No supported price-shape cells")
        x = values[:, self.columns]
        self.mean = np.nanmean(x, axis=0)
        self.scale = np.nanstd(x, axis=0)
        self.scale[self.scale < 1e-10] = 1.
        self.rank = min(rank, len(self.columns) - 1, max(0, len(x) - 2))
        mask = np.isfinite(x)
        standardized = (x - self.mean) / self.scale
        filled = np.where(mask, standardized, 0.)
        if self.rank:
            for _ in range(iterations):
                _, _, vh = np.linalg.svd(filled, full_matrices=False)
                basis = vh[:self.rank].T
                reconstruction = filled @ basis @ basis.T
                updated = np.where(mask, standardized, reconstruction)
                change = np.max(np.abs(updated - filled))
                filled = updated
                if change < 1e-6:
                    break
            _, _, vh = np.linalg.svd(filled, full_matrices=False)
            self.basis = vh[:self.rank].T
            # Fix presentation signs from TRAINING loadings only.
            for j in range(self.rank):
                largest = np.argmax(np.abs(self.basis[:, j]))
                if self.basis[largest, j] < 0: self.basis[:, j] *= -1
        else:
            self.basis = np.empty((len(self.columns), 0))
        self.training_rows = len(values)
        return self

    def encode(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(values, float)[:, self.columns] - self.mean) / self.scale
        mask = np.isfinite(x)
        codes = np.full((len(x), self.rank), np.nan)
        supported = mask.sum(axis=1) >= max(1, self.rank)
        if self.rank == 0:
            return codes, supported
        patterns, inverse = np.unique(mask, axis=0, return_inverse=True)
        for i, observed in enumerate(patterns):
            rows = np.flatnonzero((inverse == i) & supported)
            if not len(rows):
                continue
            b = self.basis[observed]
            projection = np.linalg.solve(b.T @ b + 1e-5 * np.eye(self.rank), b.T)
            codes[rows] = x[rows][:, observed] @ projection.T
        return codes, supported

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        codes, supported = self.encode(values)
        reconstructed = np.tile(self.mean, (len(values), 1)) if not self.rank else (codes @ self.basis.T) * self.scale + self.mean
        reconstructed[~supported] = np.nan
        output = np.full_like(values, np.nan, dtype=float)
        output[:, self.columns] = reconstructed
        return output

    def state(self) -> dict:
        return {"columns": self.columns, "mean": self.mean, "scale": self.scale, "basis": self.basis, "rank": self.rank,
                "training_rows": self.training_rows, "space": "NORMALIZED_QUOTED_PRICE_MASKED_GRID", "missing_values_are_observations": False}


class RobustNormalizer:
    def fit(self, values: np.ndarray) -> "RobustNormalizer":
        values = np.asarray(values, float)
        self.kept = np.flatnonzero(np.isfinite(values).sum(axis=0) >= 3)
        require(len(self.kept) > 0, "INSUFFICIENT_INDEPENDENT_SUPPORT", "No usable training features")
        x = values[:, self.kept]
        self.center = np.nanmedian(x, axis=0)
        self.scale = np.nanquantile(x, .75, axis=0) - np.nanquantile(x, .25, axis=0)
        self.constant = self.scale < 1e-12
        self.scale[self.constant] = 1.
        self.minimum, self.maximum = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, float)[:, self.kept]
        missing = ~np.isfinite(x)
        scaled = np.where(missing, 0., (x - self.center) / self.scale)
        # Remove truly constant observed values; retain their missingness indicators.
        return np.concatenate([scaled[:, ~self.constant], missing.astype(float)], axis=1)

    def within_domain(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, float)[:, self.kept]
        padding = .1 * (self.maximum - self.minimum)
        outside = np.isfinite(x) & ((x < self.minimum - padding - 1e-12) | (x > self.maximum + padding + 1e-12))
        return ~outside.any(axis=1)


def cap_group_weights(weights: np.ndarray, groups: np.ndarray, cap: float = .10) -> np.ndarray:
    weights = np.asarray(weights, float).copy()
    weights = np.maximum(weights, np.finfo(float).tiny)
    _, inverse = np.unique(groups, return_inverse=True)
    count = int(inverse.max()) + 1
    feasible_cap = max(cap, 1. / count)
    for _ in range(100):
        weights /= weights.sum()
        masses = np.bincount(inverse, weights=weights, minlength=count)
        oversized = masses > feasible_cap + 1e-10
        if not oversized.any():
            break
        weights *= np.minimum(1., feasible_cap / masses)[inverse]
    return weights / weights.sum()


def scenario_scores(scenarios: np.ndarray, weights: np.ndarray, observed: np.ndarray, scale: np.ndarray) -> dict:
    from scipy.spatial.distance import cdist
    x, y = scenarios / scale, observed / scale
    weights = weights / weights.sum()
    mean = weights @ scenarios
    energy = float(weights @ np.linalg.norm(x - y, axis=1) - .5 * weights @ cdist(x, x) @ weights)
    q05, q50, q95, crps = [], [], [], []
    for j in range(scenarios.shape[1]):
        order = np.argsort(scenarios[:, j], kind="stable")
        values, w = scenarios[order, j], weights[order]
        cumulative = np.cumsum(w)
        quantiles = np.interp([.05, .5, .95], cumulative, values)
        q05.append(quantiles[0]); q50.append(quantiles[1]); q95.append(quantiles[2])
        score = np.sum(w * np.abs(values - observed[j])) - np.sum(w * values * (2 * cumulative - w - 1))
        crps.append(float(score / scale[j]))
    probability_up = float(weights @ (scenarios[:, 0] > 0))
    error = (mean - observed) / scale
    joint_probabilities, joint_events = [], []
    for j in range(1, scenarios.shape[1]):
        joint_probabilities.append(float(weights @ ((scenarios[:, 0] < q05[0]) & (scenarios[:, j] > q95[j]))))
        joint_events.append(float(observed[0] < q05[0] and observed[j] > q95[j]))
    return {"energy_score": energy / math.sqrt(len(observed)), "mean_crps_standardized": float(np.mean(crps)),
            "mae_standardized": float(np.mean(np.abs(error))), "mse_standardized": float(np.mean(error**2)), "bias_standardized": float(np.mean(error)),
            "interval90_coverage": float(np.mean((observed >= q05) & (observed <= q95))), "interval90_width_standardized": float(np.mean((np.array(q95) - q05) / scale)),
            "spot_up_probability": probability_up, "spot_up_brier": (probability_up - float(observed[0] > 0))**2,
            "predicted_mean": mean, "predicted_q05": q05, "predicted_median": q50, "predicted_q95": q95, "observed": observed,
            "coverage90_by_target": ((observed >= q05) & (observed <= q95)).astype(float), "bias_by_target_standardized": error,
            "lower_tail_by_target": (observed < q05).astype(float), "upper_tail_by_target": (observed > q95).astype(float),
            "crps_by_target_standardized": crps, "joint_tail_probabilities": joint_probabilities, "joint_tail_observed_events": joint_events,
            "quantile_crossings": int(np.sum(np.array(q05) > q50) + np.sum(np.array(q50) > q95))}


class JointForecaster:
    """B0/B1/B2 use WHOLE joint response rows, never shuffled marginal targets."""
    def __init__(self, spec: dict, config: dict, seed: int):
        self.spec, self.config, self.seed = spec, config, seed

    def fit(self, training: pd.DataFrame, cutoff: pd.Timestamp) -> "JointForecaster":
        from scipy.spatial import cKDTree
        require((training.label_available_at < cutoff).all(), "SPLIT_LEAKAGE_DETECTED", "Training labels mature after cutoff")
        require(training.cohort_id.nunique() >= 2, "INSUFFICIENT_INDEPENDENT_SUPPORT", "Need multiple independent lifecycle clusters for exploratory fitting")
        self.cutoff = cutoff
        target_candidates = ["target_delta_cell_C_4", "target_delta_cell_P_4", "target_delta_cell_P_2", "target_delta_cell_C_6"]
        self.targets = self.spec.get("targets") or (CORE_TARGETS + [c for c in target_candidates if np.isfinite(training[c]).mean() >= .8])
        self.total_training_cases = len(training)
        complete = training.label_clock_valid & np.isfinite(training[self.targets]).all(axis=1)
        self.missing_label_rate = 1 - float(complete.mean())
        selected = training.loc[complete].copy()
        cap = self.config["max_train_anchors"]
        if len(selected) > cap:
            # Seeded, declared training-anchor subsample; panel monitoring is unchanged.
            selected = selected.sample(n=cap, random_state=self.seed).sort_values(["decision_time_ny", "cohort_id"])
        require(len(selected) >= 20 and selected.cohort_id.nunique() >= 2, "INSUFFICIENT_INDEPENDENT_SUPPORT", "Too few complete joint responses")
        self.training = selected.reset_index(drop=True)
        factor_training = training[CELL_COLUMNS].to_numpy(float)
        if len(factor_training) > 5000:
            rng = np.random.default_rng(self.seed)
            factor_training = factor_training[rng.choice(len(factor_training), 5000, replace=False)]
        self.factors = MaskedFactors().fit(factor_training, self.spec.get("rank", 0), self.seed, self.config["factor_iterations"])
        require(self.factors.rank == self.spec.get("rank", 0), "SHAPE_SUPPORT_INSUFFICIENT", "Requested factor rank is not supported; candidate remains explicitly unsupported")
        self.raw_x = self.features(self.training)
        self.normalizer = RobustNormalizer().fit(self.raw_x)
        self.x = self.normalizer.transform(self.raw_x)
        self.y = self.training[self.targets].to_numpy(float)
        self.y_scale = np.std(self.y, axis=0)
        self.constant_targets = self.y_scale < 1e-10
        self.y_scale[self.y_scale < 1e-10] = 1.
        self.tree = cKDTree(self.x)
        # A declared nearest-neighbor radius estimates support using TRAINING only.
        query_indices = np.linspace(0, len(self.x) - 1, min(1000, len(self.x))).astype(int)
        distances, _ = self.tree.query(self.x[query_indices], k=min(50, len(self.x)))
        self.distance_limit = float(np.quantile(distances[:, -1], .99)) + 1e-8
        if self.spec["name"] == "B2":
            design = np.column_stack([np.ones(len(self.x)), self.x])
            regularization = np.eye(design.shape[1]) * self.spec["ridge"]
            regularization[0, 0] = 0
            self.coefficients = np.linalg.solve(design.T @ design + regularization, design.T @ self.y)
            self.residuals = self.y - design @ self.coefficients
        return self

    def features(self, frame: pd.DataFrame) -> np.ndarray:
        factors, supported = self.factors.encode(frame[CELL_COLUMNS].to_numpy(float))
        # Missing factor encodings carry masks through RobustNormalizer.
        return np.column_stack([frame[CORE_FEATURES + ["cell_C_4", "cell_P_4", "cell_P_2", "cell_C_6"]].to_numpy(float), factors])

    def scenarios(self, query: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        require(len(query) == 1 and query.decision_time_ny.iloc[0] >= self.cutoff, "SPLIT_LEAKAGE_DETECTED", "Query precedes model cutoff")
        raw = self.features(query)
        x = self.normalizer.transform(raw)[0]
        current = query.iloc[0]
        all_indices = np.arange(len(self.training))
        stratum = self.training.calendar_date_dte.eq(current.calendar_date_dte).to_numpy()
        phase = (self.training.minutes_since_open // 30).to_numpy()
        stratum &= np.abs(phase - int(current.minutes_since_open // 30)) <= 1
        candidates = all_indices[stratum]
        pooled = False
        if len(candidates) < 20:
            candidates = all_indices[self.training.calendar_date_dte.eq(current.calendar_date_dte).to_numpy()]
            pooled = True
        if len(candidates) < 10:
            return None, None, {"status": "OUTSIDE_VALIDATED_DOMAIN", "reason": "Insufficient eligible clock stratum"}
        if not self.normalizer.within_domain(raw)[0]:
            return None, None, {"status": "OUTSIDE_VALIDATED_DOMAIN", "reason": "Feature beyond training support"}
        distance = np.linalg.norm(self.x[candidates] - x, axis=1)
        nearest = float(distance.min())
        if nearest > self.distance_limit:
            return None, None, {"status": "OUTSIDE_VALIDATED_DOMAIN", "reason": "No sufficiently close training state", "nearest_distance": nearest}
        if self.spec["name"] == "B1":
            order = np.argsort(distance, kind="stable")[:self.spec["neighbor_cap"]]
            candidates, distance = candidates[order], distance[order]
            bandwidth = max(float(np.median(distance)), 1e-8)
            weights = np.exp(-.5 * (distance / bandwidth)**2)
        else:
            weights = np.ones(len(candidates))
            bandwidth = None
        support = self.training.iloc[candidates]
        weights = cap_group_weights(weights, support.cohort_id.to_numpy())
        weights = cap_group_weights(weights, support.calendar_block.to_numpy())
        expirations, blocks = support.cohort_id.nunique(), support.calendar_block.nunique()
        concentration = 1. / float(np.sum(weights**2))
        seed = int(digest([self.seed, str(current.sample_id), self.spec])[:8], 16)
        rng = np.random.default_rng(seed)
        limit = self.config["max_scenarios"]
        if len(candidates) > limit:
            positions = rng.choice(len(candidates), size=limit, p=weights, replace=True)
            unique, counts = np.unique(positions, return_counts=True)
            candidates, weights = candidates[unique], counts.astype(float) / limit
        else:
            weights /= weights.sum()
        if self.spec["name"] == "NO_CHANGE":
            scenarios, weights = np.zeros((1, self.y.shape[1])), np.ones(1)
        elif self.spec["name"] == "B2":
            mean = np.r_[1., x] @ self.coefficients
            scenarios = mean + self.residuals[candidates]
        else:
            scenarios = self.y[candidates]
        status = "EXPLORATORY_SUPPORTED_STATE" if expirations >= self.config["min_expirations"] and blocks >= self.config["min_calendar_blocks"] else "INSUFFICIENT_INDEPENDENT_SUPPORT"
        invalid_scenario = np.zeros(len(scenarios), dtype=bool)
        for j, target in enumerate(self.targets):
            if target == "target_spread_spot_change":
                invalid_scenario |= current.mean_spread_spot + scenarios[:, j] < -1e-12
            elif target == "target_valid_fraction_change":
                future_fraction = current.valid_fraction_current + scenarios[:, j]
                invalid_scenario |= (future_fraction < -1e-12) | (future_fraction > 1 + 1e-12)
            elif target.startswith("target_delta_cell_"):
                starting_value = current[target.removeprefix("target_delta_")]
                invalid_scenario |= ~np.isfinite(starting_value) | (starting_value + scenarios[:, j] < -1e-12)
        invalid_mass = float(weights @ invalid_scenario)
        if invalid_mass > 1e-10:
            status = "FORECAST_SUMMARY_RECONSTRUCTION_DOMAIN_FAILED"
        return scenarios, weights, {"status": status, "contributing_expirations": expirations, "calendar_blocks": blocks,
                                    "weight_concentration_ess_not_independent_count": concentration, "nearest_distance": nearest,
                                    "bandwidth": bandwidth, "partial_pooling": pooled, "training_missing_label_rate": self.missing_label_rate,
                                    "population": "CONDITIONAL_ON_JOINT_RESPONSE_OBSERVABILITY", "scenario_seed": seed,
                                    "physical_summary_domain_violation_weight": invalid_mass,
                                    "targets": self.targets, "path_status": "DIRECT_ENDPOINTS_ONLY_NOT_A_PATH_PROCESS",
                                    "fixed_contract_chain_decoder": "UNSUPPORTED_DIRECT_GRID_SUMMARIES_ONLY"}

    def predict_and_score(self, validation: pd.DataFrame, fold: str) -> pd.DataFrame:
        rows = []
        for i in range(len(validation)):
            query = validation.iloc[[i]]
            current = query.iloc[0]
            base = {"sample_id": current.sample_id, "cohort_id": current.cohort_id, "session_date": current.session_date,
                    "decision_time_ny": current.decision_time_ny, "calendar_date_dte": current.calendar_date_dte,
                    "calendar_block": current.calendar_block, "horizon_minutes": int(current.horizon_minutes), "fold_id": fold,
                    "candidate_id": canonical_json(self.spec), "targets": canonical_json(self.targets), "energy_score": np.nan,
                    "origin": "CHRONOLOGICAL_OUT_OF_SAMPLE_FORECAST", "training_cutoff": self.cutoff}
            observed = query[self.targets].to_numpy(float)[0]
            scenarios, weights, info = self.scenarios(query)
            base.update(info)
            base["targets"] = canonical_json(self.targets)
            if scenarios is not None and np.isfinite(observed).all() and bool(current.label_clock_valid):
                scores = scenario_scores(scenarios, weights, observed, self.y_scale)
                for key in ("predicted_mean", "predicted_q05", "predicted_median", "predicted_q95", "observed", "coverage90_by_target", "bias_by_target_standardized", "lower_tail_by_target", "upper_tail_by_target", "crps_by_target_standardized", "joint_tail_probabilities", "joint_tail_observed_events"):
                    scores[key] = canonical_json(scores[key])
                base.update(scores)
            elif scenarios is not None:
                base["status"] = "LABEL_UNRESOLVED_FORECAST_NOT_SCORED"
            rows.append(base)
        return pd.DataFrame(rows)

    def save(self, directory: Path, fold_id: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"spec": self.spec, "training_cutoff": self.cutoff, "fold_id": fold_id, "targets": self.targets,
                    "training_sample_hash": digest(self.training.sample_id.tolist()), "factors": self.factors.state(),
                    "normalizer": {k: v for k, v in self.normalizer.__dict__.items()}, "y_scale": self.y_scale,
                    "constant_targets": self.constant_targets, "score_estimator": "WEIGHTED_EMPIRICAL_V_STATISTIC",
                    "distance_limit": self.distance_limit, "missing_label_rate": self.missing_label_rate,
                    "population": "JOINT_OBSERVED_ENDPOINTS", "claims": "EXPLORATORY_UNTIL_RELEASE_GATES", "config": self.config, "seed": self.seed}
        atomic_json(directory / "pack.json", metadata)
        atomic_frame(directory / "training_anchors.parquet", self.training[["sample_id", "cohort_id", "session_date", "decision_time_ny", "label_available_at", "calendar_block", *CORE_FEATURES, *CELL_COLUMNS, *self.targets]])
        tmp = directory / "arrays.npz.part"
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, x=self.x, y=self.y, **({"coefficients": self.coefficients, "residuals": self.residuals} if self.spec["name"] == "B2" else {}))
        os.replace(tmp, directory / "arrays.npz")


def candidate_specs(config: dict) -> list[dict]:
    result = [{"name": "NO_CHANGE", "rank": 0}, {"name": "B0", "rank": 0}]
    for rank in config["factor_candidates"]:
        result += [{"name": "B1", "neighbor_cap": cap, "rank": rank} for cap in config["neighbor_candidates"]]
        result += [{"name": "B2", "ridge": ridge, "rank": rank} for ridge in config["ridge_candidates"]]
    return result


def masked_reconstruction_validation(training: pd.DataFrame, config: dict, seed: int) -> list[dict]:
    values = training[CELL_COLUMNS].to_numpy(float)
    rng = np.random.default_rng(seed)
    if len(values) > 2000:
        values = values[rng.choice(len(values), 2000, replace=False)]
    observed = np.isfinite(values)
    hold = (rng.random(values.shape) < .10) & observed
    fitting = values.copy()
    fitting[hold] = np.nan
    rows = []
    for rank in config["factor_candidates"]:
        pack = MaskedFactors().fit(fitting, rank, seed, config["factor_iterations"])
        reconstruction = pack.reconstruct(fitting)
        for region, columns in (("ATM", [4, 13]), ("WINGS", [i for i in range(18) if i not in (4, 13)])):
            mask = np.zeros_like(hold)
            mask[:, columns] = hold[:, columns]
            mask &= np.isfinite(reconstruction)
            error = reconstruction[mask] - values[mask]
            rows.append({"rank_requested": rank, "rank_supported": pack.rank, "region": region, "withheld_cell_count": len(error),
                         "mae_normalized_price": float(np.mean(np.abs(error))) if len(error) else np.nan,
                         "status": "TRAINING_ONLY_MASKED_CELL_RECONSTRUCTION" if rank == pack.rank else "SHAPE_SUPPORT_INSUFFICIENT"})
    return rows


def paired_model_summary(predictions: pd.DataFrame, config: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows, paired_rows = [], []
    for (fold, horizon), group in predictions.groupby(["fold_id", "horizon_minutes"], sort=True):
        all_candidates = sorted(group.candidate_id.unique())
        candidates = [candidate for candidate in all_candidates if group.loc[group.candidate_id.eq(candidate), "energy_score"].notna().any()]
        valid_sets = [set(group.loc[group.candidate_id.eq(candidate) & group.energy_score.notna(), "sample_id"]) for candidate in candidates]
        common = set.intersection(*valid_sets) if valid_sets else set()
        baseline_ids = [c for c in candidates if json.loads(c)["name"] == "B0"]
        if not baseline_ids:
            continue
        baseline = group[group.candidate_id.eq(baseline_ids[0]) & group.sample_id.isin(common)].set_index("sample_id").energy_score
        for candidate in all_candidates:
            cases = group[group.candidate_id.eq(candidate)].copy()
            matched = cases[cases.sample_id.isin(common)].copy()
            matched["difference_vs_B0"] = matched.energy_score - matched.sample_id.map(baseline)
            e_means = matched.groupby("cohort_id").energy_score.mean()
            d_means = matched.groupby("cohort_id").difference_vs_B0.mean()
            block_stats = matched.groupby(["cohort_id", "calendar_block"]).difference_vs_B0.agg(["sum", "count"]).reset_index().rename(columns={"sum": "value_sum", "count": "value_count"})
            ci = block_bootstrap_expiration_mean(block_stats, config["bootstrap_repetitions"], seed) if len(block_stats) else {"lower": None, "upper": None, "status": "NO_MATCHED_CASES", "calendar_blocks": 0}
            row = {"fold_id": fold, "horizon_minutes": horizon, "candidate_id": candidate, "attempted_cases": len(cases), "scored_cases": int(cases.energy_score.notna().sum()),
                   "common_matched_cases_all_candidates": len(matched), "contributing_expirations": matched.cohort_id.nunique(),
                   "calendar_blocks": ci["calendar_blocks"], "energy_expiration_balanced": e_means.mean(), "difference_vs_B0": d_means.mean(),
                   "paired_calendar_block_ci_low": ci["lower"], "paired_calendar_block_ci_high": ci["upper"], "uncertainty_status": ci["status"],
                   "mean_interval90_coverage": matched.interval90_coverage.mean() if "interval90_coverage" in matched else np.nan,
                   "mean_brier": matched.spot_up_brier.mean() if "spot_up_brier" in matched else np.nan,
                   "abstention_or_missing_fraction": 1 - len(matched) / max(1, len(cases)), "comparison_population": "IDENTICAL_SUPPORTED_CASES_ALL_SUCCESSFUL_CANDIDATES"}
            row["reconstruction_domain_failure_fraction"] = float(cases.status.eq("FORECAST_SUMMARY_RECONSTRUCTION_DOMAIN_FAILED").mean())
            rows.append(row)
            if len(matched):
                paired_rows.append(matched)
    return pd.DataFrame(rows), pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()


def train_development(run: Run, cohorts: pd.DataFrame, calendar: Calendar) -> dict:
    require_causal_clocks(run.config)
    splits = read_json(run.path / "split_manifest.json")
    assert_sealed_available(run, splits)
    config = run.config["forecast"]
    specs = candidate_specs(config)
    atomic_json(run.path / "experiments" / "registry.json", {"specs": specs, "horizons": run.config["labels"]["horizons_minutes"],
                "question": "Do conditional historical analogs or regularized joint residual models improve direct-horizon forecasts over time/phase empirical outcomes?",
                "split_id": splits["split_id"], "multiplicity_family": "FINAL_SELECTED_HORIZON_MODELS_VS_B0", "selection": "DEVELOPMENT_ONLY_MATCHED_PROPER_SCORES_AND_RECONSTRUCTION"})
    summaries, all_failures, reconstruction_rows = [], [], []
    for fold in splits["folds"]:
        train_ids, val_ids = set(fold["training_cohorts"]), set(fold["validation_cohorts"])
        require(not train_ids & val_ids, "SPLIT_LEAKAGE_DETECTED", "Shared cohort")
        if len(train_ids) < 2 or not val_ids:
            all_failures.append({"fold_id": fold["fold_id"], "status": "INSUFFICIENT_INDEPENDENT_SUPPORT", "training_cohorts": len(train_ids), "validation_cohorts": len(val_ids)})
            continue
        for horizon in run.config["labels"]["horizons_minutes"]:
            print(f"Training {fold['fold_id']} horizon={horizon}", flush=True)
            training = chain_samples(run, cohorts, calendar, train_ids, horizon)
            validation = chain_samples(run, cohorts, calendar, val_ids, horizon)
            if training.empty or validation.empty:
                all_failures.append({"fold_id": fold["fold_id"], "horizon": horizon, "status": "INSUFFICIENT_INDEPENDENT_SUPPORT"})
                continue
            cutoff = ny_timestamp(fold["training_cutoff"])
            training = training[training.label_available_at < cutoff]
            reconstruction = masked_reconstruction_validation(training, config, run.config["seed"])
            reconstruction_rows.extend([{**r, "fold_id": fold["fold_id"], "horizon_minutes": horizon} for r in reconstruction])
            predictions = []
            for spec in specs:
                key = digest(spec)[:16]
                output_dir = run.path / "models" / fold["fold_id"] / f"h{horizon}" / key
                prediction_file = output_dir / "predictions.parquet"
                checkpoint = output_dir / "complete.json"
                task_key = digest({"semantic_id": run.semantic_id, "split": splits["split_id"], "spec": spec, "train": training.sample_id.tolist(), "validation": validation.sample_id.tolist(), "catalog": splits["catalog_identity"]})
                if checkpoint.exists() and prediction_file.exists() and read_json(checkpoint).get("task_key") == task_key and read_json(checkpoint).get("prediction_hash") == sha256_file(prediction_file):
                    predictions.append(pd.read_parquet(prediction_file))
                    continue
                try:
                    model = JointForecaster(spec, config, run.config["seed"]).fit(training, cutoff)
                    model.save(output_dir, fold["fold_id"])
                    predicted = model.predict_and_score(validation, fold["fold_id"])
                    atomic_frame(prediction_file, predicted)
                    atomic_json(checkpoint, {"task_key": task_key, "prediction_hash": sha256_file(prediction_file), "status": "DEVELOPMENT_ONLY"})
                    predictions.append(predicted)
                except ResearchError as exc:
                    failure = {"fold_id": fold["fold_id"], "horizon": horizon, "candidate": spec, "status": exc.code, "evidence": exc.evidence}
                    all_failures.append(failure)
                    run.event("train", exc.code, exc.message, canonical_json(failure))
                run.memory_check()
            if predictions:
                scores, matched = paired_model_summary(pd.concat(predictions, ignore_index=True), config, run.config["seed"])
                summaries.append(scores)
                if len(matched):
                    atomic_frame(run.path / "evaluation" / fold["fold_id"] / f"h{horizon}_matched.parquet", matched)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    atomic_frame(run.path / "evaluation" / "development_scores.csv", summary)
    atomic_frame(run.path / "evaluation" / "masked_reconstruction.csv", pd.DataFrame(reconstruction_rows))
    selection = {}
    if len(summary):
        finite = summary[summary.difference_vs_B0.notna()]
        for horizon, group in finite.groupby("horizon_minutes", sort=True):
            ranked = group.groupby("candidate_id").difference_vs_B0.mean().sort_values(kind="stable")
            if len(ranked):
                candidate = ranked.index[0]
                rows = group[group.candidate_id.eq(candidate)]
                selection[str(int(horizon))] = {"spec": json.loads(candidate), "development_mean_difference_vs_B0": float(ranked.iloc[0]),
                                               "folds": rows.fold_id.tolist(), "conclusion": "EXPLORATORY_SELECTION_NO_VALIDATED_EDGE_CLAIM"}
    result = {"prediction_status": "EXPLORATORY" if selection else "DESCRIPTIVE_ONLY", "selected_by_horizon": selection,
              "unsupported_candidates_and_folds": all_failures, "sealed_test_opened": False, "score_population": "MATCHED_JOINT_OBSERVABLE_ENDPOINTS",
              "selection_qualification": "Representation rank assessed by training masked reconstruction plus chronological forecast comparison; wing support remains separate."}
    atomic_json(run.path / "evaluation" / "development_status.json", result)
    return result


def sealed_marker(run: Run, splits: dict) -> Path:
    return run.root / "_sealed_tests" / (splits["sealed_test_identity"] + ".json")


def holdout_sessions(splits: dict) -> list[str]:
    start = ny_timestamp(splits["sealed_test_start"]).date().isoformat()
    return [day for day in splits["calendar_sessions"] if day >= start]


def assert_sealed_available(run: Run, splits: dict, resume_release_id: str | None = None) -> None:
    current_start = holdout_sessions(splits)[0]
    for path in sorted((run.root / "_sealed_tests").glob("*.json")):
        evidence = read_json(path)
        if evidence.get("status") == "IN_PROGRESS" and resume_release_id is not None and evidence.get("release_id") == resume_release_id and path == sealed_marker(run, splits):
            continue
        previous = evidence.get("test_sessions")
        if not previous and evidence.get("run_path"):
            old_splits = Path(evidence["run_path"]) / "split_manifest.json"
            if old_splits.exists(): previous = holdout_sessions(read_json(old_splits))
        require(bool(previous), "FINAL_TEST_STATE_UNVERIFIED", "Existing consumption record has no recoverable period; retain the original research metadata", marker=str(path))
        require(current_start > max(previous), "FINAL_TEST_ALREADY_CONSUMED", "A new run_id, horizon, or shifted overlapping period does not reset consumed data. Confirmation requires a genuinely later holdout.",
                previous_last_session=max(previous), requested_first_session=current_start)


def freeze_release(run: Run, catalog: Catalog, cohorts: pd.DataFrame) -> dict:
    splits = read_json(run.path / "split_manifest.json")
    assert_sealed_available(run, splits)
    development_path = run.path / "evaluation" / "development_status.json"
    require(development_path.exists(), "FORECAST_NOT_FITTED", "Run chronological development first")
    development = read_json(development_path)
    require(development["selected_by_horizon"], "INSUFFICIENT_INDEPENDENT_SUPPORT", "No supported development candidates")
    tolerances = run.config["forecast"]["release_tolerances"]
    require(isinstance(tolerances, dict) and all(k in tolerances for k in ("interval90_coverage_error_max", "standardized_bias_max", "missing_label_fraction_max", "alpha")),
            "RELEASE_TOLERANCES_UNFROZEN", "Set prospective numerical tolerances before opening the final test")
    selected = development["selected_by_horizon"]
    for horizon, choice in selected.items():
        key = digest(choice["spec"])[:16]
        packs = sorted((run.path / "models").glob(f"development_*/h{horizon}/{key}/pack.json"))
        require(bool(packs), "MODEL_ARTIFACT_MISSING", "Selected candidate has no training pack")
        choice["spec"]["targets"] = read_json(packs[-1])["targets"]
    source_hashes = {str(p.relative_to(run.path)): sha256_file(p) for p in sorted((run.path / "panels").glob("*/complete.json"))}
    result = {"semantic_id": run.semantic_id, "code_hash": run.code_hash, "catalog_identity": catalog.identity, "split_id": splits["split_id"],
              "sealed_test_identity": splits["sealed_test_identity"], "selected": selected, "tolerances": tolerances,
              "development_hash": sha256_file(development_path), "panel_checkpoints": source_hashes, "source_contracts": run.contracts,
              "null_hypotheses": "Selected horizon-specific conditional model has no lower expected energy score than B0 on matched supported cases",
              "multiplicity": "HOLM_ACROSS_SELECTED_HORIZONS_USING_CALENDAR_BLOCK_BOOTSTRAP_TESTS",
              "no_change_and_B0_included": True}
    result["release_id"] = digest(result)
    path = run.path / "frozen_release.json"
    if path.exists():
        require(read_json(path)["release_id"] == result["release_id"], "FROZEN_RELEASE_CHANGED", "Frozen release cannot be revised against same sealed test")
    else:
        atomic_json(path, result)
    return result


def holm_adjust(pvalues: list[float]) -> np.ndarray:
    order = np.argsort(pvalues, kind="stable")
    result = np.empty(len(pvalues))
    previous = 0.
    for i, position in enumerate(order):
        previous = max(previous, min(1., (len(pvalues) - i) * pvalues[position]))
        result[position] = previous
    return result


def paired_calendar_test(cases: pd.DataFrame, repetitions: int, seed: int) -> dict:
    blocks = sorted(cases.calendar_block.unique())
    if len(blocks) < 2:
        return {"mean": None, "lower": None, "upper": None, "pvalue": 1., "blocks": len(blocks)}
    stats = cases.groupby(["cohort_id", "calendar_block"]).difference_vs_B0.agg(["sum", "count"]).reset_index()
    cohorts = sorted(cases.cohort_id.unique())
    sums = stats.pivot(index="cohort_id", columns="calendar_block", values="sum").reindex(index=cohorts, columns=blocks).fillna(0).to_numpy()
    counts = stats.pivot(index="cohort_id", columns="calendar_block", values="count").reindex(index=cohorts, columns=blocks).fillna(0).to_numpy()
    estimate = float(cases.groupby("cohort_id").difference_vs_B0.mean().mean())
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repetitions):
        multiplicities = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks))
        denominator = counts @ multiplicities
        with np.errstate(divide="ignore", invalid="ignore"):
            means = (sums @ multiplicities) / denominator
        estimates.append(float(np.nanmean(means)))
    estimates = np.array(estimates)
    # Center the block-bootstrap null distribution. Approximation assumes adequate
    # stationary block support; this is not an exact randomization test.
    pvalue = (1 + np.sum(estimates - estimate <= estimate)) / (len(estimates) + 1)
    return {"mean": estimate, "lower": float(np.quantile(estimates, .025)), "upper": float(np.quantile(estimates, .975)),
            "pvalue": float(pvalue), "blocks": len(blocks), "method": "CENTERED_CALENDAR_BLOCK_BOOTSTRAP_APPROXIMATION"}


def evaluate_final(run: Run, catalog: Catalog, cohorts: pd.DataFrame, calendar: Calendar) -> dict:
    release = read_json(run.path / "frozen_release.json")
    splits = read_json(run.path / "split_manifest.json")
    assert_sealed_available(run, splits, release["release_id"])
    require(release["semantic_id"] == run.semantic_id and release["catalog_identity"] == catalog.identity and release["split_id"] == splits["split_id"], "FROZEN_RELEASE_CHANGED", "Code/config/catalog/splits changed after freezing")
    for name, expected in release["panel_checkpoints"].items():
        require(sha256_file(run.path / name) == expected, "FROZEN_RELEASE_CHANGED", "Panel checkpoint changed")
    marker = sealed_marker(run, splits)
    marker.parent.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        existing = read_json(marker)
        require(existing.get("status") == "IN_PROGRESS" and existing.get("release_id") == release["release_id"], "FINAL_TEST_ALREADY_CONSUMED", "Final test can only resume the exact unfinished frozen release")
    else:
        # Exclusive creation protects test-once state across concurrent processes/run IDs.
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json({"status": "IN_PROGRESS", "release_id": release["release_id"], "run_path": str(run.path), "test_consumed": True, "test_sessions": holdout_sessions(splits)}))
    train_ids = set(splits["final"]["training_cohorts"])
    test_ids = set(splits["final"]["validation_cohorts"])
    require(not train_ids & test_ids, "SPLIT_LEAKAGE_DETECTED", "Shared final lifecycle")
    results, tests = [], []
    for horizon_text, choice in release["selected"].items():
        horizon = int(horizon_text)
        training = chain_samples(run, cohorts, calendar, train_ids, horizon)
        testing = chain_samples(run, cohorts, calendar, test_ids, horizon)
        cutoff = ny_timestamp(splits["sealed_test_start"])
        require(len(training) and len(testing), "INSUFFICIENT_INDEPENDENT_SUPPORT", "No final train/test samples")
        training = training[training.label_available_at < cutoff]
        chosen = choice["spec"]
        baseline = {"name": "B0", "rank": 0, "targets": chosen["targets"]}
        predictions = []
        fitted = {}
        for label, spec in (("selected", chosen), ("baseline", baseline)):
            model = JointForecaster(spec, run.config["forecast"], run.config["seed"]).fit(training, cutoff)
            fitted[label] = model
            model.save(run.path / "models" / "final" / f"h{horizon}" / label, "SEALED_FINAL")
            predictions.append(model.predict_and_score(testing, "SEALED_FINAL"))
        selected, base = predictions
        common = set(selected.loc[selected.energy_score.notna(), "sample_id"]) & set(base.loc[base.energy_score.notna(), "sample_id"])
        cases = selected[selected.sample_id.isin(common)].copy()
        cases["difference_vs_B0"] = cases.energy_score - cases.sample_id.map(base.set_index("sample_id").energy_score)
        atomic_frame(run.path / "evaluation" / "final" / f"h{horizon}_selected.parquet", selected)
        atomic_frame(run.path / "evaluation" / "final" / f"h{horizon}_baseline.parquet", base)
        test = paired_calendar_test(cases, run.config["forecast"]["bootstrap_repetitions"], run.config["seed"])
        block_sensitivity = []
        for multiple in (1, 2, 4):
            length = run.config["forecast"]["block_sessions"] * multiple
            reblocked = cases.copy()
            reblocked["calendar_block"] = calendar_block_ids(reblocked.session_date, calendar, length)
            estimate = paired_calendar_test(reblocked, run.config["forecast"]["bootstrap_repetitions"], run.config["seed"])
            block_sensitivity.append({"block_sessions": length, **estimate})
        tests.append(max(value["pvalue"] for value in block_sensitivity))
        tolerances = release["tolerances"]
        row = {"horizon_minutes": horizon, "selected_spec": chosen, "matched_cases": len(cases), "attempted_cases": len(selected),
               "expirations": cases.cohort_id.nunique(), "calendar_blocks": cases.calendar_block.nunique(), "paired_test": test,
               "block_length_sensitivity": block_sensitivity,
               "interval90_coverage": float(cases.interval90_coverage.mean()) if len(cases) else None,
               "bias_standardized": float(cases.bias_standardized.mean()) if len(cases) else None,
               "missing_or_abstained_fraction": 1 - len(cases) / max(1, len(selected)),
               "local_support_or_reconstruction_failure_fraction": float((~cases.status.eq("EXPLORATORY_SUPPORTED_STATE")).mean()) if len(cases) else 1.,
               "near_expiry_mean_difference": float(cases.loc[cases.calendar_date_dte.le(1), "difference_vs_B0"].mean()) if len(cases) else None}
        target_diagnostics, joint_diagnostics = [], []
        if len(cases):
            def matrix(column: str, frame: pd.DataFrame = cases) -> np.ndarray:
                return np.vstack(frame[column].map(json.loads))
            coverage_matrix, bias_matrix = matrix("coverage90_by_target"), matrix("bias_by_target_standardized")
            low_matrix, high_matrix = matrix("lower_tail_by_target"), matrix("upper_tail_by_target")
            crps_matrix = matrix("crps_by_target_standardized")
            aligned_base = base.set_index("sample_id").loc[cases.sample_id]
            baseline_crps = matrix("crps_by_target_standardized", aligned_base)
            predicted_joint, observed_joint = matrix("joint_tail_probabilities"), matrix("joint_tail_observed_events")
            for j, target in enumerate(chosen["targets"]):
                constant = bool(fitted["selected"].constant_targets[j])
                diagnostic = {"target": target, "constant_in_training": constant, "coverage90": float(coverage_matrix[:, j].mean()),
                              "expected_coverage": 1. if constant else .9, "bias_standardized": float(bias_matrix[:, j].mean()),
                              "lower_tail_fraction": float(low_matrix[:, j].mean()), "upper_tail_fraction": float(high_matrix[:, j].mean()),
                              "crps_difference_vs_B0": float((crps_matrix[:, j] - baseline_crps[:, j]).mean())}
                diagnostic["gate"] = abs(diagnostic["coverage90"] - diagnostic["expected_coverage"]) <= tolerances["interval90_coverage_error_max"] and abs(diagnostic["bias_standardized"]) <= tolerances["standardized_bias_max"] and diagnostic["crps_difference_vs_B0"] <= tolerances["max_target_crps_increase_standardized"]
                if not constant:
                    diagnostic["gate"] &= abs(diagnostic["lower_tail_fraction"] - .05) <= tolerances["tail_probability_error_max"] and abs(diagnostic["upper_tail_fraction"] - .05) <= tolerances["tail_probability_error_max"] and len(cases) * .05 >= tolerances["min_tail_events"]
                target_diagnostics.append(diagnostic)
                if j > 0 and not constant:
                    predicted_probability, actual_probability = float(predicted_joint[:, j - 1].mean()), float(observed_joint[:, j - 1].mean())
                    event_count = int(observed_joint[:, j - 1].sum())
                    joint_diagnostics.append({"pair": [chosen["targets"][0], target], "event": "UNDERLYING_BELOW_Q05_AND_COMPONENT_ABOVE_Q95",
                                              "predicted_probability_mean": predicted_probability, "observed_fraction": actual_probability, "observed_events": event_count,
                                              "brier_score": float(np.mean((predicted_joint[:, j - 1] - observed_joint[:, j - 1])**2)),
                                              "gate": abs(predicted_probability - actual_probability) <= tolerances["joint_tail_calibration_error_max"] and event_count >= tolerances["min_tail_events"]})
        row["per_target_diagnostics"] = target_diagnostics
        row["joint_tail_diagnostics"] = joint_diagnostics
        row["support_gate"] = row["expirations"] >= run.config["forecast"]["min_expirations"] and row["calendar_blocks"] >= run.config["forecast"]["min_calendar_blocks"] and row["local_support_or_reconstruction_failure_fraction"] == 0.
        row["calibration_gate"] = bool(target_diagnostics and all(d["gate"] for d in target_diagnostics) and joint_diagnostics and all(d["gate"] for d in joint_diagnostics) and row["missing_or_abstained_fraction"] <= tolerances["missing_label_fraction_max"])
        results.append(row)
    adjusted = holm_adjust(tests)
    for row, pvalue in zip(results, adjusted):
        row["holm_adjusted_pvalue"] = float(pvalue)
        score_gate = all(value["upper"] is not None and value["upper"] < 0 and value["blocks"] >= run.config["forecast"]["min_calendar_blocks"] for value in row["block_length_sensitivity"]) and pvalue <= release["tolerances"]["alpha"]
        row["prediction_status"] = "VALIDATED_DIRECT_SUMMARY_FORECASTS_WITHIN_NARROW_OBSERVABLE_DOMAIN" if score_gate and row["support_gate"] and row["calibration_gate"] and row["near_expiry_mean_difference"] is not None and row["near_expiry_mean_difference"] <= 0 else "FORECAST_NOT_BETTER_THAN_BASELINE_OR_GATES_UNRESOLVED"
    result = {"release_id": release["release_id"], "test_consumed": True, "horizons": results, "path_status": "NOT_IMPLEMENTED", "action_status": "DISABLED",
              "conclusions_allowed": "Direct joint endpoint forecasting only, conditional on named support and observed-endpoint population. No strategy-profit, continuous-hit, account-settlement, or universal-tail claim."}
    atomic_json(run.path / "evaluation" / "final" / "status.json", result)
    atomic_json(marker, {"status": "COMPLETE", "release_id": release["release_id"], "run_path": str(run.path), "test_consumed": True, "test_sessions": holdout_sessions(splits)})
    splits["final_test_consumed"] = True
    atomic_json(run.path / "split_manifest.json", splits)
    return result


def context(run: Run) -> tuple[Catalog, Calendar, pd.DataFrame, References]:
    catalog = Catalog(run)
    require(len(catalog.frame) > 0, "MISSING_SOURCE_DATA", "No recognized local source partitions")
    references = References(run)
    options = catalog.frame[catalog.frame.expiration_date.ne("")]
    require(len(options) > 0, "MISSING_SOURCE_DATA", "No options expiration partitions")
    start = date.fromisoformat(options.session_date.min()) - timedelta(days=40)
    end = date.fromisoformat(options.expiration_date.max()) + timedelta(days=10)
    calendar = Calendar(run.config, start, end)
    cohorts = build_cohorts(run, catalog, calendar, references)
    require(len(cohorts) > 0, "NO_COHORTS_IN_RANGE", "No discovered expirations in requested range")
    return catalog, calendar, cohorts, references


def audit_sources(run: Run, catalog: Catalog, calendar: Calendar, cohorts: pd.DataFrame, references: References, deep: bool = False, hash_all: bool = False) -> dict:
    selected_units = {(s, c["expiration_date"]) for c in cohorts.to_dict("records") for s in json.loads(c["expected_sessions"])}
    selected_sessions = {s for s, _ in selected_units}
    selected = catalog.frame.apply(lambda r: (r.session_date in selected_sessions) if r.dataset_type == "underlying" else ((r.session_date, r.expiration_date) in selected_units), axis=1)
    eligible = catalog.frame.loc[selected & ~catalog.frame.validation_status.isin(["DUPLICATE_LOGICAL_PARTITION", "NONCANONICAL_FILENAME"])]
    inspect_indices = []
    for kind, group in eligible.groupby("dataset_type", sort=True):
        inspect_indices += list(group.index) if deep or hash_all else list(group.index[np.unique(np.linspace(0, len(group) - 1, min(3, len(group))).astype(int))])
    for index in inspect_indices:
        row = catalog.frame.loc[index].to_dict()
        try:
            catalog.inspect(row, full_hash=hash_all)
        except (ResearchError, OSError, EOFError, ValueError) as exc:
            code = exc.code if isinstance(exc, ResearchError) else "INVALID_SOURCE_SCHEMA"
            catalog.frame.loc[index, "validation_status"] = code
            catalog.frame.loc[index, "issue_codes"] = code
            catalog.row_dicts[index].update(catalog.frame.loc[index].to_dict())
            run.event("audit", code, str(exc), row["partition_id"], source=row["path"])
    gaps = plan_gaps(run, catalog, cohorts, calendar)
    if len(gaps):
        missing = gaps[gaps.status.isin(["MISSING_SESSION_PARTITION", "DUPLICATE_LOGICAL_PARTITION", "NONCANONICAL_FILENAME", "INVALID_SOURCE_SCHEMA"])].copy()
    else:
        missing = gaps
    prerequisites = []
    try:
        require_causal_clocks(run.config)
    except ResearchError as exc:
        prerequisites.append({"stage": "causal_panel", "status": exc.code, "remediation": exc.message})
    if not references.rules.get("profiles"):
        prerequisites.append({"stage": "intrinsic_USD_shape_IV", "status": "UNSUPPORTED_ADJUSTED_OR_UNKNOWN_DELIVERABLE", "remediation": "Supply effective-dated standard-product/deliverable/multiplier evidence; quoted price and spot-moneyness research continues."})
    if not references.rules.get("boundaries") or references.terminals.empty:
        prerequisites.append({"stage": "full_expiry_payoff", "status": "TERMINAL_REFERENCE_MISSING", "remediation": "Supply verified contractual economic/trading boundaries and an independent terminal-reference file. A 15:59 close is not substituted."})
    if references.metadata.empty:
        prerequisites.append({"stage": "listed_universe", "status": "UNVERIFIED_LISTED_UNIVERSE", "remediation": "Observed-universe results only; complete historical listings have not been supplied."})
    duplicate = catalog.frame.validation_status.eq("DUPLICATE_LOGICAL_PARTITION")
    duplicate_keys = catalog.frame.loc[duplicate, ["dataset_type", "session_date", "expiration_date"]].drop_duplicates()
    result = {"schema_version": SCHEMA_VERSION, "catalog_identity": catalog.identity,
              "catalog_partitions_by_type": catalog.frame.groupby("dataset_type").size().to_dict(),
              "duplicate_logical_partitions_by_type": duplicate_keys.groupby("dataset_type").size().to_dict(),
              "options_session_start": catalog.frame.loc[catalog.frame.dataset_type.eq("greeks"), "session_date"].min(),
              "options_session_end": catalog.frame.loc[catalog.frame.dataset_type.eq("greeks"), "session_date"].max(),
              "candidate_expirations": len(cohorts), "cohort_status_counts": cohorts.eligible_status.value_counts().to_dict(),
              "expiry_weekday_by_status": cohorts.groupby(["expiry_weekday", "eligible_status"]).size().rename("count").reset_index().to_dict("records"),
              "schema_headers_inspected": len(inspect_indices), "schema_audit_scope": "ALL_SELECTED_PARTITIONS" if deep or hash_all else "STRATIFIED_HEADER_SAMPLES_FULL_VALIDATION_ON_READ",
              "expected_session_dataset_units": len(gaps), "missing_or_conflicting_units": len(missing), "blocked_prerequisites": prerequisites,
              "source_coverage_claim": "ARCHIVE_DISCOVERED_EXPIRATIONS_AND_OBSERVED_CONTRACTS_ONLY", "acquisition": "DISABLED_LOCAL_READ_ONLY"}
    atomic_frame(run.path / "source_catalog.parquet", catalog.frame)
    atomic_frame(run.path / "cohort_registry.parquet", cohorts)
    atomic_frame(run.path / "cohort_registry.csv", cohorts)
    atomic_frame(run.path / "session_calendar.parquet", calendar.table())
    atomic_frame(run.path / "partition_coverage.csv", gaps)
    atomic_frame(run.path / "missing_data_requests.csv", missing)
    atomic_json(run.path / "audit.json", result)
    atomic_json(run.path / "blocked_prerequisites.json", prerequisites)
    return result


def expose_full_description(run: Run) -> None:
    splits_file = run.path / "split_manifest.json"
    if not splits_file.exists():
        return
    splits = read_json(splits_file)
    marker = sealed_marker(run, splits)
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json({"status": "DESCRIPTIVE_EXPOSURE", "test_consumed": True, "run_path": str(run.path),
                                         "test_sessions": holdout_sessions(splits),
                                         "reason": "Explicit full-archive descriptive output exposes the final holdout. It cannot later be claimed untouched."}))
    splits["final_test_consumed"] = True
    atomic_json(splits_file, splits)


def export_atlas(run: Run) -> dict:
    """Portable local atlas: inspectable CSVs, provenance, explicit empty cells."""
    audit = read_json(run.path / "audit.json") if (run.path / "audit.json").exists() else {}
    description = read_json(run.path / "description" / "status.json") if (run.path / "description" / "status.json").exists() else {}
    figures = []
    cells_file = run.path / "description" / "expiration_balanced_cells.csv"
    cells = pd.read_csv(cells_file) if cells_file.exists() and cells_file.stat().st_size > 4 else pd.DataFrame()
    if len(cells):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for view, label in (("FIXED_INITIAL_COORDINATE", "Fixed contracts: initial log-moneyness"), ("MOVING_CURRENT_COORDINATE", "Moving chain: current log-moneyness")):
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
                for ax, right in zip(axes, ("C", "P")):
                    part = cells[cells.view.eq(view) & cells.right.eq(right)]
                    if part.empty:
                        ax.text(.5, .5, "No supported observations", ha="center", va="center"); continue
                    value_column = "delta_mid_since_exact_t0_expiration_balanced_mean" if view == "FIXED_INITIAL_COORDINATE" else "midpoint_expiration_balanced_mean"
                    pivot = part.pivot(index="moneyness_bin", columns="calendar_date_dte", values=value_column).reindex(MONEYNESS_NAMES).sort_index(axis=1, ascending=False)
                    array = pivot.to_numpy(float)
                    if not np.isfinite(array).any():
                        ax.text(.5, .5, "Exact t0 prices unavailable", ha="center", va="center"); continue
                    limit = max(float(np.nanmax(np.abs(array))), 1e-8)
                    palette = plt.get_cmap("coolwarm").copy(); palette.set_bad("#b8b8b8")
                    picture = ax.imshow(np.ma.masked_invalid(array), aspect="auto", cmap=palette, vmin=-limit, vmax=limit, interpolation="nearest")
                    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
                    ax.set_yticks(range(len(pivot.index)), pivot.index)
                    ax.set_xlabel("Calendar dates to expiry")
                    ax.set_title(("Calls" if right == "C" else "Puts") + "; grey = unsupported")
                    fig.colorbar(picture, ax=ax, label="Mean midpoint change since exact t0 (quoted price units)" if view == "FIXED_INITIAL_COORDINATE" else "Current midpoint price (quoted price units)")
                fig.suptitle(label + "\nEqual expiration weighting; observable-conditional theoretical marks")
                target = run.path / "atlas" / (view.lower() + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(target, dpi=140); plt.close(fig)
                figures.append(target)
        except ImportError:
            run.event("atlas", "OPTIONAL_MATPLOTLIB_MISSING", "CSV/table atlas remains usable; install matplotlib for figures")
    def table_from(path: Path, limit: int = 60) -> str:
        if not path.exists() or path.stat().st_size < 4:
            return "<p>No observations in this scope.</p>"
        frame = pd.read_csv(path)
        return frame.head(limit).to_html(index=False, escape=True, na_rep="unavailable") + (f"<p>Showing {min(limit, len(frame))} of {len(frame)} rows; full table linked below.</p>" if len(frame) > limit else "")
    import base64
    plots = "".join('<figure><img alt="Lifecycle heatmap; empty cells remain unsupported" src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"></figure>' for p in figures)
    links = []
    for p in sorted(run.path.rglob("*.csv")):
        relative = p.relative_to(run.path).as_posix()
        links.append(f'<li><a href="../{html.escape(relative, quote=True)}">{html.escape(relative)}</a></li>')
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>QQQ 4-DTE lifecycle atlas</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:1500px;margin:24px auto;padding:0 20px;color:#18222b;background:#fff}}table{{border-collapse:collapse;font-size:12px;display:block;overflow:auto}}td,th{{padding:5px 8px;border:1px solid #d8dee4;white-space:nowrap}}th{{background:#eef2f6}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:16px}}img{{max-width:100%;height:auto}}.note{{border-left:4px solid #ac6b00;padding:8px 14px;background:#fff4d8}}</style>
<h1>QQQ whole-chain lifecycle atlas</h1><p>Run {html.escape(run.config['run_id'])}; engine {VERSION}; cohort {html.escape(run.config['cohort_definition'])}.</p>
<p class="note">Observed-universe research. Missing prices stay missing. Midpoint marks are theoretical; quoted-side executions, account settlement, and strategy profit have not been established. Forecast and expiry prerequisites are reported separately.</p>
<h2>Coverage, eligibility, and prerequisites</h2><pre>{html.escape(json.dumps(clean_json(audit),default=json_default,indent=2))}</pre>
<h2>Fixed-contract and moving-chain results</h2><p>Fixed-contract t0 changes retain actual identities. A moving coordinate is a cross-section and does not establish a buy-and-hold return. Grey chart cells are unsupported; no missing cells are smoothed.</p>{plots}
{table_from(cells_file)}<h2>Units, weighting, clocks, and allowed conclusions</h2><pre>{html.escape(json.dumps(clean_json(description),default=json_default,indent=2))}</pre>
<h2>Inspectable full tables</h2><ul>{''.join(links)}</ul>
<p>Source lineage: ../run_manifest.json, ../source_catalog.parquet, per-expiration contract_master.parquet, option_observations, lifecycle_observations and chain_snapshots. Unknown listed denominators, quote-update ages, and unsupported IV remain unavailable.</p></html>"""
    output = run.path / "atlas" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part"); tmp.write_text(content, encoding="utf-8"); os.replace(tmp, output)
    artifacts = {str(p.relative_to(run.path)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in sorted(run.path.rglob("*")) if p.is_file() and not p.name.endswith(".part") and p.name not in ("artifact_manifest.json", "events.jsonl")}
    atomic_json(run.path / "artifact_manifest.json", artifacts)
    return {"atlas": str(output), "artifacts": len(artifacts), "figures": len(figures)}


def self_test(output: Path | None = None) -> dict:
    report = output or Path(__file__).resolve().with_name("qqq_4dte_test_results.json")
    tests = Path(__file__).resolve().with_name("qqq_4dte_backtest_tests.py")
    require(tests.is_file(), "TEST_SUITE_MISSING", str(tests))
    process = subprocess.run([sys.executable, str(tests), "--json-output", str(report)], cwd=tests.parent, check=False)
    require(process.returncode == 0 and report.is_file(), "ACCEPTANCE_TEST_FAILURE", "Fix tests before dependent research")
    result = read_json(report)
    require(result["success"] and result["engine_sha256"] == sha256_file(Path(__file__)), "ACCEPTANCE_TEST_FAILURE", "Test report does not match engine version")
    return result


def ensure_tests() -> None:
    report = Path(__file__).resolve().with_name("qqq_4dte_test_results.json")
    tests_file = Path(__file__).resolve().with_name("qqq_4dte_backtest_tests.py")
    if not report.exists() or not read_json(report).get("success") or read_json(report).get("engine_sha256") != sha256_file(Path(__file__)) or read_json(report).get("test_sha256") != sha256_file(tests_file):
        self_test(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["init", "audit", "plan-gaps", "build-panel", "labels", "describe", "freeze-splits", "train", "freeze-release", "evaluate-final", "export-atlas", "run", "self-test"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--start", help="Inclusive EXPIRATION date YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive EXPIRATION date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="Read-only scope preview; no writes")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--deep-audit", action="store_true", help="Inspect every in-scope partition header")
    parser.add_argument("--hash-all", action="store_true", help="SHA-256 every inspected in-scope source (potentially expensive)")
    parser.add_argument("--include-final", action="store_true", help="Expose full descriptive results; consumes an untouched predictive holdout")
    parser.add_argument("--development-only", action="store_true", help="train always uses development only")
    parser.add_argument("--sealed-test", action="store_true", help="evaluate-final always enforces frozen test-once protocol")
    args = parser.parse_args(argv)
    run = None
    try:
        if args.command == "self-test":
            require(not args.dry_run, "INVALID_CONFIG", "self-test writes an explicitly separate fixture/report namespace")
            result = self_test()
        elif args.command == "init":
            require(args.config is not None, "INVALID_CONFIG", "Supply --config output.json")
            config = default_config()
            for name, path in config["source_contracts"].items():
                if Path(path).is_file():
                    config["expected_source_contract_hashes"][name] = sha256_file(Path(path))
            if args.dry_run:
                result = config
            else:
                require(not args.config.exists(), "OUTPUT_EXISTS", "Config exists; edit it explicitly instead of overwriting")
                atomic_json(args.config, config)
                result = {"config": str(args.config.resolve()), "next": "audit; run", "timestamps": config["clocks"]}
        else:
            require(args.config is not None and args.config.is_file(), "INVALID_CONFIG", "Supply an existing --config; use init to create one")
            config = merge_config(default_config(), read_json(args.config))
            if args.run_id: config["run_id"] = args.run_id
            if args.start: config["date_range"]["start"] = date.fromisoformat(args.start).isoformat()
            if args.end: config["date_range"]["end"] = date.fromisoformat(args.end).isoformat()
            run = Run(config, dry_run=args.dry_run)
            catalog, calendar, cohorts, references = context(run)
            if args.dry_run:
                gaps = plan_gaps(run, catalog, cohorts, calendar)
                result = {"dry_run": True, "intended_output": str(run.path), "command": args.command,
                          "source_partition_counts": catalog.frame.groupby("dataset_type").size().to_dict(),
                          "cohort_statuses": cohorts.eligible_status.value_counts().to_dict(), "expected_session_dataset_units": len(gaps),
                          "causal_clocks": config["clocks"], "memory_budget_gib": config["resources"]["memory_budget_gib"],
                          "acquisition": "DISABLED", "writes": ["audit/cohort/gap tables", "per-cohort derivative panels and labels", "descriptive atlas"],
                          "optional_stages": "train, freeze-release, evaluate-final are explicit; run does not open final test"}
            elif args.command in ("audit", "plan-gaps"):
                result = audit_sources(run, catalog, calendar, cohorts, references, args.deep_audit, args.hash_all)
            elif args.command == "freeze-splits":
                result = freeze_splits(run, catalog, cohorts, calendar)
            elif args.command == "export-atlas":
                result = export_atlas(run)
            else:
                ensure_tests()
                if args.command in ("train", "freeze-release", "evaluate-final"):
                    split_path = run.path / "split_manifest.json"
                    require(split_path.exists() and read_json(split_path)["catalog_identity"] == catalog.identity, "FROZEN_SPLIT_INPUT_CHANGED", "Freeze splits for this exact archive; changed source inputs require a new research run")
                if args.command == "build-panel":
                    result = build_lifecycle_panel(run, catalog, calendar, cohorts, references, args.resume)
                elif args.command == "labels":
                    result = generate_labels(run, cohorts, calendar, references)
                elif args.command == "describe":
                    if args.include_final: expose_full_description(run)
                    result = describe(run, cohorts, calendar, args.include_final)
                elif args.command == "train":
                    result = train_development(run, cohorts, calendar)
                elif args.command == "freeze-release":
                    result = freeze_release(run, catalog, cohorts)
                elif args.command == "evaluate-final":
                    result = evaluate_final(run, catalog, cohorts, calendar)
                elif args.command == "run":
                    audit_sources(run, catalog, calendar, cohorts, references, args.deep_audit, args.hash_all)
                    try:
                        freeze_splits(run, catalog, cohorts, calendar)
                    except ResearchError as exc:
                        if exc.code != "INSUFFICIENT_INDEPENDENT_SUPPORT": raise
                        run.event("splits", exc.code, exc.message)
                    built = build_lifecycle_panel(run, catalog, calendar, cohorts, references, args.resume)
                    generate_labels(run, cohorts, calendar, references)
                    if args.include_final: expose_full_description(run)
                    described = describe(run, cohorts, calendar, args.include_final)
                    result = {"panels": built, "description": described["description_status"], **export_atlas(run)}
                else:
                    raise ResearchError("INVALID_COMMAND", args.command)
        if run and not run.dry_run:
            memory = psutil.Process().memory_info()
            run.event("resources", "TASK_COMPLETE", "Measured process resource use; one partition reader and bounded numeric threads", args.command,
                      elapsed_seconds=time.perf_counter() - run.started, rss_gib=memory.rss / 2**30,
                      peak_rss_gib=getattr(memory, "peak_wset", memory.rss) / 2**30,
                      numeric_threads=run.config["resources"]["inner_numeric_threads"])
        print(json.dumps(clean_json(result), default=json_default, indent=2, allow_nan=False))
        return 0
    except (ResearchError, OSError, ValueError, KeyError) as exc:
        payload = {"status": exc.code if isinstance(exc, ResearchError) else "RUNTIME_ERROR", "message": str(exc), "evidence": exc.evidence if isinstance(exc, ResearchError) else {}}
        if run and not run.dry_run:
            run.event("command", payload["status"], payload["message"], evidence=payload["evidence"])
            atomic_json(run.path / "last_failure.json", payload)
        print(json.dumps(clean_json(payload), default=json_default, indent=2), file=sys.stderr)
        return 2
    finally:
        if run is not None: run.close()


if __name__ == "__main__":
    raise SystemExit(main())
