"""Deterministic TEST_FIXTURE_ONLY acceptance tests; never production history."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_4dte_backtest as q


FIXTURE_ROOT = Path(__file__).resolve().parent / ".tmp" / "qqq_4dte_TEST_FIXTURE_ONLY"


class Fixture(unittest.TestCase):
    def setUp(self):
        FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_FIXTURE_ONLY_", dir=FIXTURE_ROOT)
        self.base = Path(self.temp.name)
        self.config = q.default_config()
        self.config["run_id"] = "TEST_FIXTURE_ONLY"
        self.config["source_policy"] = {"version": "TEST_FIXTURE_ONLY_V1", "root_override_evidence": "Explicit isolated deterministic test inputs, never production research"}
        self.config["paths"] = {"underlying_root": str(self.base / "underlying"), "options_root": str(self.base / "options"), "research_root": str(self.base / "research")}
        for path in self.config["paths"].values(): Path(path).mkdir()
        for kind in q.OPTION_KINDS: (self.base / "options" / kind).mkdir()
        self.config["source_contracts"] = {}
        for name, filename in q.SOURCE_NAMES.items():
            path = self.base / filename
            path.write_text("TEST_FIXTURE_ONLY contract " + name, encoding="utf-8")
            self.config["source_contracts"][name] = str(path)
        self.config["clocks"]["underlying_bar_evidence"] = "TEST_FIXTURE_ONLY known bar intervals"
        self.config["clocks"]["quote_timestamp_semantics"] = "SNAPSHOT_RECORD_TIME"
        self.config["clocks"]["quote_timestamp_evidence"] = "TEST_FIXTURE_ONLY sampled record clock"
        self.config["normalization"]["strike_divisor"] = 1
        self.config["forecast"]["bootstrap_repetitions"] = 50
        cal_path = self.base / "calendar.json"
        q.atomic_json(cal_path, {"version": "TEST_FIXTURE_ONLY_CALENDAR_V1", "provenance": "TEST_FIXTURE_ONLY", "sessions": [
            {"session_date": "2024-03-07", "regular_open_ny": "2024-03-07 09:30", "regular_close_ny": "2024-03-07 16:00"},
            {"session_date": "2024-03-08", "regular_open_ny": "2024-03-08 09:30", "regular_close_ny": "2024-03-08 16:00"},
            {"session_date": "2024-03-11", "regular_open_ny": "2024-03-11 09:30", "regular_close_ny": "2024-03-11 13:00"},
            {"session_date": "2024-03-12", "regular_open_ny": "2024-03-12 09:30", "regular_close_ny": "2024-03-12 16:00"},
            {"session_date": "2024-03-13", "regular_open_ny": "2024-03-13 09:30", "regular_close_ny": "2024-03-13 16:00"},
            {"session_date": "2024-03-14", "regular_open_ny": "2024-03-14 09:30", "regular_close_ny": "2024-03-14 16:00"},
            {"session_date": "2024-03-15", "regular_open_ny": "2024-03-15 09:30", "regular_close_ny": "2024-03-15 16:00"},
        ]})
        self.config["clocks"]["calendar_file"] = str(cal_path)
        self.run = q.Run(self.config)
        self.calendar = q.Calendar(self.config, pd.Timestamp("2024-03-01").date(), pd.Timestamp("2024-03-20").date())
        self.references = q.References(self.run)
        self.catalog = q.Catalog(self.run)
        self.store = q.DataStore(self.run, self.catalog, self.calendar, self.references)
        self.cohort = {"cohort_id": "CALENDAR_DATE_4:2024-03-11", "expiration_date": "2024-03-11", "D0": "2024-03-07",
                       "scheduled_t0": q.ny_timestamp("2024-03-07 09:31"), "T_E": pd.NaT, "last_scheduled_rth_session": "2024-03-11",
                       "expected_sessions": q.canonical_json(["2024-03-07", "2024-03-08", "2024-03-11"]), "right_censored": False,
                       "terminal_status": "TERMINAL_REFERENCE_MISSING"}

    def tearDown(self):
        self.run.close()
        self.temp.cleanup()

    def quotes(self, timestamps=None, strikes=None, rights=None, bids=None, asks=None, roots=None):
        timestamps = timestamps or ["2024-03-07 09:30", "2024-03-07 09:31"]
        n = len(timestamps)
        frame = pd.DataFrame({"symbol": ["QQQ"] * n, "root": roots or ["QQQ"] * n, "expiration": ["2024-03-11"] * n,
                              "strike": pd.Series(strikes or ["100"] * n, dtype="string"), "right": pd.Series(rights or ["CALL"] * n, dtype="string"),
                              "timestamp": timestamps, "bid": bids if bids is not None else [1.] * n,
                              "ask": asks if asks is not None else [1.2] * n, "source_record_ordinal": np.arange(n),
                              "source_partition_id": ["TEST_FIXTURE_ONLY_QUOTES"] * n, "source_sha256": ["TEST_FIXTURE_ONLY"] * n})
        partition = {"dataset_type": "greeks", "session_date": "2024-03-07", "expiration_date": "2024-03-11", "partition_id": "TEST_FIXTURE_ONLY"}
        mapping = {"strike": "strike"}
        return self.store.normalize_options(frame, partition, mapping, pd.Series([100.]))

    def underlying(self, timestamps=None, close=None):
        timestamps = timestamps or ["2024-03-07 09:30", "2024-03-07 09:31", "2024-03-07 09:32"]
        prices = close or [100.] * len(timestamps)
        return pd.DataFrame({"timestamp": timestamps, "open": prices, "high": np.array(prices) + 1, "low": np.array(prices) - 1,
                             "close": prices, "volume": [10.] * len(timestamps)})

    def panel(self, quotes=None):
        observations = self.quotes() if quotes is None else quotes
        current, _ = q.current_minute_quotes(observations, self.calendar.grid("2024-03-07"), "SNAPSHOT_RECORD_TIME")
        underlying, _ = q.normalize_underlying(self.underlying(), "2024-03-07", self.calendar, "START_LABELED")
        ugrid = q.revealed_underlying_grid(underlying, self.calendar.grid("2024-03-07"), 0)
        return q.attach_spot_features(current, ugrid, self.cohort, self.calendar, "2024-03-07", 100.)


class SourceAndIdentity(Fixture):
    def test_t01_source_contract_hashes_and_missing_source(self):
        self.assertEqual(set(self.run.contracts), set(q.SOURCE_NAMES))
        for row in self.run.contracts.values(): self.assertEqual(len(row["sha256"]), 64)
        config = q.merge_config(self.config, {"source_contracts": {"underlying": str(self.base / "absent.txt")}})
        with self.assertRaisesRegex(q.ResearchError, "REQUIRED_SOURCE_CONTRACT_MISSING"): q.Run(config)

    def test_t02_duplicate_logical_partitions(self):
        folder = self.base / "options" / "greeks"
        for name in ("greeks_20240307_exp=2024-03-11.csv.gz", "copy_20240307_exp=2024-03-11 (1).csv.gz"):
            with gzip.open(folder / name, "wt") as stream: stream.write("strike,right,timestamp,bid,ask\n")
        catalog = q.Catalog(self.run)
        self.assertEqual(catalog.status("greeks", "2024-03-07", "2024-03-11"), "DUPLICATE_LOGICAL_PARTITION")
        self.assertIsNone(catalog.get("greeks", "2024-03-07", "2024-03-11"))

    def test_t03_aliases_and_missing_required_columns(self):
        mapping = q.map_columns(["strike_price", "put_call", "quote_time", "bid_price", "ask_price"], "greeks")
        self.assertEqual(mapping["strike"], "strike_price")
        with self.assertRaisesRegex(q.ResearchError, "INVALID_SOURCE_SCHEMA"): q.map_columns(["strike", "right", "price"], "greeks")

    def test_t04_exact_strikes_scale_once_ambiguous_units(self):
        self.assertEqual(q.exact_strike("100125", 1000), "100.125")
        self.assertEqual(q.resolve_strike_scale(pd.Series([100000]), pd.Series([100]))[0], 1000)
        with self.assertRaisesRegex(q.ResearchError, "AMBIGUOUS_STRIKE_SCALE"): q.resolve_strike_scale(pd.Series([3000]), pd.Series([100]))

    def test_t05_iv_units_not_guessed(self):
        decimal, status = q.normalize_iv(pd.Series([25.]), "PERCENT")
        self.assertAlmostEqual(decimal.iloc[0], .25)
        unknown, status = q.normalize_iv(pd.Series([.25, 25]), None)
        self.assertTrue(unknown.isna().all()); self.assertEqual(status, "AMBIGUOUS_IV_SCALE")

    def test_t06_t07_right_root_deliverable_identity(self):
        rows = self.quotes(strikes=["100", "100"], rights=["CALL", "PUT"])
        self.assertEqual(rows.contract_id.nunique(), 2)
        adjusted = self.quotes(roots=["QQQ", "QQQ1"])
        self.assertEqual(adjusted.contract_id.nunique(), 2)
        self.assertFalse(adjusted.standard_supported.any())

    def test_t08_authoritative_spot_and_t31_unknown_forward(self):
        observations = self.quotes(); observations["vendor_underlying_price_diagnostic"] = 999
        panel = self.panel(observations)
        self.assertTrue(panel.spot.eq(100).all())
        self.assertTrue(panel.forward_log_moneyness.isna().all())
        self.assertTrue(panel.multiplier.isna().all())

    def test_t09_sources_immutable_and_t61_changed_source_identity(self):
        path = self.base / "underlying" / "QQQ_1m_2024-03-07.parquet"
        self.underlying().rename(columns={"timestamp": "regular_time"}).to_parquet(path, index=False)
        before = q.sha256_file(path)
        catalog = q.Catalog(self.run)
        partition = catalog.get("underlying", "2024-03-07")
        store = q.DataStore(self.run, catalog, self.calendar, self.references)
        store.underlying("2024-03-07")
        self.assertEqual(before, q.sha256_file(path))
        with path.open("ab") as stream: stream.write(b"TEST_FIXTURE_ONLY_CHANGED")
        self.assertNotEqual(catalog.identity, q.Catalog(self.run).identity)

    def test_t10_midnight_selector_is_all_session(self):
        observations = self.quotes()
        for kind in ("greeks", "iv", "ohlcv", "oi_before", "oi"):
            self.assertEqual(len(q.select_intraday(observations, "00:00:00", kind)), 2)
        self.assertEqual(len(q.select_intraday(observations, "09:30:00", "greeks")), 1)

    def test_noncanonical_roots_require_explicit_versioned_policy(self):
        config = q.merge_config(self.config, {"source_policy": {"version": "CANONICAL_SOURCE_CONTRACTS_V1", "root_override_evidence": None}})
        with self.assertRaisesRegex(q.ResearchError, "SOURCE_POLICY_CONFLICT"): q.validate_config(config)


class ClocksAndCohorts(Fixture):
    def test_t11_start_bar_available_only_at_end(self):
        rows, _ = q.normalize_underlying(self.underlying(), "2024-03-07", self.calendar, "START_LABELED")
        times = pd.DatetimeIndex([q.ny_timestamp("2024-03-07 09:30"), q.ny_timestamp("2024-03-07 09:31")])
        grid = q.revealed_underlying_grid(rows, times, 0)
        self.assertTrue(pd.isna(grid.spot.iloc[0])); self.assertEqual(grid.spot.iloc[1], 100)

    def test_t12_end_bar_not_delayed_twice(self):
        rows, _ = q.normalize_underlying(self.underlying(["2024-03-07 09:31"]), "2024-03-07", self.calendar, "END_LABELED")
        self.assertEqual(rows.available_at_ny.iloc[0], q.ny_timestamp("2024-03-07 09:31"))

    def test_t13_subminute_cutoff(self):
        observations = self.quotes(timestamps=["2024-03-07 09:30:59.999", "2024-03-07 09:31:00.001"], bids=[1., 9.], asks=[2., 10.])
        current, _ = q.current_minute_quotes(observations, self.calendar.grid("2024-03-07"), "SNAPSHOT_RECORD_TIME")
        self.assertEqual(current.midpoint.iloc[0], 1.5)
        self.assertEqual(current.decision_time_ny.iloc[1], q.ny_timestamp("2024-03-07 09:32"))

    def test_t14_ny_normalization_and_dst_elapsed_seconds(self):
        stamps = q.ny_times(pd.Series(["2024-03-07T14:30:00Z", "2024-03-07 09:30"]))
        self.assertTrue(stamps.eq(q.ny_timestamp("2024-03-07 09:30")).all())
        self.assertEqual(q.actual_seconds("2024-03-08 09:31", "2024-03-11 09:31"), 3 * 86400 - 3600)

    def test_t15_early_close_and_no_file_calendar_inference(self):
        self.assertEqual(len(self.calendar.grid("2024-03-11")), 210)
        self.assertNotIn("2024-03-09", self.calendar.sessions)

    def test_t16_t17_exact_calendar_cohort(self):
        folder = self.base / "options" / "greeks"
        for expiration in ("2024-03-11", "2024-03-13"):
            with gzip.open(folder / f"greeks_20240311_exp={expiration}.csv.gz", "wt") as stream: stream.write("strike,right,timestamp,bid,ask\n")
        cohorts = q.build_cohorts(self.run, q.Catalog(self.run), self.calendar, self.references).set_index("expiration_date")
        self.assertEqual(cohorts.loc["2024-03-11", "eligible_status"], "DATA_GAP_AT_START")
        self.assertEqual(cohorts.loc["2024-03-13", "eligible_status"], "NO_EXACT_CALENDAR_4_SESSION")

    def test_t18_cohort_conventions_have_different_ids(self):
        with gzip.open(self.base / "options" / "greeks" / "greeks_20240311_exp=2024-03-14.csv.gz", "wt") as stream: stream.write("strike,right,timestamp,bid,ask\n")
        catalog = q.Catalog(self.run)
        primary = q.build_cohorts(self.run, catalog, self.calendar, self.references).iloc[0]
        alternate_config = q.merge_config(self.config, {"run_id": "TEST_FIXTURE_ONLY_ALTERNATE", "cohort_definition": "TRADING_SESSION_OFFSET_4"})
        alternate = q.build_cohorts(q.Run(alternate_config), catalog, self.calendar, self.references).iloc[0]
        self.assertEqual(primary.D0, "2024-03-10")
        self.assertEqual(alternate.D0, "2024-03-08")
        self.assertNotEqual(primary.cohort_id, alternate.cohort_id)

    def test_t19_eod_records_excluded(self):
        rows, checks = q.normalize_underlying(self.underlying(["2024-03-07 15:59", "2024-03-07 16:00"]), "2024-03-07", self.calendar, "START_LABELED")
        self.assertEqual(len(rows), 1); self.assertEqual(checks["outside_rth"], 1)
        current, rejected = q.current_minute_quotes(self.quotes(["2024-03-07 16:00"]), self.calendar.grid("2024-03-07"), "SNAPSHOT_RECORD_TIME")
        self.assertEqual(len(current), 0); self.assertEqual(len(rejected), 1)

    def test_t20_terminal_evidence_not_replaced(self):
        panel = self.panel()
        labels = q.generate_contract_labels(panel, pd.DataFrame(), self.cohort, self.calendar, self.config, self.references)
        terminal = labels[labels.horizon_id.eq("VERIFIED_EXPIRY_PAYOFF")]
        self.assertTrue(terminal.delta_mid.isna().all())
        self.assertTrue(terminal.censoring_reason.eq("TERMINAL_REFERENCE_MISSING").all())

    def test_verified_product_boundary_truncates_only_the_study_grid(self):
        cohort = {**self.cohort, "T_E": q.ny_timestamp("2024-03-07 10:00"), "official_last_option_trading_time": q.ny_timestamp("2024-03-07 09:32")}
        grid = q.cohort_grid(self.calendar, cohort, "2024-03-07")
        self.assertEqual(len(grid), 2)
        self.assertEqual(len(self.calendar.grid("2024-03-07")), 390)
        self.assertNotEqual(q.endpoint_for(grid[-1], 1, self.calendar, cohort)[1], "SCHEDULED")

    def test_verified_standard_terms_and_terminal_payoff_stay_separate(self):
        boundary = {"expiration": "2024-03-11", "economic_expiry_ny": "2024-03-11 13:00", "official_last_option_trading_time_ny": "2024-03-11 13:00", "settlement_reference_time_ny": "2024-03-11 13:00", "evidence": "TEST_FIXTURE_ONLY"}
        self.references.rules = {"profiles": [{"roots": ["QQQ"], "effective_from": "2024-03-01", "effective_to": "2024-03-31", "multiplier": 100., "deliverable_id": "TEST_FIXTURE_ONLY_STANDARD", "standard_unadjusted": True, "exercise_style": "AMERICAN", "settlement_type": "PHYSICAL", "evidence": "TEST_FIXTURE_ONLY"}], "boundaries": [boundary]}
        self.references.terminals = pd.DataFrame([{"expiration": "2024-03-11", "price": 110., "reference_time_ny": "2024-03-11 13:00", "known_at_ny": "2024-03-11 13:01", "source": "TEST_FIXTURE_ONLY_TERMINAL", "evidence": "TEST_FIXTURE_ONLY"}])
        cohort = {**self.cohort, "T_E": q.ny_timestamp("2024-03-11 13:00")}
        panel = self.panel()
        self.assertTrue(panel.multiplier.eq(100).all())
        labels = q.generate_contract_labels(panel, pd.DataFrame(), cohort, self.calendar, self.config, self.references)
        terminal = labels[labels.horizon_id.eq("VERIFIED_EXPIRY_PAYOFF")]
        self.assertTrue(terminal.outcome_mask.all())
        self.assertTrue(np.allclose(terminal.delta_mid, 8.9))
        self.assertTrue(np.allclose(terminal.delta_usd_per_contract, 890.))
        self.assertTrue(panel.midpoint.eq(1.1).all())

    def test_concurrent_writers_to_same_run_are_rejected(self):
        with self.assertRaisesRegex(q.ResearchError, "RUN_ALREADY_ACTIVE"):
            q.Run(self.config)


class MissingnessAndFeatures(Fixture):
    def test_completed_option_bar_evidence_is_known_at_entry(self):
        self.config["clocks"]["option_bar_semantics"] = "START_LABELED"
        self.config["clocks"]["option_bar_evidence"] = "TEST_FIXTURE_ONLY"
        frame = pd.DataFrame({"symbol": ["QQQ"], "expiration": ["2024-03-11"], "strike": ["100"], "right": ["CALL"], "timestamp": ["2024-03-07 09:30"], "source_partition_id": ["TEST_FIXTURE_ONLY"], "source_record_ordinal": [0]})
        partition = {"dataset_type": "ohlcv", "session_date": "2024-03-07", "expiration_date": "2024-03-11"}
        normalized = self.store.normalize_options(frame, partition, {"strike": "strike"}, pd.Series([100.]))
        inventory = q.contract_inventory([normalized], self.cohort, 100.)
        self.assertTrue(inventory.observed_at_t0.iloc[0])
        self.assertFalse(inventory.late_appearance.iloc[0])

    def test_t21_late_strike_not_at_entry_and_t23_unknown_denominator(self):
        observations = self.quotes(timestamps=["2024-03-07 09:30", "2024-03-07 09:35"], strikes=["100", "110"])
        inventory = q.contract_inventory([observations], self.cohort, 100)
        self.assertTrue(inventory.loc[inventory.strike.eq(100), "observed_at_t0"].iloc[0])
        self.assertFalse(inventory.loc[inventory.strike.eq(110), "observed_at_t0"].iloc[0])
        self.assertTrue(inventory.loc[inventory.strike.eq(110), "late_appearance"].iloc[0])
        chains = q.chain_snapshots(self.panel(), q.revealed_underlying_grid(pd.DataFrame(), self.calendar.grid("2024-03-07"), 0), self.cohort, self.calendar, "2024-03-07", 1)
        self.assertTrue(chains.listed_quote_coverage.isna().all())

    def test_t22_listed_unquoted_inventory(self):
        self.references.metadata = pd.DataFrame([{"root": "QQQ", "expiration": "2024-03-11", "exact_strike": "120", "right": "P", "multiplier": 100., "deliverable_id": "TEST_FIXTURE_ONLY_STANDARD",
                                                  "listing_effective_time": q.ny_timestamp("2024-03-01"), "listing_known_at": q.ny_timestamp("2024-03-01"), "source": "TEST_FIXTURE_ONLY"}])
        inventory = q.contract_inventory([self.quotes()], self.cohort, 100)
        result = q.add_listed_inventory(inventory, self.references, self.cohort, 100)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[result.strike.eq(120), "listing_status"].iloc[0], "LISTED_NO_OBSERVATION")

    def test_t24_disappearing_contract_and_t28_sparse_contract_retained(self):
        panel = self.panel(self.quotes(["2024-03-07 09:30"]))
        inventory = q.contract_inventory([self.quotes(["2024-03-07 09:30"])], self.cohort, 100)
        self.assertEqual(len(inventory), 1); self.assertEqual(len(panel), 1)
        labels = q.generate_contract_labels(panel, inventory, self.cohort, self.calendar, self.config, self.references)
        self.assertTrue(labels.loc[labels.horizon_id.eq("1"), "delta_mid"].isna().all())

    def test_t25_zero_bid_locked_crossed_and_zero_ask(self):
        states = q.classify_quotes(pd.Series([0., 1., 2., 0., np.nan]), pd.Series([.1, 1., 1., 0., 1.]))
        self.assertEqual(states.tolist(), ["VALID_ZERO_BID", "LOCKED_MARKET", "CROSSED_MARKET", "INVALID_NUMERIC", "INVALID_NUMERIC"])

    def test_t26_unknown_update_age_not_zero(self):
        current, _ = q.current_minute_quotes(self.quotes(), self.calendar.grid("2024-03-07"), "SNAPSHOT_RECORD_TIME")
        self.assertTrue(current.quote_age_seconds.isna().all())
        self.assertTrue(current.snapshot_record_age_seconds.eq(60).all())

    def test_t27_prior_oi_future_publication_blocked(self):
        panel = self.panel()
        oi = pd.DataFrame({"contract_id": [panel.contract_id.iloc[0]], "open_interest": [20], "source_time_ny": [q.ny_timestamp("2024-03-08 06:30")]})
        result, _ = q.join_prior_oi(panel, oi, "2024-03-07", self.calendar)
        self.assertTrue(result.prior_oi.isna().all())
        self.assertEqual(self.config["execution_research"]["enabled"], False)

    def test_t29_future_underlying_rows_do_not_change_past(self):
        early, _ = q.normalize_underlying(self.underlying(["2024-03-07 09:30"]), "2024-03-07", self.calendar, "START_LABELED")
        all_rows, _ = q.normalize_underlying(self.underlying(["2024-03-07 09:30", "2024-03-07 09:31"], [100, 999]), "2024-03-07", self.calendar, "START_LABELED")
        grid = self.calendar.grid("2024-03-07")
        a, b = q.revealed_underlying_grid(early, grid, 0), q.revealed_underlying_grid(all_rows, grid, 0)
        self.assertEqual(a.spot.iloc[0], b.spot.iloc[0]); self.assertEqual(b.spot.iloc[0], 100)

    def test_t30_fixed_identity_does_not_follow_changing_atm(self):
        data = self.quotes(timestamps=["2024-03-07 09:30", "2024-03-07 09:30", "2024-03-07 09:31", "2024-03-07 09:31"], strikes=["100", "110", "100", "110"], bids=[1, 2, 3, 4], asks=[1.2, 2.2, 3.2, 4.2])
        panel = self.panel(data)
        labels = q.generate_contract_labels(panel, pd.DataFrame(), self.cohort, self.calendar, self.config, self.references)
        one = labels[labels.horizon_id.eq("1") & labels.outcome_mask]
        self.assertTrue(np.allclose(one.delta_mid, 2.))
        self.assertEqual(one.contract_id.nunique(), 2)

    def test_t39_fixed_strike_remapped_after_spot_move_t40_vendor_greeks(self):
        observations = self.quotes()
        current, _ = q.current_minute_quotes(observations, self.calendar.grid("2024-03-07"), "SNAPSHOT_RECORD_TIME")
        bars, _ = q.normalize_underlying(self.underlying(close=[100, 110, 111]), "2024-03-07", self.calendar, "START_LABELED")
        ugrid = q.revealed_underlying_grid(bars, self.calendar.grid("2024-03-07"), 0)
        panel = q.attach_spot_features(current, ugrid, self.cohort, self.calendar, "2024-03-07", 100.)
        self.assertEqual(panel.contract_id.nunique(), 1)
        self.assertTrue(panel.strike.eq(100).all())
        self.assertNotEqual(panel.x_s.iloc[0], panel.x_s.iloc[1])
        self.assertTrue(panel.x_s0.eq(0).all())
        self.assertTrue(panel.vendor_greek_convention_status.eq("UNVERIFIED_DIAGNOSTIC_ONLY").all())


class PricingAndShapes(Fixture):
    def test_t32_terminal_no_iv_and_t33_flat_vega(self):
        self.assertEqual(q.american_tree_price(110, 100, 0, .2, "C", .03, .01), 10.)
        self.assertEqual(q.implied_vol_interval(lambda x: 10., 9, 11, 0)["status"], "TERMINAL_NO_IV")
        interval = q.implied_vol_interval(lambda x: 10., 9, 11, .1)
        self.assertIsNone(interval["iv_mid"]); self.assertEqual(interval["iv_low"], 0)
        self.assertIn("FLAT", interval["status"])

    def test_t34_independent_american_solvers_and_exercise(self):
        tree = q.american_tree_price(100, 100, 1., .2, "P", .05, 0., 500)
        fd = q.american_fd_price(100, 100, 1., .2, "P", .05, 0., space_steps=400, time_steps=350)
        self.assertAlmostEqual(tree, 6.09, delta=.025)
        self.assertAlmostEqual(tree, fd, delta=.05)
        dividend_case = q.american_tree_price(120, 100, .5, .2, "C", .02, .20, 300)
        self.assertGreaterEqual(dividend_case, 20.)

    def test_t35_t36_convex_monotone_intervals_and_raw_unchanged(self):
        strikes, bid, ask = np.array([90., 100., 110.]), np.array([11.9, 8.9, .9]), np.array([12.1, 9.1, 1.1])
        before = bid.copy()
        fit = q.fit_constrained_shape(strikes, bid, ask, 100, "C", self.config["shape"], discount_maximum=1.1)
        slopes = np.diff(fit["fitted_prices"]) / np.diff(strikes)
        self.assertTrue(np.all(slopes <= 1e-6)); self.assertTrue(np.all(np.diff(slopes) >= -1e-6))
        self.assertEqual(fit["status"], "INCONSISTENT_QUOTE_INTERVALS")
        self.assertTrue(np.array_equal(before, bid))

    def test_t37_large_gap_and_wing_extrapolation_unsupported(self):
        fit = {"strikes": np.array([90., 91., 140.]), "fitted_prices": np.array([12., 11., .1]), "unsupported_gaps": np.array([False, True])}
        self.assertTrue(np.isnan(q.interpolate_supported_shape(fit, 120)[0]))
        self.assertEqual(q.interpolate_supported_shape(fit, 150)[1], "UNSUPPORTED_EXTRAPOLATION")

    def test_t38_masked_factors_keep_absent_wing_unsupported(self):
        values = np.array([[1., 2., np.nan], [2., 3., np.nan], [3., 4., np.nan], [4., 5., np.nan]])
        pack = q.MaskedFactors().fit(values, 1, 42, 4)
        self.assertNotIn(2, pack.columns)
        self.assertTrue(np.isnan(pack.reconstruct(values)[:, 2]).all())


class LabelsAndValidation(Fixture):
    def test_t41_horizon_cannot_cross_close(self):
        endpoint, status = q.endpoint_for(q.ny_timestamp("2024-03-07 16:00"), 1, self.calendar, self.cohort)
        self.assertEqual(status, "HORIZON_OUTSIDE_SESSION")
        self.assertEqual(endpoint, q.ny_timestamp("2024-03-07 16:01"))

    def test_t42_missing_endpoint_not_next_valid_quote(self):
        panel = self.panel(self.quotes(["2024-03-07 09:30", "2024-03-07 09:32"], bids=[1., 7.], asks=[1.2, 7.2]))
        labels = q.generate_contract_labels(panel, pd.DataFrame(), self.cohort, self.calendar, self.config, self.references)
        first = labels[labels.horizon_id.eq("1")].iloc[0]
        self.assertFalse(first.outcome_mask); self.assertTrue(pd.isna(first.delta_mid))

    def test_t43_observed_hit_survives_later_loss(self):
        result = q.sampled_hit(np.array([.20, -.30]), 2, .10)
        self.assertEqual(result["status"], "SUCCESS_OBSERVED"); self.assertEqual(result["first_hit_sample_index"], 0)

    def test_t44_missing_grid_bounds(self):
        result = q.sampled_hit(np.array([np.nan, 0.]), 2, .10)
        self.assertEqual((result["lower"], result["upper"]), (0., 1.))
        self.assertEqual(q.sampled_hit(np.array([0., 0.]), 2, .1)["status"], "FAILURE_OBSERVED")

    def test_t46_t48_training_cutoff_rejects_future_labels(self):
        frame = pd.DataFrame({"label_available_at": [q.ny_timestamp("2024-03-08 10:00")]})
        model = q.JointForecaster({"name": "B0", "rank": 0}, self.config["forecast"], 42)
        with self.assertRaisesRegex(q.ResearchError, "SPLIT_LEAKAGE_DETECTED"): model.fit(frame, q.ny_timestamp("2024-03-07 16:00"))

    def test_t47_training_only_normalizer_and_t52_outside_domain(self):
        normalizer = q.RobustNormalizer().fit(np.array([[1., 9.], [2., 9.], [3., 9.]]))
        before = normalizer.center.copy()
        normalizer.transform(np.array([[1000., 9.]]))
        self.assertTrue(np.array_equal(before, normalizer.center))
        self.assertFalse(normalizer.within_domain(np.array([[1000., 9.]]))[0])
        self.assertTrue(normalizer.constant[1])

    def test_t49_t60_block_resampling_duplicate_strikes_do_not_shrink_ci(self):
        table = pd.DataFrame({"cohort_id": ["A", "B", "A", "B"], "calendar_block": [1, 1, 2, 2], "value_sum": [1., 2., 3., 5.], "value_count": [1., 1., 1., 1.]})
        first = q.block_bootstrap_expiration_mean(table, 100, 42)
        scaled = table.copy(); scaled[["value_sum", "value_count"]] *= 600
        second = q.block_bootstrap_expiration_mean(scaled, 100, 42)
        self.assertEqual(first["lower"], second["lower"]); self.assertEqual(first["upper"], second["upper"])

    def test_t50_sealed_test_identity_persists_across_run_ids(self):
        split = {"sealed_test_identity": "TEST_FIXTURE_ONLY_HOLDOUT"}
        marker = q.sealed_marker(self.run, split)
        self.assertEqual(marker.parent.name, "_sealed_tests")
        other = object.__new__(q.Run); other.root = self.run.root
        self.assertEqual(marker, q.sealed_marker(other, split))

    def test_consumed_holdout_cannot_be_reset_by_shifting_its_dates(self):
        q.atomic_json(self.run.root / "_sealed_tests" / "TEST_FIXTURE_ONLY_prior.json", {"status": "COMPLETE", "test_sessions": ["2024-03-08", "2024-03-11"]})
        overlapping = {"sealed_test_start": "2024-03-11 09:30", "calendar_sessions": ["2024-03-11", "2024-03-12"], "sealed_test_identity": "TEST_FIXTURE_ONLY_shifted"}
        with self.assertRaisesRegex(q.ResearchError, "FINAL_TEST_ALREADY_CONSUMED"): q.assert_sealed_available(self.run, overlapping)
        later = {"sealed_test_start": "2024-03-12 09:30", "calendar_sessions": ["2024-03-12", "2024-03-13"], "sealed_test_identity": "TEST_FIXTURE_ONLY_later"}
        self.assertIsNone(q.assert_sealed_available(self.run, later))

    def test_t51_joint_scenario_score_and_t53_no_signal_support_floor(self):
        result = q.scenario_scores(np.array([[-1., -2.], [1., 2.]]), np.array([.5, .5]), np.array([0., 0.]), np.ones(2))
        self.assertGreaterEqual(result["energy_score"], 0.)
        self.assertEqual(result["quantile_crossings"], 0)
        self.assertGreaterEqual(self.config["forecast"]["min_expirations"], 30)

    def test_t54_descriptive_only_is_valid_and_optional_actions_disabled(self):
        self.assertFalse(self.config["execution_research"]["enabled"])
        config = q.merge_config(self.config, {"execution_research": {"enabled": True}})
        with self.assertRaisesRegex(q.ResearchError, "EXECUTION_MODEL_UNVERIFIED"): q.validate_config(config)

    def test_t62_atomic_writes_and_no_complete_partial_file(self):
        output = self.base / "research" / "TEST_FIXTURE_ONLY.json"
        q.atomic_json(output, {"value": 1})
        self.assertTrue(output.is_file()); self.assertFalse(output.with_name(output.name + ".part").exists())
        self.assertEqual(q.read_json(output), {"value": 1})


class PipelineAndForecasts(Fixture):
    def test_real_adapter_fixture_pipeline_and_resume(self):
        for session in ("2024-03-07", "2024-03-08", "2024-03-11"):
            times = [session + " 09:30", session + " 09:31"]
            self.underlying(times, [100., 101.]).rename(columns={"timestamp": "regular_time"}).to_parquet(self.base / "underlying" / f"QQQ_1m_{session}.parquet", index=False)
            raw = pd.DataFrame({"symbol": ["QQQ", "QQQ"], "expiration": ["2024-03-11"] * 2, "strike": ["100.000"] * 2,
                                "right": ["CALL"] * 2, "timestamp": times, "bid": [1., 1.1], "ask": [1.2, 1.3]})
            raw.to_csv(self.base / "options" / "greeks" / f"greeks_{session.replace('-', '')}_exp=2024-03-11.csv.gz", index=False, compression="gzip")
        catalog = q.Catalog(self.run)
        cohorts = q.build_cohorts(self.run, catalog, self.calendar, self.references)
        result = q.build_lifecycle_panel(self.run, catalog, self.calendar, cohorts, self.references)
        self.assertEqual(result["blocked_units"], [])
        directory = self.run.path / "panels" / "2024-03-11"
        master = pd.read_parquet(directory / "contract_master.parquet")
        self.assertEqual(len(master), 1)
        self.assertTrue(master.observed_at_t0.iloc[0])
        panel = q.load_cohort_panel(directory)
        self.assertEqual(len(panel), 6)
        self.assertTrue((panel.raw_quote_time_ny < panel.decision_time_ny).all())
        self.assertTrue(panel.recomputed_iv.isna().all())
        self.assertTrue(panel.intrinsic.isna().all())
        old = (directory / "complete.json").stat().st_mtime_ns
        q.build_lifecycle_panel(self.run, catalog, self.calendar, cohorts, self.references)
        self.assertEqual(old, (directory / "complete.json").stat().st_mtime_ns)
        q.generate_labels(self.run, cohorts, self.calendar, self.references)
        labels = pd.read_parquet(directory / "labels.parquet")
        one = labels[labels.horizon_id.eq("1") & labels.outcome_mask]
        self.assertEqual(len(one), 3)
        self.assertTrue(np.allclose(one.delta_mid, .1))

    def training_frame(self):
        rng = np.random.default_rng(107)
        n = 80
        result = pd.DataFrame({column: rng.normal(size=n) for column in q.CORE_FEATURES})
        result["calendar_date_dte"] = 2
        result["minutes_since_open"] = 60.
        result["minutes_to_session_close"] = 330.
        result["remaining_regular_session_minutes"] = 720.
        result["weekday"] = 1.
        result["mean_spread_spot"] = .1
        result["valid_fraction_current"] = .5
        for cell in q.CELL_COLUMNS: result[cell] = .1 + rng.normal(scale=.0001, size=n)
        result["cohort_id"] = ["TEST_FIXTURE_ONLY_E" + str(i % 4) for i in range(n)]
        result["calendar_block"] = [i % 3 for i in range(n)]
        result["decision_time_ny"] = pd.date_range(q.ny_timestamp("2024-03-01 10:30"), periods=n, freq="min")
        result["label_available_at"] = result.decision_time_ny + pd.Timedelta(minutes=5)
        result["session_date"] = "2024-03-01"
        result["horizon_minutes"] = 5
        result["sample_id"] = ["TEST_FIXTURE_ONLY_" + str(i) for i in range(n)]
        result["label_clock_valid"] = True
        response = rng.normal(scale=.001, size=n)
        result["target_log_spot_return"] = response
        result["target_spread_spot_change"] = response * .01
        result["target_valid_fraction_change"] = response * 3
        for i, cell in enumerate(("C_4", "P_4", "P_2", "C_6")):
            result["target_delta_cell_" + cell] = response * (i + 2)
        return result

    def test_b0_b1_b2_preserve_joint_response_dependence(self):
        training = self.training_frame()
        cutoff = q.ny_timestamp("2024-03-07 09:30")
        query = training.iloc[[5]].copy()
        query["decision_time_ny"] = q.ny_timestamp("2024-03-08 10:30")
        query["sample_id"] = "TEST_FIXTURE_ONLY_QUERY"
        for spec in ({"name": "B0", "rank": 0}, {"name": "B1", "rank": 0, "neighbor_cap": 50}, {"name": "B2", "rank": 0, "ridge": 1.}):
            model = q.JointForecaster(spec, self.config["forecast"], 42).fit(training, cutoff)
            scenarios, weights, info = model.scenarios(query)
            self.assertIsNotNone(scenarios)
            self.assertAlmostEqual(weights.sum(), 1.)
            self.assertTrue(np.allclose(scenarios[:, 1], scenarios[:, 0] * .01))
            self.assertTrue(np.allclose(scenarios[:, 3], scenarios[:, 0] * 2))
            self.assertEqual(info["status"], "INSUFFICIENT_INDEPENDENT_SUPPORT")

    def test_forecast_features_ignore_all_future_label_columns(self):
        training = self.training_frame()
        model = q.JointForecaster({"name": "B0", "rank": 0}, self.config["forecast"], 42).fit(training, q.ny_timestamp("2024-03-07 09:30"))
        query = training.iloc[[0]].copy()
        before = model.features(query)
        for column in model.targets: query[column] = 999999.
        query["end_spot"] = 999999.
        self.assertTrue(np.array_equal(before, model.features(query), equal_nan=True))

    def test_forecast_pack_and_scoring_roundtrip(self):
        training = self.training_frame()
        model = q.JointForecaster({"name": "B1", "rank": 0, "neighbor_cap": 50}, self.config["forecast"], 42).fit(training, q.ny_timestamp("2024-03-07 09:30"))
        query = training.iloc[[3]].copy(); query["decision_time_ny"] = q.ny_timestamp("2024-03-08 10:30")
        scored = model.predict_and_score(query, "TEST_FIXTURE_ONLY_FOLD")
        self.assertEqual(len(scored), 1)
        self.assertTrue(np.isfinite(scored.energy_score.iloc[0]))
        model.save(self.base / "research" / "pack", "TEST_FIXTURE_ONLY_FOLD")
        saved = q.read_json(self.base / "research" / "pack" / "pack.json")
        self.assertEqual(saved["fold_id"], "TEST_FIXTURE_ONLY_FOLD")
        self.assertEqual(saved["targets"], model.targets)

    def test_actual_interval_purging_removes_straddling_lifecycles(self):
        for expiry in ("2024-03-11", "2024-03-12", "2024-03-14", "2024-03-15"):
            with gzip.open(self.base / "options" / "greeks" / f"greeks_20240307_exp={expiry}.csv.gz", "wt") as stream: stream.write("strike,right,timestamp,bid,ask\n")
        catalog = q.Catalog(self.run)
        cohorts = q.build_cohorts(self.run, catalog, self.calendar, self.references)
        splits = q.freeze_splits(self.run, catalog, cohorts, self.calendar)
        for fold in splits["folds"]:
            self.assertFalse(set(fold["training_cohorts"]) & set(fold["validation_cohorts"]))
            for cohort_id in fold["training_cohorts"]:
                cohort = cohorts[cohorts.cohort_id.eq(cohort_id)].iloc[0]
                finish = self.calendar.grid(json.loads(cohort.expected_sessions)[-1])[-1]
                self.assertLess(finish, q.ny_timestamp(fold["training_cutoff"]))

    def test_holm_multiple_horizon_family(self):
        self.assertTrue(np.allclose(q.holm_adjust([.01, .04, .03]), [.03, .06, .06]))

    def test_full_prediction_stages_and_sealed_test_once(self):
        # Short explicit fixture sessions let the complete chronological protocol
        # run in seconds. These are NOT real market-calendar observations.
        sessions = pd.bdate_range("2024-01-01", periods=70).strftime("%Y-%m-%d").tolist()
        cal_path = self.base / "full_forecast_TEST_FIXTURE_ONLY_calendar.json"
        q.atomic_json(cal_path, {"version": "TEST_FIXTURE_ONLY_FULL_FORECAST", "provenance": "TEST_FIXTURE_ONLY artificial ten-minute sessions", "sessions": [
            {"session_date": day, "regular_open_ny": day + " 09:30", "regular_close_ny": day + " 09:40"} for day in sessions]})
        config = q.merge_config(self.config, {"run_id": "TEST_FIXTURE_ONLY_FULL_FORECAST", "clocks": {"calendar_file": str(cal_path)},
                    "labels": {"horizons_minutes": [1], "anchor_spacing_minutes": 5},
                    "forecast": {"factor_candidates": [0], "neighbor_candidates": [50], "ridge_candidates": [1.], "max_scenarios": 30, "bootstrap_repetitions": 20}})
        for i in range(0, len(sessions), 5):
            with gzip.open(self.base / "options" / "greeks" / f"greeks_{sessions[i].replace('-', '')}_exp={sessions[i+4]}.csv.gz", "wt") as stream:
                stream.write("strike,right,timestamp,bid,ask\n")
        run = q.Run(config)
        calendar = q.Calendar(config, pd.Timestamp(sessions[0]).date(), pd.Timestamp(sessions[-1]).date())
        references = q.References(run)
        catalog = q.Catalog(run)
        cohorts = q.build_cohorts(run, catalog, calendar, references)
        rng = np.random.default_rng(719)
        for cohort in cohorts.to_dict("records"):
            grid = pd.DatetimeIndex([t for day in json.loads(cohort["expected_sessions"]) for t in calendar.grid(day)])
            n = len(grid)
            data = pd.DataFrame({column: np.zeros(n) for column in q.CORE_FEATURES})
            data["decision_time_ny"] = grid
            data["session_date"] = grid.strftime("%Y-%m-%d")
            data["cohort_id"] = cohort["cohort_id"]
            data["expiration_date"] = cohort["expiration_date"]
            data["cohort_entry_date"] = cohort["D0"]
            data["calendar_date_dte"] = [(pd.Timestamp(cohort["expiration_date"]).date() - t.date()).days for t in grid]
            data["minutes_since_open"] = [t.minute - 30 for t in grid]
            data["minutes_to_session_close"] = 10 - data.minutes_since_open
            data["remaining_regular_session_minutes"] = np.arange(n, 0, -1)
            data["weekday"] = [t.weekday() for t in grid]
            data["spot"] = 100 + rng.normal(scale=.01, size=n)
            data["mean_spread_spot"] = .001 + rng.normal(scale=.00001, size=n)
            data["observed_contracts"] = 40
            data["valid_quotes"] = 40
            for cell in q.CELL_COLUMNS: data[cell] = .1 + rng.normal(scale=.0001, size=n)
            directory = run.path / "panels" / cohort["expiration_date"]
            q.atomic_frame(directory / "chain_snapshots.parquet", data)
            q.atomic_json(directory / "complete.json", {"namespace": "TEST_FIXTURE_ONLY", "artifacts": {str((directory / "chain_snapshots.parquet").relative_to(run.path)): q.sha256_file(directory / "chain_snapshots.parquet")}})
        q.freeze_splits(run, catalog, cohorts, calendar)
        development = q.train_development(run, cohorts, calendar)
        self.assertTrue(development["selected_by_horizon"])
        frozen = q.freeze_release(run, catalog, cohorts)
        self.assertTrue(frozen["release_id"])
        final = q.evaluate_final(run, catalog, cohorts, calendar)
        self.assertTrue(final["test_consumed"])
        self.assertNotIn("VALIDATED", final["horizons"][0]["prediction_status"])
        with self.assertRaisesRegex(q.ResearchError, "FINAL_TEST_ALREADY_CONSUMED"):
            q.evaluate_final(run, catalog, cohorts, calendar)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"success": result.wasSuccessful(), "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "engine_sha256": q.sha256_file(Path(q.__file__)), "test_sha256": q.sha256_file(Path(__file__)), "namespace": "TEST_FIXTURE_ONLY",
              "coverage": "Core source/identity/clock/missingness/label/numerical/validation/reproducibility tests. Optional path, account, execution, cash-dividend pricing and universal IV-domain validation remain unsupported; no test report claims those modules are released."}
    if args.json_output: q.atomic_json(args.json_output, report)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
