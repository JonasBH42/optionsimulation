from datetime import date, datetime
from decimal import Decimal
import json
from types import SimpleNamespace
import pytest
from qqq_sim.config import AppConfig
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import write_synthetic_dataset, synthetic_mappings
from qqq_sim.replay.calendar import NY
from qqq_sim.services import reference_wizard as wizard
from qqq_sim.services.reference_inputs import build_reference_request, resolve_market_overrides, attach_reference_companions


def drive(monkeypatch, inputs, default="", result=None):
    prompts, outputs, calls = [], [], []
    iterator = iter(inputs)
    def read(prompt):
        prompts.append(prompt)
        return next(iterator)
    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(hash="test-job"), {"unmatched_trade_overrides": 0}
    monkeypatch.setattr(wizard, "DEFAULT_SYMBOL", default)
    monkeypatch.setattr(wizard, "build_reference_request", build)
    import qqq_sim.risk.offline.hit_oracle as oracle
    monkeypatch.setattr(oracle, "calculate_reference", lambda *args, **kwargs: result or {"status": "OK", "hold_p50": 0.5})
    assert wizard.run_wizard(AppConfig(), datetime(2026, 11, 9, 10, tzinfo=NY), read, outputs.append) == 0
    return prompts, outputs, calls


def test_interactive_symbol_is_startup_context_and_e_reaches_it(monkeypatch):
    prompts, outputs, _ = drive(monkeypatch, ["e", "QQQ", "e", "QQQ", "2026-11-13", "e", "q"])
    assert prompts[0] == prompts[1] == "Symbol: "
    assert prompts[3] == "Symbol: "
    assert prompts[2].startswith("Stage 1")
    assert not any("Stage 4" in prompt for prompt in prompts)


def test_fixed_symbol_has_no_extra_prompt_and_earliest_e_does_nothing(monkeypatch):
    prompts, outputs, _ = drive(monkeypatch, ["e", "q"], default="QQQ")
    assert all(prompt.startswith("Stage 1") for prompt in prompts)
    assert "Symbol: QQQ" in outputs


def test_edit_back_preserves_legs_undo_raw_history_and_results_clear(monkeypatch):
    values = ["2026-11-13", "P 595 s", "p 595 s", "p 595 l", "undo", "p 590 l", "done",
              "e", "undo", "p 590 l", "done", "2026-11-09 15:30", "done", "q"]
    prompts, outputs, calls = drive(monkeypatch, values, default="QQQ")
    assert calls[0][0][4] == (("P", Decimal(590), 1), ("P", Decimal(595), -2))
    assert prompts[-2].startswith("Stage 2") and prompts[-1].startswith("Stage 2")
    assert any("net position is empty" in output for output in outputs)
    assert any("position_id=QQQ|2026-11-13|QQQ|" in output for output in outputs)


def test_expiry_change_clears_old_contracts_and_dst_error_stays_stage3(monkeypatch):
    values = ["2026-11-13", "p 595 s", "e", "2026-11-20", "done", "p 590 l", "done",
              "2026-11-01 01:30", "q"]
    prompts, outputs, calls = drive(monkeypatch, values, default="QQQ")
    assert not calls
    assert any("net position is empty" in output for output in outputs)
    assert any("AMBIGUOUS_LOCAL_TIME" in output for output in outputs)
    assert prompts[-1].startswith("Stage 3")


def test_scalar_and_list_extra_globals_normalize_strictly():
    assert wizard.normalize_extra_target_pcts(25) == (Decimal(".25"), Decimal(".5"))
    assert wizard.normalize_extra_target_pcts([75, 25, 50, 25]) == (Decimal(".25"), Decimal(".5"), Decimal(".75"))
    assert wizard.normalize_extra_target_pcts(.25) == (Decimal(".0025"), Decimal(".5"))
    assert wizard.normalize_extra_day_offsets(2) == (2,)
    assert wizard.normalize_extra_day_offsets([3, 1, 1]) == (1, 3)
    for value in (True, "25", 0, 101, float("nan")):
        with pytest.raises(ValueError):
            wizard.normalize_extra_target_pcts(value)
    for value in (True, "2", 1.0, 0, -1):
        with pytest.raises(ValueError):
            wizard.normalize_extra_day_offsets(value)


def test_day_offsets_preserve_ny_clock_and_outside_horizon_not_clamped():
    start = datetime(2026, 3, 7, 10, tzinfo=NY)
    included, outside = wizard.resolve_extra_dates(start, datetime(2026, 3, 9, 10, tzinfo=NY), (1, 3))
    assert included == ({"offset_days": 1, "at": "2026-03-08T10:00:00-04:00"},)
    assert outside == ({"offset_days": 3, "at": "2026-03-10T10:00:00-04:00"},)


def test_exact_trade_override_never_carries_to_neighbor_position():
    one = wizard.position_identifier("QQQ", date(2026, 11, 13), "QQQ", (("P", Decimal(595), -1),))
    other = wizard.position_identifier("QQQ", date(2026, 11, 13), "QQQ", (("P", Decimal(594), -1),))
    values = {one: {"entry_cash_dollars": "100", "entry_context": "existing", "round_trip_cost_dollars": "10"}}
    assert wizard.resolve_trade_override(one, Decimal(75), values)["entry_cash_dollars"] == Decimal(100)
    ignored = wizard.resolve_trade_override(other, Decimal(75), values)
    assert ignored["entry_cash_dollars"] == Decimal(75) and ignored["entry_context"] == "hypothetical"
    assert ignored["unmatched_override_count"] == 1


def test_reference_midpoint_existing_basis_frozen_target_and_extra_reports(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / "data", minutes=4)
    data = HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())
    parameter_path = tmp_path / "parameters.json"
    parameter_path.write_text(json.dumps({"rate_cc": 0.04, "dividend_yield_cc": 0.01, "sigma_path": 0.25,
                                         "known_at": "2026-11-09T09:00:00-05:00", "fixing_at": "2026-11-13T16:00:00-05:00", "source": "SYNTHETIC_SCENARIO"}), encoding="utf-8")
    config = AppConfig(parameters_path=str(parameter_path))
    at, deadline, expiry = datetime(2026, 11, 9, 9, 31, tzinfo=NY), datetime(2026, 11, 9, 15, 30, tzinfo=NY), date(2026, 11, 13)
    legs = (("P", Decimal(590), 1), ("P", Decimal(595), -1))
    job, metadata = build_reference_request(config, at, "QQQ", expiry, legs, deadline, data=data)
    assert job.entry_cash_usd == Decimal(75)  # Midpoint basis, not natural-side credit65.
    assert job.target_usd == Decimal("37.50")
    identifier = metadata["position_id"]
    overrides = {identifier: {"entry_cash_dollars": "100", "entry_context": "existing", "round_trip_cost_dollars": "10"}}
    job, metadata = build_reference_request(config, at, "QQQ", expiry, legs, deadline, data=data,
        trade_overrides=overrides, extra_target_pcts=[25, 75], extra_day_offsets=[1, 7])
    assert job.entry_cash_usd == 100 and job.target_usd == 50
    assert set(job.optional_targets_usd) == {Decimal(25), Decimal(55), Decimal(75)}
    assert len(job.reporting_times_utc_us) == 1 and metadata["outside_horizon_dates"][0]["offset_days"] == 7
    assert metadata["prior_target_hit_status"] == "UNKNOWN"
    assert metadata["net_fixed_cost_companion"]["net_target_dollars"] == "45"
    product = {"QQQ": {"root": "QQQ", "eligible_intervals": [["2026-11-09T10:00:00-05:00", "2026-11-09T15:00:00-05:00"]]}}
    narrowed, metadata = build_reference_request(config, at, "QQQ", expiry, legs, deadline, data=data, product_overrides=product)
    assert narrowed.calendar_profile_id == "USER_OVERRIDE_QQQ_SESSION_SUBSET"
    assert len(narrowed.eligible_intervals) == 1
    product["QQQ"]["eligible_intervals"][0][1] = "2026-11-09T16:15:00-05:00"
    with pytest.raises(ValueError, match="exceeds supported"):
        build_reference_request(config, at, "QQQ", expiry, legs, deadline, data=data, product_overrides=product)


def test_market_override_scope_uses_snapshot_date_and_named_units():
    at = datetime(2026, 11, 9, 10, tzinfo=NY)
    entry = {"symbol": "QQQ", "effective_from": "2026-11-09T09:00:00-05:00", "effective_to": "2026-11-10T09:00:00-05:00",
             "provenance": "Explicit test scenario", "r": .03, "q": 0.0, "path_iv": .25}
    values, applied, _ = resolve_market_overrides("QQQ", date(2026, 11, 13), at, {}, [entry])
    assert values["rate_cc"] == .03 and values["dividend_yield_cc"] == 0
    assert applied[0]["label"] == "USER_OVERRIDE"
    values, applied, _ = resolve_market_overrides("SPY", date(2026, 11, 13), at, {}, [entry])
    assert not values and not applied


def test_override_boolean_and_invalid_decimal_are_actionable_errors():
    for invalid in (True, "not numeric", float("nan")):
        with pytest.raises(ValueError, match="finite numeric"):
            wizard.resolve_trade_override("exact", 1, {"exact": {"entry_cash_dollars": invalid}})
        entry = {"symbol": "QQQ", "effective_from": "2026-11-09T09:00:00-05:00", "effective_to": "2026-11-10T09:00:00-05:00", "provenance": "Test", "r": invalid}
        with pytest.raises(ValueError, match="finite numeric"):
            resolve_market_overrides("QQQ", date(2026, 11, 13), datetime(2026, 11, 9, 10, tzinfo=NY), {}, [entry])


def test_fixed_cost_companion_keeps_gross_and_terminal_events_separate():
    job = SimpleNamespace(targets=(Decimal(50), Decimal(55)), deadline_utc_us=200, reports=(200, 300))
    metadata = {"net_fixed_cost_companion": {"gross_equivalent_threshold": "55", "terminal_cost_status": "TERMINAL_COST_CONVENTION_NOT_SPECIFIED"}}
    result = {"status": "OK", "hold_p50": .7, "production_paths": 10, "first_hit_histograms": [{100: 7}, {100: 3, 250: 2}]}
    output = attach_reference_companions(job, result, metadata)
    assert output["hold_p50"] == .7
    companion = output["net_fixed_cost_companion"]
    assert companion["deadline_closing_probability"] == .3
    assert companion["through_last_trading_closing_probability"] == .5
    assert companion["terminal_inclusive_probability"] is None
    blocked = attach_reference_companions(job, {**result, "status": "RESOURCE_LIMIT"}, metadata)
    assert blocked["net_fixed_cost_companion"]["deadline_closing_probability"] is None
