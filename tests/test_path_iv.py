from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
import pytest
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import write_synthetic_dataset, synthetic_mappings
from qqq_sim.replay.calendar import NY
from qqq_sim.risk.contracts import RiskError
from qqq_sim.services.path_iv import local_atm_path_iv


@pytest.fixture
def context(tmp_path, monkeypatch):
    manifest = write_synthetic_dataset(tmp_path / "data", minutes=4)
    data = HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())
    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 34, tzinfo=NY))
    import qqq_sim.risk.features as features
    monkeypatch.setattr(features, "validated_leg_mark", lambda leg, *args, **kwargs: {"price": (leg.bid+leg.ask)/2, "error": 0., "vega_per_1pct": 1.}, raising=False)
    pack = SimpleNamespace(validation={"capabilities": {"price": {"qualified": True}}}, pack_id="qualified-test-price")
    parameters = {"rate_cc": .04, "dividend_yield_cc": .01, "fixing_at": "2026-11-13T16:00:00-05:00"}
    return snapshot, snapshot.quotes[0].contract.expiration, parameters, pack


def test_atm_fallback_is_causal_and_records_exact_contributors(context):
    snapshot, expiry, parameters, pack = context
    sigma, evidence = local_atm_path_iv(snapshot, expiry, parameters, pack, 1e-5)
    assert sigma == .25 and 4 <= len(evidence["contributors"]) <= 10
    assert sum(item["weight"] for item in evidence["contributors"]) == pytest.approx(1)
    assert {item["right"] for item in evidence["contributors"]} == {"C", "P"}
    assert all(item["quote_id"] for item in evidence["contributors"])
    first = snapshot.quotes[0]
    future = replace(first, latest_event=replace(first.latest_event, available_at=snapshot.asof+timedelta(seconds=1), iv=3.))
    other_sigma, audit = local_atm_path_iv(replace(snapshot, quotes=(future, *snapshot.quotes[1:])), expiry, parameters, pack, 1e-5)
    assert other_sigma == sigma
    assert any(item["reason"] == "INCOHERENT_SNAPSHOT" for item in audit["rejected"])


def test_unqualified_price_basis_and_insufficient_rights_are_typed_failures(context):
    snapshot, expiry, parameters, pack = context
    for missing in (None, SimpleNamespace(validation={})):
        with pytest.raises(RiskError, match="qualified packed pricing"):
            local_atm_path_iv(snapshot, expiry, parameters, missing, 1e-5)
    with pytest.raises(RiskError, match="four valid contracts"):
        local_atm_path_iv(replace(snapshot, quotes=tuple(entry for entry in snapshot.quotes if entry.contract.right == "P")), expiry, parameters, pack, 1e-5)


def test_atm_iv_dispersion_is_rejected(context):
    snapshot, expiry, parameters, pack = context
    entries = tuple(replace(entry, latest_event=replace(entry.latest_event, iv=.1 if index%2 else .9)) for index, entry in enumerate(snapshot.quotes))
    with pytest.raises(RiskError, match="dispersion exceeds"):
        local_atm_path_iv(replace(snapshot, quotes=entries), expiry, parameters, pack, 1e-5)


def test_risk_adapter_missing_path_uses_fallback_and_explicit_value_has_priority(context):
    from qqq_sim.broker.models import OrderLeg
    from qqq_sim.replay.calendar import CalendarService
    from qqq_sim.services.adapters import risk_job, broker_contract
    snapshot, expiry, parameters, pack = context
    calendar = CalendarService()
    by_strike = {entry.contract.strike: entry.contract for entry in snapshot.quotes if entry.contract.right == "P"}
    draft = SimpleNamespace(legs=(OrderLeg(broker_contract(by_strike[Decimal(590)], calendar), 1), OrderLeg(broker_contract(by_strike[Decimal(595)], calendar), -1)),
                            target_fraction=Decimal(".5"), deadline=datetime(2026, 11, 9, 15, 30, tzinfo=NY))
    parameters = {**parameters, "known_at": "2026-11-09T09:00:00-05:00"}
    estimated = risk_job(snapshot, draft, parameters, calendar, pack=pack)
    assert estimated["sigma_path"] == .25
    assert estimated["parameter_provenance"][-1]["source"] == "CAUSAL_VALIDATED_LOCAL_ATM"
    explicit = risk_job(snapshot, draft, {**parameters, "sigma_path": .33}, calendar)
    assert explicit["sigma_path"] == .33
    with pytest.raises(RiskError) as error:
        risk_job(snapshot, draft, parameters, calendar)
    assert error.value.code == "MISSING_PATH_IV"
