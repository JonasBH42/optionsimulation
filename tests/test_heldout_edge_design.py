from datetime import date,timedelta
from decimal import Decimal
import pytest

from qqq_sim.broker.service import Broker
from qqq_sim.risk.contracts import RiskError,stable_hash
from qqq_sim.risk.offline.design import ANCHORS,materialize_case
from qqq_sim.risk.offline.edge_design import FAMILIES,RESERVED_DATES,heldout_design_records
from qqq_sim.risk.payoff import payoff_extrema


def test_heldout_groups_and_actual_expiry_universes_are_separate():
    records=heldout_design_records(32768)
    assert len(records)==1000
    assert [record['index'] for record in records]==list(range(32768,33768))
    assert {record['heldout_edge_family'] for record in records}==set(FAMILIES)
    assert all(record['split']=='validation' and not record['training_eligible'] and not record['calibration_eligible']
               and not record['augmentation_seed_eligible'] for record in records)
    for expiry in map(date.fromisoformat,RESERVED_DATES):
        assert all(not date.fromisoformat(anchor)<=expiry<=date.fromisoformat(anchor)+timedelta(days=30) for anchor in ANCHORS)
    assert records==heldout_design_records(32768)


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('local_index',[0,99,199])
def test_actual_heldout_ledger_and_intended_property_are_verified(family,local_index):
    records=heldout_design_records(32768)
    record=records[FAMILIES.index(family)*200+local_index]
    case=materialize_case(record)
    audit=case['record']['edge_validation']
    assert audit['verified'] and audit['original_target_retained']
    assert case['scope']=='MANDATORY_UNTOUCHED_EDGE_DESIGN'
    broker=Broker.from_dict(case['account_context']['broker_state'])
    campaign=broker.ledger.campaigns[case['campaign']['campaign_id']]
    assert broker.ledger.external_contributions==0
    assert campaign.target==case['job'].target_usd==campaign.original_max_profit/2
    assert campaign.trade_cash==case['job'].entry_cash_usd
    assert campaign.net_cash==Decimal(case['campaign']['net_cash_usd'])
    if family in ('NARROW','TANGENT'):
        assert len(broker.orders)==2
        assert all(order.status=='FILLED' for order in broker.orders.values())
        difference=payoff_extrema(case['job'].signed_legs,case['job'].entry_cash_usd)['max_profit']-campaign.target
        assert difference==Decimal(1 if family=='NARROW' else 0)
    if family=='LOW_BUDGET':
        assert audit['operational_bound']['optional_expense_reserve']=='0.01'
        assert case['campaign']['remaining_actions']==1


def test_heldout_family_cannot_be_reclassified_for_training():
    record=heldout_design_records(32768,1)[0]
    record['split']='train'
    with pytest.raises(RiskError) as error:
        materialize_case(record)
    assert error.value.code=='HELDOUT_DESIGN_PARTITION_VIOLATION'
