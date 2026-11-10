from dataclasses import replace
from types import SimpleNamespace
from decimal import Decimal
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskError
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.universe_train import initialize_universe_arrays, make_trainable_encoder
from qqq_sim.risk.universe import universe_tokens, universe_features, hold_flag_indices,UNIVERSE_FEATURE_SIZE
from qqq_sim.risk.feature_layout import HOLD_FEATURE_COUNT,MANAGED_FEATURE_COUNT,ACTION_SCALAR_OFFSET,ACTION_FIELDS,UNIVERSE_OFFSET


def context():
    job = reference_fixture_job()
    universe = tuple(replace(leg, known_at_utc_us=job.t0_utc_us, bid=1., ask=1.1) for leg in job.signed_legs)
    return job, universe, SimpleNamespace(arrays=initialize_universe_arrays())


def test_fixed_token_order_padding_and_permutation_invariance():
    job, universe, pack = context()
    tokens = universe_tokens(job, universe)
    assert tokens.shape == (512, 16) and tokens.dtype == np.float64
    assert tokens[:2, 0].tolist() == [1, 1] and not tokens[2:].any()
    assert tokens[0, 11] == pytest.approx(.01) and tokens[0, 12] == pytest.approx(.011)
    assert tokens[0, 13] == 1 and tokens[0, 15] == .125
    assert universe_features(job, universe, pack).shape == (UNIVERSE_FEATURE_SIZE,)
    assert np.array_equal(universe_features(job, universe, pack), universe_features(job, universe[::-1], pack))


def test_different_future_opportunities_change_managed_encoding():
    job, universe, pack = context()
    expanded = (*universe, replace(universe[0], contract_id="EXTRA_C110", strike=Decimal(110)))
    assert not np.array_equal(universe_features(job, universe, pack), universe_features(job, expanded, pack))


def test_future_or_unknown_listing_never_leaks():
    job, universe, pack = context()
    with pytest.raises(RiskError, match="FUTURE_UNIVERSE_RECORD"):
        universe_tokens(job, [replace(universe[0], known_at_utc_us=job.t0_utc_us + 1)])
    with pytest.raises(RiskError, match="UNIVERSE_AVAILABILITY_UNKNOWN"):
        universe_tokens(job, [replace(universe[0], known_at_utc_us=None)])


def test_online_and_trainable_encoder_agree_and_gradients_flow():
    import torch
    job, universe, pack = context()
    encoder = make_trainable_encoder(pack.arrays)
    actual = encoder(torch.tensor(universe_tokens(job, universe)[None], dtype=torch.float64))
    assert np.allclose(actual.detach().numpy()[0], universe_features(job, universe, pack), atol=1e-12)
    actual.square().sum().backward()
    assert encoder.first.weight.grad is not None
    assert torch.count_nonzero(encoder.first.weight.grad) > 0
    assert np.array_equal(encoder.export_arrays()["universe_w0"], pack.arrays["universe_w0"])
    assert np.array_equal(universe_features(job, (), pack), np.zeros(UNIVERSE_FEATURE_SIZE))


def test_hold_flags_are_distinct_in_range_without_price_features():
    indexes = hold_flag_indices()
    assert len(indexes) == len(set(indexes)) and min(indexes) >= 0 and max(indexes) < HOLD_FEATURE_COUNT
    assert not any(108 <= index < 498 for index in indexes)


def test_action_features_include_exact_lag_and_opportunity_encoder():
    from qqq_sim.risk.contracts import YEAR_US
    from qqq_sim.risk.policy import action_features
    job, universe, pack = context()
    job = replace(job, signed_legs=tuple(replace(leg, exercise_style="EUROPEAN") for leg in job.signed_legs))
    action = {"family": "HOLD", "premium_cost_usd": "0", "fees_usd": "0", "eligible_utc_us": job.t0_utc_us+60_000_000, "legs": []}
    features = action_features(job, {}, action, pack, universe)
    assert features.shape == (MANAGED_FEATURE_COUNT,)
    assert features[ACTION_SCALAR_OFFSET+ACTION_FIELDS.index('eligible_years')] == pytest.approx(60_000_000/YEAR_US)
    assert np.array_equal(features[UNIVERSE_OFFSET:], universe_features(job, universe, pack))
