import math
from pathlib import Path
import numpy as np
import pytest

from qqq_sim.risk.contracts import RiskError
from qqq_sim.risk.offline import pde_grid_cache as module


class ManufacturedGrid:
    calls = 0
    temporal_bias = 1e-4
    spatial_bias = 1e-4

    def __init__(self, right, strike, sigma, rate, carry, domain, nodes, eps, scale, deadline):
        type(self).calls += 1
        self.spots = np.exp(np.linspace(math.log(domain[0]), math.log(domain[1]), nodes))
        self.lower = np.zeros(nodes)
        self.upper = np.zeros(nodes)
        self.scale, self.tau, self.steps = scale, 0., 0
        self.residual = 0.
        self.max_iterations = 1

    def _step(self, dt):
        self.tau += dt
        self.steps += 1
        # Linear-in-spot extraction is exact; the controlled mesh errors are
        # independently visible across each space/time refinement family.
        self.lower = self.spots+self.tau+self.spatial_bias/len(self.spots)**2+self.temporal_bias/self.scale**2
        self.upper = self.lower.copy()


@pytest.fixture
def fake_grid(monkeypatch):
    ManufacturedGrid.calls = 0
    ManufacturedGrid.spatial_bias = 1e-4
    ManufacturedGrid.temporal_bias = 1e-4
    monkeypatch.setattr(module, "AmericanGrid", ManufacturedGrid)
    monkeypatch.setattr(module, "normalized_step_schedule", lambda scale, rate: (i/(8*scale) for i in range(1, 8*scale+1)))


def cache(tmp_path, **kwargs):
    return module.NormalizedGridCache(tmp_path, epsilon=.001, kernel_hash="test", initial_nodes=5, max_nodes=65, **kwargs)


def test_one_converged_law_grid_supplies_all_x_and_survives_reload(tmp_path, fake_grid):
    first = cache(tmp_path)
    low = first.value("PUT", (-.1, .2, .03, .01), (-.2, .2), max_seconds=10)
    assert low.status == "CONVERGED"
    assert ManufacturedGrid.calls == 9
    high = first.value("PUT", (.1, .2, .03, .01), (-.2, .2), max_seconds=0)
    assert ManufacturedGrid.calls == 9
    assert high.price-low.price == pytest.approx(math.exp(.1)-math.exp(-.1))
    again = cache(tmp_path).value("PUT", (0., .2, .03, .01), (-.2, .2), max_seconds=0)
    assert ManufacturedGrid.calls == 9
    assert again.price == pytest.approx(2+1e-4/17**2+1e-4/4**2)
    assert len(low.diagnostics["refinements"][-1]["space_differences"]) == 2
    assert len(low.diagnostics["refinements"][-1]["time_differences"]) == 2
    assert low.diagnostics["requested_x"] == [-.2, .2]


def test_independent_time_refinement_cannot_be_replaced_by_space(tmp_path, fake_grid):
    ManufacturedGrid.temporal_bias = 10
    item = cache(tmp_path, max_time_scale=4)
    with pytest.raises(RiskError) as caught:
        item.value("PUT", (0., .2, .03, .01), (-.1, .1), max_seconds=10)
    assert caught.value.code == "PRICER_NOT_CONVERGED"
    assert not list(tmp_path.rglob("accepted.json"))


def test_grid_artifact_tamper_is_rejected_even_after_memory_hit(tmp_path, fake_grid):
    item = cache(tmp_path)
    item.value("PUT", (0., .2, .03, .01), (-.1, .1), max_seconds=10)
    receipt = next(tmp_path.rglob("accepted.json"))
    import json
    numeric = receipt.parent/json.loads(receipt.read_text())["numeric"]["file"]
    with numeric.open("r+b") as stream:
        stream.seek(30)
        stream.write(b"corrupt")
    with pytest.raises(RiskError, match="CHECKPOINT_MISMATCH"):
        item.value("PUT", (0., .2, .03, .01), (-.1, .1), max_seconds=10)


def test_partial_grid_restores_original_partition_and_exact_values(tmp_path, fake_grid, monkeypatch):
    item = cache(tmp_path/"resumed")
    folder = item.directory/"case"
    folder.mkdir()
    law = {"right":"PUT", "w":.2, "a":.03, "b":.01}
    clock = iter([0., 0., 0., 2.])
    with monkeypatch.context() as patch:
        patch.setattr(module.time, "monotonic", lambda: next(clock))
        with pytest.raises(RiskError) as caught:
            item._solve_grid(folder, law, (.5, 2.), 17, 1, 1.)
        assert caught.value.code == "RESOURCE_LIMIT"
    import json
    saved = json.loads(next(folder.glob("*.json")).read_text())
    assert saved["completed_schedule_steps"] == 3
    assert saved["tau"] == 3/8
    resumed, record = item._solve_grid(folder, law, (.5, 2.), 17, 1, math.inf)
    fresh = cache(tmp_path/"fresh")
    fresh_folder = fresh.directory/"case"
    fresh_folder.mkdir()
    expected, full = fresh._solve_grid(fresh_folder, law, (.5, 2.), 17, 1, math.inf)
    assert item.resumed_steps == 3
    assert record["steps"] == full["steps"] == 8
    for name in expected:
        assert np.array_equal(resumed[name], expected[name])


def test_memory_guard_runs_before_grid_allocation(tmp_path, fake_grid):
    item = cache(tmp_path, memory_bytes=100)
    with pytest.raises(RiskError) as caught:
        item.value("PUT", (0., .2, .03, .01), (-.1, .1), max_seconds=10)
    assert caught.value.code == "RESOURCE_LIMIT"
    assert caught.value.details["resource"] == "normalized_pde_memory"
    assert ManufacturedGrid.calls == 0


def test_same_law_concurrent_writer_refuses_and_os_lock_releases(tmp_path):
    with module._law_lock(tmp_path):
        with pytest.raises(RiskError) as caught:
            with module._law_lock(tmp_path):
                pass
        assert caught.value.details["resource"] == "normalized_pde_cache_busy"
    with module._law_lock(tmp_path):
        pass


def test_schedule_matches_actual_solver_initial_time_partition(monkeypatch):
    from qqq_sim.risk.offline.pricing_oracle import AmericanGrid
    for scale in (1, 2, 4):
        grid = AmericanGrid("PUT", 1, .2, .03, .01, (.5, 2.), 5, 1e-5, scale)
        seen = []
        def step(dt):
            grid.tau += dt
            seen.append(grid.tau)
        monkeypatch.setattr(grid, "_step", step)
        grid.advance(1.)
        assert np.array_equal(seen, list(module.normalized_step_schedule(scale, .03)))


def test_oracle_cache_routes_general_law_to_shared_grid_and_exact_limits_to_formula(tmp_path, monkeypatch):
    from qqq_sim.risk.offline.price_basis import OraclePriceCache
    from qqq_sim.risk.offline.pricing_oracle import PriceResult
    item = OraclePriceCache(tmp_path)
    item.set_requested_domain([-.25, .25])
    calls = []
    def value(right, point, requested_x, **kwargs):
        calls.append((right, point, requested_x))
        return PriceResult(.25, 1e-12, "CONVERGED", {})
    monkeypatch.setattr(item.grid_cache, "value", value)
    assert item.value("PUT", [.1,.2,.03,.01]) == (.25,1e-12)
    assert calls[0][2] == (-.252,.252)
    exact, uncertainty = item.value("CALL", [.1,.2,.03,0.])
    assert exact > 0 and uncertainty == 0
    assert len(calls) == 1


def test_real_lcp_checkpoint_restarts_without_changing_a_time_step(tmp_path):
    import time
    item = module.NormalizedGridCache(tmp_path/"resumed", epsilon=.01, kernel_hash="real", initial_nodes=17, max_nodes=65)
    law = {"right":"PUT", "w":.2, "a":.03, "b":.01}
    folder = item.directory/"case"
    folder.mkdir()
    with pytest.raises(RiskError) as caught:
        item._solve_grid(folder, law, (.4, 2.5), 17, 1, time.monotonic()+.03)
    assert caught.value.code == "RESOURCE_LIMIT"
    resumed, record = item._solve_grid(folder, law, (.4, 2.5), 17, 1, time.monotonic()+60)
    fresh = module.NormalizedGridCache(tmp_path/"fresh", epsilon=.01, kernel_hash="real", initial_nodes=17, max_nodes=65)
    fresh_folder = fresh.directory/"case"
    fresh_folder.mkdir()
    expected, full = fresh._solve_grid(fresh_folder, law, (.4, 2.5), 17, 1, time.monotonic()+60)
    assert item.resumed_steps > 0
    assert record["steps"] == full["steps"]
    assert record["tau"] == full["tau"] == 1.
    for name in expected:
        assert np.array_equal(resumed[name], expected[name])
