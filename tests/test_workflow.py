"""Config handling, reports, tolerances, handoff and the closed loop (smoke tests)."""

from pathlib import Path

import numpy as np
import pytest

from maglev_sim import Trajectory, build_system, evaluate_design, load_config, make_config, simulate, with_changes
from maglev_sim.handoff import fade_weights, straight_line_sweep
from maglev_sim.sweep import parse_values, sweep
from maglev_sim.tolerance import TOLERANCES, as_built, required_tolerances, run_trials

ROOT = Path(__file__).resolve().parents[1]
QUICK = ROOT / "configs" / "examples" / "single_tile_quick.yaml"


@pytest.fixture(scope="module")
def quick():
    return build_system(load_config(QUICK))


@pytest.mark.parametrize("path", sorted((ROOT / "configs").rglob("*.yaml")))
def test_all_configs_load_and_build(path):
    s = build_system(load_config(path))
    assert s.n_coils > 0 and s.mover.mass > 0


def test_typo_is_caught():
    with pytest.raises(ValueError, match="did you mean 'operating'"):
        make_config({"operatng": {"gap_mm": 3}})


def test_unknown_type_is_caught():
    with pytest.raises(KeyError, match="Did you mean 'halbach_2d'"):
        build_system(make_config({"mover": {"type": "halbach2d"}}))


def test_with_changes_does_not_modify_original(quick):
    cfg2 = with_changes(quick.cfg, {"operating.gap_mm": 9})
    assert cfg2["operating"]["gap_mm"] == 9 and quick.cfg["operating"]["gap_mm"] == 3


def test_parse_values():
    assert parse_values("2:6:2") == [2, 4, 6]
    assert parse_values("square,offset") == ["square", "offset"]
    assert parse_values("0.5,1") == [0.5, 1]


def test_design_report_quick(quick):
    rep = evaluate_design(quick, per_tile=3, gaps_mm=[3, 6])
    assert rep.summary["hover_ok_fraction"] > 0.5
    assert rep.summary["lift_margin_worst"] > 1
    assert "Verdicts" in rep.text()


def test_sweep_runs(quick):
    df = sweep(quick.cfg, {"operating.gap_mm": [3, 6]}, per_tile=2, progress=False)
    assert len(df) == 2
    assert df.lift_margin_worst.iloc[0] > df.lift_margin_worst.iloc[1]  # bigger gap -> less lift


def test_fade_weights_smooth():
    xy = np.array([[0.0, 0], [0.02, 0], [0.04, 0], [0.06, 0]])
    w = fade_weights(xy, [0, 0], 0.01, 0.05)
    assert w[0] == 1 and w[-1] == 0 and 0 < w[2] < 1


def test_handoff_sweep_is_continuous():
    s = build_system(load_config(ROOT / "configs" / "baseline.yaml"))
    # 2.5 mm steps; the coil pitch is 25 mm = 10 steps. The tile edge is at x = 100 mm.
    df = straight_line_sweep(s, [0.07, 0.05], [0.13, 0.05], n=25)
    assert not df.saturated.any()
    # A seamless tile edge means the pattern simply repeats every coil pitch:
    # crossing x = 100 mm looks exactly like crossing x = 75 mm inside a tile.
    p = df.power_W.values
    assert np.allclose(p[:-10], p[10:], rtol=1e-6)


def test_as_built_zero_tolerance_is_identity(quick):
    plant, perr = as_built(quick, {}, np.random.default_rng(0))
    p = quick.pose()
    assert np.allclose(plant.wrench_matrix(p), quick.wrench_matrix(p))
    assert not perr.any()


def test_bigger_tolerance_bigger_impact(quick):
    small = run_trials(quick, {"coil_position_mm": 0.1}, trials=4)
    large = run_trials(quick, {"coil_position_mm": 1.0}, trials=4)
    assert large.force_error_pct.mean() > small.force_error_pct.mean()


def test_required_tolerances_shape(quick):
    cfg = with_changes(quick.cfg, {"tolerances.values": {k: 0 for k in TOLERANCES} | {"coil_turns_pct": 2}})
    s = build_system(cfg)
    req, curves = required_tolerances(s, factors=(1, 4), trials=3)
    assert set(req.tolerance) == set(TOLERANCES)
    assert (req.max_allowed >= 0).all()


def test_closed_loop_hover_and_move(quick):
    p0 = np.r_[quick.grid.tile_center(0) - [0.01, 0], quick.gap]
    p1 = p0 + [0.02, 0, 0]
    res = simulate(quick, Trajectory.move(p0, p1, 0.05, 0.2), duration=0.4)
    assert not res.info["touchdown"]
    assert res.info["max_error_mm"] < 1.0
    assert res.info["final_error_mm"] < 0.2
