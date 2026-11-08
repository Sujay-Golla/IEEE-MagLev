"""Physics sanity checks. Run with:  python -m pytest"""

import magpylib as magpy
import numpy as np
import pytest

from maglev_sim import build_system, make_config
from maglev_sim.allocation import allocate, condition_number, max_wrench
from maglev_sim.forcemap import FastForceModel
from maglev_sim.magnets import build_mover

ONE_COIL = {
    "mover": {"type": "disk", "params": {"diameter_mm": 20, "height_mm": 10}},
    "coil": {
        "type": "round",
        "params": {"inner_diameter_mm": 10, "outer_diameter_mm": 30, "height_mm": 12, "turns": 100, "awg": 24},
        "mesh": {"radial": 1, "axial": 1, "segments": 64},
    },
    "tile": {"layout": "square", "params": {"n": 1, "pitch_mm": 35}},
}

SMALL = {
    "mover": {"type": "cube_grid", "params": {"nx": 2, "ny": 2, "size_mm": 8, "pitch_mm": 16}},
    "coil": {"type": "round", "params": {"inner_diameter_mm": 5, "outer_diameter_mm": 24, "height_mm": 12,
                                          "awg": 26, "turns": "auto"}},
    "tile": {"layout": "square", "params": {"n": 3, "pitch_mm": 25}},
    "grid": {"rows": 1, "cols": 2},
    "operating": {"gap_mm": 3},
    "analysis": {"cutoff_mm": 50},
}


@pytest.fixture(scope="module")
def one_coil():
    return build_system(make_config(ONE_COIL))


@pytest.fixture(scope="module")
def small():
    return build_system(make_config(SMALL))


@pytest.mark.parametrize("dx", [0.0, 0.005, 0.012])
def test_force_matches_magpylib_getFT(one_coil, dx):
    """Our Lorentz sum vs Magpylib's independent force solver (Newton's 3rd law)."""
    s = one_coil
    pose = s.pose(x=0.0175 + dx, z=0.005)
    K = s.wrench_matrix(pose)[:, 0]
    loop = magpy.current.Circle(current=100, diameter=0.02, position=(0.0175, 0.0175, -0.006))
    F, T = magpy.getFT(loop, s.mover.sources(pose)[0], meshing=(24, 8, 8), pivot=s.mover.com_world(pose))
    assert np.allclose(K[:3], F, rtol=0.01, atol=2e-4)
    assert np.allclose(K[3:], T, rtol=0.02, atol=5e-6)


def test_force_linear_in_current(small):
    pose = small.pose(x=0.06, y=0.04)
    K = small.wrench_matrix(pose)
    I = np.random.default_rng(0).uniform(-1, 1, small.n_coils)
    assert np.allclose(small.wrench(pose, 2 * I), 2 * small.wrench(pose, I))
    assert np.allclose(small.wrench(pose, I), K @ I)


def test_symmetric_pose_has_no_lateral_force(one_coil):
    """Disk centred over a coil: pure vertical force, no torque."""
    K = one_coil.wrench_matrix(one_coil.pose(z=0.004))[:, 0]
    assert abs(K[2]) > 0.05
    assert np.allclose(K[[0, 1, 3, 4, 5]], 0, atol=1e-9)


def test_force_falls_with_gap(one_coil):
    f = [abs(one_coil.wrench_matrix(one_coil.pose(x=0.0175 + 0.008, z=z))[2, 0]) for z in (0.004, 0.008, 0.016)]
    assert f[0] > f[1] > f[2]


def test_batched_equals_single(small):
    poses = np.array([small.pose(x=x, y=0.04) for x in (0.03, 0.05, 0.07)])
    Kb = small.wrench_matrices(poses)
    for p, K in zip(poses, Kb):
        assert np.allclose(K, small.wrench_matrix(p))


def test_fast_model_matches_exact(small):
    fm = FastForceModel(small)
    rng = np.random.default_rng(3)
    for _ in range(3):
        pose = small.pose(x=0.03 + 0.09 * rng.random(), y=0.025 + 0.025 * rng.random(), z=0.0028, yaw=0.1)
        Ke, Kf = small.wrench_matrix(pose), fm.wrench_matrix(pose)
        assert np.linalg.norm(Kf - Ke) / np.linalg.norm(Ke) < 0.02


def test_halbach_strong_side_faces_coils():
    m = build_mover({"type": "halbach_2d", "params": {"n": 5, "size_mm": 6}})
    assert m.info["bottom_to_top_field_ratio"] > 1.5
    m1 = build_mover({"type": "halbach_1d", "params": {"n": 5, "size_mm": 6}})
    assert m1.info["bottom_to_top_field_ratio"] > 1.5


def test_allocation_hits_target(small):
    K = small.wrench_matrix(small.pose(x=0.06, y=0.04))
    want = small.hover_wrench()
    exact = allocate(K, want, small.control_axes, i_max=None, length=small.mover.size, damping=0)
    assert exact.error < 1e-6 * small.weight
    damped = allocate(K, want, small.control_axes, i_max=None, length=small.mover.size)
    assert damped.error < 0.005 * small.weight  # default damping costs < 0.5 % of weight
    assert not damped.saturated


def test_allocation_respects_limit(small):
    K = small.wrench_matrix(small.pose(x=0.06, y=0.04))
    want = small.hover_wrench() * 50  # impossible
    a = allocate(K, want, small.control_axes, i_max=1.0, length=small.mover.size)
    assert a.max_abs_current <= 1.0 + 1e-9
    assert a.saturated


def test_max_lift_consistent_with_allocation(small):
    K = small.wrench_matrix(small.pose(x=0.06, y=0.04))
    axes, imax = small.control_axes, small.max_current
    fz = max_wrench(K, 2, axes, np.zeros(6), imax)
    want = np.zeros(6)
    want[2] = 0.95 * fz
    a = allocate(K, want, axes, i_max=imax, length=small.mover.size)
    assert not a.saturated
    want[2] = 1.1 * fz
    assert allocate(K, want, axes, i_max=imax, length=small.mover.size).saturated


def test_condition_number_finite(small):
    K = small.wrench_matrix(small.pose(x=0.06, y=0.04))
    assert np.isfinite(condition_number(K, small.control_axes, length=small.mover.size))
