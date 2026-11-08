"""Closed-loop simulation of the mover (the start of the digital twin).

Each control step (``control.rate_Hz``):
  1. sensor reads the position (+ noise + optional bias), velocity is a filtered difference
  2. controller -> wanted wrench (PD + gravity)
  3. allocation with handoff weights -> coil current commands (clipped to the limit)
  4. tile network decides the owner and delays neighbour commands
  5. physics runs ``simulation.substeps`` small steps: coil drivers chase their
     commands (limited by inductance and supply voltage), force = K_true @ I + gravity

``plant`` lets the *real* table differ from the model the controller uses
(e.g. a random as-built table from tolerance.as_built) - the most direct way to
see whether tolerances actually break levitation.

v1 limits: the mover translates only (x, y, z); torques are commanded to zero
and logged as ``residual_torque`` but do not rotate the mover yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocation import allocate
from .control import PDController, Trajectory
from .handoff import system_weights
from .materials import G, mm
from .tiles import CentralController, TileNetwork


@dataclass
class SimResult:
    t: np.ndarray
    pos: np.ndarray  # true position (N, 3)
    ref: np.ndarray  # target position (N, 3)
    currents: np.ndarray  # applied coil currents (N, C)
    power: np.ndarray  # total I^2 R (N,)
    owner: np.ndarray | None  # owner tile (N,)
    residual_torque: np.ndarray  # (N, 2) roll/pitch torque actually produced [N*m]
    info: dict = field(default_factory=dict)

    def summary(self) -> str:
        i = self.info
        return (
            f"tracking error: rms {i['rms_error_mm']:.3f} mm, max {i['max_error_mm']:.3f} mm | "
            f"peak coil current {i['peak_current_A']:.2f} A | mean power {i['mean_power_W']:.1f} W | "
            f"saturated steps {i['saturated_steps']} | handoffs {i['handoffs']} | "
            f"link messages {i['messages']} | touched down: {i['touchdown']}"
        )


def simulate(
    system,
    trajectory: Trajectory | None = None,
    duration: float | None = None,
    plant=None,
    plant_pose_error=None,
    controller=None,
    decentralized: bool = True,
    sensor_bias=None,
    fast: bool = True,
) -> SimResult:
    """Run the closed loop. ``fast=True`` uses the FastForceModel (field grid,
    a few seconds to build); ``fast=False`` uses the exact calculation every step
    (slow, for checking)."""
    cfg = system.cfg
    sim, ctl = cfg["simulation"], cfg["control"]
    plant = plant or system
    perr = np.zeros(6) if plant_pose_error is None else np.asarray(plant_pose_error, float)
    bias = np.zeros(3) if sensor_bias is None else np.asarray(sensor_bias, float)
    trajectory = trajectory or Trajectory.hold(system.pose()[:3])
    duration = float(duration or sim["duration_s"])
    controller = controller or PDController(system)
    net = TileNetwork(system) if decentralized else CentralController(system)
    rng = np.random.default_rng(sim["seed"])

    dt = 1.0 / ctl["rate_Hz"]
    nsub = int(sim["substeps"])
    h = dt / nsub
    noise = mm(sim["sensor_noise_mm"])
    alpha = dt / (dt + 1 / (2 * np.pi * ctl["velocity_filter_Hz"]))
    tau_d = sim["driver_time_constant_ms"] * 1e-3
    R, L, Vs = plant.coil.resistance, plant.coil.inductance, cfg["electrical"]["supply_voltage_V"]
    imax, axes, Lm = system.max_current, system.control_axes, system.mover.size
    m = plant.mover.mass

    if fast:
        from .forcemap import FastForceModel

        model = FastForceModel(system)
        plant_model = model if plant.mover is system.mover else FastForceModel(plant)
        model_K = model.wrench_matrix
        plant_K = lambda pose: plant_model.wrench_matrix(pose, plant=plant)  # noqa: E731
    else:
        model_K, plant_K = system.wrench_matrix, plant.wrench_matrix

    p, _, _ = trajectory(0.0)
    p = p.copy()
    v = np.zeros(3)
    K0 = model_K(np.r_[p, 0, 0, 0])
    I = allocate(K0, system.hover_wrench(), axes, system_weights(system, p), imax, Lm).currents
    p_meas_prev, v_est = p + bias, np.zeros(3)

    n = int(round(duration / dt))
    T, P, REF, CUR, POW, OWN, TRQ = [], [], [], [], [], [], []
    sat_steps, touchdown = 0, False
    for k in range(n):
        t = k * dt
        # 1. sensing
        p_meas = p + bias + rng.normal(0, noise, 3)
        v_est = (1 - alpha) * v_est + alpha * (p_meas - p_meas_prev) / dt
        p_meas_prev = p_meas
        # 2. control
        p_ref, v_ref, a_ref = trajectory(t)
        wanted = controller.wrench(p_meas, v_est, p_ref, v_ref, a_ref)
        # 3. allocation on the controller's model
        K_model = model_K(np.r_[p_meas, 0, 0, 0])
        a = allocate(K_model, wanted, axes, system_weights(system, p_meas), imax, Lm)
        sat_steps += a.saturated
        # 4. tiles
        net.update_owner(p_meas[:2])
        I_cmd = net.apply(np.clip(a.currents, -imax, imax))
        # 5. physics on the real plant
        K_true = plant_K(np.r_[p, 0, 0, 0] + perr)
        for _ in range(nsub):
            dI = (I_cmd - I) * (1 - np.exp(-h / tau_d))
            V = np.clip(R * I + L * dI / h, -Vs, Vs)
            I = I + (V - R * I) / L * h
            F = K_true[:3] @ I + np.array([0, 0, -m * G])
            v = v + F / m * h
            p = p + v * h
            if p[2] <= 0:
                p[2], v[2], touchdown = 0.0, max(v[2], 0.0), True
        T.append(t + dt)
        P.append(p.copy())
        REF.append(p_ref)
        CUR.append(I.copy())
        POW.append(float(np.sum(I**2) * R))
        OWN.append(net.owner)
        TRQ.append(K_true[3:5] @ I)

    pos, ref = np.array(P), np.array(REF)
    err = np.linalg.norm(pos - ref, axis=1) * 1e3
    cur = np.array(CUR)
    info = {
        "rms_error_mm": float(np.sqrt(np.mean(err**2))),
        "max_error_mm": float(err.max()),
        "final_error_mm": float(err[-1]),
        "peak_current_A": float(np.abs(cur).max()),
        "mean_power_W": float(np.mean(POW)),
        "saturated_steps": int(sat_steps),
        "handoffs": net.handoffs,
        "messages": net.messages,
        "touchdown": touchdown,
        "max_residual_torque_mNm": float(np.abs(np.array(TRQ)).max() * 1e3),
        "handoff_log": net.log,
    }
    return SimResult(np.array(T), pos, ref, cur, np.array(POW), np.array(OWN), np.array(TRQ), info)
