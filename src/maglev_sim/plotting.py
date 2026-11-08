"""Plots. Every function takes an optional matplotlib ``ax`` and returns the figure
so it works the same in notebooks and scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from scipy.spatial.transform import Rotation

from .handoff import system_weights


def _ax(ax, size=(6, 6)):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
    return ax.figure, ax


def plot_layout(system, pose=None, ax=None, show_weights=True, label_coils=False):
    """Top view: tiles, coils (shaded by handoff weight) and the mover."""
    fig, ax = _ax(ax)
    g = system.grid
    pose = system.pose() if pose is None else np.asarray(pose)
    w = system_weights(system, pose[:2]) if show_weights else np.zeros(g.n_coils)
    for t in range(g.n_tiles):
        r, c = g.tile_rc(t)
        ax.add_patch(Rectangle((c * g.tile_size[0] * 1e3, r * g.tile_size[1] * 1e3), g.tile_size[0] * 1e3,
                               g.tile_size[1] * 1e3, fill=False, lw=2, ec="0.3"))
        ctr = g.tile_center(t) * 1e3
        ax.text(ctr[0], (r + 1) * g.tile_size[1] * 1e3 - 4, f"tile {t}", ha="center", va="top", fontsize=8, color="0.3")
    o = system.coil.outline
    for i, (x, y) in enumerate(g.coil_xy * 1e3):
        face = plt.cm.Oranges(0.15 + 0.75 * w[i])
        if o["shape"] == "circle":
            ax.add_patch(Circle((x, y), o["r_out"] * 1e3, fc=face, ec="0.4", lw=0.8))
            ax.add_patch(Circle((x, y), o["r_in"] * 1e3, fc="white", ec="0.4", lw=0.5))
        else:
            (wo, lo), (wi, li) = o["outer"], o["inner"]
            ax.add_patch(Rectangle((x - wo * 500, y - lo * 500), wo * 1e3, lo * 1e3, fc=face, ec="0.4", lw=0.8))
            ax.add_patch(Rectangle((x - wi * 500, y - li * 500), wi * 1e3, li * 1e3, fc="white", ec="0.4", lw=0.5))
        if label_coils:
            ax.text(x, y, str(i), ha="center", va="center", fontsize=6)
    _draw_mover(ax, system.mover, pose)
    ex, ey = g.extent
    ax.set_xlim(-5, ex * 1e3 + 5)
    ax.set_ylim(-5, ey * 1e3 + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"{system.cfg['name']}: coils (orange = active weight), mover (red N-up / blue S-up)")
    return fig


def _draw_mover(ax, mover, pose):
    rot = Rotation.from_euler("z", pose[5])
    for m in mover.magnets:
        c = (pose[:3] + rot.apply(m.center))[:2] * 1e3
        p = m.polarization
        col = "tab:red" if p[2] > 1e-6 else ("tab:blue" if p[2] < -1e-6 else "0.6")
        if m.shape == "cylinder":
            ax.add_patch(Circle(c, m.dimension[0] * 500, fc=col, ec="k", alpha=0.6, lw=0.8))
        else:
            a, b = m.dimension[0] * 1e3, m.dimension[1] * 1e3
            ax.add_patch(Rectangle(c - [a / 2, b / 2], a, b, fc=col, ec="k", alpha=0.6, lw=0.8))
        if abs(p[2]) < 1e-6:  # horizontal magnetisation: arrow
            d = rot.apply(p)[:2]
            d = d / np.linalg.norm(d) * m.dimension[0] * 350
            ax.arrow(c[0] - d[0] / 2, c[1] - d[1] / 2, d[0], d[1], head_width=1.5, color="k", length_includes_head=True)


def plot_map(df, column, ax=None, title=None, cmap="viridis"):
    """Colour map of one column of a design report's ``samples`` table."""
    fig, ax = _ax(ax, (6, 5))
    xs, ys = np.unique(df.x_mm), np.unique(df.y_mm)
    Z = df.pivot_table(index="y_mm", columns="x_mm", values=column).reindex(index=ys, columns=xs).values
    im = ax.pcolormesh(xs, ys, Z.astype(float), shading="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, label=column)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title or column)
    return fig


def plot_report(report, system=None):
    """Four-panel summary of a DesignReport."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    df = report.samples
    plot_map(df, "lift_margin", axs[0, 0], "Lift margin (max lift / weight)")
    plot_map(df, "hover_power_W", axs[0, 1], "Hover power [W]", "magma")
    plot_map(df.assign(condition=np.minimum(df.condition, 200)), "condition", axs[1, 0], "Condition number (capped 200)", "cividis")
    c = report.lift_curve
    a = axs[1, 1]
    a.plot(c.gap_mm, c.lift_margin, "o-", label="lift margin")
    a.axhline(1, color="r", ls="--", lw=1, label="= weight")
    a.set_xlabel("gap [mm]")
    a.set_ylabel("max lift / weight")
    a2 = a.twinx()
    a2.plot(c.gap_mm, c.hover_power_W, "s--", color="tab:orange", label="hover power")
    a2.set_ylabel("hover power at centre [W]", color="tab:orange")
    a.set_title("Lift vs gap (table centre)")
    a.legend(loc="upper right")
    fig.suptitle(report.summary["name"])
    fig.tight_layout()
    return fig


def plot_sweep(df, x, metrics=None):
    metrics = metrics or ["lift_margin_worst", "hover_power_W_worst", "hover_coil_temp_C_worst", "condition_worst"]
    fig, axs = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5))
    for a, m in zip(np.atleast_1d(axs), metrics):
        a.plot(df[x], df[m], "o-")
        a.set_xlabel(x)
        a.set_title(m)
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tolerance(tol_report):
    """Left: impact of each tolerance at its assumed value. Right: impact vs tolerance size."""
    sens, curves = tol_report.sensitivity, tol_report.curves
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    a = axs[0]
    s = sens.iloc[::-1]
    colors = ["tab:red" if v > 1 else "tab:green" for v in s.worst_vs_threshold]
    a.barh(s.tolerance + "  " + s.assumed, s.worst_vs_threshold, color=colors)
    a.axvline(1, color="k", ls="--", lw=1)
    a.set_xlabel("worst impact / threshold   (>1 = breaks a requirement)")
    a.set_title("Which tolerances matter most")
    a = axs[1]
    for name, c in curves.groupby("tolerance"):
        a.plot(c.value / (c.value.max() / 4 if c.value.max() else 1), c.worst_vs_threshold, "o-", label=name, ms=3)
    a.axhline(1, color="k", ls="--", lw=1)
    a.set_xlabel("tolerance as a multiple of the assumed value")
    a.set_ylabel("worst impact / threshold")
    a.set_yscale("symlog", linthresh=0.1)
    a.set_ylim(bottom=0)
    a.legend(fontsize=7, ncol=2)
    a.set_title("Impact vs tolerance size")
    fig.tight_layout()
    return fig


def plot_simulation(result):
    """Position tracking, coil currents, power and owner tile over time."""
    t = result.t
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for k, lab in enumerate("xyz"):
        axs[0, 0].plot(t, result.pos[:, k] * 1e3, label=f"{lab} actual")
        axs[0, 0].plot(t, result.ref[:, k] * 1e3, "--", lw=1, label=f"{lab} target")
    axs[0, 0].set_ylabel("position [mm]")
    axs[0, 0].legend(fontsize=7, ncol=3)
    err = np.linalg.norm(result.pos - result.ref, axis=1) * 1e3
    axs[0, 1].plot(t, err)
    axs[0, 1].set_ylabel("tracking error [mm]")
    axs[1, 0].plot(t, result.currents)
    axs[1, 0].set_ylabel("coil currents [A]")
    axs[1, 0].set_xlabel("time [s]")
    axs[1, 1].plot(t, result.power, label="power [W]")
    axs[1, 1].set_xlabel("time [s]")
    axs[1, 1].set_ylabel("total coil power [W]")
    if result.owner is not None:
        a2 = axs[1, 1].twinx()
        a2.step(t, result.owner, color="tab:green", where="post")
        a2.set_ylabel("owner tile", color="tab:green")
    fig.tight_layout()
    return fig


def plot_handoff(df, system=None):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    cols = [c for c in df.columns if c.startswith("I") and c[1:].isdigit()]
    active = [c for c in cols if df[c].abs().max() > 1e-3]
    axs[0].plot(df.x_mm, df[active])
    axs[0].set_ylabel("coil currents [A]")
    axs[1].plot(df.x_mm, df.power_W)
    axs[1].set_ylabel("power [W]")
    axs[2].plot(df.x_mm, df.current_step_A)
    axs[2].set_ylabel("current jump between steps [A]")
    axs[2].set_xlabel("x [mm]")
    if system is not None:
        for c in range(1, system.grid.cols):
            for a in axs:
                a.axvline(c * system.grid.tile_size[0] * 1e3, color="k", ls=":", lw=1)
    axs[0].set_title("Quasi-static handoff along a line (dotted = tile edges)")
    fig.tight_layout()
    return fig
