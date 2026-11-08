"""maglev-sim: simulation backbone for the 2D MagLev Tile project.

Quick start (in a notebook):

    from maglev_sim import load_config, build_system, evaluate_design
    system = build_system(load_config("configs/baseline.yaml"))
    print(evaluate_design(system).text())
"""

from . import coils, layouts, magnets  # noqa: F401  (registers the built-in types)
from .config import get_path, load_config, make_config, with_changes  # noqa: F401
from .control import PDController, Trajectory  # noqa: F401
from .dynamics import simulate  # noqa: F401
from .metrics import evaluate_design, lift_vs_gap  # noqa: F401
from .registry import register_coil, register_layout, register_mover  # noqa: F401
from .system import AXES, System, build_system  # noqa: F401

__version__ = "0.1.0"
