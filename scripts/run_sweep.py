"""Compare variations of a design.

    python scripts/run_sweep.py configs/baseline.yaml --vary coil.params.awg=24,26,28
    python scripts/run_sweep.py configs/baseline.yaml --vary operating.gap_mm=2:6:1 --vary tile.params.pitch_mm=22,25
"""
import sys

from maglev_sim.cli import sweep_main

sys.exit(sweep_main())
