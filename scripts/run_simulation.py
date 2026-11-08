"""Closed-loop simulation (hover, or move across tiles).

    python scripts/run_simulation.py configs/baseline.yaml --to 150,50
"""
import sys

from maglev_sim.cli import simulate_main

sys.exit(simulate_main())
