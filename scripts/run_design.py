"""Design report for one config.

    python scripts/run_design.py configs/baseline.yaml
    python scripts/run_design.py configs/baseline.yaml --tolerances
"""
import sys

from maglev_sim.cli import design_main

sys.exit(design_main())
