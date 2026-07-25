"""
Simulation-based and Property-Based Testing Infrastructure (Phase 2)

This package validates uSTAT's statistical methods at an intermediate-to-advanced
level.

Contents:
- `data_generators.py` -> synthetic data from known parameters
- `test_property_based.py` -> property-based tests via Hypothesis
- `test_*_simulation.py` -> classic simulation-recovery tests
- `conftest.py` -> Hypothesis strategies and settings

Usage:
    PYTHONPATH=. python -m pytest tests/simulation/ -q -m "simulation or not simulation"

Planned:
- More models (survival, PSM/IPTW, GEE)
- Automated "method vs ground truth" reports
- Behaviour tests under assumption violations
"""