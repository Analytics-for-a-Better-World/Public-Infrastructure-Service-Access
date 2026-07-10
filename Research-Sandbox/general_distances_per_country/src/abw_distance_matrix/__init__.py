"""Installable public facade for the ABW distance-matrix pipeline.

The historical research modules remain importable as ``distance_pipeline`` and
the command-line scripts remain available as ``run_pipeline.py`` and
``calibrate_speeds.py``.  This package adds a stable package name and console
entry points for installed use.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
