"""Visualization helpers for kim2026."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from kim2026.viz.d2nn_beamreducer_sweep import generate_figures as generate_d2nn_beamreducer_sweep_figures
from kim2026.viz.fd2nn_sweep import generate_figures as generate_fd2nn_sweep_figures
