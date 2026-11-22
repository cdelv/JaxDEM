# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Spheres-in-a-box throughput benchmarks for JaxDEM."""

from .cases import (
    COLLIDERS,
    DIMS,
    DOMAINS,
    FORCES,
    MODES,
    PROFILES,
    REGIMES,
    BenchmarkCase,
    all_cases,
    build_spheres_in_box,
    select_cases,
    validate_phi_vel,
)
from .dashboard import render_dashboard_html, render_standalone_html
from .runner import check_throttle_policy, run_custom, run_suite
from .scaling import format_scaling_table, plot_scaling, run_scaling
from .store import DEFAULT_HISTORY_PATH, load_history, upsert_run

__all__ = [
    "BenchmarkCase",
    "COLLIDERS",
    "DEFAULT_HISTORY_PATH",
    "DIMS",
    "DOMAINS",
    "FORCES",
    "MODES",
    "PROFILES",
    "REGIMES",
    "all_cases",
    "build_spheres_in_box",
    "check_throttle_policy",
    "format_scaling_table",
    "load_history",
    "plot_scaling",
    "render_dashboard_html",
    "render_standalone_html",
    "run_custom",
    "run_scaling",
    "run_suite",
    "select_cases",
    "upsert_run",
    "validate_phi_vel",
]
