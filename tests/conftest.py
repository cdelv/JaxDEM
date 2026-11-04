# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Configure test precision, compilation-cache cleanup, and extended validation."""

import jax

import os
import gc

import pytest

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")


@pytest.fixture(scope="module", autouse=True)
def release_compiled_executables():
    """Bound compilation memory across the complete CPU suite."""
    yield
    jax.clear_caches()
    gc.collect()


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False, help="run full/slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: extended trajectory or learning validation"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--full"):
        return
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="extended validation: use --full"))
