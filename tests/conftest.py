# conftest.py
import os

import jax

jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1") == "1")


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False, help="run full/slow tests"
    )
