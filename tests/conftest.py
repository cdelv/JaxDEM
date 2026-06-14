# conftest.py
import jax

jax.config.update("jax_enable_x64", True)


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False, help="run full/slow tests"
    )
