import jax
import jax.numpy as jnp
import jaxdem as jdem
import time
import numpy as np
import os
import sys
import importlib
import inspect
from typing import Any, Callable

# Ensure current directory is in sys.path
sys.path.insert(0, os.getcwd())

from benchmarks.utils import (
    get_git_commit,
    get_hardware_info,
    update_results,
    get_commit_date,
)
from benchmarks.base import SkipBenchmark

RESULTS_FILE = "benchmarks/results.json"


def benchmark_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    n_warmup: int = 3,
    n_runs: int = 10,
) -> tuple[float, float]:
    for _ in range(n_warmup):
        args_copy = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, jax.Array) else x, args
        )
        kwargs_copy = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, jax.Array) else x, kwargs
        )
        res = func(*args_copy, **kwargs_copy)
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
    times = []
    for _ in range(n_runs):
        args_copy = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, jax.Array) else x, args
        )
        kwargs_copy = jax.tree_util.tree_map(
            lambda x: jnp.array(x) if isinstance(x, jax.Array) else x, kwargs
        )
        start = time.perf_counter()
        res = func(*args_copy, **kwargs_copy)
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)
    return float(np.mean(times)), float(np.std(times))


def run_all_benchmarks() -> None:
    commit = get_git_commit()
    hw = get_hardware_info()
    date = get_commit_date(commit)

    benchmark_dir = "benchmarks"
    modules = [
        f[:-3]
        for f in os.listdir(benchmark_dir)
        if f.endswith(".py")
        and f not in ["__init__.py", "run_benchmarks.py", "utils.py", "base.py"]
    ]

    for mod_name in modules:
        print(f"Running benchmarks from module: {mod_name}")
        module = importlib.import_module(f"benchmarks.{mod_name}")

        # Reload module to ensure we get dynamic functions if any
        importlib.reload(module)

        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("benchmark_"):
                print(f"  Benchmark: {name}")
                try:
                    # Execute setup function
                    result = func()
                    if result is None:
                        continue

                    target_func, args, kwargs, m_type, category, sys_name = result

                    mean, std = benchmark_function(target_func, args, kwargs)

                    update_results(
                        RESULTS_FILE,
                        {
                            "commit": commit,
                            "date": date,
                            "hardware": hw,
                            "function": (
                                target_func.__name__
                                if hasattr(target_func, "__name__")
                                else str(target_func)
                            ),
                            "module_type": m_type,
                            "system": sys_name,
                            "mean": mean,
                            "std": std,
                        },
                    )
                except SkipBenchmark as e:
                    print(f"    Skipped: {e}")
                except Exception as e:
                    print(f"    Failed: {e}")


if __name__ == "__main__":
    run_all_benchmarks()
