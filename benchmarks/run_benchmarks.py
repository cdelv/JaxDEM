# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
import importlib
import inspect
import sys
import timeit
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

# Ensure current directory is in sys.path
sys.path.insert(0, str(Path.cwd()))

from benchmarks.base import SkipBenchmark
from benchmarks.utils import (
    get_commit_date,
    get_git_commit,
    get_hardware_info,
    update_results,
)

RESULTS_FILE = Path("benchmarks") / "results.json"


def benchmark_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    number: int = 1,
    repeat: int = 100,
) -> tuple[float, float]:

    def validate_result(res: Any) -> None:
        for leaf in jax.tree.leaves(res):
            try:
                arr = np.asarray(leaf)
            except (TypeError, ValueError):
                continue
            if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
                raise ValueError("benchmark returned non-finite output")

        values = res if isinstance(res, tuple) else (res,)
        if isinstance(res, tuple) and len(res) == 4:
            possible_overflow = np.asarray(res[-1])
            if possible_overflow.dtype == np.dtype(bool) and bool(possible_overflow):
                raise RuntimeError("benchmark neighbor query reported overflow")
        for value in values:
            if hasattr(value, "check_overflow"):
                value.check_overflow()
            collider = getattr(value, "collider", None)
            if collider is not None and bool(np.asarray(collider.overflow)):
                raise RuntimeError("benchmark collider reported overflow")

    def run_once() -> None:
        res = func(*args, **kwargs)
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        elif hasattr(res, "device_buffer"):
            res.device_buffer.block_until_ready()
        else:
            jax.tree.map(
                lambda x: (
                    x.block_until_ready() if hasattr(x, "block_until_ready") else x
                ),
                res,
            )
        validate_result(res)

    run_once()
    times = timeit.repeat(stmt=run_once, number=number, repeat=repeat)
    return float(np.mean(times)), float(np.std(times))


def run_all_benchmarks() -> None:
    commit = get_git_commit()
    hw = get_hardware_info()
    date = get_commit_date(commit)

    benchmark_dir = Path("benchmarks")
    modules = [
        f.name[:-3]
        for f in benchmark_dir.iterdir()
        if f.is_file()
        and f.name.endswith(".py")
        and f.name not in ["__init__.py", "run_benchmarks.py", "utils.py", "base.py"]
    ]

    failures: list[str] = []
    for mod_name in modules:
        print(f"Running benchmarks from module: {mod_name}")
        module = importlib.import_module(f"benchmarks.{mod_name}")
        importlib.reload(module)

        for name, func in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("benchmark_"):
                print(f"  Benchmark: {name}")
                try:
                    result = func()
                    if result is None:
                        continue

                    target_func, args, kwargs, m_type, category, sys_name = result
                    mean, std = benchmark_function(target_func, args, kwargs)
                    print(f"    Result: {mean:.6f} \u00b1 {std:.6f} s")

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
                            "benchmark_category": category,
                            "system": sys_name,
                            "mean": mean,
                            "std": std,
                        },
                    )
                except SkipBenchmark as e:
                    print(f"    Skipped: {e}")
                except Exception as e:
                    print(f"    Failed: {e}")
                    failures.append(f"{mod_name}.{name}: {type(e).__name__}: {e}")

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(
            f"{len(failures)} benchmark(s) failed or returned invalid output:\n{details}"
        )


if __name__ == "__main__":
    run_all_benchmarks()
