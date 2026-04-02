# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
import jaxdem as jdem
from benchmarks.base import get_state_factory
from typing import Any, Callable


def _benchmark_force_model(
    method: str, force_key: str
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    state_factory = get_state_factory("spheres")
    state = state_factory()
    # Use celllist as default collider to test force models
    system = jdem.System.create(
        state.pos_c.shape,
        force_model_type=force_key,
        collider_type="celllist",
        collider_kw={"state": state},
    )

    if method == "force":
        func = system.collider.compute_force
        args = (state, system)
    elif method == "energy":
        func = system.collider.compute_potential_energy
        args = (state, system)
    else:
        raise ValueError(f"Unknown method {method}")

    return func, args, {}, force_key, "Force Model", "spheres"


# Create functions for each combination
for method in ["force", "energy"]:
    for f_key in ["spring", "hertz", "cundallstrack", "wca", "lennardjones"]:
        func_name = f"benchmark_{f_key}_{method}"
        globals()[func_name] = lambda m=method, f=f_key: _benchmark_force_model(m, f)
