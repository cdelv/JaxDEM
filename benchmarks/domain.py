# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
import jaxdem as jdem
from benchmarks.base import get_state_factory
from typing import Any, Callable


def _benchmark_domain(
    method: str, system_type: str, domain_key: str
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    state_factory = get_state_factory(system_type)
    state = state_factory()
    system = jdem.System.create(state.pos_c.shape, domain_type=domain_key)
    func = getattr(system.domain, method)
    args: tuple[Any, ...]
    if method == "displacement":
        args = (state.pos_c, state.pos_c, system)
    else:
        args = (state, system)
    return func, args, {}, system.domain.type_name, "Domain", system_type


# Create functions for each combination
for method in ["apply", "displacement", "shift"]:
    for sys_type in ["spheres", "clumps", "deformable", "mixed"]:
        for d_key in jdem.Domain._registry.keys():
            func_name = f"benchmark_{d_key}_{method}_{sys_type}"
            globals()[func_name] = (
                lambda m=method, s=sys_type, d=d_key: _benchmark_domain(m, s, d)
            )
