# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from typing import Any, Callable

import jaxdem as jdem
from benchmarks.base import get_state_factory


def _benchmark_force_manager(
    system_type: str,
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    state_factory = get_state_factory(system_type)
    state = state_factory()
    system = jdem.System.create(state.pos_c.shape)
    func = system.force_manager.apply
    args = (state, system)
    return func, args, {}, "ForceManager", "Force Manager", system_type


# Create functions for each combination
for sys_type in ["spheres", "clumps"]:
    func_name = f"benchmark_apply_{sys_type}"
    globals()[func_name] = lambda s=sys_type: _benchmark_force_manager(s)
