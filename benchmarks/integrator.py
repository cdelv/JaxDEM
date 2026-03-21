import jaxdem as jdem
from benchmarks.base import get_state_factory
from typing import Any, Callable


def _benchmark_integrator(
    method: str, system_type: str, integrator_key: str
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    state_factory = get_state_factory(system_type)
    state = state_factory()
    system = jdem.System.create(
        state.pos_c.shape, linear_integrator_type=integrator_key
    )
    func = getattr(system.linear_integrator, method)
    args = (state, system)
    return func, args, {}, system.linear_integrator.type_name, "Integrator", system_type


# Create functions for each combination
for method in ["step_before_force", "step_after_force"]:
    for sys_type in ["spheres", "clumps", "deformable", "mixed"]:
        for i_key in ["verlet", "euler"]:
            func_name = f"benchmark_{i_key}_{method}_{sys_type}"
            globals()[func_name] = (
                lambda m=method, s=sys_type, i=i_key: _benchmark_integrator(m, s, i)
            )
