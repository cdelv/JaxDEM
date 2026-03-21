import jaxdem as jdem
from benchmarks.base import get_state_factory
from typing import Any, Callable


def _benchmark_collider(
    method: str, system_type: str, collider_key: str
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    state_factory = get_state_factory(system_type)
    state = state_factory()

    collider_kw: dict[str, Any] = {}
    if collider_key in ["celllist", "staticcelllist"]:
        collider_kw = {"state": state}
    elif collider_key == "neighborlist":
        collider_kw = {"state": state, "cutoff": 0.5}

    system = jdem.System.create(
        state.pos_c.shape, collider_type=collider_key, collider_kw=collider_kw
    )
    func = getattr(system.collider, method)
    args: tuple[Any, ...]
    if method == "compute_force":
        args = (state, system)
    elif method == "create_neighbor_list":
        args = (state, system, 0.5, 100)
    else:
        args = (state, system)
    return func, args, {}, system.collider.type_name, "Collider", system_type


# Create functions for each combination
for method in ["compute_force", "create_neighbor_list"]:
    for sys_type in ["spheres", "clumps", "deformable", "mixed"]:
        for c_key in ["celllist", "naive", "neighborlist"]:
            func_name = f"benchmark_{c_key}_{method}_{sys_type}"
            globals()[func_name] = (
                lambda m=method, s=sys_type, c=c_key: _benchmark_collider(m, s, c)
            )
