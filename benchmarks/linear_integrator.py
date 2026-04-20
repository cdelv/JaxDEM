# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from typing import Any, Callable

import jaxdem as jdem
from benchmarks.base import SkipBenchmark, get_state_factory

EXCLUDED_INTEGRATORS = {"vicsek_extrinsic", "vicsek_intrinsic"}


def _benchmark_linear_integrator(
    method: str, system_type: str, integrator_key: str
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    if integrator_key in EXCLUDED_INTEGRATORS:
        raise SkipBenchmark(f"Skipping excluded integrator: {integrator_key}")

    state_factory = get_state_factory(system_type)
    state = state_factory()

    kw: dict[str, Any] = {}
    if integrator_key == "optax":
        import optax

        kw = {"optimizer": optax.adam(1e-3), "state": state}
    elif integrator_key == "langevin":
        kw = {"gamma": 1.0, "k_B": 1.0, "temperature": 1.0}
    elif "rescaling" in integrator_key:
        kw = {
            "k_B": 1.0,
            "temperature": 1.0,
            "rescale_every": 10,
            "can_rotate": True,
            "subtract_drift": True,
        }

    system = jdem.System.create(
        state.pos_c.shape,
        linear_integrator_type=integrator_key,
        linear_integrator_kw=kw,
        rotation_integrator_type="",
        collider_type="celllist",
        collider_kw={"state": state},
    )
    state, system = system.linear_integrator.initialize(state, system)
    func = getattr(system.linear_integrator, method)
    args = (state, system)
    return (
        func,
        args,
        {},
        system.linear_integrator.type_name,
        "Linear Integrator",
        system_type,
    )


for method in ["step_before_force", "step_after_force"]:
    for sys_type in ["spheres", "clumps", "deformable", "mixed"]:
        for i_key in jdem.LinearIntegrator._registry.keys():
            func_name = f"benchmark_{i_key}_{method}_{sys_type}"
            globals()[func_name] = lambda m=method, s=sys_type, i=i_key: (
                _benchmark_linear_integrator(m, s, i)
            )
