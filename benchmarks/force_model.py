# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from dataclasses import replace
from typing import Any, Callable

import jax.numpy as jnp

import jaxdem as jdem
from benchmarks.base import SkipBenchmark, get_state_factory

EXCLUDED_FORCEMODELS = {"lawcombiner", "forcerouter"}
LJ_MATERIAL_ID = 0
ELASTIC_FRICTION_MATERIAL_ID = 1

BENCHMARK_MATERIAL_TABLE = jdem.MaterialTable.from_materials(
    [
        jdem.Material.create("lj", density=1.0, epsilon=1.0),
        jdem.Material.create(
            "elasticfrict",
            density=1.0,
            young=1.0e4,
            poisson=0.3,
            mu=0.5,
            e=0.8,
            mu_r=0.05,
        ),
    ]
)


def _benchmark_force_model(
    method: str, force_key: str
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any], str, str, str]:
    if force_key in EXCLUDED_FORCEMODELS:
        raise SkipBenchmark(f"Skipping excluded force model: {force_key}")

    force_model = jdem.ForceModel.create(force_key)
    required_material_properties = set(force_model.required_material_properties)
    mat_id = (
        LJ_MATERIAL_ID
        if "epsilon_eff" in required_material_properties
        else ELASTIC_FRICTION_MATERIAL_ID
    )

    state_factory = get_state_factory("spheres")
    state = state_factory()
    state = replace(state, mat_id=jnp.full_like(state.mat_id, mat_id))

    system = jdem.System.create(
        state.pos_c.shape,
        force_model_type=force_key,
        mat_table=BENCHMARK_MATERIAL_TABLE,
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

    return func, args, {}, system.force_model.type_name, "Force Model", "spheres"


# Create functions for each combination
for method in ["force", "energy"]:
    for f_key in jdem.ForceModel._registry.keys():
        func_name = f"benchmark_{f_key}_{method}"
        globals()[func_name] = lambda m=method, f=f_key: _benchmark_force_model(m, f)
