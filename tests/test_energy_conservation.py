# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unit tests to verify energy conservation of spheres, clumps, and facets under Euler and Verlet integrators."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.utils.thermal import compute_energy

# Import setups from collider invariance test
from tests.test_colliders_invariance import (
    set_up_clump,
    set_up_facets_and_spheres,
    set_up_spheres,
)


@pytest.fixture(autouse=True)
def skip_if_not_full(request):
    if not request.config.getoption("--full"):
        pytest.skip("Skipped energy conservation tests. Use --full to run them.")


def run_conservation_test(
    name: str,
    state_init,
    system_init,
    integrator_type: str,
    total_time: float = 5.0,
    n_samples: int = 100,
):
    dts = np.logspace(np.log10(5e-3), np.log10(3e-5), 5)

    if integrator_type == "euler":
        lin_int = "euler"
        rot_int = "spiral"
    else:
        lin_int = "verlet"
        rot_int = "verletspiral"

    stds = []

    for dt in dts:
        dt_val = float(dt)
        steps = int(total_time / dt_val)
        stride = steps // n_samples
        if stride < 1:
            stride = 1

        system = dataclasses.replace(
            system_init,
            dt=jnp.asarray(dt_val, dtype=float),
            linear_integrator=jdem.integrators.LinearIntegrator.create(lin_int),
            rotation_integrator=jdem.integrators.RotationIntegrator.create(rot_int),
        )
        state_copy = jax.tree.map(lambda x: x, state_init)
        _, _, (state_traj, system_traj) = system.trajectory_rollout(
            state_copy, system, n=n_samples, stride=stride
        )
        energy = jax.vmap(compute_energy)(state_traj, system_traj)
        stds.append(jnp.std(energy))

    log_dts = np.log(dts)
    log_stds = np.log(stds)
    slope, _ = np.polyfit(log_dts, log_stds, 1)

    # Facet contacts switch regions (face/edge/vertex) non-smoothly, which
    # limits the observable convergence order to one even under Verlet.
    is_first_order = integrator_type == "euler" or name == "facet"

    if is_first_order:
        assert (
            0.7 < slope
        ), f"First-order slope {slope} out of range 0.7<.\n dts={dts},\n stds={stds}"
    else:
        assert (
            1.9 < slope
        ), f"Second-order slope {slope} out of range 1.9<.\n dts={dts},\n stds={stds}"


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "domain_type", ["periodic", "reflect", "free", "reflectsphere"]
)
@pytest.mark.parametrize("integrator_type", ["euler", "verlet"])
def test_spheres_energy_conservation(dim: int, domain_type: str, integrator_type: str):
    state, system = set_up_spheres(
        dim=dim,
        domain_type=domain_type,
        collider_type="naive",
    )
    run_conservation_test(
        "sphere",
        state,
        system,
        integrator_type,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("domain_type", ["periodic", "reflect", "free"])
@pytest.mark.parametrize("integrator_type", ["euler", "verlet"])
def test_clumps_energy_conservation(dim: int, domain_type: str, integrator_type: str):
    state, system = set_up_clump(
        dim=dim,
        collider_type="naive",
        domain_type=domain_type,
    )
    run_conservation_test(
        "clump",
        state,
        system,
        integrator_type,
        n_samples=100,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("domain_type", ["periodic", "reflect", "free"])
@pytest.mark.parametrize("integrator_type", ["euler", "verlet"])
def test_facets_energy_conservation(dim: int, domain_type: str, integrator_type: str):
    state, system = set_up_facets_and_spheres(
        dim=dim,
        collider_type="naive",
        domain_type=domain_type,
    )
    run_conservation_test(
        "facet",
        state,
        system,
        integrator_type,
    )
