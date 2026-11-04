# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check scalar spring forces and their closed-form position derivatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd

_RTOL = 1e-12 if jax.config.jax_enable_x64 else 2e-6
_ATOL = 1e-12 if jax.config.jax_enable_x64 else 2e-7


def _spring_case(displacement):
    displacement = jnp.asarray(displacement, dtype=float)
    state = jd.State.create(
        pos=jnp.stack((jnp.zeros_like(displacement), displacement)),
        rad=jnp.full((2,), 0.5),
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.SpringForce(),
        collider_type="naive",
    )
    history = system.force_model.init_history((), state.dim)

    def scalar_force(pos):
        force, _, _ = jd.forces.SpringForce.force(0, 1, pos, state, system, history)
        return force

    def pair_energy(pos):
        return jd.forces.SpringForce.energy(0, 1, pos, state, system, history)

    return state.pos, scalar_force, pair_energy, float(system.mat_table.young_eff[0, 0])


@pytest.mark.parametrize(
    "displacement",
    [
        (0.0, 0.0),
        (6e-11, 8e-11),
        (0.45, 0.6),
        (1.0, 0.0),
        (0.75, 1.0),
        (2e-11, -4e-11, 8e-11),
        (0.3, -0.4, 0.5),
        (0.6, 0.8, -0.75),
    ],
    ids=[
        "zero-2d",
        "tiny-2d",
        "overlap-2d",
        "touching-2d-axis",
        "separated-2d",
        "tiny-3d",
        "overlap-3d",
        "separated-3d",
    ],
)
def test_scalar_spring_matches_radial_force_and_closed_form_jacobian(displacement):
    pos, scalar_force, pair_energy, stiffness = _spring_case(displacement)

    actual_force = scalar_force(pos)
    rij = np.asarray(pos[0] - pos[1], dtype=float)
    radius = np.linalg.norm(rij)
    expected_force = np.zeros_like(rij)
    jacobian = np.zeros((len(rij), len(rij)))
    if 0 < radius < 1:
        expected_force = stiffness * (1 - radius) * rij / radius
        jacobian = stiffness * (
            (1 / radius - 1) * np.eye(len(rij)) - np.outer(rij, rij) / radius**3
        )
    elif radius == 1:
        # At contact, max(0, overlap) selects the midpoint subgradient.
        jacobian = -0.5 * stiffness * np.outer(rij, rij)
    energy_force = -jax.grad(pair_energy)(pos)[0]
    actual_gradient = jax.jacrev(scalar_force)(pos)
    expected_gradient = np.stack((jacobian, -jacobian), axis=1)

    assert jnp.all(jnp.isfinite(actual_force))
    assert jnp.all(jnp.isfinite(actual_gradient))
    assert jnp.allclose(actual_force, expected_force, rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual_force, energy_force, rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual_gradient, expected_gradient, rtol=_RTOL, atol=_ATOL)
