# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jd
from jaxdem.utils.linalg import unit_and_norm

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

    def original_force(pos):
        rij = system.domain._displacement(pos[0], pos[1], system)
        normal, distance = unit_and_norm(rij)
        stiffness = system.mat_table.young_eff[state.mat_id[0], state.mat_id[1]]
        overlap = jnp.maximum(0.0, state.rad[0] + state.rad[1] - distance)
        return (stiffness * overlap)[..., None] * normal

    return state.pos, scalar_force, original_force


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
def test_scalar_spring_matches_unit_and_norm_force_and_gradient(displacement):
    pos, scalar_force, original_force = _spring_case(displacement)

    actual_force = scalar_force(pos)
    expected_force = original_force(pos)
    actual_gradient = jax.jacrev(scalar_force)(pos)
    expected_gradient = jax.jacrev(original_force)(pos)

    assert jnp.all(jnp.isfinite(actual_force))
    assert jnp.all(jnp.isfinite(actual_gradient))
    assert jnp.allclose(actual_force, expected_force, rtol=_RTOL, atol=_ATOL)
    assert jnp.allclose(actual_gradient, expected_gradient, rtol=_RTOL, atol=_ATOL)
