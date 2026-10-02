# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Contact normals remain normalized at tiny nonzero separations."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import jaxdem as jd


_SEPARATION = np.array([3.0e-10, 4.0e-10])
_DISTANCE = 5.0e-10
_NORMAL_0_FROM_1 = np.array([-0.6, -0.8])


def _state() -> jd.State:
    return jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], _SEPARATION]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )


def test_near_coincident_spring_contact_uses_unit_normal() -> None:
    state = _state()
    young = 12.0
    material = jd.Material.create(
        "elastic", density=1.0, young=young, poisson=0.25
    )
    law = jd.forces.SpringForce()
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="naive",
    )

    force, _, _ = law.force(
        0, 1, state.pos, state, system, law.init_history((), state.dim)
    )

    expected = young * (1.0 - _DISTANCE) * _NORMAL_0_FROM_1
    np.testing.assert_allclose(force, expected, rtol=2e-6, atol=1e-6)


def test_near_coincident_hertz_contact_uses_unit_normal() -> None:
    state = _state()
    young = 12.0
    poisson = 0.25
    material = jd.Material.create(
        "elastic", density=1.0, young=young, poisson=poisson
    )
    law = jd.forces.HertzianForce()
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="naive",
    )

    force, _, _ = law.force(
        0, 1, state.pos, state, system, law.init_history((), state.dim)
    )

    effective_young = young / (2.0 * (1.0 - poisson**2))
    effective_radius = 0.25
    overlap = 1.0 - _DISTANCE
    magnitude = (
        (4.0 / 3.0)
        * effective_young
        * overlap
        * np.sqrt(effective_radius * overlap)
    )
    np.testing.assert_allclose(
        force, magnitude * _NORMAL_0_FROM_1, rtol=2e-6, atol=1e-6
    )


def test_near_coincident_cundall_contact_uses_unit_normal() -> None:
    state = _state()
    normal_stiffness = 10.0
    material = jd.Material.create(
        "cundallstrackparams",
        density=1.0,
        k_n=normal_stiffness,
        k_t=4.0,
        b_n=0.0,
        b_t=0.0,
        mu=0.0,
        mu_r=0.0,
    )
    law = jd.forces.CundallStrackForce(parameterization="coefficients")
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 1,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )

    force, _, _ = law.force(
        0,
        1,
        state.pos,
        state,
        system,
        law.init_history((), state.dim),
        advance_history=False,
    )

    expected = normal_stiffness * (1.0 - _DISTANCE) * _NORMAL_0_FROM_1
    np.testing.assert_allclose(force, expected, rtol=2e-6, atol=1e-6)
