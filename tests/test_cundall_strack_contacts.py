# SPDX-License-Identifier: BSD-3-Clause
"""Resolved per-contact diagnostics for the Cundall-Strack force law."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import pair_sources


def _system(
    tangential_displacement: float,
) -> tuple[jd.State, jd.System, jnp.ndarray]:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0], [4.0, 4.0]]),
        rad=jnp.full(3, 0.5),
        mass=jnp.ones(3),
        mat_id=jnp.array([0, 1, 0]),
        ang_vel=jnp.array([[1.0], [-1.0], [0.0]]),
    )
    materials = [
        jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=100.0,
            k_t=20.0,
            b_n=0.0,
            b_t=0.0,
            mu=0.2,
            mu_r=0.1,
        ),
        jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=300.0,
            k_t=60.0,
            b_n=0.0,
            b_t=0.0,
            mu=0.8,
            mu_r=0.3,
        ),
    ]
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(parameterization="coefficients"),
        mat_table=jd.MaterialTable.from_materials(materials),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 1,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )
    state, system = jd.System.initialize(state, system)
    collider = system.collider
    sources = pair_sources(collider)
    destinations = collider.neighbor_list
    valid = destinations >= 0
    signed_displacement = jnp.where(
        sources == 0, tangential_displacement, -tangential_displacement
    )
    history = collider.history.at[:, 1].set(jnp.where(valid, signed_displacement, 0.0))
    system = replace(system, collider=replace(collider, history=history))
    return state, system, history


@pytest.mark.parametrize(
    "displacement,tangential_magnitude,mobilized",
    [(0.1, 3.0, False), (0.2, 6.0, True), (0.3, 6.0, True)],
)
def test_resolved_contacts_report_components_coefficients_and_mobilization(
    displacement: float,
    tangential_magnitude: float,
    mobilized: bool,
) -> None:
    state, system, original_history = _system(displacement)

    returned_state, returned_system, contacts = jd.forces.get_cundall_strack_contacts(
        state, system
    )

    assert returned_state is state
    np.testing.assert_array_equal(contacts.pair_ids, [[0, 1], [1, 0]])
    np.testing.assert_allclose(
        contacts.normal_forces, [[-30.0, 0.0], [30.0, 0.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        contacts.tangential_forces,
        [[0.0, -tangential_magnitude], [0.0, tangential_magnitude]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        contacts.torques,
        [
            [-0.5 * tangential_magnitude - 0.75],
            [-0.5 * tangential_magnitude + 0.75],
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(contacts.friction_coefficients, [0.2, 0.2])
    np.testing.assert_array_equal(contacts.mobilized, [mobilized, mobilized])
    np.testing.assert_array_equal(returned_system.collider.history, original_history)

    for row, (i, j) in enumerate(np.asarray(contacts.pair_ids)):
        force, torque, unchanged = system.force_model.force(
            i,
            j,
            state.pos,
            state,
            returned_system,
            returned_system.collider.history[
                (pair_sources(returned_system.collider) == i)
                & (returned_system.collider.neighbor_list == j)
            ][0],
            advance_history=False,
        )
        np.testing.assert_allclose(
            force,
            contacts.normal_forces[row] + contacts.tangential_forces[row],
            atol=1e-6,
        )
        np.testing.assert_allclose(torque, contacts.torques[row], atol=1e-6)
        np.testing.assert_array_equal(
            unchanged,
            returned_system.collider.history[
                (pair_sources(returned_system.collider) == i)
                & (returned_system.collider.neighbor_list == j)
            ][0],
        )


def test_resolved_contacts_require_a_direct_cundall_strack_model() -> None:
    state = jd.State.create(pos=[[0.0, 0.0], [0.8, 0.0]], rad=jnp.full(2, 0.5))
    system = jd.System.create(state=state)

    with pytest.raises(TypeError, match="CundallStrackForce"):
        jd.forces.get_cundall_strack_contacts(state, system)
