# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Force reach, contact kinematics, and initial-force regressions."""

import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd


def test_explicit_initialization_supplies_first_verlet_acceleration():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]), vel=jnp.array([[0.2, 0.3]]))
    gravity = jnp.array([0.0, -2.0])
    system = jd.System.create(
        state=state, dt=0.1, collider_type="", force_manager_kw={"gravity": gravity}
    )
    state, system = jd.System.initialize(state, system)
    result, updated = jd.System.step(state, system)
    np.testing.assert_allclose(
        result.pos_c, state.pos_c + 0.1 * state.vel + 0.005 * gravity
    )
    np.testing.assert_allclose(result.vel, state.vel + 0.1 * gravity)
    assert int(updated.step_count) == 1


def test_initialization_replaces_stale_stored_forces():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.5, 0.0]]), rad=jnp.full(2, 0.5)
    )
    system = jd.System.create(state=state, collider_type="CellList")
    state, system = system.collider.compute_force(state, system)
    assert float(jnp.linalg.norm(state.force)) > 0.0
    moved = replace(state, pos_c=jnp.array([[0.0, 0.0], [3.0, 0.0]]))
    prepared, system = jd.System.initialize(moved, system)
    np.testing.assert_array_equal(prepared.force, 0.0)
    np.testing.assert_array_equal(prepared.pos, moved.pos)
    assert int(system.step_count) == 0
    assert float(system.time) == 0.0


def test_initialization_applies_queued_loads_without_reflection():
    state = jd.State.create(pos=jnp.array([[2.0, 0.0]]), vel=jnp.array([[1.0, 0.0]]))
    system = jd.System.create(
        state=state, collider_type="", domain_type="ReflectSphere"
    )
    system = system.force_manager.add_force(state, system, jnp.array([[3.0, 4.0]]))
    queued_force = system.force_manager.external_force.copy()
    initialized, updated = jd.System.initialize(state, system)
    np.testing.assert_array_equal(initialized.pos, state.pos)
    np.testing.assert_array_equal(initialized.vel, state.vel)
    np.testing.assert_array_equal(initialized.force, queued_force)
    np.testing.assert_array_equal(updated.force_manager.external_force, 0.0)


def test_overflow_stays_failed_after_later_successful_step():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.5, 0.0]]),
        rad=jnp.full(2, 0.5),
        fixed=jnp.ones(2, dtype=bool),
    )
    system = jd.System.create(
        state=state,
        collider_type="NeighborList",
        collider_kw={"max_neighbors": 0, "cutoff": 1.0, "skin": 0.1},
    )
    state, failed = jd.System.step(state, system)
    assert bool(failed.search_overflow)
    retry = replace(failed, collider=jd.Collider.create("naive"))
    _, retry = jd.System.step(state, retry)
    with pytest.raises(RuntimeError, match="pre-overflow"):
        retry.check_overflow()


def test_packing_callback_matches_jax_integer_contract():
    from jaxdem.utils.packing_utils import scale_to_packing_fraction

    state = jd.State.create(
        pos=jnp.array([[0.5, 0.5], [1.5, 1.5]]), rad=jnp.full(2, 0.1)
    )
    system = jd.System.create(
        state=state,
        domain_type="periodic",
        domain_kw={"box_size": jnp.array([2.0, 2.0])},
    )
    scaled, updated = scale_to_packing_fraction(state, system, 0.2)
    np.testing.assert_allclose(
        jd.utils.compute_packing_fraction(scaled, updated), 0.2, rtol=1e-6
    )


@pytest.mark.parametrize("collider", ["CellList", "MultiCellList", "NeighborList"])
def test_flexible_facet_bounds_follow_deformation(collider):
    state = jd.State.create(pos=jnp.array([[3.0, 0.15]]), rad=jnp.array([0.1]))
    state = state.add_facet(
        state, jnp.array([[0.0, 0.0], [1.0, 0.0]]), thickness=0.1, rigid=False
    )
    # Stretch a flexible facet after construction without manually inflating _rad.
    state = replace(state, pos_c=state.pos_c.at[2].set(jnp.array([4.0, 0.0])))
    system = jd.System.create(state=state, force_model_type="sphere_facet_spring")
    expected, _ = system.collider.compute_force(state, system)
    kw = (
        {"max_neighbors": 4, "cutoff": 0.2, "skin": 0.1}
        if collider == "NeighborList"
        else {}
    )
    actual_system = replace(
        system, collider=jd.Collider.create(collider, state=state, **kw)
    )
    actual, actual_system = actual_system.collider.compute_force(state, actual_system)
    assert np.linalg.norm(expected.force) > 0
    np.testing.assert_allclose(actual.force, expected.force, rtol=1e-6, atol=1e-6)
    assert not bool(actual_system.collider.overflow)


@pytest.mark.parametrize(
    "dim,collider",
    [(2, "CellList"), (2, "MultiCellList"), (2, "NeighborList"), (3, "NeighborList")],
)
def test_shear_image_force_and_energy_are_physical(dim, collider):
    pos = jnp.array([[1.0, 0.1], [6.0, 9.9]])
    if dim == 3:
        pos = jnp.pad(pos, ((0, 0), (0, 1)))
    state = jd.State.create(pos=pos, rad=jnp.full(2, 0.2))
    kw = (
        {"max_neighbors": 2, "cutoff": 0.4, "skin": 0.1}
        if collider == "NeighborList"
        else {}
    )
    system = jd.System.create(
        state=state,
        collider_type=collider,
        collider_kw=kw,
        domain_type="leesedwards",
        domain_kw={"box_size": jnp.full(dim, 10.0), "gamma": 0.5},
    )
    result, updated = system.collider.compute_force(state, system)
    _, updated, energy = updated.collider.compute_potential_energy(result, updated)
    stiffness = system.mat_table.young_eff[0, 0]
    force = np.zeros((2, dim))
    force[:, 1] = np.array([1.0, -1.0]) * float(stiffness) * 0.2
    np.testing.assert_allclose(result.force, force, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(energy, 0.5 * stiffness * 0.2**2, rtol=1e-5)
    assert not bool(updated.collider.overflow)
