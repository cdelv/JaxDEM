from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd


@pytest.mark.parametrize("dim,alpha,beta", [(2, 0, 1), (2, 1, 0), (3, 2, 0)])
@pytest.mark.parametrize("dgamma", [-0.07, 0.07])
def test_shear_deforms_periodic_displacements(dim, alpha, beta, dgamma):
    box = jnp.arange(dim, dtype=float) + 10.0
    anchor = jnp.arange(dim, dtype=float) - 2.0
    gamma = 0.23
    ri = anchor + 0.2
    displacement = jnp.full(dim, 0.1).at[beta].set(0.3)
    image = jnp.zeros(dim).at[beta].set(2 * box[beta])
    image = image.at[alpha].set(2 * gamma * box[beta] - box[alpha])
    state = jd.State.create(pos=jnp.stack([ri, ri - displacement + image]))
    system = jd.System.create(
        state=state,
        domain_type="leesedwards",
        domain_kw={
            "box_size": box,
            "anchor": anchor,
            "gamma": gamma,
            "alpha": alpha,
            "beta": beta,
        },
    )
    sheared, updated = system.domain.shear(state, system, dgamma)
    strain = np.eye(dim)
    strain[alpha, beta] = dgamma
    actual = updated.domain.displacement(sheared.pos[0], sheared.pos[1], updated)
    np.testing.assert_allclose(actual, strain @ displacement, atol=1e-12)
    np.testing.assert_allclose(updated.domain.gamma, gamma + dgamma)
    np.testing.assert_array_equal(updated.domain.box_size, box)
    np.testing.assert_array_equal(updated.time, system.time)
    np.testing.assert_array_equal(updated.domain.gamma_dot, system.domain.gamma_dot)
    np.testing.assert_array_equal(sheared.vel, state.vel)
    restored, restored_system = updated.domain.shear(sheared, updated, -dgamma)
    np.testing.assert_allclose(restored.pos, state.pos, atol=1e-12)
    np.testing.assert_allclose(restored_system.domain.gamma, gamma)


def test_shear_preserves_rigid_clump_geometry():
    state = jd.State.create(
        pos=jnp.array([[2.0, 3.0], [2.0, 3.0]]),
        pos_p=jnp.array([[-0.2, 0.3], [0.2, -0.3]]),
        clump_id=jnp.zeros(2, dtype=int),
    )
    system = jd.System.create(state=state, domain_type="leesedwards")
    sheared, _ = system.domain.shear(state, system, 0.1)
    np.testing.assert_allclose(sheared.pos_c, [[2.3, 3.0], [2.3, 3.0]])
    np.testing.assert_allclose(
        sheared.pos[1] - sheared.pos[0], state.pos[1] - state.pos[0], atol=1e-12
    )
    np.testing.assert_array_equal(sheared.pos_p, state.pos_p)


def _contact_system(state, domain, *, restitution=0.7):
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=restitution,
        mu=10.0,
        mu_r=0.0,
    )
    return jd.System.create(
        state=state,
        domain=domain,
        force_model_type="cundallstrack",
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="NeighborList",
        collider_kw={"max_neighbors": 2, "cutoff": 1.0, "skin": 0.1},
        dt=0.01,
    )


@pytest.mark.parametrize("dim,alpha,beta", [(2, 0, 1), (2, 1, 0), (3, 2, 0)])
@pytest.mark.parametrize("image_index", [-1, 0, 1])
def test_affine_shear_advances_contact_history(dim, alpha, beta, image_index):
    box = jnp.arange(dim, dtype=float) + 10.0
    gamma = 0.23
    dgamma = 0.01
    displacement = jnp.zeros(dim).at[alpha].set(0.12).at[beta].set(0.9)
    local_pos = jnp.stack([jnp.full(dim, 0.45), 0.45 - displacement])
    image_offset = jnp.zeros(dim).at[beta].set(image_index * box[beta])
    image_offset = image_offset.at[alpha].set(
        image_index * gamma * box[beta]
    )
    state = jd.State.create(
        pos=local_pos.at[1].add(image_offset),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    domain = jd.Domain.create(
        "leesedwards",
        dim=dim,
        box_size=box,
        gamma=gamma,
        gamma_dot=0.37,
        alpha=alpha,
        beta=beta,
    )
    system = _contact_system(state, domain, restitution=1.0)
    state, system = system.initialize(state, system)

    sheared, updated = system.domain.shear(state, system, dgamma)

    neighbors = jnp.array([[1], [0]])
    history = updated.collider.get_history(sheared, updated, neighbors)
    rij = updated.domain.displacement(sheared.pos[0], sheared.pos[1], updated)
    normal = rij / jnp.linalg.norm(rij)
    affine_displacement = jnp.zeros(dim).at[alpha].set(
        dgamma * displacement[beta]
    )
    expected = (
        affine_displacement
        - jnp.vdot(affine_displacement, normal) * normal
    )
    np.testing.assert_allclose(history[0, 0, :dim], expected, atol=1e-12)
    np.testing.assert_allclose(history[1, 0, :dim], -expected, atol=1e-12)
    np.testing.assert_allclose(history[0, 0, dim:], normal, atol=1e-12)
    np.testing.assert_allclose(history[1, 0, dim:], -normal, atol=1e-12)
    np.testing.assert_array_equal(sheared.vel, state.vel)
    np.testing.assert_array_equal(sheared.ang_vel, state.ang_vel)
    np.testing.assert_array_equal(updated.domain.gamma_dot, domain.gamma_dot)
    assert not bool(updated.search_overflow | updated.collider.overflow)


def test_zero_affine_shear_does_not_advance_history_or_rebuild() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.9]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    domain = jd.Domain.create(
        "leesedwards", dim=2, box_size=jnp.full(2, 10.0)
    )
    system = _contact_system(state, domain, restitution=1.0)
    state, system = system.initialize(state, system)

    sheared, updated = system.domain.shear(state, system, 0.0)

    np.testing.assert_array_equal(sheared.pos_c, state.pos_c)
    np.testing.assert_array_equal(updated.collider.history, system.collider.history)
    np.testing.assert_array_equal(
        updated.collider.n_build_times, system.collider.n_build_times
    )


def test_minimization_does_not_repeat_affine_history_increment() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.9]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    domain = jd.Domain.create(
        "leesedwards", dim=2, box_size=jnp.full(2, 10.0)
    )
    system = _contact_system(state, domain, restitution=1.0)
    state, system = system.initialize(state, system)
    sheared, system = system.domain.shear(state, system, 0.01)
    history = system.collider.history

    result = system.minimize(
        sheared,
        system,
        max_steps=0,
        force_tol=-1.0,
        torque_tol=-1.0,
    )

    np.testing.assert_allclose(result.system.collider.history, history, atol=1e-15)


def test_stateless_affine_shear_does_not_traverse_neighbor_list() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.9]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    system = jd.System.create(
        state=state,
        domain_type="leesedwards",
        domain_kw={"box_size": jnp.full(2, 10.0)},
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )
    state, system = system.initialize(state, system)
    builds = system.collider.n_build_times

    _, updated = system.domain.shear(state, system, 0.01)

    np.testing.assert_array_equal(updated.collider.n_build_times, builds)


@pytest.mark.parametrize("dim,alpha,beta", [(2, 0, 1), (2, 1, 0), (3, 2, 0)])
@pytest.mark.parametrize("image_index", [-2, 0, 1])
@pytest.mark.parametrize("gamma_dot", [-0.1, 0.0, 0.1])
@pytest.mark.parametrize("advance_history", [False, True])
def test_sheared_contact_matches_explicit_moving_image(
    dim, alpha, beta, image_index, gamma_dot, advance_history
):
    """Match force, torque and history to a nearby image with its true velocity."""
    box = jnp.arange(dim, dtype=float) + 10.0
    gamma = 0.23
    displacement = jnp.zeros(dim).at[alpha].set(0.12).at[beta].set(0.9)
    local_pos = jnp.stack([jnp.full(dim, 0.45), 0.45 - displacement])
    image_offset = jnp.zeros(dim).at[beta].set(image_index * box[beta])
    image_offset = image_offset.at[alpha].set(image_index * gamma * box[beta])
    state = jd.State.create(
        pos=local_pos.at[1].add(image_offset),
        vel=jnp.zeros((2, dim)),
        ang_vel=jnp.full((2, 1 if dim == 2 else 3), 0.03),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    domain = jd.Domain.create(
        "leesedwards",
        dim=dim,
        box_size=box,
        gamma=gamma,
        gamma_dot=gamma_dot,
        alpha=alpha,
        beta=beta,
    )
    system = _contact_system(state, domain)
    # The nearby replica is translated by -image_offset and consequently
    # moves at -image_index * gamma_dot * L_beta relative to particle 1.
    reference = replace(
        state,
        pos_c=local_pos,
        vel=state.vel.at[1, alpha].add(-image_index * gamma_dot * box[beta]),
    )
    reference_system = replace(system, domain=jd.Domain.create("free", dim=dim))
    history = system.force_model.init_history((2,), dim)
    sources, targets = jnp.array([0, 1]), jnp.array([1, 0])

    def forces(st, sys, h):
        return jax.vmap(
            lambda i, j, pair_history: sys.force_model.force(
                i,
                j,
                st.pos,
                st,
                sys,
                pair_history,
                advance_history=advance_history,
            )
        )(sources, targets, h)

    actual = forces(state, system, history)
    expected = forces(reference, reference_system, history)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, wanted, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual[0][0], -actual[0][1], atol=1e-7)
    if not advance_history:
        np.testing.assert_array_equal(actual[2], history)
    else:
        # A second evaluation also checks accumulation of the corrected slip.
        for result, wanted in zip(
            forces(state, system, actual[2]),
            forces(reference, reference_system, expected[2]),
            strict=True,
        ):
            np.testing.assert_allclose(result, wanted, rtol=1e-6, atol=1e-7)


def test_shear_rate_change_updates_cached_contact_force():
    state = jd.State.create(
        pos=jnp.array([[1.0, 0.45], [3.0, 9.55]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    domain = jd.Domain.create(
        "leesedwards", dim=2, box_size=jnp.full(2, 10.0), gamma=0.2
    )
    system = _contact_system(state, domain)
    state, system = system.initialize(state, system)
    np.testing.assert_allclose(state.force[:, 0], 0.0, atol=1e-10)
    builds = system.collider.n_build_times
    system = replace(system, domain=replace(domain, gamma_dot=jnp.asarray(0.1)))
    actual, system = system.evaluate_forces(state, system)
    reference = replace(
        state,
        pos_c=state.pos_c.at[1].add(jnp.array([-2.0, -10.0])),
        vel=state.vel.at[1, 0].set(-1.0),
    )
    reference_system = replace(system, domain=jd.Domain.create("free", dim=2))
    expected, _ = reference_system.evaluate_forces(reference, reference_system)
    np.testing.assert_allclose(actual.force, expected.force, rtol=1e-6, atol=1e-7)
    assert abs(float(actual.force[0, 0])) > 0.1
    assert int(system.collider.n_build_times) == int(builds)


def test_pre_step_shear_uses_current_strain_and_rate():
    """The documented callback keeps the returned force at the returned strain."""

    def advance_shear(state, system):
        domain = replace(
            system.domain,
            gamma=system.domain.gamma + system.domain.gamma_dot * system.dt,
        )
        return state, replace(system, domain=domain)

    state = jd.State.create(
        pos=jnp.array([[1.0, 0.45], [3.0, 9.55]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
        fixed=jnp.ones(2, dtype=bool),
    )
    domain = jd.Domain.create(
        "leesedwards",
        dim=2,
        box_size=jnp.full(2, 10.0),
        gamma=0.2,
        gamma_dot=0.1,
    )
    # Isolate strain timing using elastic normal forces without contact history.
    system = jd.System.create(
        state=state, domain=domain, dt=0.01, user_pre_step_actions=advance_shear
    )
    state, system = system.initialize(state, system)
    state, system = system.step(state, system, n=3)
    expected, _ = system.evaluate_forces(state, system)
    np.testing.assert_allclose(system.domain.gamma, 0.203)
    np.testing.assert_allclose(state.force, expected.force, rtol=1e-6, atol=1e-7)
    assert float(state.force[0, 0]) > 0
