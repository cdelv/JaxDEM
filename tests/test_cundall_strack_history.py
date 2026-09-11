from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd


def _system(
    state: jd.State, *, mu: float = 10.0, restitution: float = 1.0, dt: float = 0.1
) -> jd.System:
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=restitution,
        mu=mu,
        mu_r=0.0,
    )
    table = jd.MaterialTable.from_materials([material])
    base = jd.System.create(state=state, mat_table=table, dt=dt)
    return replace(base, force_model=jd.forces.CundallStrackForce())


def _state(dim: int = 2, *, tangent_speed: float = 1.0) -> jd.State:
    pos = jnp.zeros((2, dim)).at[1, 0].set(0.8)
    vel = jnp.zeros_like(pos).at[0, 1].set(tangent_speed)
    return jd.State.create(pos=pos, vel=vel, rad=jnp.full(2, 0.5), mass=jnp.ones(2))


def _force(state: jd.State, system: jd.System, history: jnp.ndarray, *, advance=True):
    return system.force_model.force(
        0,
        1,
        state.pos,
        state,
        system,
        history,
        advance_history=advance,
    )


def test_elastic_tangent_accumulates_and_persists_after_motion_stops() -> None:
    state = _state()
    system = _system(state)
    history = system.force_model.init_history((), state.dim)

    force1, _, history1 = _force(state, system, history)
    force2, _, history2 = _force(state, system, history1)
    stopped = replace(state, vel=jnp.zeros_like(state.vel))
    force_stopped, _, history_stopped = _force(stopped, system, history2)

    np.testing.assert_allclose(force1[1], -2.5, atol=1e-6)
    np.testing.assert_allclose(force2[1], -5.0, atol=1e-6)
    np.testing.assert_allclose(force_stopped[1], -5.0, atol=1e-6)
    np.testing.assert_allclose(history_stopped[:2], jnp.array([0.0, 0.2]))


def test_contact_velocity_includes_clump_member_and_surface_arms() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
        pos_p=jnp.array([[0.0, 0.25], [0.0, 0.25]]),
        ang_vel=jnp.array([[1.0], [0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    surface_arm = jnp.array([0.5, 0.0])
    np.testing.assert_allclose(
        state.velocity_at(0, surface_arm), [-0.25, 0.5], atol=1e-7
    )
    batch = jd.State.stack([state, state])
    np.testing.assert_allclose(
        batch.velocity_at(0, surface_arm), [[-0.25, 0.5], [-0.25, 0.5]], atol=1e-7
    )

    system = _system(state)
    history = system.force_model.init_history((), state.dim)
    force, _, _ = _force(state, system, history)
    np.testing.assert_allclose(force[1], -1.25, atol=1e-6)


def test_coulomb_return_mapping_prevents_spring_windup() -> None:
    state = _state(tangent_speed=10.0)
    system = _system(state, mu=0.1)
    history = system.force_model.init_history((), state.dim)

    force, _, returned = _force(state, system, history)
    np.testing.assert_allclose(jnp.abs(force[1]), 1.0, atol=1e-6)
    stopped = replace(state, vel=jnp.zeros_like(state.vel))
    stopped_force, _, _ = _force(stopped, system, returned)
    np.testing.assert_allclose(jnp.abs(stopped_force[1]), 1.0, atol=1e-6)


def test_damped_sliding_return_mapping_stays_on_coulomb_surface() -> None:
    state = _state(tangent_speed=10.0)
    system = _system(state, mu=0.1, restitution=0.5)
    history = system.force_model.init_history((), state.dim)
    force, _, returned = _force(state, system, history)

    normal_force = jnp.abs(force[0])
    np.testing.assert_allclose(jnp.abs(force[1]), 0.1 * normal_force, rtol=1e-6)
    _, _, returned_again = _force(state, system, returned)
    np.testing.assert_allclose(returned_again[:2], returned[:2], atol=1e-6)


def test_separation_resets_only_advancing_history_and_recontact_is_fresh() -> None:
    state = _state()
    system = _system(state)
    _, _, history = _force(
        state, system, system.force_model.init_history((), state.dim)
    )
    separated = replace(state, pos_c=state.pos_c.at[1, 0].set(2.0))

    force_frozen, _, frozen = _force(separated, system, history, advance=False)
    force_reset, _, reset = _force(separated, system, history, advance=True)
    assert jnp.all(force_frozen == 0.0)
    assert jnp.all(force_reset == 0.0)
    np.testing.assert_array_equal(frozen, history)
    np.testing.assert_array_equal(reset, jnp.zeros_like(reset))

    stopped = replace(state, vel=jnp.zeros_like(state.vel))
    recontact_force, _, _ = _force(stopped, system, reset)
    np.testing.assert_allclose(recontact_force[1], 0.0, atol=1e-7)


@pytest.mark.parametrize("dim", [2, 3])
def test_history_follows_rotating_contact_normal(dim: int) -> None:
    state = _state(dim, tangent_speed=0.0)
    system = _system(state)
    pos = jnp.zeros((2, dim)).at[1, 1].set(0.8)
    rotated = replace(state, pos_c=pos)
    xi = jnp.zeros(dim).at[1].set(0.1)
    old_normal = jnp.zeros(dim).at[0].set(-1.0)
    history = jnp.concatenate([xi, old_normal])

    force, _, new_history = _force(rotated, system, history)
    np.testing.assert_allclose(new_history[:dim], old_normal * 0.1, atol=1e-6)
    np.testing.assert_allclose(force[0], 2.5, atol=1e-6)
    np.testing.assert_allclose(force[1], -10.0, atol=5e-6)


def test_vectorized_antiparallel_transport_uses_per_pair_axes() -> None:
    from jaxdem.forces.cundall_strack import _transport_tangent

    old = -jnp.eye(3)
    new = -old
    xi = jnp.roll(jnp.eye(3), shift=1, axis=1)
    transported = _transport_tangent(xi, old, new)
    individual = jnp.stack(
        [_transport_tangent(xi[k], old[k], new[k]) for k in range(3)]
    )
    np.testing.assert_allclose(transported, individual, atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(transported, axis=-1), 1.0)
    np.testing.assert_allclose(jnp.sum(transported * new, axis=-1), 0.0, atol=1e-6)


def test_3d_common_spin_corotates_tangential_spring() -> None:
    state = _state(3, tangent_speed=0.0)
    normal = jnp.array([-1.0, 0.0, 0.0])
    omega = normal * (0.5 * jnp.pi / 0.1)
    state = replace(state, ang_vel=jnp.broadcast_to(omega, (2, 3)))
    system = _system(state)
    history = jnp.array([0.0, 0.1, 0.0, -1.0, 0.0, 0.0])
    _, _, rotated = _force(state, system, history)
    np.testing.assert_allclose(rotated[:3], jnp.array([0.0, 0.0, -0.1]), atol=1e-6)
    np.testing.assert_allclose(jnp.linalg.norm(rotated[:3]), 0.1, atol=1e-6)


def test_directed_pair_forces_are_antisymmetric() -> None:
    state = _state()
    system = _system(state)
    history_ij = jnp.array([0.0, 0.1, 1.0, 0.0])
    history_ji = -history_ij
    force_ij, _, _ = _force(state, system, history_ij, advance=False)
    force_ji, _, _ = system.force_model.force(
        1,
        0,
        state.pos,
        state,
        system,
        history_ji,
        advance_history=False,
    )
    np.testing.assert_allclose(force_ij, -force_ji, atol=1e-6)


def test_readonly_force_returns_history_exactly_unchanged() -> None:
    state = _state()
    system = _system(state)
    history = jnp.array([0.03, 0.1, 0.8, 0.6])
    _, _, returned = _force(state, system, history, advance=False)
    np.testing.assert_array_equal(returned, history)


def test_neighbor_list_initializes_and_advances_contact_history() -> None:
    state = _state()
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=1.0,
        mu=10.0,
        mu_r=0.0,
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(),
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )
    state, system = jd.System.initialize(state, system)
    assert system.collider.history.shape == (2, 2, 4)
    np.testing.assert_array_equal(system.collider.history, 0.0)

    _, stepped = jd.System.step(state, system)
    valid = stepped.collider.neighbor_list >= 0
    assert bool(jnp.any(jnp.abs(stepped.collider.history[valid, :2]) > 0.0))


def test_excluded_cached_pair_does_not_accumulate_hidden_history() -> None:
    state = replace(_state(), clump_id=jnp.zeros(2, dtype=int))
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=1.0,
        mu=10.0,
        mu_r=0.0,
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(),
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )
    _, system = jd.System.initialize(state, system)
    _, system = system.collider.compute_force(state, system)
    np.testing.assert_array_equal(system.collider.history, 0.0)

    enabled = replace(state, clump_id=jnp.arange(2))
    _, system = system.collider.compute_force(enabled, system)
    valid = system.collider.neighbor_list >= 0
    tangential = system.collider.history[..., :2][valid]
    np.testing.assert_allclose(jnp.abs(tangential[:, 1]), 0.1, atol=1e-6)


def test_checkpoint_preserves_history_and_next_step(tmp_path) -> None:
    state = _state()
    material = jd.Material.create(
        "elasticfrict", density=1.0, young=100.0, poisson=0.0, e=1.0, mu=10.0, mu_r=0.0
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(),
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )
    state, system = jd.System.initialize(state, system)
    state, system = jd.System.step(state, system)
    path = tmp_path / "cundall-history"
    with jd.CheckpointWriter(path) as writer:
        writer.save(state, system)
    restored_state, restored_system = jd.CheckpointLoader(path).load()
    assert type(restored_system.force_model) is jd.forces.CundallStrackForce
    np.testing.assert_array_equal(
        restored_system.collider.history, system.collider.history
    )

    expected = jd.System.step(state, system)
    actual = jd.System.step(restored_state, restored_system)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
