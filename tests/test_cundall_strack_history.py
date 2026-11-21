# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check Cundall-Strack friction memory, contact motion, and return mapping."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd


def quadratic_mean_friction(mu_i: jax.Array, mu_j: jax.Array) -> jax.Array:
    """Importable custom rule used to exercise callable friction mixing."""
    return jnp.sqrt(0.5 * (mu_i * mu_i + mu_j * mu_j))


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


def _coefficient_system(
    state: jd.State,
    *,
    friction_mixing=jd.forces.minimum_friction,
    rolling_friction_mixing=jd.forces.minimum_friction,
    damping: bool = False,
) -> jd.System:
    materials = [
        jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=100.0,
            k_t=20.0,
            b_n=2.0 if damping else 0.0,
            b_t=4.0 if damping else 0.0,
            mu=0.2,
        ),
        jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=300.0,
            k_t=60.0,
            b_n=6.0 if damping else 0.0,
            b_t=12.0 if damping else 0.0,
            mu=0.8,
        ),
    ]
    return jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(
            parameterization="coefficients",
            friction_mixing=friction_mixing,
            rolling_friction_mixing=rolling_friction_mixing,
        ),
        mat_table=jd.MaterialTable.from_materials(materials),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )


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


def _tiny_contact(
    *, tangential_displacement: float
) -> tuple[jd.State, jd.System, jax.Array]:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.0 - 1.0e-9, 0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    material = jd.Material.create(
        "cundallstrackparams",
        density=1.0,
        k_n=1.0,
        k_t=1.0,
        b_n=0.0,
        b_t=0.0,
        mu=1.0,
        mu_r=0.0,
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(parameterization="coefficients"),
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 1,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
    )
    history = jnp.array([0.0, tangential_displacement, -1.0, 0.0])
    return state, system, history


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


@pytest.mark.parametrize("tangential_displacement", [1.0e-10, 2.0e-9])
def test_tiny_coulomb_forces_use_the_true_trial_norm(
    tangential_displacement: float,
) -> None:
    state, system, history = _tiny_contact(
        tangential_displacement=tangential_displacement
    )

    force, _, returned = _force(state, system, history)
    coulomb_limit = abs(float(force[0]))
    expected_tangential = min(tangential_displacement, coulomb_limit)
    np.testing.assert_allclose(force[1], -expected_tangential, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(returned[1], expected_tangential, rtol=1.0e-12, atol=0.0)
    # Read-only snapshot force matches and leaves returned history unchanged.
    snapshot_force, _, frozen = _force(state, system, returned, advance=False)
    np.testing.assert_allclose(snapshot_force, force, rtol=1.0e-12, atol=0.0)
    np.testing.assert_array_equal(frozen, returned)


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


@pytest.mark.parametrize("parameterization", ["elastic", "coefficients"])
def test_checkpoint_preserves_history_parameterization_and_next_step(
    tmp_path, parameterization
) -> None:
    pytest.importorskip("orbax.checkpoint")
    if parameterization == "elastic":
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
    else:
        state = replace(_state(), mat_id=jnp.array([0, 1]))
        system = replace(
            _coefficient_system(
                state,
                friction_mixing=jd.forces.arithmetic_mean_friction,
                rolling_friction_mixing=jd.forces.maximum_friction,
            ),
            dt=jnp.asarray(0.01),
        )
    state, system = jd.System.initialize(state, system)
    state, system = jd.System.step(state, system)
    assert bool(jnp.any(system.collider.history != 0.0))

    path = tmp_path / f"cundall-{parameterization}"
    with jd.CheckpointWriter(path) as writer:
        writer.save(state, system)
    with jd.CheckpointLoader(path) as loader:
        restored_state, restored_system = loader.load()

    assert type(restored_system.force_model) is jd.forces.CundallStrackForce
    assert restored_system.force_model.parameterization == parameterization
    if parameterization == "coefficients":
        assert (
            restored_system.force_model.friction_mixing
            is jd.forces.arithmetic_mean_friction
        )
        assert (
            restored_system.force_model.rolling_friction_mixing
            is jd.forces.maximum_friction
        )
    np.testing.assert_array_equal(
        restored_system.collider.history, system.collider.history
    )
    expected = jd.System.step(state, system)
    actual = jd.System.step(restored_state, restored_system)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_direct_coefficients_use_harmonic_particle_mixing() -> None:
    state = replace(
        _state(tangent_speed=0.0),
        mat_id=jnp.array([0, 1]),
        vel=jnp.array([[-1.0, 1.0], [0.0, 0.0]]),
    )
    system = _coefficient_system(
        state, friction_mixing=jd.forces.maximum_friction, damping=True
    )
    history = system.force_model.init_history((), state.dim)
    force, _, _ = system.force_model.force(
        0, 1, state.pos, state, system, history, advance_history=False
    )

    # Harmonic means: kn=150, kt=30, bn=3, bt=6. With overlap 0.2,
    # separating speed 1 and tangential speed 1, F=(-27, -6).
    np.testing.assert_allclose(force, jnp.array([-27.0, -6.0]), atol=1e-6)


def test_coefficient_force_accepts_custom_friction_mixing_callable() -> None:
    state = replace(
        _state(tangent_speed=0.0),
        mat_id=jnp.array([0, 1]),
    )
    system = _coefficient_system(state, friction_mixing=quadratic_mean_friction)
    # kn=150 and overlap=0.2 give Fn=30. A unit tangential spring gives an
    # uncapped trial magnitude of kt=30, so the mixed mu sets the result.
    history = jnp.array([0.0, 1.0, -1.0, 0.0])
    force, _, _ = system.force_model.force(
        0, 1, state.pos, state, system, history, advance_history=False
    )
    expected_mu = np.sqrt(0.5 * (0.2**2 + 0.8**2))
    np.testing.assert_allclose(force, [-30.0, -30.0 * expected_mu], atol=1e-6)


@pytest.mark.parametrize("mode", ["elastic", "coefficients", "combiner", "router"])
def test_neighbor_energy_includes_stored_tangential_spring(mode) -> None:
    if mode == "coefficients":
        state = replace(_state(tangent_speed=0.0), mat_id=jnp.array([0, 1]))
        system = _coefficient_system(state)
        expected = 0.5 * 150.0 * 0.2**2 + 0.5 * 30.0 * 0.1**2
    else:
        state = _state(tangent_speed=0.0)
        material = jd.Material.create(
            "elasticfrict", density=1.0, young=100.0, poisson=0.0, e=1.0, mu=10.0
        )
        law = jd.forces.CundallStrackForce()
        if mode == "combiner":
            law = jd.LawCombiner(laws=(law,))
        elif mode == "router":
            law = jd.ForceRouter.from_dict(1, {(0, 0): law})
        system = jd.System.create(
            state=state,
            force_model=law,
            mat_table=jd.MaterialTable.from_materials([material]),
            dt=0.1,
            collider_type="NeighborList",
            collider_kw={
                "max_neighbors": 2,
                "cutoff": 1.0,
                "secondary_collider_type": "naive",
            },
        )
        expected = 0.5 * 50.0 * 0.2**2 + 0.5 * 25.0 * 0.1**2
    state, system = jd.System.initialize(state, system)
    valid = system.collider.neighbor_list >= 0
    history = system.collider.history.at[valid, 1].set(0.1)
    system = replace(system, collider=replace(system.collider, history=history))
    _, _, energy = system.collider.compute_potential_energy(state, system)
    np.testing.assert_allclose(energy, expected)


def test_advance_history_clears_lost_contacts() -> None:
    state = replace(
        _state(tangent_speed=0.0),
        mat_id=jnp.array([0, 1]),
    )
    system = _coefficient_system(state)
    state, system = jd.System.initialize(state, system)
    valid = system.collider.neighbor_list >= 0
    history = system.collider.history.at[valid, 1].set(0.1)
    system = replace(system, collider=replace(system.collider, history=history))
    separated = replace(state, pos_c=state.pos_c.at[1, 0].set(1.01))

    _, advanced = jd.colliders.advance_history(state, system, separated, system)

    np.testing.assert_array_equal(advanced.collider.history, 0.0)


def test_advance_history_does_not_depend_on_dt_or_damping() -> None:
    state = replace(_state(tangent_speed=0.0), mat_id=jnp.array([0, 1]))
    results = []
    for damping in (False, True):
        system = _coefficient_system(state, damping=damping)
        initialized, system = jd.System.initialize(state, system)
        moved = replace(initialized, pos_c=initialized.pos_c.at[0, 1].add(0.05))
        for dt in (1e-3, 1.0):
            timed = replace(system, dt=jnp.asarray(dt))
            moved_state, advanced = jd.colliders.advance_history(
                initialized, timed, moved, timed
            )
            valid = advanced.collider.neighbor_list >= 0
            assert bool(jnp.any(advanced.collider.history[valid, 1] != 0.0))
            results.append((moved_state.force, advanced.collider.history))
    for force, history in results[1:]:
        np.testing.assert_allclose(force, results[0][0], rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(history, results[0][1], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "optimizer,parameterization,history_update",
    [
        (jd.minimizers.fire, "elastic", "path"),
        (jd.minimizers.fire, "coefficients", "endpoint"),
        (jd.minimizers.damped_newtonian, "coefficients", "frozen"),
    ],
)
def test_force_minimizers_advance_contact_history(
    optimizer, parameterization, history_update
) -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0], [0.0, 0.85]]),
        rad=jnp.full(3, 0.5),
        mass=jnp.ones(3),
        fixed=jnp.array([False, True, True]),
    )
    if parameterization == "elastic":
        material = jd.Material.create(
            "elasticfrict",
            density=1.0,
            young=100.0,
            poisson=0.0,
            e=0.9,
            mu=10.0,
        )
    else:
        material = jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=100.0,
            k_t=20.0,
            b_n=0.5,
            b_t=0.25,
            mu=10.0,
        )
    system = jd.System.create(
        state=state,
        dt=1.0e-3,
        force_model=jd.forces.CundallStrackForce(parameterization=parameterization),
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
        minimizer=optimizer,
        minimizer_kw={"dt": 1.0e-3},
    )

    _, system = jd.System.evaluate_forces(state, system)
    col = system.collider
    initial = np.asarray(col.history).copy()
    sources = np.repeat(np.arange(state.N), np.diff(np.asarray(col.row_offsets)))
    targets = np.asarray(col.neighbor_list)[: len(sources)]
    old_normals = np.asarray(state.pos)[sources] - np.asarray(state.pos)[targets]
    old_normals /= np.linalg.norm(old_normals, axis=-1, keepdims=True)
    tangents = np.stack([-old_normals[:, 1], old_normals[:, 0]], axis=-1)
    initial[: len(sources), :2] = 0.002 * tangents
    initial[: len(sources), 2:] = old_normals
    system = replace(system, collider=replace(col, history=jnp.asarray(initial)))
    result = system.minimize(
        state,
        system,
        max_steps=1,
        force_tol=-1.0,
        torque_tol=-1.0,
        history_update=history_update,
    )

    valid = result.system.collider.neighbor_list >= 0
    tangential = result.system.collider.history[valid, : state.dim]
    assert int(result.steps) == 1
    # After one update, both advancing options take the same single step.
    if history_update != "frozen":
        pos = np.asarray(result.state.pos)
        normal = pos[sources] - pos[targets]
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        tangent = np.stack([-normal[:, 1], normal[:, 0]], axis=-1)
        translation = np.asarray(result.state.pos_c - state.pos_c)
        angle = 2 * np.arctan2(
            np.asarray(result.state.q.xyz)[:, 2], np.asarray(result.state.q.w)[:, 0]
        )
        motion = translation[sources] - translation[targets]
        motion -= (
            np.asarray(state.rad)[sources] * angle[sources]
            + np.asarray(state.rad)[targets] * angle[targets]
        )[:, None] * tangent
        expected = (
            0.002 * tangent
            + motion
            - np.sum(motion * normal, axis=-1, keepdims=True) * normal
        )
        np.testing.assert_allclose(tangential, expected, rtol=2e-6, atol=1e-10)
        np.testing.assert_allclose(
            result.system.collider.history[valid, 2:], normal, atol=2e-6
        )
    else:
        np.testing.assert_array_equal(result.system.collider.history, initial)
    assert bool(jnp.isfinite(result.energy))


def _pair_history(collider, n: int) -> dict[tuple[int, int], np.ndarray]:
    sources = np.repeat(np.arange(n), np.diff(np.asarray(collider.row_offsets)))
    targets = np.asarray(collider.neighbor_list)[: len(sources)]
    history = np.asarray(collider.history)[: len(sources)]
    return {
        (int(i), int(j)): history[k]
        for k, (i, j) in enumerate(zip(sources, targets, strict=True))
        if j >= 0
    }


def test_endpoint_history_update_is_one_step_from_the_initial_configuration() -> None:
    optimizer = jd.minimizers.fire
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0], [0.0, 0.85]]),
        rad=jnp.full(3, 0.5),
        mass=jnp.ones(3),
        fixed=jnp.array([False, True, True]),
    )
    material = jd.Material.create(
        "cundallstrackparams",
        density=1.0,
        k_n=100.0,
        k_t=20.0,
        b_n=0.5,
        b_t=0.25,
        mu=10.0,
    )
    system = jd.System.create(
        state=state,
        dt=1.0e-3,
        force_model=jd.forces.CundallStrackForce(parameterization="coefficients"),
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "secondary_collider_type": "naive",
        },
        minimizer=optimizer,
        minimizer_kw={"dt": 3.0e-3},
    )
    _, system = jd.System.evaluate_forces(state, system)
    col = system.collider
    initial = np.asarray(col.history).copy()
    sources = np.repeat(np.arange(state.N), np.diff(np.asarray(col.row_offsets)))
    targets = np.asarray(col.neighbor_list)[: len(sources)]
    normals = np.asarray(state.pos)[sources] - np.asarray(state.pos)[targets]
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    initial[: len(sources), :2] = 0.02 * np.stack(
        [-normals[:, 1], normals[:, 0]], axis=-1
    )
    initial[: len(sources), 2:] = normals
    system = replace(system, collider=replace(col, history=jnp.asarray(initial)))

    results = {
        mode: system.minimize(
            state,
            system,
            max_steps=40,
            force_tol=-1.0,
            torque_tol=-1.0,
            history_update=mode,
        )
        for mode in ("path", "endpoint")
    }
    result = results["endpoint"]
    expected_state, expected_system = jd.colliders.advance_history(
        state, system, result.state, system
    )

    # The relaxation crosses several rebuilds while both contacts persist.
    assert int(result.system.collider.n_build_times) > int(col.n_build_times) + 2
    expected = _pair_history(expected_system.collider, state.N)
    assert set(expected) == {(0, 1), (1, 0), (0, 2), (2, 0)}
    assert all(np.abs(h[:2]).max() > 0.05 for h in expected.values())

    actual = _pair_history(result.system.collider, state.N)
    assert set(actual) == set(expected)
    for pair, history in expected.items():
        np.testing.assert_allclose(actual[pair], history, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(
        result.state.force, expected_state.force, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        result.state.torque, expected_state.torque, rtol=1e-6, atol=1e-8
    )

    # Accumulating along the optimizer path rotates the spring step by step.
    path = _pair_history(results["path"].system.collider, state.N)
    assert max(np.abs(path[p] - expected[p]).max() for p in expected) > 1e-3


def _scripted_history_optimizer(path, rotations):
    """Force a known optimizer path independently of the contact forces."""
    from jaxdem.minimizers.optimizers import CustomGradientTransformation

    path_key = tuple(tuple(row) for row in path)
    rotation_key = tuple(rotations)
    translations = jnp.asarray(path_key)
    angles = jnp.asarray(rotation_key)

    def init(params):
        return jnp.asarray(0)

    def update(grads, step, params=None, **kwargs):
        translation = jnp.zeros_like(params["pos_c"]).at[0].set(translations[step])
        # 3D optimizer coordinates are raw quaternions with initial w = 2;
        # choose z so their normalized orientation has the requested angle.
        angle = angles[step]
        coordinate = (
            2.0 * jnp.tan(angle / 2.0)
            if params["rotation"].shape[-1] == 4
            else angle
        )
        rotation = jnp.zeros_like(params["rotation"]).at[0, -1].set(coordinate)
        return {"pos_c": translation, "rotation": rotation}, step + 1

    return CustomGradientTransformation(
        init,
        update,
        _scripted_history_optimizer,
        {"path": path_key, "rotations": rotation_key},
        requires_value=False,
    )


def _history_regression_system(backend, separation, dim=2):
    state = jd.State.create(
        pos=jnp.zeros((2, dim)).at[1, 0].set(separation),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
        fixed=jnp.array([False, True]),
    )
    material = jd.Material.create(
        "cundallstrackparams",
        density=1.0,
        k_n=1.0,
        k_t=1.0,
        b_n=0.0,
        b_t=0.0,
        mu=10.0,
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(parameterization="coefficients"),
        mat_table=jd.MaterialTable.from_materials([material]),
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0)},
        collider_type=backend,
        collider_kw=(
            {} if backend == "naive" else {
                "max_neighbors": 2,
                "cutoff": 1.0,
                "skin": 0.05,
                "secondary_collider_type": "naive",
            }
        ),
    )
    _, system = jd.System.evaluate_forces(state, system)
    return state, system


@pytest.mark.parametrize("backend", ["naive", "NeighborList"])
@pytest.mark.parametrize("history_update", ["endpoint", "path"])
def test_endpoint_history_recovers_pair_that_left_the_trial_cache(backend, history_update):
    from jaxdem.colliders._neighbor_cache import pair_sources

    state, system = _history_regression_system(backend, 0.8)
    col = system.collider
    if backend == "naive":
        history = col.history.at[0, 1].set(jnp.array([0.0, 0.02, -1.0, 0.0]))
        history = history.at[1, 0].set(jnp.array([0.0, -0.02, 1.0, 0.0]))
    else:
        source = pair_sources(col)
        sign = jnp.where(source == 0, 1.0, -1.0)
        values = jnp.stack([0.0 * sign, 0.02 * sign, -sign, 0.0 * sign], axis=-1)
        history = jnp.where((col.neighbor_list >= 0)[:, None], values, 0.0)
    system = replace(
        system,
        collider=replace(col, history=history),
        minimizer=_scripted_history_optimizer([[-1.0, 0.0], [1.0, 0.0]], [0.0, 0.0]),
    )
    result = jd.minimize(
        state, system, max_steps=2, force_tol=-1.0, torque_tol=-1.0,
        history_update=history_update,
    )
    if backend == "NeighborList":
        assert int(result.system.collider.n_build_times - col.n_build_times) == 2
    np.testing.assert_allclose(result.state.pos_c, state.pos_c, atol=1e-12)
    _, _, contacts = jd.forces.get_cundall_strack_contacts(result.state, result.system)
    # A numerical excursion cannot erase initial history. Incremental history
    # intentionally forgets the contact when the disks physically separate.
    expected = 0.02 if history_update == "endpoint" else 0.0
    np.testing.assert_allclose(
        contacts.tangential_forces, [[0.0, -expected], [0.0, expected]], atol=1e-12
    )
    np.testing.assert_allclose(result.state.force[0, 1], -expected, atol=1e-12)


@pytest.mark.parametrize(
    "backend,dim,history_update",
    [("naive", 2, "endpoint"), ("NeighborList", 3, "path")],
)
def test_quasistatic_new_contact_counts_only_rotation_after_touch(
    backend, dim, history_update
):
    state, system = _history_regression_system(backend, 1.2, dim)
    path = [[0.4] + [0.0] * (dim - 1)]
    system = replace(system, minimizer=_scripted_history_optimizer(path, [0.4]))
    result = jd.minimize(
        state, system, max_steps=1, force_tol=-1.0, torque_tol=-1.0,
        history_update=history_update,
    )
    _, _, contacts = jd.forces.get_cundall_strack_contacts(result.state, result.system)
    # First touch is halfway through the approach (1.2 -> 0.8). Of the 0.4
    # radians, only 0.2 occur in contact: |xi| = R * 0.2 = 0.1, with kt = 1.
    np.testing.assert_allclose(
        np.linalg.norm(contacts.tangential_forces, axis=-1), 0.1, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(result.state.torque, axis=-1), 0.05, atol=1e-12
    )


@pytest.mark.parametrize(
    "backend,shrink_box",
    [("naive", True), ("NeighborList", False), ("NeighborList", True)],
)
def test_quasistatic_contact_birth_uses_periodic_image_motion(backend, shrink_box):
    from jaxdem.utils.quaternion import Quaternion

    state, system = _history_regression_system(backend, 8.8)
    rotation = jnp.zeros((2, 3)).at[0, 2].set(0.4)
    moved = replace(state, q=Quaternion.from_rotvec(rotation))
    if shrink_box:
        changed = replace(
            system, domain=replace(system.domain, box_size=jnp.full(2, 9.6))
        )
    else:
        moved = replace(moved, pos_c=moved.pos_c.at[0, 0].set(-0.4))
        changed = system
    advanced, changed = jd.colliders.advance_history(state, system, moved, changed)
    _, _, contacts = jd.forces.get_cundall_strack_contacts(advanced, changed)
    # The image separation goes from 1.2 to 0.8 in both cases. Motion of the
    # particle and of its periodic image must give the same halfway birth.
    np.testing.assert_allclose(
        np.linalg.norm(contacts.tangential_forces, axis=-1), 0.1, atol=1e-12
    )


def test_native_bisection_jams_cundall_strack() -> None:
    pos = (
        jnp.stack(
            jnp.meshgrid(jnp.arange(3), jnp.arange(3), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        * 1.1
    )
    state = jd.State.create(
        pos=pos,
        rad=jnp.full(9, 0.5),
        mass=jnp.ones(9),
    )
    material = jd.Material.create(
        "cundallstrackparams",
        density=1.0,
        k_n=1.0,
        k_t=0.5,
        b_n=0.0,
        b_t=0.0,
        mu=0.5,
    )
    system = jd.System.create(
        state=state,
        dt=1.0e-2,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 3.3)},
        force_model=jd.forces.CundallStrackForce(parameterization="coefficients"),
        mat_table=jd.MaterialTable.from_materials([material]),
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "max_neighbors": 8,
            "secondary_collider_type": "naive",
        },
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-2},
    )

    result = jd.utils.jamming.bisection_jam(
        state,
        system,
        n_jamming_steps=40,
        pe_tol=1.0e-8,
        packing_fraction_increment=0.03,
        packing_fraction_tolerance=1.0e-6,
        minimizer_kwargs={
            "max_steps": 100,
            "force_tol": 1.0e-8,
            "torque_tol": 1.0e-8,
        },
        verbose=False,
    )

    assert result.converged
    assert result.potential_energy > 1.0e-8
    assert result.info.minimization.converged


def test_coefficient_force_accepts_rolling_friction_mixing_callable() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        ang_vel=jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
        mat_id=jnp.array([0, 1]),
    )
    materials = [
        jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=100.0,
            k_t=20.0,
            b_n=0.0,
            b_t=0.0,
            mu=0.0,
            mu_r=0.2,
        ),
        jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=100.0,
            k_t=20.0,
            b_n=0.0,
            b_t=0.0,
            mu=0.0,
            mu_r=0.8,
        ),
    ]
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce(
            parameterization="coefficients",
            rolling_friction_mixing=jd.forces.maximum_friction,
        ),
        mat_table=jd.MaterialTable.from_materials(materials),
        dt=0.1,
        collider_type="NeighborList",
        collider_kw={"max_neighbors": 1, "cutoff": 1.0},
    )
    history = system.force_model.init_history((), state.dim)

    _, torque, _ = system.force_model.force(
        0, 1, state.pos, state, system, history, advance_history=False
    )

    np.testing.assert_allclose(torque, [-4.0, 0.0, 0.0], atol=1e-6)


def test_near_coincident_cundall_contact_uses_unit_normal() -> None:
    separation = np.array([3.0e-10, 4.0e-10])
    distance = 5.0e-10
    normal = np.array([-0.6, -0.8])
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], separation]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
    )
    material = jd.Material.create(
        "cundallstrackparams",
        density=1.0,
        k_n=10.0,
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
        collider_type="naive",
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
    np.testing.assert_allclose(
        force, 10.0 * (1.0 - distance) * normal, rtol=2e-6, atol=1e-6
    )


def _resolved_contacts_system(
    tangential_displacement: float,
) -> tuple[jd.State, jd.System, jnp.ndarray]:
    from jaxdem.colliders._neighbor_cache import pair_sources

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
    valid = collider.neighbor_list >= 0
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
    displacement: float, tangential_magnitude: float, mobilized: bool
) -> None:
    from jaxdem.colliders._neighbor_cache import pair_sources

    state, system, original_history = _resolved_contacts_system(displacement)
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
        slot_history = returned_system.collider.history[
            (pair_sources(returned_system.collider) == i)
            & (returned_system.collider.neighbor_list == j)
        ][0]
        force, torque, unchanged = system.force_model.force(
            i, j, state.pos, state, returned_system, slot_history, advance_history=False
        )
        np.testing.assert_allclose(
            force, contacts.normal_forces[row] + contacts.tangential_forces[row], atol=1e-6
        )
        np.testing.assert_allclose(torque, contacts.torques[row], atol=1e-6)
        np.testing.assert_array_equal(unchanged, slot_history)


def test_resolved_contacts_convert_to_generic_contact_data() -> None:
    state, system, _ = _resolved_contacts_system(0.1)
    state = replace(state, pos_p=jnp.zeros_like(state.pos_p).at[:2, 1].set(0.2))

    _, system, expected = jd.utils.get_contacts(state, system)
    _, system, resolved = jd.forces.get_cundall_strack_contacts(state, system)
    history = np.asarray(system.collider.history).copy()

    contacts = resolved.as_contact_data(state, system)
    np.testing.assert_array_equal(contacts.pair_ids, expected.pair_ids)
    np.testing.assert_allclose(contacts.forces, expected.forces, atol=1e-6)
    np.testing.assert_allclose(contacts.torques, expected.torques, atol=1e-6)
    np.testing.assert_allclose(
        contacts.displacements, expected.displacements, atol=1e-6
    )
    assert np.max(np.abs(np.asarray(contacts.torques - resolved.torques))) > 0.0
    np.testing.assert_array_equal(system.collider.history, history)

    _, _, counts = jd.utils.count_sphere_contacts(state, system, contacts=contacts)
    _, _, rattlers, non_rattlers = jd.utils.get_sphere_rattler_ids(
        state, system, zc=1, contacts=contacts
    )
    np.testing.assert_array_equal(counts, [1, 1, 0])
    np.testing.assert_array_equal(rattlers, [2])
    np.testing.assert_array_equal(non_rattlers, [0, 1])
