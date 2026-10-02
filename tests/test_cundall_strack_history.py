from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
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
    assert system.collider.history.shape == (4, 4)
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


def test_coefficient_neighbor_energy_includes_stored_tangential_spring() -> None:
    state = replace(
        _state(tangent_speed=0.0),
        mat_id=jnp.array([0, 1]),
    )
    system = _coefficient_system(state)
    state, system = jd.System.initialize(state, system)
    valid = system.collider.neighbor_list >= 0
    history = system.collider.history.at[valid, 1].set(0.1)
    system = replace(system, collider=replace(system.collider, history=history))

    _, _, energy = system.collider.compute_potential_energy(state, system)

    # Harmonic means kn=150 and kt=30, with overlap=0.2 and |xi|=0.1.
    np.testing.assert_allclose(energy, 0.5 * 150.0 * 0.2**2 + 0.5 * 30.0 * 0.1**2)


def test_elastic_neighbor_energy_includes_stored_tangential_spring() -> None:
    state = _state(tangent_speed=0.0)
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=1.0,
        mu=10.0,
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
    valid = system.collider.neighbor_list >= 0
    history = system.collider.history.at[valid, 1].set(0.1)
    system = replace(system, collider=replace(system.collider, history=history))

    _, _, energy = system.collider.compute_potential_energy(state, system)

    np.testing.assert_allclose(
        energy, 0.5 * 50.0 * 0.2**2 + 0.5 * 25.0 * 0.1**2
    )


@pytest.mark.parametrize("composition", ["combiner", "router"])
def test_composed_cundall_strack_uses_history_energy(composition) -> None:
    state = _state(tangent_speed=0.0)
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=1.0,
        mu=10.0,
    )
    law = jd.forces.CundallStrackForce()
    if composition == "combiner":
        force_model = jd.LawCombiner(laws=(law,))
    else:
        force_model = jd.ForceRouter.from_dict(1, {(0, 0): law})
    system = jd.System.create(
        state=state,
        force_model=force_model,
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
    valid = system.collider.neighbor_list >= 0
    history = system.collider.history.at[valid, 1].set(0.1)
    system = replace(system, collider=replace(system.collider, history=history))

    _, _, energy = system.collider.compute_potential_energy(state, system)

    assert system.force_model.has_history_dependent_energy
    np.testing.assert_allclose(
        energy, 0.5 * 50.0 * 0.2**2 + 0.5 * 25.0 * 0.1**2
    )


def test_coefficient_relaxation_clears_lost_contact_history_without_steps() -> None:
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

    result = system.minimize(separated, system, max_steps=0)

    valid = result.system.collider.neighbor_list >= 0
    np.testing.assert_array_equal(
        result.system.collider.history[valid], 0.0
    )


@pytest.mark.parametrize(
    "optimizer", [jd.minimizers.fire, jd.minimizers.damped_newtonian]
)
@pytest.mark.parametrize("parameterization", ["elastic", "coefficients"])
def test_force_minimizers_advance_contact_history(optimizer, parameterization) -> None:
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

    result = system.minimize(
        state, system, max_steps=1, force_tol=-1.0, torque_tol=-1.0
    )

    valid = result.system.collider.neighbor_list >= 0
    tangential = result.system.collider.history[valid, : state.dim]
    assert int(result.steps) == 1
    assert bool(jnp.any(jnp.abs(tangential) > 0.0))
    assert bool(jnp.isfinite(result.energy))


def test_history_minimizer_uses_one_force_traversal_per_evaluation(
    monkeypatch,
) -> None:
    from jaxdem.minimizers import routines

    state = _state(tangent_speed=0.0)
    system = replace(
        _coefficient_system(state),
        minimizer=jd.minimizers.fire(dt=1.0e-3),
    )
    collider_type = type(system.collider)
    original = collider_type.compute_force
    evaluations = []

    def counted_force(state, system, **kwargs):
        state, system = original(state, system, **kwargs)
        jax.debug.callback(
            lambda _: evaluations.append(None), state.force, ordered=True
        )
        return state, system

    routines.minimize.clear_cache()
    monkeypatch.setattr(collider_type, "compute_force", staticmethod(counted_force))
    try:
        result = routines.minimize(
            state,
            system,
            max_steps=2,
            force_tol=-1.0,
            torque_tol=-1.0,
        )
        jax.block_until_ready(result)
        jax.effects_barrier()
        assert int(result.steps) == 2
        assert len(evaluations) == 3
    finally:
        routines.minimize.clear_cache()


@pytest.mark.parametrize("parameterization", ["elastic", "coefficients"])
def test_cundall_strack_has_history_dependent_energy(parameterization) -> None:
    law = jd.forces.CundallStrackForce(parameterization=parameterization)
    assert law.has_history_dependent_energy


def test_cundall_strack_parameterization_contract() -> None:
    elastic = jd.forces.CundallStrackForce()
    coefficients = jd.forces.CundallStrackForce(parameterization="coefficients")
    assert elastic.required_material_properties == (
        "young",
        "poisson",
        "e",
        "mu",
        "mu_r",
    )
    assert coefficients.required_material_properties == (
        "k_n",
        "k_t",
        "b_n",
        "b_t",
        "mu",
        "mu_r",
    )
    with pytest.raises(ValueError, match="parameterization"):
        jd.forces.CundallStrackForce(parameterization="invalid")


@pytest.mark.parametrize("parameterization", ["elastic", "coefficients"])
def test_native_bisection_jams_cundall_strack(parameterization) -> None:
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
    if parameterization == "elastic":
        material = jd.Material.create(
            "elasticfrict",
            density=1.0,
            young=2.0,
            poisson=0.0,
            e=1.0,
            mu=0.5,
        )
    else:
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
        force_model=jd.forces.CundallStrackForce(parameterization=parameterization),
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
        n_minimization_steps=100,
        n_jamming_steps=40,
        pe_tol=1.0e-8,
        packing_fraction_increment=0.03,
        packing_fraction_tolerance=1.0e-6,
        force_tol=1.0e-8,
        torque_tol=1.0e-8,
        verbose=False,
    )

    assert result.converged
    assert result.potential_energy > 1.0e-8
    assert result.info.minimization.converged


def test_coefficient_zero_mu_r_has_zero_rolling_torque() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        ang_vel=jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.ones(2),
        mat_id=jnp.array([0, 1]),
    )
    system = _coefficient_system(state)
    history = system.force_model.init_history((), state.dim)
    _, torque, _ = system.force_model.force(
        0, 1, state.pos, state, system, history, advance_history=False
    )
    np.testing.assert_allclose(torque, 0.0, atol=1e-7)


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


def test_coefficient_checkpoint_preserves_parameterization_and_mixing(tmp_path) -> None:
    state = replace(_state(), mat_id=jnp.array([0, 1]))
    system = _coefficient_system(
        state,
        friction_mixing=jd.forces.arithmetic_mean_friction,
        rolling_friction_mixing=jd.forces.maximum_friction,
    )
    state, system = jd.System.initialize(state, system)
    path = tmp_path / "coefficient-cundall-history"
    with jd.CheckpointWriter(path) as writer:
        writer.save(state, system)

    restored_state, restored_system = jd.CheckpointLoader(path).load()
    assert isinstance(restored_system.force_model, jd.forces.CundallStrackForce)
    assert restored_system.force_model.parameterization == "coefficients"
    assert (
        restored_system.force_model.friction_mixing
        is jd.forces.arithmetic_mean_friction
    )
    assert (
        restored_system.force_model.rolling_friction_mixing
        is jd.forces.maximum_friction
    )
    expected = jd.System.step(state, system)
    actual = jd.System.step(restored_state, restored_system)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
