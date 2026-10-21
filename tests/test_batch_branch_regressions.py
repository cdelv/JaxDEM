# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Heterogeneous replicas retain independent physics through batch dispatch."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd


def _constant_torque(pos, state, system):
    torque = jnp.zeros_like(state.ang_vel).at[..., 0].set(0.6)
    return jnp.zeros_like(pos), torque


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider", ["naive", "CellList"])
def test_batched_contact_loads_and_energy_keep_replica_and_particle_axes(dim, collider):
    states, systems, predicted_force, predicted_energy = [], [], [], []
    for replica in range(3):
        pos = jnp.zeros((3, dim)).at[:, 0].set(jnp.array([3.8, 0.3, 1.1]))
        pos = pos.at[0, 1].set(3.8).at[1:, 1].set(0.2)
        state = jd.State.create(
            pos=pos, rad=jnp.array([0.2, 0.5, 0.6]), mass=jnp.array([1.0, 2.0, 3.0])
        )
        gravity = jnp.arange(dim, dtype=float) * (replica + 1) / 10
        system = jd.System.create(
            state=state,
            collider_type=collider,
            collider_kw={"cell_size": 1.2} if collider == "CellList" else {},
            domain_type="periodic",
            domain_kw={"box_size": jnp.full(dim, 5.0)},
            force_model=jd.forces.LawCombiner(
                laws=(jd.forces.SpringForce(), jd.forces.SpringForce())
            ),
            force_manager_kw={"gravity": gravity},
        )
        system = replace(
            system,
            mat_table=replace(
                system.mat_table,
                pair={
                    **system.mat_table.pair,
                    "young_eff": system.mat_table.young_eff * (replica + 1),
                },
            ),
        )
        external = jnp.arange(3 * dim, dtype=float).reshape(3, dim) / 100
        system = replace(
            system,
            force_manager=replace(system.force_manager, external_force=external),
        )
        # Only particles 1 and 2 touch: overlap .3 along x, with two spring laws.
        stiffness = float(system.mat_table.young_eff[0, 0])
        contact = np.zeros((3, dim))
        contact[1, 0], contact[2, 0] = -2 * stiffness * 0.3, 2 * stiffness * 0.3
        predicted_force.append(
            contact
            + np.asarray(external)
            + np.asarray(state.mass)[:, None] * np.asarray(gravity)
        )
        predicted_energy.append(stiffness * 0.3**2)
        states.append(state)
        systems.append(system)

    batched, configs = jd.System.initialize(
        jd.State.stack(states), jd.System.stack(systems)
    )
    np.testing.assert_allclose(batched.force, predicted_force, rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(batched.torque, 0.0, atol=1e-9)
    np.testing.assert_array_equal(
        batched.rad, jnp.stack([state.rad for state in states])
    )
    np.testing.assert_array_equal(
        batched.pos, jnp.stack([state.pos for state in states])
    )
    energy = jax.jit(
        jax.vmap(
            lambda state, system: system.collider.compute_potential_energy(
                state, system
            )[2]
        )
    )(batched, configs)
    assert energy.shape == (3,)
    np.testing.assert_allclose(energy, predicted_energy, rtol=3e-6)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "linear,rotation", [("euler", "spiral"), ("verlet", "verletspiral")]
)
def test_batched_rollout_keeps_per_replica_dt_gravity_and_torque(dim, linear, rotation):
    states, systems = [], []
    time_steps = np.array([0.01, 0.025, 0.04])
    inertia = np.array([1.0, 2.0, 3.0])
    initial_pos = np.zeros((3, dim))
    initial_pos[:, 0] = np.arange(3) * 3.0
    initial_vel = np.full((3, dim), 0.2)
    gravity = np.stack(
        [np.arange(1, dim + 1) * (replica + 1) / 10 for replica in range(3)]
    )
    for replica, dt in enumerate(time_steps):
        state = jd.State.create(
            pos=jnp.asarray(initial_pos),
            vel=jnp.asarray(initial_vel),
            inertia=jnp.broadcast_to(
                jnp.asarray(inertia[:, None]), (3, 1 if dim == 2 else 3)
            ),
        )
        system = jd.System.create(
            state=state,
            dt=float(dt),
            # The unused minimizer stays shared static configuration while
            # the physical timestep varies independently between replicas.
            minimizer_kw={"dt": 0.01},
            collider_type="",
            linear_integrator_type=linear,
            rotation_integrator_type=rotation,
            force_manager_kw={
                "gravity": gravity[replica],
                "force_functions": [(_constant_torque, None)],
            },
        )
        states.append(state)
        systems.append(system)
    state, system = jd.System.initialize(
        jd.State.stack(states), jd.System.stack(systems)
    )
    final, config, (trajectory, _) = jd.System.trajectory_rollout(state, system, n=3)
    elapsed = 3 * time_steps
    expected_velocity = (
        initial_vel[None, :, :] + elapsed[:, None, None] * gravity[:, None, :]
    )
    drift_factor = 6 * time_steps**2 if linear == "euler" else 0.5 * elapsed**2
    expected_position = (
        initial_pos[None, :, :]
        + elapsed[:, None, None] * initial_vel[None, :, :]
        + drift_factor[:, None, None] * gravity[:, None, :]
    )
    np.testing.assert_allclose(final.vel, expected_velocity, rtol=3e-6, atol=1e-8)
    np.testing.assert_allclose(final.pos_c, expected_position, rtol=3e-6, atol=1e-8)
    expected_spin = np.zeros((3, 3, 1 if dim == 2 else 3))
    expected_spin[..., 0] = 0.6 * elapsed[:, None] / inertia[None, :]
    np.testing.assert_allclose(final.ang_vel, expected_spin, rtol=3e-6, atol=1e-8)
    angle = 0.5 * (0.6 / inertia[None, :]) * elapsed[:, None] ** 2
    expected_quat = np.zeros((3, 3, 4))
    expected_quat[..., 0] = np.cos(angle / 2)
    expected_quat[..., 3 if dim == 2 else 1] = np.sin(angle / 2)
    actual_quat = np.concatenate((final.q.w, final.q.xyz), axis=-1)
    np.testing.assert_allclose(
        np.abs(np.sum(actual_quat * expected_quat, axis=-1)), 1.0, atol=2e-6
    )
    np.testing.assert_allclose(config.time, elapsed, rtol=3e-6)
    assert trajectory.pos_c.shape == (3, 3, 3, dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_batched_reflection_keeps_per_replica_geometry_and_restitution(dim):
    states, systems, expected = [], [], []
    radii = np.array([0.1, 0.15, 0.2])
    for replica, (dt, restitution) in enumerate(
        zip([0.8, 1.0, 1.2], [0.25, 0.5, 1.0], strict=True)
    ):
        anchor = np.full(dim, replica * 0.2)
        pos = np.ones((3, dim))
        pos[:, 0] = radii + 0.5
        pos[:, 1] = np.arange(1, 4)
        velocity = np.zeros_like(pos)
        velocity[:, 0] = -1
        state = jd.State.create(
            pos=jnp.asarray(pos + anchor),
            vel=jnp.asarray(velocity),
            rad=jnp.asarray(radii),
        )
        system = jd.System.create(
            state=state,
            dt=dt,
            minimizer_kw={"dt": 0.01},
            collider_type="",
            rotation_integrator_type="",
            domain_type="reflect",
            domain_kw={
                "box_size": jnp.full(dim, 5.0),
                "anchor": jnp.asarray(anchor),
                "restitution_coefficient": restitution,
            },
        )
        expected_pos = pos + anchor
        # Every particle first touches x=anchor+radius at t=.5.
        expected_pos[:, 0] = anchor[0] + radii + restitution * (2 * dt - 0.5)
        expected.append(expected_pos)
        states.append(state)
        systems.append(system)
    final, _ = jd.System.step(jd.State.stack(states), jd.System.stack(systems), n=2)
    np.testing.assert_allclose(final.pos, expected, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(
        final.vel[..., 0],
        np.array([0.25, 0.5, 1.0])[:, None] * np.ones((1, 3)),
        rtol=3e-6,
    )
    np.testing.assert_allclose(final.vel[..., 1:], 0.0, atol=1e-9)
