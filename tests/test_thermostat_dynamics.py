# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Exact free-particle thermostat transitions, including small reduced units."""

import jax
import jax.numpy as jnp

import dataclasses
from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.thermal import (
    compute_potential_energy,
    compute_temperature,
    compute_translational_kinetic_energy,
    compute_translational_kinetic_energy_per_particle,
    scale_to_temperature,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(slots=True)
class _TerminalRotationalKick(jd.RotationIntegrator):
    @staticmethod
    def step_after_force(state: jd.State, system: jd.System):
        state.ang_vel = jnp.full_like(state.ang_vel, 2.0)
        return state, system


@pytest.mark.parametrize(
    "dim,x64,gamma_dt",
    [(2, False, 0.0), (2, False, 1e-8), (3, True, 1e-8), (3, False, 0.4)],
)
def test_langevin_matches_exact_ou_transition(dim, x64, gamma_dt):
    with jax.enable_x64(x64):
        velocity = np.zeros((4, dim))
        velocity[2:, 0] = [0.3, -0.2]
        state = jd.State.create(
            pos=jnp.ones((4, dim)),
            vel=jnp.asarray(velocity),
            mass=jnp.array([2.0, 2.0, 3.0, 4.0]),
            clump_id=jnp.array([0, 0, 1, 2]),
            fixed=jnp.array([False, False, False, True]),
        )
        system = jd.System.create(
            state=state,
            seed=42,
            dt=0.2,
            collider_type="",
            rotation_integrator_type="",
            linear_integrator_type="langevin",
            linear_integrator_kw={
                "gamma": gamma_dt / 0.2,
                "temperature": 0.7,
                "k_B": 1.3,
            },
        )
        next_key, noise_key = jax.random.split(system.key)
        noise = np.asarray(
            jax.random.normal(noise_key, state.vel.shape, dtype=state.vel.dtype)
        )[np.asarray(state.clump_id)]
        expected = (
            np.exp(-gamma_dt) * velocity
            + np.sqrt(
                0.91 / np.asarray(state.mass, dtype=float) * -np.expm1(-2 * gamma_dt)
            )[:, None]
            * noise
        )
        expected[-1] = velocity[-1]
        advanced, updated = jd.System.step(state, system)
        np.testing.assert_allclose(advanced.vel, expected, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(
            advanced.pos_c,
            np.asarray(state.pos_c) + 0.1 * (velocity + expected),
            rtol=2e-6,
            atol=2e-7,
        )
        np.testing.assert_array_equal(advanced.vel[0], advanced.vel[1])
        np.testing.assert_array_equal(updated.key, next_key)


@pytest.mark.parametrize(
    "speed,temperature", [(0.0, 2.0), (1e-16, 2.0), (1.0, 2.0), (1.0, 0.0)]
)
def test_rescaling_reaches_target_without_a_temperature_floor(speed, temperature):
    state = jd.State.create(
        pos=jnp.ones((3, 2)),
        vel=jnp.array([[speed, 0.0], [-speed, 0.0], [0.7, 0.0]]),
        mass=jnp.array([1.0, 3.0, 1.0]),
        fixed=jnp.array([False, False, True]),
    )
    system = jd.System.create(
        state=state,
        collider_type="",
        rotation_integrator_type="",
        linear_integrator_type="verlet_rescaling",
        linear_integrator_kw={"temperature": temperature},
    )
    advanced, _ = jd.System.step(state, system)
    target_speed = np.sqrt(temperature) if speed > 0 else 0.0
    np.testing.assert_allclose(
        advanced.vel, [[target_speed, 0], [-target_speed, 0], [0.7, 0]], atol=1e-7
    )
    np.testing.assert_allclose(
        jd.utils.compute_temperature(advanced, False, False),
        temperature if speed > 0 else 0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("utility", [False, True])
@pytest.mark.parametrize("target", [0.0, 2.0])
def test_temperature_scaling_has_the_analytical_velocity_jacobian(utility, target):
    with jax.enable_x64(False):
        state = jd.State.create(
            pos=jnp.ones((3, 2)),
            mass=jnp.array([1.0, 3.0, 1.0]),
            fixed=jnp.array([False, False, True]),
        )
        system = jd.System.create(
            state=state,
            collider_type="",
            linear_integrator_type="verlet_rescaling",
            linear_integrator_kw={"temperature": target},
        )

        def rescale(velocity):
            trial = replace(state, vel=velocity)
            if utility:
                return jd.utils.scale_to_temperature(trial, target, False, False).vel
            return system.linear_integrator.step_after_force(trial, system)[0].vel

        for speed in (0.0, 1e-16, 1.0):
            velocity = jnp.array([[speed, 0.0], [-speed, 0.0], [0.7, 0.0]])
            jacobian = np.asarray(jax.jacrev(rescale)(velocity)).reshape(6, 6)
            assert np.isfinite(jacobian).all()
            np.testing.assert_allclose(jacobian[4:, 4:], np.eye(2))
            np.testing.assert_allclose(jacobian[:4, 4:], 0.0)
            if target == 0:
                np.testing.assert_array_equal(jacobian[:4], 0.0)
            elif speed > 0:
                expected = (
                    np.eye(4)
                    - np.outer([1.0, 0.0, -1.0, 0.0], [1.0, 0.0, -3.0, 0.0]) / 4
                )
                np.testing.assert_allclose(
                    jacobian[:4, :4] * (speed / np.sqrt(target)),
                    expected,
                    rtol=3e-5,
                    atol=2e-6,
                )


def test_degree_counts_follow_logical_bodies_and_available_drift():
    from jaxdem.utils.thermal import count_dynamic_dofs

    cases = [
        ([], 0),
        ([True, True, True], 0),
        ([False, False, True], 1),
        ([False, False, False], 2),
    ]
    for dim in (2, 3):
        for fixed, free_bodies in cases:
            state = jd.State.create(
                pos=jnp.zeros((len(fixed), dim)),
                fixed=jnp.asarray(fixed, dtype=bool),
                clump_id=jnp.asarray([0, 0, 1][: len(fixed)], dtype=int),
            )
            for rotate in (False, True):
                for drift in (False, True):
                    translation = (free_bodies - int(drift and free_bodies > 0)) * dim
                    rotation = free_bodies * dim * (dim - 1) // 2 * rotate
                    np.testing.assert_array_equal(
                        count_dynamic_dofs(state, rotate, drift),
                        [translation + rotation, translation, rotation],
                    )


def test_thermal_reductions_preserve_all_leading_snapshot_axes() -> None:
    first = jd.State.create(
        pos=jnp.zeros((2, 2)), vel=jnp.array([[1.0, 0.0], [0.0, 2.0]])
    )
    second = replace(first, vel=2.0 * first.vel)
    batch = jd.State.stack([first, second])
    trajectory = jax.tree.map(lambda x: jnp.stack([x, x]), batch)

    assert compute_translational_kinetic_energy_per_particle(batch).shape == (2, 2)
    assert compute_translational_kinetic_energy(batch).shape == (2,)
    assert compute_temperature(batch, False, False).shape == (2,)
    assert compute_translational_kinetic_energy(trajectory).shape == (2, 2)
    np.testing.assert_allclose(
        compute_translational_kinetic_energy(batch), jnp.array([2.5, 10.0])
    )

    first_system = jd.System.create(state=first, dt=0.1, rotation_integrator_type=None)
    second_system = jd.System.create(
        state=second, dt=0.1, rotation_integrator_type=None
    )
    second_table = replace(
        second_system.mat_table,
        props={
            name: 2.0 * value for name, value in second_system.mat_table.props.items()
        },
        pair={name: 2.0 * value for name, value in second_system.mat_table.pair.items()},
    )
    second_system = replace(second_system, mat_table=second_table)
    systems = jd.System.stack([first_system, second_system])
    assert compute_potential_energy(batch, systems).shape == (2,)


def test_rescaling_and_temperature_scaling_remove_sub_unit_clump_drift() -> None:
    # Body 0 has two members with total mass 0.2; body 1 has one member with mass 0.6.
    state = jd.State.create(
        pos=jnp.zeros((3, 2)),
        vel=jnp.array([[2.0, 0.0], [2.0, 0.0], [-1.0, 0.0]]),
        mass=jnp.array([0.2, 0.2, 0.6]),
        clump_id=jnp.array([0, 0, 1]),
    )
    member_mass = jnp.array([0.1, 0.1, 0.6])

    integrator = jd.LinearIntegrator.create(
        "verlet_rescaling", temperature=1.0, subtract_drift=True
    )
    system = jd.System.create(
        state=state,
        dt=0.1,
        collider_type="",
        rotation_integrator_type=None,
        linear_integrator=integrator,
    )
    system = replace(system, step_count=jnp.asarray(1))
    rescaled, _ = integrator.step_after_force(state, system)
    scaled = scale_to_temperature(state, 1.0, False, True)

    for result in (rescaled, scaled):
        drift = jnp.sum(result.vel * member_mass[:, None], axis=0) / member_mass.sum()
        np.testing.assert_allclose(drift, 0.0, atol=1e-6)
        np.testing.assert_allclose(
            0.5 * jnp.sum(member_mass[:, None] * result.vel**2), 1.0, rtol=2e-6
        )


def test_rescaling_runs_after_the_rotational_terminal_kick() -> None:
    state = jd.State.create(pos=jnp.zeros((1, 2)), mass=jnp.ones(1))
    integrator = jd.LinearIntegrator.create(
        "verlet_rescaling", temperature=3.0, can_rotate=True
    )
    system = jd.System.create(
        state=state,
        dt=0.1,
        collider_type="",
        linear_integrator=integrator,
        rotation_integrator=_TerminalRotationalKick(),
    )
    state, system = jd.System.step(state, system)
    np.testing.assert_allclose(
        compute_temperature(state, can_rotate=True, subtract_drift=False),
        3.0,
        rtol=1e-6,
    )
