# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check force and integrator initialization and batched simulation setup."""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace

import numpy as np
import pytest

import jaxdem as jd


def constant_torque(pos, state, system):
    return jnp.zeros_like(pos), jnp.full_like(state.torque, 2.0)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class BackwardHalfKick(jd.LinearIntegrator):
    @staticmethod
    def initialize(state, system):
        return (
            replace(
                state,
                vel=state.vel - 0.5 * system.dt * state.force / state.mass[..., None],
            ),
            system,
        )


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class BackwardAngularHalfKick(jd.RotationIntegrator):
    @staticmethod
    def initialize(state, system):
        return (
            replace(
                state,
                ang_vel=state.ang_vel - 0.5 * system.dt * state.torque / state.inertia,
            ),
            system,
        )


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class InitializationMarker(jd.LinearIntegrator):
    @staticmethod
    def initialize(state, system):
        return replace(state, vel=state.vel + 17.0), system


def test_initialize_computes_loads_and_invokes_both_integrator_hooks():
    state = jd.State.create(
        pos=jnp.array([[1.0, 2.0]]),
        vel=jnp.array([[3.0, 4.0]]),
        ang_vel=jnp.array([[5.0]]),
        mass=jnp.array([2.0]),
        inertia=jnp.array([[4.0]]),
    )
    system = jd.System.create(
        state=state,
        dt=0.2,
        collider_type="",
        linear_integrator=BackwardHalfKick(),
        rotation_integrator=BackwardAngularHalfKick(),
        force_manager_kw={
            "gravity": jnp.array([0.0, -3.0]),
            "force_functions": (constant_torque,),
        },
    )
    initial_pos = state.pos
    initial_vel = state.vel
    initial_ang_vel = state.ang_vel
    initial_time = system.time
    initial_step = system.step_count

    initialized, initialized_system = jd.System.initialize(state, system)

    np.testing.assert_array_equal(initialized.pos, initial_pos)
    np.testing.assert_array_equal(initialized_system.time, initial_time)
    np.testing.assert_array_equal(initialized_system.step_count, initial_step)
    np.testing.assert_allclose(initialized.force, [[0.0, -6.0]])
    np.testing.assert_allclose(initialized.torque, [[2.0]])
    np.testing.assert_allclose(
        initialized.vel,
        initial_vel - 0.5 * system.dt * initialized.force / state.mass[..., None],
    )
    np.testing.assert_allclose(
        initialized.ang_vel,
        initial_ang_vel - 0.5 * system.dt * initialized.torque / state.inertia,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_initialize_supports_batched_states(batch_size):
    states = [
        jd.State.create(
            pos=jnp.zeros((2, 2)),
            mass=jnp.array([1.0, 2.0]) * (i + 1),
            vel=jnp.full((2, 2), 0.1 * (i + 1)),
        )
        for i in range(batch_size)
    ]
    systems = [
        jd.System.create(
            state=state_i,
            collider_type="",
            dt=0.1 * (i + 1),
            linear_integrator=BackwardHalfKick(),
            minimizer=jd.fire,
            minimizer_kw={"dt": 0.01},
            force_manager_kw={"gravity": jnp.array([1.0, -2.0]) * (i + 1)},
        )
        for i, state_i in enumerate(states)
    ]
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    system = jax.tree.map(lambda *xs: jnp.stack(xs), *systems)

    initialized, initialized_system = jd.System.initialize(state, system)

    assert initialized.pos.shape == (batch_size, 2, 2)
    for i in range(batch_size):
        np.testing.assert_allclose(
            initialized.force[i],
            np.array([1.0, 2.0])[:, None] * np.array([1.0, -2.0]) * (i + 1) ** 2,
        )
        np.testing.assert_allclose(
            initialized.vel[i],
            np.broadcast_to(
                0.1 * (i + 1) - 0.05 * (i + 1) ** 2 * np.array([1.0, -2.0]), (2, 2)
            ),
        )
    np.testing.assert_array_equal(initialized_system.time, system.time)
    np.testing.assert_array_equal(initialized_system.step_count, system.step_count)


def test_step_does_not_implicitly_initialize_integrators():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]))
    system = jd.System.create(
        state=state,
        collider_type="",
        linear_integrator=InitializationMarker(),
    )

    stepped, stepped_system = jd.System.step(state, system)

    np.testing.assert_array_equal(stepped.vel, state.vel)
    np.testing.assert_array_equal(stepped_system.step_count, system.step_count + 1)


def test_step_does_not_implicitly_initialize_forces():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]))
    gravity = jnp.array([0.0, -2.0])
    system = jd.System.create(
        state=state, dt=0.1, collider_type="", force_manager_kw={"gravity": gravity}
    )
    # Deliberately skip initialization: the first kick must use the supplied
    # zero force, not a hidden extra force evaluation.
    stepped, _ = jd.System.step(state, system)
    np.testing.assert_array_equal(stepped.pos, state.pos)
    np.testing.assert_allclose(stepped.vel, 0.5 * system.dt * gravity[None, :])
