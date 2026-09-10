# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Regression tests for rigid-body Langevin noise and step layout dispatch."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem


def _free_system(state: jdem.State) -> jdem.System:
    return jdem.System.create(
        state=state,
        dt=0.01,
        rotation_integrator_type=None,
    )


def _assert_pytrees_allclose(actual: object, expected: object) -> None:
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf)


def test_step_dispatches_unbatched_and_all_batch_sizes() -> None:
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [10.0, 0.0]]),
        vel=jnp.array([[0.25, -0.5], [-0.75, 1.0]]),
    )
    system = _free_system(state)

    expected_state, _ = jdem.System.step(state, system, n=3)
    assert expected_state.shape == (2, 2)
    np.testing.assert_allclose(
        expected_state.pos_c, state.pos_c + 3 * system.dt * state.vel
    )

    for batch_size in (1, 2):
        states = [
            dataclasses.replace(state, vel=state.vel * (batch_index + 1))
            for batch_index in range(batch_size)
        ]
        expected = [jdem.System.step(item, system, n=3) for item in states]
        expected_states = jdem.State.stack([item[0] for item in expected])
        expected_systems = jdem.System.stack([item[1] for item in expected])

        batched_state = jdem.State.stack(states)
        batched_system = jdem.System.stack([system] * batch_size)
        actual_state, actual_system = jdem.System.step(
            batched_state, batched_system, n=3
        )

        assert actual_state.shape == (batch_size, 2, 2)
        _assert_pytrees_allclose(actual_state, expected_states)
        _assert_pytrees_allclose(actual_system, expected_systems)


@pytest.mark.parametrize("stack_state", [False, True])
def test_step_rejects_mismatched_state_and_system_layouts(stack_state: bool) -> None:
    state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
    system = _free_system(state)
    if stack_state:
        state = jdem.State.stack([state])
    else:
        system = jdem.System.stack([system])

    with pytest.raises(ValueError, match="matching state and system layouts"):
        jdem.System.step(state, system)


def test_langevin_shares_translational_noise_within_each_clump() -> None:
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0], [5.0, 0.0]]),
        pos_p=jnp.array([[-0.2, 0.0], [0.2, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        clump_id=jnp.array([0, 0, 1, 2]),
        fixed=jnp.array([False, False, False, True]),
        vel=jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.5, -0.5]]),
    )
    system = jdem.System.create(
        state=state,
        dt=0.1,
        seed=7,
        linear_integrator_type="langevin",
        linear_integrator_kw={"gamma": 1.0, "k_B": 1.0, "temperature": 1.0},
        rotation_integrator_type=None,
    )

    actual, _ = jdem.System.step(state, system)

    np.testing.assert_allclose(actual.vel[0], actual.vel[1])
    np.testing.assert_allclose(actual.pos_c[0], actual.pos_c[1])
    assert not np.allclose(actual.vel[0], actual.vel[2])
    np.testing.assert_allclose(actual.vel[3], state.vel[3])


def test_langevin_noise_has_body_mass_temperature_covariance() -> None:
    particle_count = 4096
    state = jdem.State.create(
        pos=jnp.zeros((particle_count, 2)),
        mass=jnp.full(particle_count, 4.0),
    )
    system = jdem.System.create(
        state=state,
        dt=0.1,
        seed=13,
        linear_integrator_type="langevin",
        linear_integrator_kw={"gamma": 100.0, "k_B": 1.0, "temperature": 2.0},
        rotation_integrator_type=None,
    )

    actual, _ = system.linear_integrator.step_before_force(state, system)

    # With exp(-gamma * dt) effectively zero, each velocity component is an
    # equilibrium sample with variance k_B * T / body_mass = 0.5.
    np.testing.assert_allclose(jnp.var(actual.vel), 0.5, rtol=0.08)
