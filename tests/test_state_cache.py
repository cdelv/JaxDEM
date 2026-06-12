# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unit tests for the State rotation cache and attribute setting under JIT."""

import jax
import jax.numpy as jnp
from dataclasses import replace

import jaxdem as jdem


def test_jit_update_q_then_pos_p():
    pos_c = jnp.array([[1.0, 2.0, 3.0]])
    pos_p = jnp.array([[1.0, 0.0, 0.0]])
    q_init = jdem.utils.quaternion.Quaternion.create(
        jnp.array([[1.0]]), jnp.array([[0.0, 0.0, 0.0]])
    )

    state = jdem.State.create(pos=pos_c, pos_p=pos_p, q=q_init)

    @jax.jit
    def update_state(s, new_q, new_pos_p):
        s.q = new_q
        s.pos_p = new_pos_p
        return s

    # Rotate 90 degrees around Z axis: w = cos(45), z = sin(45)
    q_new = jdem.utils.quaternion.Quaternion.create(
        jnp.array([[0.70710678]]), jnp.array([[0.0, 0.0, 0.70710678]])
    )
    pos_p_new = jnp.array([[0.0, 2.0, 0.0]])

    out_state = update_state(state, q_new, pos_p_new)

    # Compute expected rotation: pos_p_new ([0, 2, 0]) rotated by q_new (90 deg around Z).
    # Since q_new rotates about Z by 90 degrees: [0, 2, 0] should become [-2, 0, 0].
    expected_rot = q_new.rotate(q_new, pos_p_new)

    assert jnp.allclose(out_state._pos_p_rot, expected_rot, atol=1e-5)
    assert jnp.allclose(out_state.pos, pos_c + expected_rot, atol=1e-5)


def test_replace_updates_cache():
    pos_c = jnp.array([[1.0, 2.0, 3.0]])
    pos_p = jnp.array([[1.0, 0.0, 0.0]])
    q_init = jdem.utils.quaternion.Quaternion.create(
        jnp.array([[1.0]]), jnp.array([[0.0, 0.0, 0.0]])
    )

    state = jdem.State.create(pos=pos_c, pos_p=pos_p, q=q_init)

    # Rotate 90 degrees around Z axis
    q_new = jdem.utils.quaternion.Quaternion.create(
        jnp.array([[0.70710678]]), jnp.array([[0.0, 0.0, 0.70710678]])
    )

    # Using dataclasses.replace
    state_replaced = replace(state, q=q_new)

    expected_rot = q_new.rotate(q_new, pos_p)
    assert jnp.allclose(state_replaced._pos_p_rot, expected_rot, atol=1e-5)
    assert jnp.allclose(state_replaced.pos, pos_c + expected_rot, atol=1e-5)


def test_simulation_rotation_cache():
    # Create 3D particles with non-zero relative offsets (pos_p) and initial angular velocity
    pos_c = jnp.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    pos_p = jnp.array([[0.2, 0.1, -0.1], [-0.2, -0.1, 0.1]])
    ang_vel = jnp.array([[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]])
    rad = jnp.array([1.0, 1.0])

    state = jdem.State.create(
        pos=pos_c,
        pos_p=pos_p,
        rad=rad,
        ang_vel=ang_vel,
    )

    system = jdem.System.create(
        state.shape,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        dt=1e-3,
    )

    # Run for 300 steps
    state_end, system_end = system.step(state, system, n=300)

    # Check that the cache array is correct at the end
    expected_rot = state_end.q.rotate(state_end.q, state_end.pos_p)
    assert jnp.allclose(state_end._pos_p_rot, expected_rot, atol=1e-6)
    assert jnp.allclose(state_end.pos, state_end.pos_c + expected_rot, atol=1e-6)
