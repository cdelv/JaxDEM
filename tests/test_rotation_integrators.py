# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp
import pytest
import numpy as np

import jaxdem as jdem


def body_torque(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> tuple[jax.Array, jax.Array]:
    """Return zero force and a torque that is constant in the body frame."""
    tx_val = 1.0
    # Create torque in body frame: [tx_val, 0, 0]
    torque_body = jnp.zeros_like(state.ang_vel)
    torque_body = torque_body.at[..., 0].set(tx_val)
    # Rotate body-frame torque to lab frame
    torque_lab = state.q.rotate(state.q, torque_body)
    return jnp.zeros_like(pos), torque_lab


@pytest.mark.parametrize("rot_integrator", ["spiral", "verletspiral"])
def test_aspherical_rotation(rot_integrator):
    # Dimensionless symmetric top: I=(1,2,2), unit torque along body x.
    initial = np.array([0.3, -0.9, 0.6])
    state = jdem.State.create(
        pos=jnp.zeros((1, 3)),
        ang_vel=jnp.asarray(initial[None]),
        inertia=jnp.array([[1.0, 2.0, 2.0]]),
        mass=jnp.ones(1),
    )
    system = jdem.System.create(
        state=state,
        collider_type="",
        dt=1 / 512,
        rotation_integrator_type=rot_integrator,
        force_manager_kw={"force_functions": [(body_torque, None)]},
    )
    state, system = jdem.System.initialize(state, system)
    final, _ = jdem.System.step(state, system, n=512)
    # Euler equations give wx=.3+t and a rotation of (wy,wz) by
    # phase=(J-Ix)/J * integral(wx dt)=.5*(.3+.5)=.4 at t=1.
    phase = 0.4
    expected = np.array(
        [
            1.3,
            initial[1] * np.cos(phase) + initial[2] * np.sin(phase),
            initial[2] * np.cos(phase) - initial[1] * np.sin(phase),
        ]
    )
    w, x, y, z = float(final.q.w[0, 0]), *np.asarray(final.q.xyz[0])
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + 2 * w * skew + 2 * skew @ skew
    body = rotation.T @ np.asarray(final.ang_vel[0])
    np.testing.assert_allclose(body, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        body[1:] @ body[1:], initial[1:] @ initial[1:], rtol=2e-5
    )
    np.testing.assert_allclose(w * w + x * x + y * y + z * z, 1.0, atol=2e-6)


@pytest.mark.parametrize("rot_integrator", ["spiral", "verletspiral"])
def test_asymmetric_free_top_conserves_inertial_momentum_under_refinement(
    rot_integrator,
):
    inertia, initial = np.array([1.0, 2.0, 3.0]), np.array([0.3, -0.9, 0.6])
    momentum0 = inertia * initial
    energy0 = 0.5 * np.sum(inertia * initial**2)
    errors = []
    for steps in [16, 32, 64]:
        state = jdem.State.create(
            pos=jnp.zeros((1, 3)),
            inertia=jnp.asarray(inertia[None]),
            ang_vel=jnp.asarray(initial[None]),
        )
        system = jdem.System.create(
            state=state,
            collider_type="",
            linear_integrator_type="",
            rotation_integrator_type=rot_integrator,
            dt=1 / steps,
        )
        state, system = jdem.System.initialize(state, system)
        _, _, (trajectory, _) = jdem.System.trajectory_rollout(state, system, n=steps)
        w = np.asarray(trajectory.q.w[:, 0, 0])
        xyz = np.asarray(trajectory.q.xyz[:, 0])
        skew = np.zeros((steps, 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -xyz[:, 2], xyz[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = xyz[:, 2], -xyz[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -xyz[:, 1], xyz[:, 0]
        rotation = np.eye(3) + 2 * w[:, None, None] * skew + 2 * skew @ skew
        body_velocity = np.einsum(
            "tji,tj->ti", rotation, np.asarray(trajectory.ang_vel[:, 0])
        )
        momentum = np.einsum("tij,tj->ti", rotation, inertia * body_velocity)
        energy = 0.5 * np.sum(inertia * body_velocity**2, axis=-1)
        error = np.max(np.linalg.norm(momentum - momentum0, axis=-1)) / np.linalg.norm(
            momentum0
        )
        errors.append(error)
        assert error < 2 / steps**2
        np.testing.assert_allclose(energy, energy0, rtol=2 / steps**2)
        assert np.linalg.norm(body_velocity[-1] - initial) > 0.1
    assert np.all(np.asarray(errors[:-1]) / errors[1:] > 2.5)
