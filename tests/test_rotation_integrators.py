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
