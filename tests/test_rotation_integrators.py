# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jdem


def body_torque(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> tuple[jax.Array, jax.Array]:
    """Constant torque applied in the body frame."""
    tx_val = 0.025  # Rad * 0.5 = 0.05 * 0.5
    # Create torque in body frame: [tx_val, 0, 0]
    torque_body = jnp.zeros_like(state.ang_vel)
    torque_body = torque_body.at[..., 0].set(tx_val)
    # Rotate body-frame torque to lab frame
    torque_lab = state.q.rotate(state.q, torque_body)
    return jnp.zeros_like(pos), torque_lab


@pytest.mark.parametrize("rot_integrator", ["spiral", "verletspiral"])
def test_aspherical_rotation(rot_integrator):
    # Parameters from checkRotAlgorithmNonSpherical.py
    rho = 7750.0
    Rad = 0.05
    Height = 0.15
    Mass = rho * jnp.pi * Rad * Rad * Height
    Ix = 0.5 * Mass * Rad * Rad
    Iy = Mass * Height * Height / 12.0 + 0.25 * Mass * Rad * Rad
    Iz = Iy
    tx = Rad * 0.5
    wx0 = 0.3
    wy0 = -0.9
    wz0 = 0.6

    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0, 0.0]]),
        ang_vel=jnp.array([[wx0, wy0, wz0]]),
        inertia=jnp.array([[Ix, Iy, Iz]]),
        mass=jnp.array([Mass]),
        rad=jnp.array([Rad]),
    )

    dt = 1.0e-5
    steps = 70000  # 0.7s total time

    system = jdem.System.create(
        state.shape,
        linear_integrator_type="verlet",
        rotation_integrator_type=rot_integrator,
        dt=dt,
        force_manager_kw={
            "force_functions": [(body_torque, None)],
        },
        collider_type="naive",
    )

    @jax.jit
    def run_sim(state, system):
        def body_fn(i, val):
            s, sys = val
            return sys.step(s, sys)

        return jax.lax.fori_loop(0, steps, body_fn, (state, system))

    final_state, final_system = run_sim(state, system)

    # Calculate analytical solution at t = 0.7
    t = 0.7
    A = (Ix - Iy) * (Iz - Ix) / (Iy * Iz)
    B = Iy / (Iz - Ix)
    sqrt_A = jnp.sqrt(-A)
    E = 2.0 * tx * B / Ix
    wx = wx0 + tx * t / Ix

    eta = 0.5 * Ix * sqrt_A / tx
    C = E * eta
    K1 = (C * wy0 * jnp.cos(eta * wx0**2) - wz0 * jnp.sin(eta * wx0**2)) / C
    K2 = (C * wy0 * jnp.sin(eta * wx0**2) + wz0 * jnp.cos(eta * wx0**2)) / C
    D = eta * wx**2

    wy = K1 * jnp.cos(D) + K2 * jnp.sin(D)
    wz = C * (K2 * jnp.cos(D) - K1 * jnp.sin(D))
    ww = jnp.array([wx, wy, wz])

    # Get current angular velocity in body frame
    q = final_state.q
    w_body = q.rotate_back(q, final_state.ang_vel[0])

    error = jnp.log10(jnp.linalg.norm(w_body - ww) / jnp.linalg.norm(ww))

    assert (
        error < -4.0
    ), f"Error in aspherical {rot_integrator} algorithm too big: {error}"
