"""Exact free-particle thermostat transitions, including small reduced units."""

from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("gamma_dt", [0.0, 1e-8, 0.4])
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


@pytest.mark.parametrize("speed", [0.0, 1e-16, 1.0])
@pytest.mark.parametrize("temperature", [0.0, 2.0])
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
            return system.linear_integrator.finalize_step(trial, system)[0].vel

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
