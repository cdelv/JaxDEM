# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Known collision times, restitution and rigid-body wall impulses."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.domains._toc import verlet_collision_fraction


@pytest.mark.parametrize("x64", [False, True])
def test_collision_fraction_is_independent_of_length_units(x64):
    with jax.enable_x64(x64):
        scale = jnp.array([1e-30, 1e-12, 1.0, 1e12, 1e30])[:, None]
        # Constant speed, accelerating toward a wall, and initially receding.
        velocity = scale * jnp.array([-1.0, -2.0, -3.0])
        acceleration = scale * jnp.array([0.0, -1.0, -4.0])
        penetration = scale * jnp.array([0.5, 0.875, 0.625])
        result = jax.jit(verlet_collision_fraction)(
            velocity, acceleration, penetration, jnp.ones_like(velocity), jnp.array(1.0)
        )
        np.testing.assert_allclose(
            result, np.broadcast_to([0.5, 0.5, 0.75], result.shape), rtol=2e-6
        )

        # Inactive coordinates and the zero discriminant must not poison AD.
        def solve(values):
            return verlet_collision_fraction(*values, jnp.array(1.0), jnp.array(1.0))

        for values in ([0.0, 0.0, 0.0], [-1.0, 0.0, 0.5], [0.0, -2.0, 0.0]):
            assert np.isfinite(jax.jacrev(solve)(jnp.asarray(values))).all()
        derivatives = jax.vmap(jax.jacrev(solve))(scale * jnp.array([-1.0, 0.0, 0.5]))
        np.testing.assert_allclose(
            derivatives * scale,
            np.broadcast_to([-0.5, 0.125, -1.0], derivatives.shape),
            rtol=3e-6,
        )


@pytest.mark.parametrize("domain", ["reflect", "reflectsphere"])
@pytest.mark.parametrize("restitution", [0.5, 1.0])
def test_wall_collision_uses_remaining_step_time(domain, restitution):
    scale = 1e-12
    state = jd.State.create(
        pos=jnp.array([[0.6, 1.0]]) * scale,
        rad=jnp.array([0.1]) * scale,
        vel=jnp.array([[-1.0, 0.0]]) * scale,
    )
    system = jd.System.create(
        state=state,
        dt=1.0,
        collider_type="",
        rotation_integrator_type="",
        domain_type=domain,
        domain_kw={
            "box_size": jnp.array([2.0, 2.0]) * scale,
            "restitution_coefficient": restitution,
        },
    )
    # Reflection corrects the preceding drift at the start of the next step.
    advanced, _ = jd.System.step(state, system, n=2)
    np.testing.assert_allclose(advanced.vel / scale, [[restitution, 0]], atol=1e-7)
    np.testing.assert_allclose(
        advanced.pos / scale, [[0.1 + 1.5 * restitution, 1.0]], atol=1e-7
    )


@pytest.mark.parametrize("restitution", [0.5, 1.0])
def test_rigid_wall_impulse_transfers_translation_into_rotation(restitution):
    state = jd.State.create(
        pos=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
        pos_p=jnp.array([[-0.1, 0.5], [0.1, -0.5]]),
        clump_id=jnp.zeros(2, dtype=int),
        mass=jnp.full(2, 2.0),
        inertia=jnp.full((2, 1), 0.5),
        vel=jnp.array([[-1.0, 0.0], [-1.0, 0.0]]),
        rad=jnp.full(2, 0.1),
    )
    system = jd.System.create(
        state=state,
        dt=0.4,
        collider_type="",
        domain_type="reflect",
        domain_kw={
            "box_size": jnp.array([4.0, 4.0]),
            "restitution_coefficient": restitution,
        },
    )
    reflected, _ = system.domain.apply(state, system)
    impulse = 1 + restitution  # 1/m + r_y^2/I = 1.
    vx, omega = -1 + impulse / 2, -impulse
    np.testing.assert_allclose(reflected.vel, [[vx, 0], [vx, 0]], atol=1e-7)
    np.testing.assert_allclose(reflected.ang_vel, [[omega], [omega]], atol=1e-7)
    np.testing.assert_allclose(vx - 0.5 * omega, restitution)
    kinetic = vx**2 + 0.25 * omega**2
    np.testing.assert_allclose(kinetic, 1 - 0.5 * (1 - restitution**2))
