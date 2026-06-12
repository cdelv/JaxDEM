# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Shared Verlet-consistent time-of-collision (TOC) root solver for reflective domains."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def verlet_collision_fraction(
    v_end: jax.Array,
    acc: jax.Array,
    delta: jax.Array,
    wall_sign: jax.Array,
    dt: jax.Array,
) -> jax.Array:
    r"""Solve the Verlet-consistent time-of-collision fraction :math:`\alpha`.

    Under Velocity Verlet, with the end-of-step velocity :math:`v` and
    acceleration :math:`a` known after the step, the velocities entering the
    position update are reconstructed as

    .. math::
        v_0 = v - a \Delta t, \qquad v_{mid} = v - \tfrac{1}{2} a \Delta t

    and the collision time fraction :math:`\alpha \in [0, 1]` solves the
    quadratic

    .. math::
        A \alpha^2 + B \alpha + C = 0

    with :math:`A = \tfrac{1}{2} a_n \Delta t^2`, :math:`B = v_{0,n} \Delta t`,
    :math:`C = -(\delta + v_{mid,n} \Delta t)`, evaluated with the numerically
    stable form

    .. math::
        \alpha = \frac{2 (-C)}{B + \sqrt{B^2 - 4 A C}}

    when :math:`B < 0` and the standard form otherwise. The square root uses
    the double-``where`` idiom so reverse-mode gradients stay finite when the
    discriminant is clamped to zero.

    Parameters
    ----------
    v_end : jax.Array
        Velocity (of the contact point or particle) at the end of the step.
    acc : jax.Array
        Acceleration (of the contact point or particle) during the step.
    delta : jax.Array
        Penetration depth past the wall (positive value).
    wall_sign : jax.Array
        Sign of the inward wall normal per coordinate (+1 lower wall,
        -1 upper wall, 0 no contact).
    dt : jax.Array
        The time step.

    Returns
    -------
    jax.Array
        The collision time fraction :math:`\alpha`, clipped to ``[0, 1]``.

    Notes
    -----
    The velocity at the moment of collision is then reconstructed (also
    Verlet-consistently) as ``v_col = v_end + (alpha - 1) * dt * acc``.
    """
    v_0 = v_end - dt * acc
    v_mid = v_end - 0.5 * dt * acc

    v_0_n = v_0 * wall_sign
    v_mid_n = v_mid * wall_sign
    acc_n = acc * wall_sign

    A_pos = 0.5 * acc_n * dt * dt
    B_pos = v_0_n * dt
    d_wall_pos = delta + v_mid_n * dt

    disc = B_pos * B_pos + 4.0 * A_pos * d_wall_pos
    disc = jnp.maximum(0.0, disc)
    # Double-where: sqrt must never see 0, even in the untaken branch,
    # otherwise reverse-mode gradients at disc=0 are NaN.
    safe_disc = jnp.where(disc > 0.0, disc, 1.0)
    sqrt_disc = jnp.where(disc > 0.0, jnp.sqrt(safe_disc), 0.0)

    alpha_stable = jnp.where(
        B_pos < 0,
        2.0
        * d_wall_pos
        / jnp.where(B_pos - sqrt_disc < -1e-10, B_pos - sqrt_disc, -1.0),
        (-B_pos - sqrt_disc)
        / jnp.where(jnp.abs(2.0 * A_pos) > 1e-10, 2.0 * A_pos, 1.0),
    )
    return jnp.clip(alpha_stable, 0.0, 1.0)


__all__ = ["verlet_collision_fraction"]
