# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reusable custom force/energy functions for checkpoint tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxdem.state import State
    from jaxdem.system import System


def harmonic_trap(
    pos: jax.Array, state: State, system: System
) -> tuple[jax.Array, jax.Array]:
    """Harmonic restoring force toward the origin."""
    k = 1.0
    return -k * pos, jnp.zeros_like(state.torque)


def harmonic_trap_energy(pos: jax.Array, state: State, system: System) -> jax.Array:
    """Potential energy for :func:`harmonic_trap`."""
    k = 1.0
    return 0.5 * k * jnp.sum(pos**2, axis=-1)


def constant_push(
    pos: jax.Array, state: State, system: System
) -> tuple[jax.Array, jax.Array]:
    """Constant rightward push."""
    f = jnp.broadcast_to(jnp.array([1.0, 0.0]), pos.shape)
    return f, jnp.zeros_like(state.torque)
