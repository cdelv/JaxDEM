# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Utility functions to compute thermodynamic quantities."""

from __future__ import annotations

import math

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

from .linalg import dot, norm2

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@jax.jit
def _translational_ke_snapshot(state: State) -> jax.Array:
    count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
    weight = state.mass / count
    return 0.5 * weight * norm2(state.vel)


def _map_snapshots(
    fn: Callable[..., Any], state: State, system: System | None = None
) -> Any:
    """Map a single-snapshot kernel over every leading state axis."""
    mapped = fn
    for _ in range(state.pos_c.ndim - 2):
        mapped = jax.vmap(mapped)
    return mapped(state) if system is None else mapped(state, system)


@partial(
    jax.named_call, name="thermal.compute_translational_kinetic_energy_per_particle"
)
def compute_translational_kinetic_energy_per_particle(state: State) -> jax.Array:
    r"""Compute the translational kinetic energy per particle.

    .. math::
        E_{trans} = \frac{1}{2} m |v|^2

    Notes
    -----
    - The function divides the energy of clump members by the number of
      spheres in the clump.

    Parameters
    ----------
    state : State
        The current state of the system containing particle masses and velocities.

    Returns
    -------
    jax.Array
        Energy with shape ``(..., N)``. Every leading batch or trajectory axis
        is preserved; the last axis is always the particle axis.

    """
    return _map_snapshots(_translational_ke_snapshot, state)


@jax.jit
def _rotational_ke_snapshot(state: State) -> jax.Array:
    count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
    if state.dim == 2:
        w_body = state.ang_vel
    else:
        w_body = state.q.rotate_back(state.q, state.ang_vel)
    return 0.5 * dot(w_body, state.inertia * w_body) / count


@partial(jax.named_call, name="thermal.compute_rotational_kinetic_energy_per_particle")
def compute_rotational_kinetic_energy_per_particle(state: State) -> jax.Array:
    r"""Compute the rotational kinetic energy per particle.

    .. math::
        E_{rot} = \frac{1}{2} \vec{\omega}^T I \vec{\omega}

    Notes
    -----
    - The function divides the energy of clump members by the number of
      spheres in the clump.

    Parameters
    ----------
    state : State
        The current state of the system containing inertia, orientation, and angular velocity.

    Returns
    -------
    jax.Array
        Energy with shape ``(..., N)``. Leading batch and trajectory axes are
        preserved.

    """
    return _map_snapshots(_rotational_ke_snapshot, state)


@jax.jit
@partial(jax.named_call, name="thermal.compute_translational_kinetic_energy")
def compute_translational_kinetic_energy(state: State) -> jax.Array:
    r"""Compute the total translational kinetic energy of the system.

    .. math::
        E_{trans, total} = \sum_{i} \frac{1}{2} m_i |v_i|^2

    Parameters
    ----------
    state : State
        The current state of the system.

    Returns
    -------
    jax.Array
        The scalar sum of translational kinetic energy across all particles.

    """
    return jnp.sum(compute_translational_kinetic_energy_per_particle(state), axis=-1)


@jax.jit
@partial(jax.named_call, name="thermal.compute_rotational_kinetic_energy")
def compute_rotational_kinetic_energy(state: State) -> jax.Array:
    r"""Compute the total rotational kinetic energy of the system.

    .. math::
        E_{rot, total} = \sum_{i} \frac{1}{2} \vec{\omega}_i^T I_i \vec{\omega}_i

    Parameters
    ----------
    state : State
        The current state of the system.

    Returns
    -------
    jax.Array
        The scalar sum of rotational kinetic energy across all particles.

    """
    return jnp.sum(compute_rotational_kinetic_energy_per_particle(state), axis=-1)


@jax.jit(inline=True)
def _potential_energy_snapshot(state: State, system: System) -> jax.Array:
    pe_force_manager = system.force_manager.compute_potential_energy(state, system)
    _, _, pe_collider = system.collider.compute_potential_energy(state, system)
    return pe_force_manager + pe_collider


@partial(jax.named_call, name="thermal.compute_potential_energy")
def compute_potential_energy(state: State, system: System) -> jax.Array:
    r"""Compute the total potential energy of the system.
    The function sums the potential energy from the force models in the
    collider. It also sums the gravity and force functions in the force
    manager that have a potential energy.

    .. math::
        E_{pot, total} = \sum_{i} U(r_i)

    Parameters
    ----------
    state : State
        The current state of the system.
    system : System
        The system definition containing the collider.

    Returns
    -------
    jax.Array
        Potential energy for each snapshot. The result is scalar for an
        unbatched state and has the state's leading shape for stacked states;
        the system must have matching leading axes.

    """
    if system.dt.ndim != state.pos_c.ndim - 2:
        raise ValueError("State and system leading axes must match.")
    return _map_snapshots(_potential_energy_snapshot, state, system)


@jax.jit
@partial(jax.named_call, name="thermal.compute_energy")
def compute_energy(state: State, system: System) -> jax.Array:
    """Compute the total mechanical energy of the system.

    .. math::
        E_{total} = E_{pot, total} + E_{trans, total} + E_{rot, total}

    Parameters
    ----------
    state : State
        The current state of the system.
    system : System
        The system definition containing physics parameters and colliders.

    Returns
    -------
    jax.Array
        The total energy (scalar) of the system.

    """
    Pe = compute_potential_energy(state, system)
    Ke_t = compute_translational_kinetic_energy(state)
    Ke_r = compute_rotational_kinetic_energy(state)
    return Pe + Ke_t + Ke_r


def _count_dynamic_dofs_snapshot(
    state: State, can_rotate: bool, subtract_drift: bool
) -> tuple[jax.Array, jax.Array, jax.Array]:
    counts = jnp.bincount(state.clump_id, length=state.N)
    fixed_counts = jnp.bincount(
        state.clump_id, weights=state.fixed.astype(int), length=state.N
    )
    free_count = jnp.sum((counts > 0) & (fixed_counts == 0))
    n_dof_v = (free_count - subtract_drift) * state.vel.shape[-1]
    n_dof_w = free_count * state.ang_vel.shape[-1] * can_rotate
    n_dof = n_dof_v + n_dof_w
    return n_dof, n_dof_v, n_dof_w


def count_dynamic_dofs(
    state: State, can_rotate: bool, subtract_drift: bool
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Count the number of degrees of freedom for the dynamics.

    Parameters
    ----------
    state : State
        Current simulation state.
    can_rotate : bool
        Whether to include rigid body rotations.
    subtract_drift : bool
        If True, subtract the center-of-mass drift degrees of freedom
        (usually only relevant for small systems).

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        Total, translational, and rotational counts. Each is scalar for one
        snapshot and has the leading state shape for stacked snapshots.
    """
    kernel = partial(
        _count_dynamic_dofs_snapshot,
        can_rotate=can_rotate,
        subtract_drift=subtract_drift,
    )
    return _map_snapshots(kernel, state)


def _assign_random_velocities(
    state: State, subtract_drift: bool, seed: int | None = 0
) -> State:
    """Assign random velocities to one ``(N, dim)`` snapshot.

    Parameters
    ----------
    state : State
        Current simulation state.
    subtract_drift : bool
        Whether to remove center-of-mass drift.
    seed : int, optional
        RNG seed.

    """
    if state.pos_c.ndim != 2:
        raise ValueError(
            "Random velocity assignment accepts one snapshot; use jax.vmap "
            "with independent seeds for stacked states."
        )
    if seed is None:
        seed = 0
    key = jax.random.PRNGKey(seed)
    v_k, w_k = jax.random.split(key, 2)
    counts = jnp.bincount(state.clump_id, length=state.N)
    exists = counts > 0
    fixed_counts = jnp.bincount(
        state.clump_id, weights=state.fixed.astype(int), length=state.N
    )
    free_mask = (fixed_counts == 0) & exists
    v_clump = jax.random.normal(v_k, (state.N, state.dim)) * free_mask[:, None]
    if subtract_drift:
        num_free = jnp.sum(free_mask)
        v_clump_mean = jnp.sum(v_clump, axis=0) / jnp.maximum(num_free, 1)
        v_clump -= v_clump_mean * free_mask[:, None]
    vel = v_clump[state.clump_id]
    w_clump = (
        jax.random.normal(w_k, (state.N, state.ang_vel.shape[-1])) * free_mask[:, None]
    )  # body frame
    w = w_clump[state.clump_id]
    if state.dim == 2:
        ang_vel = w
    else:  # rotate to lab frame
        ang_vel = state.q.rotate(state.q, w)
    state.vel = vel
    state.ang_vel = ang_vel
    return state


def compute_temperature(
    state: State, can_rotate: bool, subtract_drift: bool, k_B: float = 1.0
) -> jax.Array:
    """Compute the temperature for a state.

    Parameters
    ----------
    state : State
        Current simulation state.
    can_rotate : bool
        Whether to include rigid body rotations.
    subtract_drift : bool
        If True, remove the center-of-mass drift degrees of freedom
        (usually only relevant for small systems).
    k_B : float, optional
        Boltzmann constant (default is 1.0).

    """
    if not math.isfinite(k_B) or k_B <= 0:
        raise ValueError("`k_B` must be finite and positive.")
    n_dof, _, _ = count_dynamic_dofs(state, can_rotate, subtract_drift)
    free = ~state.fixed
    ke = jnp.sum(
        compute_translational_kinetic_energy_per_particle(state) * free, axis=-1
    )
    if can_rotate:
        ke += jnp.sum(
            compute_rotational_kinetic_energy_per_particle(state) * free, axis=-1
        )
    return jnp.where(n_dof > 0, 2 * ke / (k_B * n_dof), 0.0)


def set_temperature(
    state: State,
    target_temperature: float,
    can_rotate: bool,
    subtract_drift: bool,
    seed: int | None = 0,
    k_B: float = 1.0,
) -> State:
    """Randomize the velocities of a state according to a desired temperature.

    Parameters
    ----------
    state : State
        Current simulation state.
    target_temperature : float
        Desired target temperature.
    can_rotate : bool
        Whether to include rigid body rotations.
    subtract_drift : bool
        If True, remove the center-of-mass drift degrees of freedom
        (usually only relevant for small systems).
    seed : int, optional
        RNG seed.
    k_B : float, optional
        Boltzmann constant (default is 1.0).

    """
    state = _assign_random_velocities(state, subtract_drift, seed)
    return scale_to_temperature(
        state, target_temperature, can_rotate, subtract_drift, k_B
    )


def scale_to_temperature(
    state: State,
    target_temperature: float,
    can_rotate: bool,
    subtract_drift: bool,
    k_B: float = 1.0,
) -> State:
    """Scale the velocities of one snapshot to a desired temperature.

    Parameters
    ----------
    state : State
        Current simulation state.
    target_temperature : float
        Desired target temperature.
    can_rotate : bool
        Whether to include rigid body rotations.
    subtract_drift : bool
        If True, remove the center-of-mass drift degrees of freedom
        (usually only relevant for small systems).
    k_B : float, optional
        Boltzmann's constant (default is 1.0).
    """
    if not math.isfinite(target_temperature) or target_temperature < 0:
        raise ValueError("`target_temperature` must be finite and nonnegative.")
    if not math.isfinite(k_B) or k_B <= 0:
        raise ValueError("`k_B` must be finite and positive.")
    if state.pos_c.ndim != 2:
        raise ValueError(
            "Velocity scaling accepts one snapshot; use jax.vmap for stacked states."
        )
    free = ~state.fixed
    if subtract_drift:
        count = jnp.bincount(state.clump_id, length=state.N)[state.clump_id]
        free_mass = state.mass / count * free
        total_free_mass = jnp.sum(free_mass, axis=-1, keepdims=True)
        # Guard only the all-fixed (zero total mass) case; clamping to 1.0
        # would silently corrupt the drift for total masses < 1.
        safe_total_free_mass = jnp.where(total_free_mass == 0, 1.0, total_free_mass)
        v_drift = (
            jnp.sum(state.vel * free_mass[..., None], axis=-2, keepdims=True)
            / safe_total_free_mass[..., None]
        )
        vel = state.vel - v_drift * free[..., None]
    else:
        vel = state.vel
    old_vel = state.vel
    state.vel = vel
    temperature = compute_temperature(state, can_rotate, subtract_drift, k_B)
    state.vel = old_vel
    scale = jnp.where(temperature > 0, jnp.sqrt(target_temperature / temperature), 1.0)
    vel = jnp.where(free[..., None], vel * scale, old_vel)
    # Angular velocities are scaled only when rotations participate in the
    # temperature; otherwise they are left untouched.
    ang_vel = jnp.where(
        free[..., None] & jnp.asarray(can_rotate),
        state.ang_vel * scale,
        state.ang_vel,
    )
    state.vel = vel
    state.ang_vel = ang_vel
    return state
