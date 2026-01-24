# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to compute thermodynamic quantitites.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING, Optional, Tuple
from functools import partial

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@jax.jit
@partial(
    jax.named_call, name="thermal.compute_translational_kinetic_energy_per_particle"
)
def compute_translational_kinetic_energy_per_particle(state: State) -> jax.Array:
    r"""
    compute the translational kinetic energy per particle.

    .. math::
        E_{trans} = \frac{1}{2} m |v|^2

    Notes
    ------
    - The energy of clump members is divided by the number of spheres in the clump.

    Parameters
    ----------
    state : State
        The current state of the system containing particle masses and velocities.

    Returns
    -------
    jax.Array
        An array containing the translational kinetic energy for each particle.
    """
    count = jnp.bincount(state.clump_ID, length=state.N)[state.clump_ID]
    weight = state.mass / count
    return 0.5 * weight * jnp.sum(state.vel * state.vel, axis=-1)


@jax.jit
@partial(jax.named_call, name="thermal.compute_rotational_kinetic_energy_per_particle")
def compute_rotational_kinetic_energy_per_particle(state: State) -> jax.Array:
    r"""
    compute the rotational kinetic energy per particle.

    .. math::
        E_{rot} = \frac{1}{2} \vec{\omega}^T I \vec{\omega}

    Notes
    ------
    - The energy of clump members is divided by the number of spheres in the clump.

    Parameters
    ----------
    state : State
        The current state of the system containing inertia, orientation, and angular velocity.

    Returns
    -------
    jax.Array
        An array containing the rotational kinetic energy for each particle.
    """
    count = jnp.bincount(state.clump_ID, length=state.N)[state.clump_ID]
    if state.dim == 2:
        w_body = state.angVel
    else:
        w_body = state.q.rotate_back(state.q, state.angVel)  # to body frame
    return 0.5 * jnp.vecdot(w_body, state.inertia * w_body) / count


@jax.jit
@partial(jax.named_call, name="thermal.compute_translational_kinetic_energy")
def compute_translational_kinetic_energy(state: State) -> jax.Array:
    r"""
    compute the total translational kinetic energy of the system.

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
    return jnp.sum(compute_translational_kinetic_energy_per_particle(state))


@jax.jit
@partial(jax.named_call, name="thermal.compute_rotational_kinetic_energy")
def compute_rotational_kinetic_energy(state: State) -> jax.Array:
    r"""
    compute the total rotational kinetic energy of the system.

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
    return jnp.sum(compute_rotational_kinetic_energy_per_particle(state))


@jax.jit
@partial(jax.named_call, name="thermal.compute_potential_energy_per_particle")
def compute_potential_energy_per_particle(state: State, system: System) -> jax.Array:
    """
    compute the potential energy per particle based on system interactions.
    Energy is computed from the force models in the collider, and gravity and force functions
    that have potential energy associated with them in the force manager.

    Parameters
    ----------
    state : State
        The current state of the system.
    system : System
        The system definition containing the collider and potential energy functions.

    Returns
    -------
    jax.Array
        An array containing the potential energy for each particle.
    """
    pe_force_manager = system.force_manager.compute_potential_energy(state, system)
    pe_collider = system.collider.compute_potential_energy(state, system)
    return pe_force_manager + pe_collider


@jax.jit
@partial(jax.named_call, name="thermal.compute_potential_energy")
def compute_potential_energy(state: State, system: System) -> jax.Array:
    r"""
    compute the total potential energy of the system. Energy is computed from the force models in the collider, and gravity and force functions
    that have potential energy associated with them in the force manager.

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
        The scalar sum of potential energy across all particles.
    """
    return jnp.sum(compute_potential_energy_per_particle(state, system))


@jax.jit
@partial(jax.named_call, name="thermal.compute_energy")
def compute_energy(state: State, system: System) -> jax.Array:
    """
    compute the total mechanical energy of the system.

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


def count_dynamic_dofs(
    state: State, subtract_drift: bool, is_rigid: bool
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Count the number of degrees of freedom for the dynamics
    state: State
    subtract_drift: bool - whether to include center of mass drift (usually only relevant for small systems)
    is_rigid: bool - whether to include rigid body rotations
    """
    counts = jnp.bincount(state.clump_ID, length=state.N)
    fixed_counts = jnp.bincount(
        state.clump_ID, weights=state.fixed.astype(jnp.int32), length=state.N
    )
    free_count = jnp.sum((counts > 0) & (fixed_counts == 0))
    n_dof_v = (free_count - subtract_drift) * state.vel.shape[-1]
    n_dof_w = free_count * state.angVel.shape[-1] * is_rigid
    n_dof = n_dof_v + n_dof_w
    return n_dof, n_dof_v, n_dof_w


def _assign_random_velocities(
    state: State, subtract_drift: bool, seed: Optional[int] = None
) -> State:
    """
    Assign random translational and angular velocities
    state: State
    subtract_drift: bool - whether to remove center of mass drift
    seed: Optional[int] - rng seed
    """
    if seed is None:
        seed = np.random.randint(0, 1e9)
    key = jax.random.PRNGKey(seed)
    v_k, w_k = jax.random.split(key, 2)
    counts = jnp.bincount(state.clump_ID, length=state.N)
    exists = counts > 0
    fixed_counts = jnp.bincount(
        state.clump_ID, weights=state.fixed.astype(jnp.int32), length=state.N
    )
    free_mask = (fixed_counts == 0) & exists
    v_clump = jax.random.normal(v_k, (state.N, state.dim)) * free_mask[:, None]
    if subtract_drift:
        num_clumps = jnp.sum(exists)
        v_clump_mean = jnp.sum(v_clump, axis=0) / jnp.maximum(num_clumps, 1)
        v_clump -= v_clump_mean * exists[:, None]
    state.vel = v_clump[state.clump_ID]
    w_clump = (
        jax.random.normal(w_k, (state.N, state.angVel.shape[-1])) * free_mask[:, None]
    )  # body frame
    w = w_clump[state.clump_ID]
    if state.dim == 2:
        state.angVel = w
    else:  # rotate to lab frame
        state.angVel = state.q.rotate(state.q, w)
    return state


def compute_temperature(
    state: State, is_rigid: bool, subtract_drift: bool, k_B: Optional[float] = 1.0
) -> float:
    """
    compute the temperature for a state
    state: State
    is_rigid: bool - whether to include the rigid body rotations
    subtract_drift: bool - whether to remove center of mass drift (usually only relevant for small systems)
    k_B: Optional[float] - boltzmanns constant, default is 1.0
    """
    n_dof, _, _ = count_dynamic_dofs(state, subtract_drift, is_rigid)
    ke_t = compute_translational_kinetic_energy(state)
    if is_rigid:
        ke_r = compute_rotational_kinetic_energy(state)
    else:
        ke_r = 0.0
    ke = ke_t + ke_r
    return 2 * ke / (k_B * n_dof)


def set_temperature(
    state: State,
    target_temperature: float,
    is_rigid: bool,
    subtract_drift: bool,
    seed: Optional[int] = None,
    k_B: Optional[float] = 1.0,
) -> State:
    """
    Randomize the velocities of a state according to a desired temperature
    state: State
    target_temperature: float - desired target temperature
    is_rigid: bool - whether to include the rigid body rotations
    subtract_drift: bool - whether to remove center of mass drift (usually only relevant for small systems)
    seed: Optional[int] - rng seed
    k_B: Optional[float] - boltzmanns constant, default is 1.0
    """
    # assign random
    state = _assign_random_velocities(state, subtract_drift, seed)
    # compute temperature
    temperature = compute_temperature(state, is_rigid, subtract_drift, k_B)
    # scale to temperature
    scale = jnp.sqrt(target_temperature / temperature)
    state.vel *= scale
    state.angVel *= scale * is_rigid
    return state


def scale_to_temperature(
    state: State,
    target_temperature: float,
    is_rigid: bool,
    subtract_drift: bool,
    k_B: Optional[float] = 1.0,
) -> State:
    """
    Scale the velocities of a state to a desired temperature
    state: State
    target_temperature: float - desired target temperature
    is_rigid: bool - whether to include the rigid body rotations
    subtract_drift: bool - whether to remove center of mass drift (usually only relevant for small systems)
    k_B: Optional[float] - boltzmanns constant, default is 1.0
    """
    # subtract drift
    state.vel -= jnp.mean(state.vel, axis=-2) * subtract_drift
    # compute temperature
    temperature = compute_temperature(state, is_rigid, subtract_drift, k_B)
    # scale to temperature
    scale = jnp.sqrt(target_temperature / temperature)
    state.vel *= scale
    state.angVel *= scale * is_rigid
    return state
