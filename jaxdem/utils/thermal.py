# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to set and calculate temperature and kinetic energies.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ..state import State


def count_dynamic_dofs(
    state: State, subtract_drift: bool, is_rigid: bool
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Count the number of degrees of freedom for the dynamics
    state: State
    subtract_drift: bool - whether to include center of mass drift (usually only relevant for small systems)
    is_rigid: bool - whether to include rigid body rotations
    """
    cids, offsets = jnp.unique(state.clump_ID, return_index=True)
    free_mask = 1 - state.fixed[offsets]
    free_count = jnp.sum(free_mask)
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
    cids, offsets = jnp.unique(state.clump_ID, return_index=True)
    free_mask = 1 - state.fixed[offsets]
    v_clump = jax.random.normal(v_k, (cids.size, state.dim)) * free_mask[:, None]
    v_clump -= jnp.mean(v_clump, axis=-2) * subtract_drift
    state.vel = v_clump[state.clump_ID]
    w_clump = (
        jax.random.normal(w_k, (cids.size, state.angVel.shape[-1])) * free_mask[:, None]
    )  # body frame
    w = w_clump[state.clump_ID]
    if state.dim == 2:
        state.angVel = w
    else:  # rotate to lab frame
        state.angVel = state.q.rotate(state.q, w)
    return state


def calculate_translational_kinetic_energy(state: State) -> jax.Array:
    """
    Calculate kinetic energy for translations
    KE = 1 / 2 \Sigma_i m_i v_i^2
    """
    cids, offsets = jnp.unique(state.clump_ID, return_index=True)
    return 0.5 * jnp.sum(
        (((1 - state.fixed) * state.mass)[:, None] * (state.vel**2))[offsets], axis=-1
    )


def calculate_rotational_kinetic_energy(state: State) -> jax.Array:
    """
    Calculate kinetic energy for rotations
    KE = 1 / 2 \Sigma_i I_i \omega_i^2
    """
    cids, offsets = jnp.unique(state.clump_ID, return_index=True)
    if state.dim == 2:
        w_body = state.angVel
    else:
        w_body = state.q.rotate_back(state.q, state.angVel)  # to body frame
    return 0.5 * jnp.sum(
        (((1 - state.fixed)[:, None] * state.inertia) * (w_body**2))[offsets], axis=-1
    )


def calculate_temperature(
    state: State, is_rigid: bool, subtract_drift: bool, k_B: Optional[float] = 1.0
) -> float:
    """
    Calculate the temperature for a state
    state: State
    is_rigid: bool - whether to include the rigid body rotations
    subtract_drift: bool - whether to remove center of mass drift (usually only relevant for small systems)
    k_B: Optional[float] - boltzmanns constant, default is 1.0
    """
    n_dof, _, _ = count_dynamic_dofs(state, subtract_drift, is_rigid)
    ke_t = calculate_translational_kinetic_energy(state)
    if is_rigid:
        ke_r = calculate_rotational_kinetic_energy(state)
    else:
        ke_r = 0.0
    ke = jnp.sum(ke_t + ke_r, axis=-1)
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
    # calculate temperature
    temperature = calculate_temperature(state, is_rigid, subtract_drift, k_B)
    # scale to temperature
    scale = jnp.sqrt(target_temperature / temperature)
    state.vel *= scale
    state.angVel *= scale
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
    # calculate temperature
    temperature = calculate_temperature(state, is_rigid, subtract_drift, k_B)
    # scale to temperature
    scale = jnp.sqrt(target_temperature / temperature)
    state.vel *= scale
    state.angVel *= scale
    return state
