# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Sweep and prune :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax import tree_util

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Tuple, cast
from functools import partial

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

_jit = cast(Callable[..., Any], jax.jit)
_named_call = cast(Callable[..., Any], jax.named_call)


@_jit
@partial(_named_call, name="pad_to_power2")
def pad_to_power2(x: jax.Array) -> jax.Array:
    """
    Pad odd-dimensional vectors to an even size (Pallas kernel limitation).
    """
    if x.ndim != 2:
        return x
    n, dim = x.shape
    target_dim = dim + dim % 2
    return jnp.pad(x, ((0, 0), (0, target_dim - dim)), constant_values=0.0)


@partial(jax.profiler.annotate_function, name="sap_kernel_full")
def sap_kernel_full(
    state_ref: Any,
    system_ref: Any,
    aabb_ref: jax.Array,
    m_ref: jax.Array,
    M_ref: jax.Array,
    HASH_ref: jax.Array,
    forces_ref: jax.Array,
) -> None:
    i = pl.num_programs(1) * pl.program_id(0) + pl.program_id(1)
    n = state_ref.N
    dim = state_ref.dim
    M_i = M_ref[i]
    # pos = state_ref.pos_c[i] + state_ref.q.rotate(state_ref.q[i], state_ref.pos_p[i])
    pos_i = state_ref.pos_c[i]
    aabb_i = aabb_ref[i]
    HASH_i = HASH_ref[i]
    forces_ref[i] = jnp.zeros_like(pos_i)

    def cond(j: jax.Array) -> jax.Array:
        return jnp.logical_and(j < n, m_ref[j] <= M_i)

    def body(j: jax.Array) -> jax.Array:
        pos_j = state_ref.pos_c[j]
        aabb_j = aabb_ref[j]
        r_ij = system_ref.domain.displacement(pos_i, pos_j, system_ref)
        overlap = jnp.sum(jnp.abs(r_ij) <= (aabb_i + aabb_j)) == dim
        f, t = force(i, j, state_ref, system_ref)
        f *= overlap

        pl.atomic_add(forces_ref, (i, slice(None)), f)  # type: ignore[attr-defined]
        pl.atomic_add(forces_ref, (j, slice(None)), -f)  # type: ignore[attr-defined]
        return j + 1

    jax.lax.while_loop(cond, body, i + 1)


@_jit
@partial(jax.profiler.annotate_function, name="compute_hash")
def compute_hash(
    state: Any, proj_perp: jax.Array, aabb: jax.Array, shift: jax.Array
) -> jax.Array:
    cell_size = 4 * jnp.max(aabb)
    proj_min = proj_perp.min(axis=0)
    proj_max = proj_perp.max(axis=0)
    grid_dims = jnp.maximum(
        1, jnp.ceil((proj_max - proj_min + 2 * cell_size) / cell_size).astype(int)
    )
    multipliers = jnp.concatenate([jnp.ones(1, dtype=int), jnp.cumprod(grid_dims[:-1])])
    cell_idx = jnp.floor((proj_perp + shift * cell_size / 2) / cell_size).astype(int)
    return jnp.dot(cell_idx, multipliers)


@_jit
@partial(jax.profiler.annotate_function, name="compute_virtual_shift")
def compute_virtual_shift(
    m: jax.Array, M: jax.Array, HASH: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    shift = M.max() - m.min()
    virtual_shift1 = 2 * HASH * shift
    return m + virtual_shift1, M + virtual_shift1


@_jit
@partial(jax.profiler.annotate_function, name="sort")
def sort(
    state: "State", iota: jax.Array, m: jax.Array, M: jax.Array
) -> Tuple["State", jax.Array, jax.Array, jax.Array]:
    m, M, perm = jax.lax.sort([m, M, iota], num_keys=1)
    state = tree_util.tree_map(lambda x: x[perm], state)
    return state, m, M, perm


@_jit
@partial(jax.profiler.annotate_function, name="pad_state")
def pad_state(state: "State") -> "State":
    return tree_util.tree_map(pad_to_power2, state)


@partial(_jit, inline=True)
@partial(_named_call, name="SpringForce.force")
def force(
    i: int, j: int, state: "State", system: "System"
) -> Tuple[jax.Array, jax.Array]:
    """
    Compute linear spring-like interaction force acting on particle :math:`i` due to particle :math:`j`.

    Returns zero when :math:`i = j`.

    Parameters
    ----------
    i : int
        Index of the first particle.
    j : int
        Index of the second particle.
    state : State
        Current state of the simulation.
    system : System
        Simulation system configuration.

    Returns
    -------
    jax.Array
        Force vector acting on particle :math:`i` due to particle :math:`j`.
    """
    mi, mj = state.mat_id[i], state.mat_id[j]
    k = system.mat_table.young_eff[mi, mj]
    R = state.rad[i] + state.rad[j]

    rij = system.domain.displacement(state.pos_c[i], state.pos_c[j], system)
    r = jnp.sum(rij**2, axis=-1)
    r = jnp.where(r == 0, 1.0, jnp.sqrt(r))
    # s = jnp.maximum(0.0, R / r - 1.0)
    s = R / r - 1.0
    s *= s > 0
    return (k * s)[..., None] * rij, jnp.zeros_like(state.angVel[i])


@Collider.register("sap")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SweepAndPrune(Collider):
    @staticmethod
    @partial(_jit, static_argnames=("max_neighbors",))
    @partial(_named_call, name="SweepAndPrune.create_neighbor_list")
    def create_neighbor_list(
        state: "State",
        system: "System",
        cutoff: float,
        max_neighbors: int,
    ) -> Tuple["State", "System", jax.Array, jax.Array]:
        raise NotImplementedError(
            "SweepAndPrune does not implement create_neighbor_list"
        )

    @staticmethod
    @partial(_jit, donate_argnames=("state", "system"))
    @partial(_named_call, name="SweepAndPrune.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        aabb = state.rad[:, None] * jnp.ones((1, state.pos.shape[1]))
        chunk_size = 1
        n, dim = state.pos.shape
        iota = jax.lax.iota(int, n)
        quantization = jnp.min(aabb) / 4.0
        m = state.pos - aabb
        M = state.pos + aabb

        # 1) PCA sweep direction
        # v, v_perp = PCA_decomposition(m)
        I = jnp.eye(dim, dtype=state.pos.dtype)
        v = I[:, 0]
        v_perp = I[:, 1:]

        # 2) Project into perpendicular plane
        proj_perp = jnp.dot(m, v_perp)

        # 3) Create grid in perpendicular directions
        HASH1 = compute_hash(state, proj_perp, aabb, 0.0)
        HASH2 = compute_hash(state, proj_perp, aabb, 1.0)

        # project into sweeping direction and quantize to integers for performance
        m = (jnp.dot(m / quantization, v)).astype(int)
        M = (jnp.dot(M / quantization, v)).astype(int)

        m1, M1 = compute_virtual_shift(m, M, HASH1)
        m2, M2 = compute_virtual_shift(m, M, HASH2)

        # Sort particles by shifted sweep coordinates
        state1, m1, M1, perm1 = sort(state, iota, m1, M1)
        state2, m2, M2, perm2 = sort(state, iota, m2, M2)

        # First SaP pass - compute all interactions in the cell
        state_padded1 = pad_state(state1)
        state_padded1.force = pl.pallas_call(
            sap_kernel_full,
            out_shape=state_padded1.force,
            grid=(n // chunk_size + 1, chunk_size),
            interpret=False,
            name="First pass",
        )(state_padded1, system, aabb, m1, M1, iota)

        # Second SaP pass - skip same hash interactions
        state_padded2 = pad_state(state2)
        state_padded2.force = pl.pallas_call(
            sap_kernel_full,
            out_shape=state_padded2.force,
            grid=(n // chunk_size + 1, chunk_size),
            interpret=False,
            name="Second pass",
        )(state_padded2, system, aabb, m2, M2, HASH1[perm2])

        # Combine forces and unpermute
        perm2 = perm2.at[perm2].set(iota)
        state_padded2.force = state_padded2.force[:, :dim][perm2]  # unpad
        state1.force = (
            state_padded1.force[:, :dim] + state_padded2.force[perm1]
        ) / state_padded1.mass[:, None]

        return state1, system


# Backwards-compatible alias.
SweeAPrune = SweepAndPrune
