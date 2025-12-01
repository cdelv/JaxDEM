# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Sweep and prune :math:`O(N log N)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax import tree_util

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING, cast
from functools import partial

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit
@partial(jax.named_call, name="pad_to_power2")
def pad_to_power2(x):
    """
    Pad 3D simulations to 4D (Pallas Kernel limitations)
    """
    if x.ndim != 2:
        return x
    n, dim = x.shape
    target_dim = dim + dim % 2
    return jnp.pad(x, ((0, 0), (0, target_dim - dim)), constant_values=0.0)


from dataclasses import replace


from dataclasses import replace


# 1. Define RefProxy (Make sure this is in your file, outside the kernel)
# --- 1. Define and Register RefProxy ---
# Place this at the top level of your file (before the kernel)


class RefProxy:
    """Wraps a Pallas Ref to behave like a JAX Array with auto-loading."""

    def __init__(self, ref):
        self.ref = ref

    # Properties to mimic array attributes
    @property
    def shape(self):
        return self.ref.shape

    @property
    def dtype(self):
        return self.ref.dtype

    @property
    def ndim(self):
        return len(self.ref.shape)

    # Auto-load on indexing
    def __getitem__(self, idx):
        # Heuristic: If 2D and index is scalar, load row. Otherwise load element.
        if self.ndim == 2 and not isinstance(idx, tuple):
            return pl.load(self.ref, (idx, slice(None)))
        return pl.load(self.ref, (idx,) if not isinstance(idx, tuple) else idx)

    # Helper to load the full array for math ops
    def _load_all(self):
        return pl.load(self.ref, (slice(None),) * self.ndim)

    # Math operators (RefProxy + x, x / RefProxy, etc.)
    def __add__(self, o):
        return self._load_all() + o

    def __radd__(self, o):
        return o + self._load_all()

    def __sub__(self, o):
        return self._load_all() - o

    def __rsub__(self, o):
        return o - self._load_all()

    def __mul__(self, o):
        return self._load_all() * o

    def __rmul__(self, o):
        return o * self._load_all()

    def __truediv__(self, o):
        return self._load_all() / o

    def __rtruediv__(self, o):
        return o / self._load_all()

    def __neg__(self):
        return -self._load_all()

    # Catch-all for other attributes (like .T)
    def __getattr__(self, name):
        return getattr(self._load_all(), name)


# --- CRITICAL: JAX PyTree Registration ---
def _ref_proxy_flatten(proxy):
    # Tell JAX: "I contain this ref"
    return (proxy.ref,), None


def _ref_proxy_unflatten(aux, children):
    # Tell JAX: "Rebuild me using this ref"
    return RefProxy(children[0])


jax.tree_util.register_pytree_node(RefProxy, _ref_proxy_flatten, _ref_proxy_unflatten)


# --- 2. The Universal Kernel ---


@partial(jax.profiler.annotate_function, name="sap_kernel_full")
def sap_kernel_full(
    state_ref, system_ref, aabb_ref, m_ref, M_ref, HASH_ref, forces_ref
):
    i = pl.num_programs(1) * pl.program_id(0) + pl.program_id(1)

    # Wrap Inputs in Proxies
    # We use duck typing (hasattr "shape") to find Refs, as pl.Ref isn't always exposed
    state_proxy = jax.tree.map(
        lambda x: RefProxy(x) if hasattr(x, "shape") else x, state_ref
    )
    system_proxy = jax.tree.map(
        lambda x: RefProxy(x) if hasattr(x, "shape") else x, system_ref
    )

    M_i = pl.load(M_ref, (i,))
    pos_i = pl.load(state_ref.pos, (i, slice(None)))
    aabb_i = pl.load(aabb_ref, (i, slice(None)))

    pl.store(forces_ref, (i, slice(None)), jnp.zeros_like(pos_i))

    def cond(j):
        n = state_ref.pos.shape[0]
        return (j < n) * (pl.load(m_ref, (j,)) <= M_i)

    def body(j):
        pos_j = pl.load(state_ref.pos, (j, slice(None)))
        aabb_j = pl.load(aabb_ref, (j, slice(None)))

        # Now works seamlessly with JIT-ed displacement
        r_ij = system_proxy.domain.displacement(pos_i, pos_j, system_proxy)

        overlap = jnp.sum(jnp.abs(r_ij) <= (aabb_i + aabb_j)) == state_ref.pos.shape[1]

        def compute_force_wrapper(_):
            # Now works seamlessly with JIT-ed force
            # JAX flattens system_proxy -> sees Ref -> passes Ref -> rebuilds RefProxy inside
            return system_ref.force_model.force(state_proxy, system_proxy, i, j)[0]

        f = jax.lax.cond(
            overlap,
            compute_force_wrapper,
            lambda _: jnp.zeros_like(pos_i),
            operand=None,
        )

        pl.atomic_add(forces_ref, (i, slice(None)), f)
        pl.atomic_add(forces_ref, (j, slice(None)), -f)
        return j + 1

    jax.lax.while_loop(cond, body, i + 1)


@jax.jit
@partial(jax.profiler.annotate_function, name="compute_hash")
def compute_hash(state, proj_perp, aabb, shift):
    cell_size = 4 * jnp.max(aabb)
    proj_min = proj_perp.min(axis=0)
    proj_max = proj_perp.max(axis=0)
    grid_dims = jnp.maximum(
        1, jnp.ceil((proj_max - proj_min + 2 * cell_size) / cell_size).astype(int)
    )
    multipliers = jnp.concatenate([jnp.ones(1, dtype=int), jnp.cumprod(grid_dims[:-1])])
    cell_idx = jnp.floor((proj_perp + shift * cell_size / 2) / cell_size).astype(int)
    return jnp.dot(cell_idx, multipliers)


@jax.jit
@partial(jax.profiler.annotate_function, name="compute_virtual_shift")
def compute_virtual_shift(m, M, HASH):
    shift = M.max() - m.min()
    virtual_shift1 = 2 * HASH * shift
    return m + virtual_shift1, M + virtual_shift1


@jax.jit
@partial(jax.profiler.annotate_function, name="sort")
def sort(state, iota, m, M):
    m, M, perm = jax.lax.sort([m, M, iota], num_keys=1)
    state = tree_util.tree_map(lambda x: x[perm], state)
    return state, m, M, perm


@jax.jit
@partial(jax.profiler.annotate_function, name="padd")
def padd(state):
    return tree_util.tree_map(pad_to_power2, state)


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="SpringForce.force")
def force(
    i: int, j: int, state: "State", system: "System"
) -> Tuple[jax.Array, jax.Array]:
    # 1. Load data from Refs
    mat_id_i = pl.load(state.mat_id, (i,))
    mat_id_j = pl.load(state.mat_id, (j,))
    rad_i = pl.load(state.rad, (i,))
    rad_j = pl.load(state.rad, (j,))
    pos_i = pl.load(state.pos, (i, slice(None)))
    pos_j = pl.load(state.pos, (j, slice(None)))

    # 2. Lookup Stiffness (Must use pl.load on the Ref)
    k = pl.load(system.mat_table.young_eff, (mat_id_i, mat_id_j))

    # 3. Calculate Displacement
    # (Works because system.domain.box_size is a Value in the hybrid object)
    rij = system.domain.displacement(pos_i, pos_j, system)

    # --- THE FIX FOR THE ERROR ---
    # Old: r = jnp.vecdot(rij, rij)  <-- Causes "must be 2D" error in Triton
    # New: Element-wise multiply + sum
    r_sq = jnp.sum(rij * rij)
    r = jnp.sqrt(r_sq + jnp.finfo(pos_i.dtype).eps)

    # 4. Force Calculation
    R = rad_i + rad_j
    s = R / r - 1.0
    s *= s > 0

    return k * s * rij, jnp.zeros_like(pos_i)


@Collider.register("sap")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SweeAPrune(Collider):
    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SweeAPrune.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System"):
        pass

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="SweeAPrune.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        aabb = state.rad[:, None] * jnp.ones((1, state.pos.shape[1]))
        chunk_size = 16
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

        # project into sweeping direction and quantizatize to integers for performance
        m = (jnp.dot(m / quantization, v)).astype(int)
        M = (jnp.dot(M / quantization, v)).astype(int)

        m1, M1 = compute_virtual_shift(m, M, HASH1)
        m2, M2 = compute_virtual_shift(m, M, HASH2)

        # Sort particles by shifted sweep coordinates
        state1, m1, M1, perm1 = sort(state, iota, m1, M1)
        state2, m2, M2, perm2 = sort(state, iota, m2, M2)

        # First SaP pass - compute all interactions in the cell
        state_padded1 = padd(state1)
        state_padded1.force = pl.pallas_call(
            sap_kernel_full,
            out_shape=state_padded1.force,
            grid=(n // chunk_size + 1, chunk_size),
            interpret=False,
            name="First pass",
        )(state_padded1, system, aabb, m1, M1, iota)

        # Second SaP pass - skip same hash interactions
        state_padded2 = padd(state2)
        state_padded2.force = pl.pallas_call(
            sap_kernel_full,
            out_shape=state_padded2.force,
            grid=(n // chunk_size + 1, chunk_size),
            interpret=False,
            name="Second pass",
        )(state_padded2, system, aabb, m2, M2, HASH1[perm2])

        # Combine forces and unpermute
        perm2 = perm2.at[perm2].set(iota)
        state_padded2.force = state_padded2.force[:, :dim][perm2]  # unpadd
        state1.force = (
            state_padded1.force[:, :dim] + state_padded2.force[perm1]
        ) / state_padded1.mass[:, None]

        return state1, system
