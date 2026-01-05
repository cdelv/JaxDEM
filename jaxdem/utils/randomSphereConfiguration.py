# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Generates a random, energy-minimized configurations of spheres in 2D or 3D.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from typing import Sequence, Optional, Tuple

from ..materials import Material, MaterialTable
from ..material_matchmakers import MaterialMatchmaker
from ..minimizers import minimize
from ..state import State
from ..system import System


def _broadcast(arr, is_scalar):
    arr = jnp.asarray(arr)
    assert not (arr.ndim == 0 and not is_scalar), f"This is not a scalar array!"
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if is_scalar:
            return arr[:, None]
        else:
            return arr[None, :]
    return arr

def _pad(arr, size):
    assert not (arr.shape[0] != size and arr.shape[0] != 1)
    return arr * jnp.ones((size, arr.shape[1]), dtype=arr.dtype)

def random_sphere_configuration(
    particle_radii: Sequence[float] | Sequence[Sequence[float]],
    phi: float | Sequence[float],
    dim: int,
    seed: Optional[int] = None,
    collider_type="naive",
    box_aspect: Optional[Sequence[float] | Sequence[Sequence[float]]] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate one or more random sphere packings at a target packing fraction.

    This builds periodic systems with spherical particles, initializes particle
    positions uniformly at random inside a rectangular periodic box, and then minimizes
    the potential energy to obtain a mechanically stable configuration.

    The function supports **batching over multiple independent "systems"** by treating
    the leading axis as the system index and broadcasting any length-1 inputs to match
    the maximum number of systems inferred from the inputs.

    Parameters
    ----------
    particle_radii
        Particle radii for one system or multiple systems.

        - **Single system**: a 1D sequence of length ``N`` (radii for each particle).
        - **Multiple systems**: a 2D sequence with shape ``(S, N)`` (one radii list per system).

        Internally, this is converted to a JAX array with shape ``(S, N)``.
    phi
        Target packing fraction(s).

        - **Scalar**: a single float applied to all systems.
        - **Per-system**: a 1D sequence of length ``S``.

        Internally, this is converted to a JAX array with shape ``(S, 1)`` and then
        broadcast/padded to match the inferred number of systems.
    dim
        Spatial dimension (e.g. 2 or 3).
    seed
        RNG seed used to initialize particle positions. If ``None``, a random seed is
        drawn via NumPy.

        Note: a **single** JAX PRNGKey is used to generate the full position array of
        shape ``(S, N, dim)``.
    collider_type
        Collision detection backend. Must be one of ``"naive"`` or ``"celllist"``.
    box_aspect
        Box aspect ratios for the periodic domain.

        - If ``None``, defaults to ``jnp.ones(dim)``.
        - Otherwise must be a 1D sequence of length ``dim``.

        Internally broadcast/padded to shape ``(S, dim)``.

        (Even though the type annotation allows a sequence-of-sequences, the current
        implementation asserts ``len(box_aspect) == dim`` before broadcasting, so
        per-system ``(S, dim)`` input is not accepted here.)

    Returns
    -------
    pos
        Particle positions after minimization.

        - If ``S > 1``: shape ``(S, N, dim)``.
        - If ``S == 1``: shape ``(N, dim)`` due to ``squeeze()``.
    box_size
        Periodic box size vectors.

        - If ``S > 1``: shape ``(S, dim)``.
        - If ``S == 1``: shape ``(dim,)`` due to ``squeeze()``.

    Notes
    -----
    - **Broadcasting rule**: any input provided for a single system (leading dimension 1)
      is replicated to match the maximum ``S`` inferred from ``particle_radii``, ``phi``,
      and ``box_aspect``.
    - The final ``squeeze()`` calls can also drop other singleton dimensions (e.g. if
      ``N == 1``). If you need stable rank/shape, remove the squeezes.
    """

    # handle seed assignment
    if seed is None:
        seed = np.random.randint(0, 1e9)

    assert collider_type in ["naive", "celllist"], f"Collider type {collider_type} not understood.  Must be one of [naive, celllist]"
    if box_aspect is None:
        box_aspect = jnp.ones(dim)
    else:
        box_aspect = jnp.asarray(box_aspect)
    assert dim == len(box_aspect), f"Box aspect ({len(box_aspect)}) and spatial dimension ({dim}) do not match."

    # broadcast to leading dimension
    particle_radii = _broadcast(particle_radii, is_scalar=False)
    phi = _broadcast(phi, is_scalar=True)
    box_aspect = _broadcast(box_aspect, is_scalar=False)

    # pad to proper sizing
    N_systems = max(arr.shape[0] for arr in [particle_radii, phi, box_aspect])
    particle_radii = _pad(particle_radii, N_systems)
    phi = _pad(phi, N_systems)
    box_aspect = _pad(box_aspect, N_systems)

    e_int = 1.0
    mass = 1.0
    dt = 1e-2
    N = particle_radii.shape[1]

    key = jax.random.PRNGKey(seed)
    mats = [Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = MaterialMatchmaker.create("harmonic")
    mat_table = MaterialTable.from_materials(mats, matcher=matcher)
    pos = jax.random.uniform(key, (N_systems, N, dim), minval=0, maxval=1) * box_aspect[:, None, :]

    def _build_state(i):
        # create system and state
        state = State.create(pos=pos[i], rad=particle_radii[i], mass=mass * jnp.ones(N))

        # box aspect = [a, b, c]
        # box size = l * box aspect
        l = (jnp.sum(state.volume) / (phi[i] * jnp.prod(box_aspect))) ** (1 / dim)
        box_size = l * jnp.array(box_aspect[i])
        state.pos_c *= l

        collider_kw = dict()
        if collider_type == "celllist":
            collider_kw = dict(state=state)

        system = System.create(
            state_shape=state.shape,
            dt=dt,
            linear_integrator_type="linearfire",
            rotation_integrator_type="",
            domain_type="periodic",
            force_model_type="spring",
            collider_type=collider_type,
            collider_kw=collider_kw,
            mat_table=mat_table,
            domain_kw=dict(
                box_size=box_size,
            ),
        )
        return state, system

    state, system = jax.vmap(_build_state)(jnp.arange(N_systems))
    assert jnp.all(jnp.isclose(jnp.sum(state.volume, axis=-1) / jnp.prod(system.domain.box_size, axis=-1), phi.squeeze()))
    state, system, steps, final_pe = jax.vmap(lambda st, sys: minimize(st, sys, max_steps=1_000_000, pe_tol=1e-16, pe_diff_tol=1e-16, initialize=True))(state, system)

    return state.pos.squeeze(), system.domain.box_size.squeeze()
