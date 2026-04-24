# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Generates a random, energy-minimized configurations of spheres in 2D or 3D."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from typing import Any
from collections.abc import Sequence

from ..materials import Material, MaterialTable
from ..material_matchmakers import MaterialMatchmaker
from ..minimizers import minimize
from ..state import State
from ..system import System


def _broadcast(arr: Any, is_scalar: bool) -> jax.Array:
    arr = jnp.asarray(arr)
    assert not (arr.ndim == 0 and not is_scalar), "This is not a scalar array!"
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if is_scalar:
            return arr[:, None]
        return arr[None, :]
    return arr


def _pad(arr: jax.Array, size: int) -> jax.Array:
    assert not (arr.shape[0] != size and arr.shape[0] != 1)
    return arr * jnp.ones((size, arr.shape[1]), dtype=arr.dtype)


def random_sphere_configuration(
    particle_radii: Sequence[float] | Sequence[Sequence[float]],
    phi: float | Sequence[float],
    dim: int,
    seed: int | None = None,
    collider_type: str = "naive",
    box_aspect: Sequence[float] | Sequence[Sequence[float]] | None = None,
    max_avg_pe: float | None = 1e-16,
    domain_type: str = "periodic",
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    max_avg_pe
        Maximum potential energy per particle allowed in the configuration.  The minimizer will attempt
        to adjust the sphere positions until this is met.  If far above the jamming density, the minimizer
        will likely run for the maximum number of steps, and may take unnecessarily long to terminate.

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
        seed = int(np.random.randint(0, int(1e9)))

    assert collider_type in [
        "naive",
        "celllist",
    ], f"Collider type {collider_type} not understood.  Must be one of [naive, celllist]"
    box_aspect_input = (
        jnp.ones(dim) if box_aspect is None else jnp.asarray(box_aspect, dtype=float)
    )
    assert dim == len(
        box_aspect_input
    ), f"Box aspect ({len(box_aspect_input)}) and spatial dimension ({dim}) do not match."

    # Broadcast inputs so we can precompute per-system box sizes here, draw
    # random positions uniformly inside each box, and then delegate the
    # state/system build + minimize to `minimize_sphere_configuration`.
    particle_radii_arr = _broadcast(particle_radii, is_scalar=False)
    phi_arr = _broadcast(phi, is_scalar=True)
    box_aspect_arr = _broadcast(box_aspect_input, is_scalar=False)

    N_systems = max(
        arr.shape[0] for arr in [particle_radii_arr, phi_arr, box_aspect_arr]
    )
    particle_radii_arr = _pad(particle_radii_arr, N_systems)
    phi_arr = _pad(phi_arr, N_systems)
    box_aspect_arr = _pad(box_aspect_arr, N_systems)

    N = particle_radii_arr.shape[1]

    # V_d(r) = pi^(d/2) / Gamma(d/2 + 1) * r^d (same formula State.create uses
    # for the default volume field).
    vol_per_particle = jnp.exp(
        0.5 * dim * jnp.log(jnp.pi)
        + dim * jnp.log(particle_radii_arr)
        - jax.scipy.special.gammaln(0.5 * dim + 1.0)
    )
    total_vol = jnp.sum(vol_per_particle, axis=1)  # (S,)
    # l chosen so that prod(box_size) = prod(l * aspect) = total_vol / phi.
    l = (total_vol / (phi_arr[:, 0] * jnp.prod(box_aspect_arr, axis=1))) ** (1 / dim)
    box_size = l[:, None] * box_aspect_arr  # (S, dim)

    key = jax.random.PRNGKey(seed)
    pos = (
        jax.random.uniform(key, (N_systems, N, dim), minval=0, maxval=1)
        * box_size[:, None, :]
    )

    return minimize_sphere_configuration(
        particle_radii=particle_radii_arr,
        pos=pos,
        phi=phi_arr,
        dim=dim,
        collider_type=collider_type,
        box_aspect=box_aspect,
        max_avg_pe=max_avg_pe,
        domain_type=domain_type,
    )


def minimize_sphere_configuration(
    particle_radii: Sequence[float] | Sequence[Sequence[float]],
    pos: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]] | jax.Array,
    phi: float | Sequence[float],
    dim: int,
    collider_type: str = "naive",
    box_aspect: Sequence[float] | Sequence[Sequence[float]] | None = None,
    max_avg_pe: float | None = 1e-16,
    domain_type: str = "periodic",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Minimize a user-supplied sphere configuration at a target packing fraction.

    Builds a periodic box whose volume is ``sum(particle_volume) / phi`` (with
    the requested aspect ratio), places the particles at ``pos`` in that box,
    and FIRE-minimizes the potential energy.

    Parameters
    ----------
    particle_radii
        Particle radii for one or multiple systems.

        - **Single system**: 1D sequence of length ``N``.
        - **Multiple systems**: 2D sequence of shape ``(S, N)``.
    pos
        Particle centers in **absolute** coordinates (same frame as the final
        box anchored at the origin). Shape ``(N, dim)`` or ``(S, N, dim)``.
        Positions outside the computed box are wrapped by the periodic
        domain during minimization.
    phi
        Target packing fraction. Scalar or 1D sequence of length ``S``.
    dim
        Spatial dimension (2 or 3).
    collider_type
        Collision-detection backend, ``"naive"`` or ``"celllist"``.
    box_aspect
        Aspect ratios of the periodic box. ``None`` defaults to
        ``jnp.ones(dim)``. Shape ``(dim,)``; broadcast to ``(S, dim)``.
    max_avg_pe
        Convergence tolerance used for both ``pe_tol`` and ``pe_diff_tol``.
    domain_type
        Boundary condition for the analogue sphere system. Must be a
        registered :class:`Domain` type (e.g. ``"periodic"`` or
        ``"reflect"``). Default ``"periodic"``.

    Returns
    -------
    pos
        Minimized particle positions (shape ``(S, N, dim)``, squeezed if
        ``S == 1``).
    box_size
        Periodic box size vector(s) (shape ``(S, dim)``, squeezed if
        ``S == 1``).
    """
    assert collider_type in [
        "naive",
        "celllist",
    ], f"Collider type {collider_type} not understood.  Must be one of [naive, celllist]"
    box_aspect_input = (
        jnp.ones(dim) if box_aspect is None else jnp.asarray(box_aspect, dtype=float)
    )
    assert dim == len(
        box_aspect_input
    ), f"Box aspect ({len(box_aspect_input)}) and spatial dimension ({dim}) do not match."

    particle_radii_arr = _broadcast(particle_radii, is_scalar=False)
    phi_arr = _broadcast(phi, is_scalar=True)
    box_aspect_arr = _broadcast(box_aspect_input, is_scalar=False)

    pos_arr = jnp.asarray(pos, dtype=float)
    if pos_arr.ndim == 2:
        pos_arr = pos_arr[None, :, :]
    if pos_arr.ndim != 3:
        raise ValueError(
            f"pos must have shape (N, dim) or (S, N, dim); got {pos_arr.shape}"
        )
    if pos_arr.shape[-1] != dim:
        raise ValueError(
            f"pos last axis must equal dim={dim}; got pos.shape={pos_arr.shape}."
        )
    if pos_arr.shape[1] != particle_radii_arr.shape[1]:
        raise ValueError(
            f"pos has N={pos_arr.shape[1]} particles but particle_radii has "
            f"N={particle_radii_arr.shape[1]}."
        )

    N_systems = max(
        arr.shape[0] for arr in [particle_radii_arr, phi_arr, box_aspect_arr, pos_arr]
    )
    particle_radii_arr = _pad(particle_radii_arr, N_systems)
    phi_arr = _pad(phi_arr, N_systems)
    box_aspect_arr = _pad(box_aspect_arr, N_systems)
    if pos_arr.shape[0] == 1 and N_systems > 1:
        pos_arr = jnp.broadcast_to(pos_arr, (N_systems, *pos_arr.shape[1:]))
    assert (
        pos_arr.shape[0] == N_systems
    ), f"pos leading axis {pos_arr.shape[0]} does not match N_systems={N_systems}."

    e_int = 1.0
    mass = 1.0
    dt = 1e-2
    N = particle_radii_arr.shape[1]

    mats = [Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = MaterialMatchmaker.create("harmonic")
    mat_table = MaterialTable.from_materials(mats, matcher=matcher)

    def _build_state(i: jax.Array) -> tuple[State, System]:
        state = State.create(
            pos=pos_arr[i], rad=particle_radii_arr[i], mass=mass * jnp.ones(N)
        )

        # box aspect = [a, b, c] -> box size = l * box_aspect, with l chosen
        # so that prod(box_size) = sum(particle_volume) / phi.
        l = (jnp.sum(state.volume) / (phi_arr[i] * jnp.prod(box_aspect_arr[i]))) ** (
            1 / dim
        )
        box_size = l * jnp.array(box_aspect_arr[i])

        collider_kw = {}
        if collider_type == "celllist":
            collider_kw = {"state": state}

        system = System.create(
            state_shape=state.shape,
            dt=dt,
            linear_integrator_type="linearfire",
            rotation_integrator_type="",
            domain_type=domain_type,
            force_model_type="spring",
            collider_type=collider_type,
            collider_kw=collider_kw,
            mat_table=mat_table,
            domain_kw={
                "box_size": box_size,
            },
        )
        return state, system

    state, system = jax.vmap(_build_state)(jnp.arange(N_systems))
    assert jnp.all(
        jnp.isclose(
            jnp.sum(state.volume, axis=-1) / jnp.prod(system.domain.box_size, axis=-1),
            phi_arr.squeeze(),
        )
    )
    state, system, _steps, _final_pe = jax.vmap(
        lambda st, sys: minimize(
            st,
            sys,
            max_steps=1_000_000,
            pe_tol=max_avg_pe,
            pe_diff_tol=max_avg_pe,
            initialize=True,
        )
    )(state, system)

    return state.pos.squeeze(), system.domain.box_size.squeeze()
