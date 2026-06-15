# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for calculating and changing the packing fraction."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from ..colliders import NeighborList

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@jax.jit
def compute_particle_volume(
    state: State,
) -> jax.Array:  # This is not the proper instantaneous volume for DPs
    """Return the total particle volume."""
    seg = jax.ops.segment_max(state.volume, state.clump_id, num_segments=state.N)
    return jnp.sum(jnp.maximum(seg, 0.0))


@jax.jit
def compute_packing_fraction(state: State, system: System) -> jax.Array:
    """Return the packing fraction ``total_particle_volume / box_volume``.

    Assumes the domain anchor is at the origin, so the box volume is simply
    ``prod(system.domain.box_size)``.
    """
    return compute_particle_volume(state) / jnp.prod(system.domain.box_size)


def _host_body_grouping(clump_id: Any, bond_id: Any) -> Any:
    """Host-side per-particle body labels (int32, shape ``(N,)``).

    Uses the bond graph's connected components when the state contains DPs,
    otherwise falls back to ``clump_id``. Batch/trajectory leading dimensions
    are flattened (the bond topology is static across a batch).
    """
    import numpy as np

    from .particle_creation import _bond_graph_components

    clump_ids_np = np.asarray(clump_id)
    bond_ids_np = np.asarray(bond_id)
    n = clump_ids_np.shape[-1]

    # Flatten batch/trajectory dimensions if present
    clump_ids_np = clump_ids_np.reshape(-1, n)[0]
    bond_ids_np = bond_ids_np.reshape(-1, n, bond_ids_np.shape[-1])[0]

    n_unique_bond, bond_group_id = _bond_graph_components(bond_ids_np)
    has_dps = n_unique_bond < n

    if has_dps:
        return bond_group_id.astype(int)
    return clump_ids_np.astype(int)


@jax.jit
def _scale_to_packing_fraction_grouped(
    state: State, system: System, new_packing_fraction: float, group_id: jax.Array
) -> tuple[State, System]:
    """Rescale the box (preserving its aspect ratio) to ``new_packing_fraction``.

    ``group_id`` is the per-particle body label (see :func:`_host_body_grouping`);
    it is taken as an argument so callers running compression/jamming loops can
    compute it once (the bond topology is static) instead of paying a host
    callback per iteration.
    """
    # this assumes that the domain anchor is 0.
    # All box dimensions are scaled by a single common factor so anisotropic
    # boxes keep their aspect ratio and positions stay inside the box.
    box_size = system.domain.box_size
    scale_factor = (
        compute_particle_volume(state) / (new_packing_fraction * jnp.prod(box_size))
    ) ** (1.0 / state.dim)
    new_domain = replace(system.domain, box_size=box_size * scale_factor)

    # For spheres and clumps, we can just rescale the positions via state.pos_c * scale_factor
    # But, for DPs, we need to scale the com positions
    # Both behaviors can be generalized by scaling the DP com positions, finding the offset
    # before and after the scaling, and applying the offset to state.pos_c
    # This preserves the size of the DPs, clumps, and spheres, uniformly
    total_pos = jax.ops.segment_sum(state.pos_c, group_id, num_segments=state.N)
    dp_counts = jax.ops.segment_sum(
        jnp.ones((state.N,), dtype=state.pos_c.dtype),
        group_id,
        num_segments=state.N,
    )
    dp_com = total_pos / jnp.maximum(
        dp_counts[:, None], 1.0
    )  # avoid divide by zero errors for empty clumps (MAY NOT BE NEEDED)
    offset = dp_com * scale_factor - dp_com

    state.pos_c = state.pos_c + offset[group_id]
    new_system = replace(system, domain=new_domain)

    # force rebuild the neighbor list if using it
    if isinstance(new_system.collider, NeighborList):
        new_system = replace(
            new_system,
            collider=replace(
                new_system.collider, n_build_times=jnp.array(0, dtype=int)
            ),
        )

    return state, new_system


@jax.jit
def scale_to_packing_fraction(
    state: State, system: System, new_packing_fraction: float
) -> tuple[State, System]:
    """Rescale the box to ``new_packing_fraction``, preserving its aspect ratio."""
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), int),  # type: ignore[no-untyped-call]
        state.clump_id,
        state.bond_id,
        vmap_method="sequential",
    )
    return _scale_to_packing_fraction_grouped(
        state, system, new_packing_fraction, group_id
    )


def quasistatic_compress_to_packing_fraction(
    state: State,
    system: System,
    target_phi: float,
    *,
    step: float = 1e-3,
    phi_tolerance: float = 1e-10,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    max_n_min_steps_per_outer: int = 1_000_000,
    max_n_outer_steps: int = 1_000_000,
    progress: bool = False,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Quasi-statically compress (or decompress) toward ``target_phi``.

    Alternates :func:`scale_to_packing_fraction` with :func:`minimize` in
    steps no larger than ``step`` in packing fraction. The final step is
    truncated so the target is hit exactly (within ``phi_tolerance``).
    Works in both directions: if ``target_phi > current_phi`` the box
    shrinks and particles are pushed closer; if ``target_phi <
    current_phi`` the box grows and the system relaxes.

    The state is minimized once up front, so a non-equilibrium input is
    safe. Above the jamming point the minimizer may exit with residual
    PE — the final PE is returned so the caller can detect this.

    Parameters
    ----------
    state, system
        Current state/system; any domain type that :func:`scale_to_packing_fraction`
        supports is allowed.
    target_phi
        Target packing fraction.
    step
        Maximum magnitude of the per-outer-step increment in phi. Smaller
        values are more quasistatic (costlier); 1e-3 is a reasonable
        default for dense compressions.
    phi_tolerance
        Absolute tolerance on the terminal packing fraction.
    pe_tol, pe_diff_tol
        Minimizer convergence tolerances.
    max_n_min_steps_per_outer
        FIRE iterations allowed per minimization (per outer step).
    max_n_outer_steps
        Hard cap on outer iterations (safety net).
    progress
        If ``True`` and ``tqdm`` is importable, wraps the outer loop in a
        progress bar. Otherwise silent.

    Returns
    -------
    state, system, final_phi, final_pe
        ``final_phi`` is ``compute_packing_fraction(state, system)`` at
        exit; ``final_pe`` is the PE after the last minimization.
    """
    state, system, _, pe = system.minimize(
        state,
        system,
        max_steps=max_n_min_steps_per_outer,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )
    current_phi = float(compute_packing_fraction(state, system))
    step_mag = abs(float(step))

    # Body grouping depends only on the (static) bond/clump topology:
    # compute it once on the host instead of once per outer iteration.
    group_id = jnp.asarray(_host_body_grouping(state.clump_id, state.bond_id))

    iter_range: Any = range(int(max_n_outer_steps))
    if progress:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]

            iter_range = tqdm(
                iter_range, total=int(max_n_outer_steps), desc="Compressing"
            )
        except ImportError:
            pass

    for _ in iter_range:
        remaining = float(target_phi) - current_phi
        if abs(remaining) <= phi_tolerance:
            break
        delta = (1.0 if remaining > 0.0 else -1.0) * min(step_mag, abs(remaining))
        new_phi = current_phi + delta
        state, system = _scale_to_packing_fraction_grouped(
            state, system, new_phi, group_id
        )
        state, system, _, pe = system.minimize(
            state,
            system,
            max_steps=max_n_min_steps_per_outer,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
        )
        current_phi = new_phi

    return state, system, jnp.asarray(current_phi), jnp.asarray(pe)
