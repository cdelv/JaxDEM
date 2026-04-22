# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for calculating and changing the packing fraction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import replace

from typing import TYPE_CHECKING, Any

from ..colliders import NeighborList
from ..minimizers import minimize

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
    # this assumes that the domain anchor is 0
    return compute_particle_volume(state) / jnp.prod(system.domain.box_size)


@jax.jit
def scale_to_packing_fraction(
    state: State, system: System, new_packing_fraction: float
) -> tuple[State, System]:
    # this assumes that the domain anchor is 0
    new_box_size_scalar = (compute_particle_volume(state) / new_packing_fraction) ** (
        1 / state.dim
    )
    current_box_L = system.domain.box_size[0]
    scale_factor = new_box_size_scalar / current_box_L

    new_box_size = jnp.ones_like(system.domain.box_size) * new_box_size_scalar
    new_domain = replace(system.domain, box_size=new_box_size)

    # For spheres and clumps, we can just rescale the positions via state.pos_c * scale_factor
    # But, for DPs, we need to scale the com positions
    # Both behaviors can be generalized by scaling the DP com positions, finding the offset
    # before and after the scaling, and applying the offset to state.pos_c
    # This preserves the size of the DPs, clumps, and spheres, uniformly
    total_pos = jax.ops.segment_sum(state.pos_c, state.bond_id, num_segments=state.N)
    dp_counts = jax.ops.segment_sum(
        jnp.ones((state.N,), dtype=state.pos_c.dtype),
        state.bond_id,
        num_segments=state.N,
    )
    dp_com = total_pos / jnp.maximum(
        dp_counts[:, None], 1.0
    )  # avoid divide by zero errors for empty clumps (MAY NOT BE NEEDED)
    offset = dp_com * scale_factor - dp_com

    new_state = replace(
        state, pos_c=state.pos_c + offset[state.bond_id]
    )  # broadcast back and apply shift
    new_system = replace(system, domain=new_domain)

    # force rebuild the neighbor list if using it
    if isinstance(new_system.collider, NeighborList):
        new_system = replace(
            new_system,
            collider=replace(
                new_system.collider, n_build_times=jnp.array(0, dtype=int)
            ),
        )

    return new_state, new_system


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
    state, system, _, pe = minimize(
        state,
        system,
        max_steps=max_n_min_steps_per_outer,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
        initialize=True,
    )
    current_phi = float(compute_packing_fraction(state, system))
    step_mag = abs(float(step))

    iter_range: Any = range(int(max_n_outer_steps))
    if progress:
        try:
            from tqdm import tqdm

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
        state, system = scale_to_packing_fraction(state, system, new_phi)
        state, system, _, pe = minimize(
            state,
            system,
            max_steps=max_n_min_steps_per_outer,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
            initialize=True,
        )
        current_phi = new_phi

    return state, system, jnp.asarray(current_phi), pe
