# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Packing summaries, contact networks, and energy spectra."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from . import contacts as contact
from ._pair_candidates import _check_search, _validate_state
from .dynamical_matrix import (
    clump_non_bonded_hessian,
    non_bonded_hessian,
    zero_mode_mask,
)
from .packing_utils import compute_packing_fraction

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@dataclass(frozen=True, slots=True)
class SphereContactNetwork:
    """Packed directed sphere contacts, sorted by source then destination ID.

    ``pair_ids`` has shape ``(M, 2)``. For sphere ``i``, slice every pair
    array with ``row_offsets[i]:row_offsets[i + 1]``. Offsets have length
    ``state.N + 1``; isolated spheres have empty rows. There is no padding.

    ``forces`` and ``displacements`` have shape ``(M, dim)``. Displacement
    points from destination to source using the domain's image rule.
    ``torques`` has shape ``(M, angular_dim)`` and measures each contact's
    moment about the source clump COM, including the member lever arm.

    ``overlap`` is the signed radius sum minus sphere separation, shape
    ``(M,)``. Attractive contacts can have negative overlap. ``mu`` is
    ``|F_t| / |F_n|`` along the sphere separation: zero for zero force,
    infinity for purely tangential force, and NaN for nonzero force at zero
    separation. ``pair_species`` is None or an integer array ``(M, 2)``
    inherited from the clump labels supplied to :func:`analyze_packing`.

    Entries exist when force or torque is nonzero. Reciprocal contacts have
    both directions; ``pair_ids[:, 0] < pair_ids[:, 1]`` selects one edge
    for drawing or counting each physical contact once.
    """

    pair_ids: jax.Array
    row_offsets: jax.Array
    forces: jax.Array
    torques: jax.Array
    displacements: jax.Array
    overlap: jax.Array
    mu: jax.Array
    pair_species: jax.Array | None


@dataclass(frozen=True, slots=True)
class ClumpContactNetwork:
    """Packed directed clump contacts with aligned geometry and pair statistics.

    ``pair_ids`` contains clump IDs from the accompanying state, shape
    ``(K, 2)``. ``row_offsets`` has length ``max(clump_id) + 2`` (one for
    an empty state). Unoccupied IDs and isolated clumps have empty rows.

    ``forces`` is the net force on the source clump; ``torques`` is its
    total moment about the source COM. ``displacements`` points from
    destination COM to source COM using the domain's image rule.
    Their shapes are ``(K, dim)``, ``(K, angular_dim)``, and ``(K, dim)``.

    ``mu`` has shape ``(K,)`` and uses the COM axis with the same zero,
    infinity, and NaN conventions as :class:`SphereContactNetwork`.
    ``sphere_counts`` has shape ``(K, 2)`` and counts distinct participating
    spheres on each side. ``contact_counts`` has shape ``(K,)`` and counts
    constituent directed contacts, including torque-only contacts.
    Net-force cancellation does not erase an interaction.

    ``pair_species`` is None or the supplied source/destination species
    labels, shape ``(K, 2)``. All pair arrays share the same row ordering.
    Reciprocal interactions have both directions.
    """

    pair_ids: jax.Array
    row_offsets: jax.Array
    forces: jax.Array
    torques: jax.Array
    displacements: jax.Array
    mu: jax.Array
    sphere_counts: jax.Array
    contact_counts: jax.Array
    pair_species: jax.Array | None


@dataclass(frozen=True, slots=True)
class PackingData:
    """Analysis of one configuration, with all indices relative to its state.

    Attributes
    ----------
    state, system
        Configuration with refreshed forces and collider caches. Positions,
        velocities, time, queued loads, and contact history are not advanced.
    sphere_contacts, clump_contacts
        Packed contact networks. Pair indices address this object's state.
    original_sphere_ids, original_clump_ids
        Maps from current indices to the input state's indices. Unoccupied
        clump ID slots contain -1. ``clump_ids`` lists the occupied slots.
    clump_positions
        COM positions indexed by clump ID, shape ``(max(clump_id) + 1, dim)``.
        Unoccupied slots contain NaN. Network edges can be drawn from these
        positions and the domain-aware pair displacements.
    potential_energy, pressure
        Non-bonded pair potential energy and contact virial pressure. Managed
        external energies, kinetic stress, and boundary reactions are excluded.
    packing_fraction, effective_packing_fraction
        Physical particle volume divided by box volume, and a contact-distance
        estimate. The latter assigns each clump a disk/sphere whose diameter
        is its mean contacting-clump COM distance. Zero mean distance uses the
        clump's stored physical volume. It is not a union-volume measurement.
    sphere_contact_counts, vertex_contact_counts, clump_contact_counts
        Force-bearing coordination per sphere, constituent contacts per clump,
        and distinct contacting clumps per clump. Clump arrays are indexed by
        clump ID; means include only occupied clumps, including isolated ones.
    mean_vertex_contacts, mean_clump_contacts
        Mean per-clump coordination. NaN for an empty state.
    coordinates, dof_per_clump, global_modes
        Hessian coordinate convention and degrees used in the count criterion.
        Sphere coordinates are translational; clump coordinates include rotation.
    isostatic_coordination, satisfies_isostatic_count
        ``2 * (G * dof_per_clump - global_modes) / G`` and its comparison with
        mean vertex coordination. This is a count criterion, not a mechanical
        stability test. Empty states have NaN coordination and a False result.
    hessian, eigenvalues, eigenvectors
        Dense non-bonded energy Hessian, ascending eigenvalues, and eigenvectors
        stored as columns.
        Sphere coordinates follow sphere-array order; clump coordinate blocks
        follow ``clump_ids``. Arrays for an empty state have shapes ``(0, 0)``,
        ``(0,)``, and ``(0, 0)``.
    zero_mode_count, negative_mode_count
        Counts from the spectrum. Negative modes exclude eigenvalues
        classified as numerical zero. These describe the potential energy,
        not the full response of a dissipative or history-dependent force law.
    """

    state: State
    system: System
    sphere_contacts: SphereContactNetwork
    clump_contacts: ClumpContactNetwork
    original_sphere_ids: jax.Array
    original_clump_ids: jax.Array
    clump_ids: jax.Array
    clump_positions: jax.Array
    potential_energy: jax.Array
    pressure: jax.Array
    packing_fraction: jax.Array
    effective_packing_fraction: jax.Array
    sphere_contact_counts: jax.Array
    vertex_contact_counts: jax.Array
    clump_contact_counts: jax.Array
    mean_vertex_contacts: float
    mean_clump_contacts: float
    coordinates: str
    dof_per_clump: int
    global_modes: int
    isostatic_coordination: float
    satisfies_isostatic_count: bool
    hessian: jax.Array
    eigenvalues: jax.Array
    eigenvectors: jax.Array
    zero_mode_count: int
    negative_mode_count: int


@dataclass(frozen=True, slots=True)
class PackingAnalysis:
    """Full and non-rattler packing data from the same input configuration.

    ``full`` and ``non_rattlers`` are :class:`PackingData` objects with the
    same fields. When nothing is removed, both refer to the same object.
    ``rattler_ids`` and ``non_rattler_ids`` contain original input clump IDs.
    ``rattler_proportion`` is the fraction of input clumps removed, or zero
    for an empty input. No files are written by the analysis.

    :func:`jaxdem.utils.h5.save` and :func:`jaxdem.utils.h5.load` preserve the
    nested data objects, contact networks, and their states and systems.
    """

    full: PackingData
    non_rattlers: PackingData
    rattler_ids: jax.Array
    non_rattler_ids: jax.Array
    rattler_proportion: float


def _row_offsets(pairs: jax.Array, n: int) -> jax.Array:
    counts = jnp.bincount(pairs[:, 0], length=n)
    return jnp.concatenate((jnp.zeros(1, dtype=counts.dtype), jnp.cumsum(counts)))


@jax.jit
def _refresh_forces(
    state: State, system: System, data: contact.ContactData
) -> tuple[State, System]:
    """Sum collected contacts and observe managed forces without consuming loads."""
    sources = data.pair_ids[:, 0]
    state = replace(
        state,
        force=jnp.zeros_like(state.force).at[sources].add(data.forces),
        torque=jnp.zeros_like(state.torque).at[sources].add(data.torques),
    )
    manager = system.force_manager
    working = replace(system, force_manager=replace(manager))
    state, working = working.force_manager.apply(state, working)
    return state, replace(working, force_manager=manager)


def _packing_data(
    state: State,
    system: System,
    data: contact.ContactData,
    original_sphere_ids: jax.Array,
    original_clump_ids: jax.Array,
    species: jax.Array | None,
    coordinates: str,
    rotation_scale: jax.Array | None,
    global_modes: int,
    zero_mode_rel_gap: float,
    zero_mode_atol: float,
) -> PackingData:
    state, system = _refresh_forces(state, system, data)
    occupied, first = np.unique(np.asarray(state.clump_id), return_index=True)
    clump_ids = jnp.asarray(occupied)
    n_clumps = clump_ids.size
    n_slots = original_clump_ids.size
    clump_positions = jnp.full((n_slots, state.dim), jnp.nan, dtype=state.pos_c.dtype)
    clump_positions = clump_positions.at[clump_ids].set(state.pos_c[first])
    state, system, groups = contact.get_group_contacts(state, system, contacts=data)
    sphere_network = SphereContactNetwork(
        pair_ids=data.pair_ids,
        row_offsets=_row_offsets(data.pair_ids, state.N),
        forces=data.forces,
        torques=data.torques,
        displacements=data.displacements,
        overlap=state.rad[data.pair_ids].sum(axis=1)
        - jnp.linalg.norm(data.displacements, axis=1),
        mu=contact._pair_friction(data.forces, data.displacements),
        pair_species=None
        if species is None
        else species[state.clump_id[data.pair_ids]],
    )
    clump_network = ClumpContactNetwork(
        pair_ids=groups.pair_ids,
        row_offsets=_row_offsets(groups.pair_ids, n_slots),
        forces=groups.forces,
        torques=groups.torques,
        displacements=groups.displacements,
        mu=groups.friction,
        sphere_counts=groups.sphere_counts,
        contact_counts=groups.contact_counts,
        pair_species=None if species is None else species[groups.pair_ids],
    )
    state, system, pressure = contact.compute_contact_pressure(
        state, system, contacts=data
    )
    state, system, sphere_counts = contact.count_sphere_contacts(
        state, system, contacts=data
    )
    vertex_counts = jax.ops.segment_sum(
        sphere_counts, state.clump_id, num_segments=n_slots
    )
    state, system, clump_counts = contact.count_clump_contacts(
        state, system, contacts=data
    )
    phi = compute_packing_fraction(state, system)
    distance_sum = (
        jnp.zeros(n_slots, dtype=state.pos_c.dtype)
        .at[groups.pair_ids[:, 0]]
        .add(jnp.linalg.norm(groups.displacements, axis=1))
    )
    neighbors = jnp.diff(clump_network.row_offsets)
    diameter = distance_sum / jnp.maximum(neighbors, 1)
    physical_volume = jax.ops.segment_max(
        state.volume, state.clump_id, num_segments=n_slots
    )
    ball_volume = math.pi ** (state.dim / 2) / math.gamma(state.dim / 2 + 1)
    effective_volume = jnp.where(
        diameter > 0,
        ball_volume * (diameter / 2) ** state.dim,
        jnp.maximum(physical_volume, 0),
    )
    effective_phi = jnp.sum(effective_volume) / jnp.prod(system.domain.box_size)
    energy = jnp.asarray(0.0, dtype=state.pos_c.dtype)
    if state.N:
        state, system, energy = system.collider.compute_potential_energy(state, system)
        _check_search(system)
    mean_vertex = float(jnp.mean(vertex_counts[clump_ids])) if n_clumps else math.nan
    mean_clump = float(jnp.mean(clump_counts[clump_ids])) if n_clumps else math.nan
    dof = state.dim + (state.ang_vel.shape[-1] if coordinates == "clump" else 0)
    n_global = global_modes if n_clumps else 0
    z_iso = 2 * (n_clumps * dof - n_global) / n_clumps if n_clumps else math.nan
    if coordinates == "clump":
        state, system, matrix = clump_non_bonded_hessian(
            state, system, rotation_scale=rotation_scale
        )
        rows = (clump_ids[:, None] * dof + jnp.arange(dof)).reshape(-1)
        matrix = matrix[rows[:, None], rows[None, :]]
    else:
        state, system, matrix = non_bonded_hessian(state, system)
    matrix = 0.5 * (matrix + matrix.T)
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError("The packing Hessian contains nonfinite values.")
    values, vectors = jnp.linalg.eigh(matrix)
    if values.size > 1 and bool(jnp.any(values != 0)):
        zeros = zero_mode_mask(values, rel_gap=zero_mode_rel_gap)
    else:
        zeros = values == 0
    zeros = zeros | (jnp.abs(values) <= zero_mode_atol)
    n_zero = int(jnp.sum(zeros))
    n_negative = int(jnp.sum((values < 0) & ~zeros))
    return PackingData(
        state=state,
        system=system,
        sphere_contacts=sphere_network,
        clump_contacts=clump_network,
        original_sphere_ids=original_sphere_ids,
        original_clump_ids=original_clump_ids,
        clump_ids=clump_ids,
        clump_positions=clump_positions,
        potential_energy=energy,
        pressure=pressure,
        packing_fraction=phi,
        effective_packing_fraction=effective_phi,
        sphere_contact_counts=sphere_counts,
        vertex_contact_counts=vertex_counts,
        clump_contact_counts=clump_counts,
        mean_vertex_contacts=mean_vertex,
        mean_clump_contacts=mean_clump,
        coordinates=coordinates,
        dof_per_clump=dof,
        global_modes=n_global,
        isostatic_coordination=z_iso,
        satisfies_isostatic_count=bool(n_clumps and mean_vertex >= z_iso),
        hessian=matrix,
        eigenvalues=values,
        eigenvectors=vectors,
        zero_mode_count=n_zero,
        negative_mode_count=n_negative,
    )


def analyze_packing(
    state: State,
    system: System,
    *,
    clump_species_ids: jax.typing.ArrayLike | None = None,
    coordinates: Literal["auto", "sphere", "clump"] = "auto",
    rotation_scale: jax.typing.ArrayLike | None = None,
    zc: int | None = None,
    check_contact_rank: bool = False,
    contact_rank_tol: float | None = None,
    global_modes: int | None = None,
    zero_mode_rel_gap: float = 1e4,
    zero_mode_atol: float = 0.0,
) -> PackingAnalysis:
    """Analyze a sphere or rigid-clump packing before and after rattler removal.

    Both results include contact networks, packing statistics, the dense energy
    Hessian, its eigenvalues and eigenvectors, and zero/negative mode counts.

    Parameters
    ----------
    state, system
        One 2D or 3D configuration with its configured collider. Contact
        snapshots are reused for network geometry, coordination, pressure,
        effective packing fraction, and rattler detection. History is never
        advanced. Bonded models are unsupported because removal would require
        remapping their topology.
    clump_species_ids
        Optional integer labels, one per occupied clump in ascending clump-ID
        order. They label both networks' pairs without changing force routing.
        No species masks are stored. Omit to return None for pair species.
    coordinates
        ``auto`` uses sphere coordinates when each clump is a single sphere
        at its COM; otherwise it uses clump translations and rotations. This
        selects both the rattler criterion and Hessian coordinates. The same
        convention is used for both returned configurations.
    rotation_scale
        Optional positive finite length per occupied clump in ascending ID
        order, for clump coordinates ``(delta r_c, R*omega)``. It is mapped
        to surviving clumps after removal. Requires clump coordinates.
    zc, check_contact_rank, contact_rank_tol
        Passed to the matching sphere or clump rattler helper. The default
        contact threshold is ``dof_per_clump + 1``.
    global_modes
        Modes subtracted in the coordination count, default ``dim`` for global
        translations of a periodic packing. Set explicitly for a different
        constraint convention. Fixed bodies are included in coordination and
        Hessian coordinates; this function does not impose their constraints.
    zero_mode_rel_gap, zero_mode_atol
        Relative gap for :func:`zero_mode_mask` and an additional absolute
        zero threshold. All-zero spectra are counted entirely as zero modes.

    Returns
    -------
    PackingAnalysis
        Matching ``full`` and ``non_rattlers`` data, plus original rattler and
        non-rattler clump IDs and the removed fraction. Empty results use the
        same types. No simulation steps, minimization, or file writes occur.

    Notes
    -----
    This is a host operation. Contact storage scales with active pairs; dense
    Hessian and eigenvector storage grows quadratically with coordinate count.
    The two results share one PackingData object when no clumps are removed.
    """
    _validate_state(state)
    if system.bonded_force_model is not None:
        raise NotImplementedError(
            "Packing analysis requires unbonded spheres or rigid clumps."
        )
    if coordinates not in ("auto", "sphere", "clump"):
        raise ValueError("coordinates must be 'auto', 'sphere', or 'clump'.")
    for name, value, lower in (
        ("zero_mode_rel_gap", zero_mode_rel_gap, 1.0),
        ("zero_mode_atol", zero_mode_atol, 0.0),
    ):
        if not math.isfinite(value) or value < lower:
            raise ValueError(f"{name} must be finite and at least {lower}.")
    if global_modes is None:
        global_modes = state.dim
    if (
        isinstance(global_modes, bool)
        or not isinstance(global_modes, (int, np.integer))
        or global_modes < 0
    ):
        raise ValueError("global_modes must be a nonnegative integer.")
    ids = np.unique(np.asarray(state.clump_id))
    n_slots = int(ids[-1]) + 1 if ids.size else 0
    single_spheres = ids.size == state.N and not bool(jnp.any(state.pos_p != 0))
    if coordinates == "auto":
        coordinates = "sphere" if single_spheres else "clump"
    if coordinates == "sphere" and not single_spheres:
        raise ValueError("Sphere coordinates require one sphere at each clump COM.")
    species = None
    if clump_species_ids is not None:
        labels = np.asarray(clump_species_ids)
        if labels.shape != (ids.size,) or (
            labels.size and not np.issubdtype(labels.dtype, np.integer)
        ):
            raise ValueError(
                "clump_species_ids must contain one integer per occupied clump."
            )
        species = (
            jnp.zeros(n_slots, dtype=int).at[ids].set(jnp.asarray(labels, dtype=int))
        )
    scales = None
    if rotation_scale is not None:
        if coordinates != "clump":
            raise ValueError("rotation_scale requires clump coordinates.")
        supplied = jnp.asarray(rotation_scale, dtype=state.pos_c.dtype)
        if supplied.shape != (ids.size,) or not bool(
            jnp.all(jnp.isfinite(supplied) & (supplied > 0))
        ):
            raise ValueError(
                "rotation_scale must contain one positive finite length per occupied clump."
            )
        scales = jnp.ones(n_slots, dtype=state.pos_c.dtype).at[ids].set(supplied)
    state, system, data = contact.get_contacts(state, system)
    if coordinates == "sphere":
        _, _, rattler_spheres, _ = contact.get_sphere_rattler_ids(
            state,
            system,
            zc=zc,
            check_contact_rank=check_contact_rank,
            contact_rank_tol=contact_rank_tol,
            contacts=data,
        )
        rattlers = jnp.sort(state.clump_id[rattler_spheres])
    else:
        _, _, rattlers, _ = contact.get_clump_rattler_ids(
            state,
            system,
            zc=zc,
            check_contact_rank=check_contact_rank,
            contact_rank_tol=contact_rank_tol,
            contacts=data,
        )
    non_rattlers = jnp.asarray(ids[~np.isin(ids, np.asarray(rattlers))])
    original_clumps = jnp.full(n_slots, -1, dtype=int).at[ids].set(ids)
    full = _packing_data(
        state,
        system,
        data,
        jnp.arange(state.N),
        original_clumps,
        species,
        coordinates,
        scales,
        global_modes,
        zero_mode_rel_gap,
        zero_mode_atol,
    )
    reduced = full
    if rattlers.size:
        keep = jnp.asarray(
            np.flatnonzero(~np.isin(np.asarray(state.clump_id), np.asarray(rattlers)))
        )
        reduced_state, reduced_system = contact.remove_rattlers(
            full.state, full.system, rattlers
        )
        manager = full.system.force_manager
        reduced_system = replace(
            reduced_system,
            force_manager=replace(
                reduced_system.force_manager,
                external_force=manager.external_force[keep],
                external_force_com=manager.external_force_com[keep],
                external_torque=manager.external_torque[keep],
            ),
        )
        reduced_state, reduced_system, reduced_contacts = contact.get_contacts(
            reduced_state, reduced_system
        )
        reduced = _packing_data(
            reduced_state,
            reduced_system,
            reduced_contacts,
            keep,
            non_rattlers,
            None if species is None else species[non_rattlers],
            coordinates,
            None if scales is None else scales[non_rattlers],
            global_modes,
            zero_mode_rel_gap,
            zero_mode_atol,
        )
    return PackingAnalysis(
        full,
        reduced,
        rattlers,
        non_rattlers,
        float(rattlers.size / ids.size) if ids.size else 0.0,
    )


__all__ = [
    "SphereContactNetwork",
    "ClumpContactNetwork",
    "PackingData",
    "PackingAnalysis",
    "analyze_packing",
]
