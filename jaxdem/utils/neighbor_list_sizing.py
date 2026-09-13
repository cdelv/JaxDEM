# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Measure neighbor candidates and estimate capacity from relaxed bounding spheres."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..minimizers import MinimizeInfo
    from ..state import State
    from ..system import System


@dataclass(frozen=True)
class CandidateStatistics:
    """Candidate counts for one configuration, before applying a capacity limit.

    Attributes
    ----------
    degree : numpy.ndarray
        Directed candidate count for each constituent sphere, shape ``(N,)``.
    directed_pair_count : int
        Sum of ``degree``; each directed entry occupies one pooled cache slot.
    max_degree, mean_degree, p95_degree, p99_degree : float or int
        Maximum, mean, and degree percentiles across constituent spheres.
    list_cutoff : float
        Effective search distance, including the absolute skin.
    """

    degree: NDArray[np.int64]
    directed_pair_count: int
    max_degree: int
    mean_degree: float
    p95_degree: float
    p99_degree: float
    list_cutoff: float


@dataclass(frozen=True)
class NeighborCapacityEstimate:
    """Capacity recommendations from a relaxed bounding-sphere configuration.

    Attributes
    ----------
    max_neighbors : int
        Average per-sphere budget for a NeighborList pool of size
        ``N * max_neighbors``, including the requested margin and rounding.
    query_max_neighbors : int
        Per-row capacity for dense neighbor queries at the sampled list cutoff.
    samples : tuple[CandidateStatistics, ...]
        Counts for the restored input orientations followed by random rotations.
    bounding_radii : numpy.ndarray
        Bounding radius of each clump, ordered by increasing clump ID.
    box_size : numpy.ndarray
        Probe box lengths in input length units, shape ``(dim,)``.
    packing_fraction : float
        Sum of bounding-sphere volumes divided by the probe box volume.
    cutoff, skin : float
        Configured cutoff and absolute skin suitable for NeighborList.Create.
        The force model can enlarge the effective cutoff reported in ``samples``.
    relaxation_steps : int
        Number of FIRE updates performed on the bounding spheres.
    relaxation_info : MinimizeInfo
        Force and torque convergence diagnostics in analogue units. The largest
        bounding radius, sphere mass, and contact stiffness are each one.
    """

    max_neighbors: int
    query_max_neighbors: int
    samples: tuple[CandidateStatistics, ...]
    bounding_radii: NDArray[np.float64]
    box_size: NDArray[np.float64]
    packing_fraction: float
    cutoff: float
    skin: float
    relaxation_steps: int
    relaxation_info: MinimizeInfo


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _integer(value: int, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be a Python integer >= {minimum}.")


def _validate_state(state: State) -> None:
    if state.pos_c.ndim != 2 or state.dim not in (2, 3):
        raise ValueError("A single 2D or 3D state is required.")
    if not np.isfinite(np.asarray(state.pos)).all():
        raise ValueError("Particle positions must be finite.")
    radii = np.asarray(state.rad)
    if not np.isfinite(radii).all() or np.any(radii <= 0):
        raise ValueError("Particle radii must be finite and positive.")


def _search_distances(
    state: State,
    system: System,
    cutoff: float | None,
    skin: float | None,
    skin_fraction: float | None,
) -> tuple[float, float, float]:
    if skin is not None and skin_fraction is not None:
        raise ValueError("Pass either skin or skin_fraction, not both.")
    radii = np.asarray(system.force_model.search_radii(state, system))
    if not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError("Force-model search radii must be finite and nonnegative.")
    physical_cutoff = 2.0 * float(radii.max(initial=0.0))
    if cutoff is None:
        cutoff = float(getattr(system.collider, "cutoff", physical_cutoff))
    cutoff = _nonnegative(cutoff, "cutoff")
    if skin_fraction is not None:
        skin = _nonnegative(skin_fraction, "skin_fraction") * cutoff
    elif skin is None:
        skin = float(getattr(system.collider, "skin", 0.05 * cutoff))
    skin = _nonnegative(skin, "skin")
    return cutoff, skin, max(cutoff, physical_cutoff) + skin


def measure_neighbor_candidates(
    state: State,
    system: System,
    *,
    cutoff: float | None = None,
    skin: float | None = None,
    skin_fraction: float | None = None,
) -> CandidateStatistics:
    """Count candidates in a snapshot without allocating a neighbor-pair buffer.

    Parameters
    ----------
    state, system
        One configuration and its domain, force model, and interaction rules.
        Positions, forces, and contact history are unchanged.
    cutoff
        Configured cutoff. Defaults to the collider's cutoff if present,
        otherwise twice the largest force-model search radius. The effective
        cutoff is at least twice that radius, matching NeighborList rebuilds.
    skin
        Absolute buffer distance. Defaults to the collider's stored skin if
        present, otherwise ``0.05 * cutoff``.
    skin_fraction
        Buffer as a fraction of the configured cutoff; mutually exclusive
        with ``skin``. Overrides the collider's stored skin.

    Returns
    -------
    CandidateStatistics
        Directed per-sphere counts with the domain's distance convention and
        the collider's clump and bond exclusions. Empty states have zero counts.

    Raises
    ------
    RuntimeError
        The spatial hash overflows, so counts would be incomplete.

    Notes
    -----
    This is a host-side utility. A JIT-compiled search counts candidates without
    evaluating forces or energy. At most 64 particles use the all-pairs backend
    to avoid grid setup. Larger configurations use a cell search, except custom
    domains without a declared cell-search geometry, which use all pairs.
    """
    from ..colliders import Collider
    from ..colliders._neighbor_cache import count_pairs

    _validate_state(state)
    _, _, list_cutoff = _search_distances(state, system, cutoff, skin, skin_fraction)
    if state.N <= 64 or system.domain.search_geometry is None:
        collider = Collider.create("naive")
    else:
        collider = Collider.create(
            "celllist", state=state, cell_size=max(list_cutoff, 1e-12), search_range=1
        )
    degree, overflow = count_pairs(
        state, replace(system, collider=collider), jnp.asarray(list_cutoff)
    )
    if bool(overflow):
        raise RuntimeError("Spatial hash overflow during neighbor candidate counting.")
    degree_np = np.array(degree, dtype=np.int64)
    degree_np.setflags(write=False)
    return CandidateStatistics(
        degree=degree_np,
        directed_pair_count=int(degree_np.sum()),
        max_degree=int(degree_np.max(initial=0)),
        mean_degree=float(degree_np.mean()) if degree_np.size else 0.0,
        p95_degree=float(np.percentile(degree_np, 95)) if degree_np.size else 0.0,
        p99_degree=float(np.percentile(degree_np, 99)) if degree_np.size else 0.0,
        list_cutoff=list_cutoff,
    )


def _capacity(observed: float, safety_factor: float, extra: int, round_to: int) -> int:
    requested = math.ceil(max(observed * safety_factor, observed + extra))
    return ((requested + round_to - 1) // round_to) * round_to


def estimate_neighbor_capacity(
    state: State,
    system: System,
    packing_fraction: float,
    *,
    cutoff: float | None = None,
    skin: float | None = None,
    skin_fraction: float | None = None,
    max_steps: int = 1_000_000,
    force_tol: float = 1e-8,
    orientation_samples: int = 3,
    safety_factor: float = 1.2,
    extra_neighbors: int = 2,
    round_to: int = 8,
    seed: int = 0,
) -> NeighborCapacityEstimate:
    """Estimate neighbor capacity using compressed, relaxed bounding spheres.

    Parameters
    ----------
    state, system
        One periodic 2D or 3D configuration of spheres or rigid clumps. Bodies
        are identified by ``clump_id``. All analogue bodies can move, including
        those marked fixed in the input. Input arrays and history are unchanged.
    packing_fraction
        Positive target sum of bounding-sphere volumes divided by box volume.
        Values above one are permitted for overlapping soft bounding spheres.
    cutoff, skin, skin_fraction
        Search settings as in :func:`measure_neighbor_candidates`.
    max_steps
        FIRE update budget for one analogue relaxation. Default: 1,000,000.
    force_tol
        Maximum force norm tolerance in analogue units: the largest bounding
        radius, sphere mass, and spring stiffness are each one. Default: 1e-8.
    orientation_samples
        Total orientation samples, including the input orientations first.
        Later samples independently rotate each clump. Default: three.
        Single-sphere states require only one sample.
    safety_factor
        Multiplier >= 1 applied to the largest observed mean and maximum degree.
    extra_neighbors
        Minimum additive margin on each of those statistics. Default: two.
        The larger of the additive and multiplicative margins is used.
    round_to
        Round each recommendation upward to this integer multiple. Default: eight.
    seed
        Seed for the initial small center perturbation and random orientations.

    Returns
    -------
    NeighborCapacityEstimate
        Pooled and dense-query capacities, sample counts, probe geometry, and
        analogue convergence diagnostics. Capacities are Python integers.

    Raises
    ------
    ValueError
        Inputs are invalid, empty, or nonperiodic.
    RuntimeError
        Analogue relaxation fails to converge or candidate counting overflows.

    Notes
    -----
    Body centers and box lengths are scaled once to the requested packing
    fraction, preserving the box aspect ratio and periodic shear. A small
    seeded perturbation separates coincident centers before relaxation.
    Frictionless spheres are relaxed with FIRE and a CellList. Only force
    evaluations are needed during its iterations. Clump geometry is then
    restored at the relaxed centers; orientation samples reuse that relaxation.

    Counts are exact for the sampled configurations. The recommendations are
    estimates for subsequent motion, so production overflow checks remain
    necessary. Dense queries with a different cutoff require separate sizing.
    This host-side preparation utility cannot be called inside ``jax.jit``.
    """
    from ..material_matchmakers import MaterialMatchmaker
    from ..materials import Material, MaterialTable
    from ..minimizers import TerminationReason
    from ..state import State
    from ..system import System
    from .particle_creation import _randomize_body_orientations

    _validate_state(state)
    if state.N == 0:
        raise ValueError("At least one particle is required for a packing probe.")
    if not system.domain.periodic:
        raise ValueError("Bounding-sphere packing probes require a periodic domain.")
    packing_fraction = _nonnegative(packing_fraction, "packing_fraction")
    if packing_fraction == 0:
        raise ValueError("packing_fraction must be positive.")
    force_tol = _nonnegative(force_tol, "force_tol")
    safety_factor = _nonnegative(safety_factor, "safety_factor")
    if safety_factor < 1:
        raise ValueError("safety_factor must be at least one.")
    _integer(max_steps, "max_steps", 0)
    _integer(orientation_samples, "orientation_samples", 1)
    _integer(extra_neighbors, "extra_neighbors", 0)
    _integer(round_to, "round_to", 1)
    _integer(seed, "seed", 0)
    cutoff, skin, _ = _search_distances(state, system, cutoff, skin, skin_fraction)

    _, group_np = np.unique(np.asarray(state.clump_id), return_inverse=True)
    group = jnp.asarray(group_np)
    n_bodies = int(group_np.max()) + 1
    counts = jnp.bincount(group, length=n_bodies)
    centers = jax.ops.segment_sum(state.pos_c, group, num_segments=n_bodies)
    centers = centers / counts[:, None]
    extents = jnp.linalg.norm(state.pos - centers[group], axis=-1) + state.rad
    radii = jax.ops.segment_max(extents, group, num_segments=n_bodies)
    length_unit = float(jnp.max(radii))
    normalized_radii = radii / length_unit
    sphere_volume = math.pi ** (state.dim / 2) / math.gamma(state.dim / 2 + 1)
    total_volume = sphere_volume * float(jnp.sum(normalized_radii**state.dim))
    box = system.domain.box_size / length_unit
    scale = (total_volume / (packing_fraction * float(jnp.prod(box)))) ** (
        1 / state.dim
    )
    box = box * scale
    domain = replace(
        system.domain, box_size=box, inv_box_size=1 / box, anchor=jnp.zeros_like(box)
    )
    key, orientation_key = jax.random.split(jax.random.PRNGKey(seed))
    positions = (centers - system.domain.anchor) / length_unit * scale
    positions = positions + 1e-3 * jnp.min(normalized_radii) * jax.random.uniform(
        key, positions.shape, minval=-1.0, maxval=1.0
    )
    analogue = State.create(
        pos=positions, rad=normalized_radii, mass=jnp.ones(n_bodies)
    )
    material = Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    table = MaterialTable.from_materials(
        [material], matcher=MaterialMatchmaker.create("harmonic")
    )
    analogue_system = System.create(
        state=analogue,
        domain=domain,
        collider_type="celllist",
        collider_kw={"cell_size": 2.0, "search_range": 1},
        mat_table=table,
        dt=0.01,
    )
    relaxed = analogue_system.minimize(
        analogue, analogue_system, max_steps=max_steps, force_tol=force_tol
    )
    if not bool(relaxed.converged):
        reason = TerminationReason(int(relaxed.reason)).name
        raise RuntimeError(
            f"Bounding-sphere relaxation failed: {reason} after {int(relaxed.steps)} "
            f"steps (force_max={float(relaxed.info.force_max):.6g})."
        )

    probe_box = box * length_unit
    probe_system = replace(
        system,
        domain=replace(system.domain, box_size=probe_box, inv_box_size=1 / probe_box),
    )
    new_centers = relaxed.state.pos_c * length_unit + system.domain.anchor
    probe = replace(state, pos_c=state.pos_c + (new_centers - centers)[group])
    if n_bodies == state.N:
        orientation_samples = 1
    samples = []
    for i in range(orientation_samples):
        sample = probe
        if i:
            sample = _randomize_body_orientations(
                probe, group, n_bodies, jax.random.fold_in(orientation_key, i)
            )
        samples.append(
            measure_neighbor_candidates(sample, probe_system, cutoff=cutoff, skin=skin)
        )
    return NeighborCapacityEstimate(
        max_neighbors=_capacity(
            max(s.mean_degree for s in samples),
            safety_factor,
            extra_neighbors,
            round_to,
        ),
        query_max_neighbors=_capacity(
            max(s.max_degree for s in samples), safety_factor, extra_neighbors, round_to
        ),
        samples=tuple(samples),
        bounding_radii=np.asarray(radii, dtype=np.float64),
        box_size=np.asarray(probe_box, dtype=np.float64),
        packing_fraction=packing_fraction,
        cutoff=cutoff,
        skin=skin,
        relaxation_steps=int(relaxed.steps),
        relaxation_info=relaxed.info,
    )
