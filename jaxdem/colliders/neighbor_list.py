# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Neighbor List Collider implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from ..utils.linalg import cross, norm2
from . import Collider, valid_interaction_mask

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


@jax.jit(inline=True)
def _remap_history_array(
    old_hist: jax.Array | None,
    old_nl: jax.Array,
    new_nl: jax.Array,
    old_uid: jax.Array,
    new_uid: jax.Array,
) -> jax.Array | None:
    if old_hist is None:
        return None

    sort_old = jnp.argsort(old_uid)
    idx_in_sort = jnp.searchsorted(old_uid[sort_old], new_uid)
    old_idx = sort_old[idx_in_sort]

    def map_particle(
        h_old_i: jax.Array, nl_old_i: jax.Array, nl_new_i: jax.Array
    ) -> jax.Array:
        uid_old_neighbors = jnp.where(nl_old_i != -1, old_uid[nl_old_i], -1)
        uid_new_neighbors = jnp.where(nl_new_i != -1, new_uid[nl_new_i], -1)

        matches = uid_new_neighbors[:, None] == uid_old_neighbors[None, :]
        valid_matches = matches * (uid_new_neighbors[:, None] != -1)

        has_match = jnp.any(valid_matches, axis=-1)
        idx = jnp.argmax(valid_matches, axis=-1)

        gathered = h_old_i[idx]

        has_match_exp = has_match
        for _ in range(gathered.ndim - has_match.ndim):
            has_match_exp = jnp.expand_dims(has_match_exp, -1)

        return jnp.where(has_match_exp, gathered, jnp.zeros_like(gathered))

    h_old_permuted = old_hist[old_idx]
    nl_old_permuted = old_nl[old_idx]

    return jax.vmap(map_particle)(h_old_permuted, nl_old_permuted, new_nl)


@jax.jit(inline=True)
def _check_and_rebuild(
    state: State, system: System, collider: "NeighborList"
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
    """Check the displacement criterion and conditionally rebuild the list.

    Triggers a rebuild when any particle has moved farther than half the
    skin distance since the last build, or when the list has never been
    built (``n_build_times == 0``). Otherwise returns the cached buffers.

    Returns
    -------
    tuple
        ``(state, neighbor_list, old_pos, n_build_times, overflow)``.

    Notes
    -----
    Under ``jax.vmap`` the ``jax.lax.cond`` below lowers to ``select`` and
    both branches execute every step (full rebuild every step). See the
    class docstring of :class:`NeighborList`.
    """
    # Intentionally not a periodic displacement: the list is built from
    # absolute positions, so unwrapped motion is what invalidates it.
    disp = state.pos - collider.old_pos
    max_disp_sq = jnp.max(norm2(disp))
    trigger_dist_sq = collider.skin**2 / 4

    # Force rebuild if displacement is large OR if this is the first step (count == 0)
    should_rebuild = (max_disp_sq > trigger_dist_sq) + (collider.n_build_times == 0)

    def rebuild_branch(
        operands: tuple[State, System, NeighborList],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
        s, sys, col = operands
        nl_new, old_pos_new, n_build_new, overflow_new = col._rebuild(col, s, sys)

        def init_hist(_: Any) -> Any:
            shape = s.pos_c.shape[:-1] + (col.max_neighbors,)
            return sys.force_model.init_history(shape)

        def remap_hist(_: Any) -> Any:
            if col.history is None:
                return init_hist(None)
            return jax.tree.map(
                lambda h: _remap_history_array(
                    h, col.neighbor_list, nl_new, s.unique_id, s.unique_id
                ),
                col.history,
            )

        new_history = jax.lax.cond(col.n_build_times == 0, init_hist, remap_hist, None)

        return nl_new, old_pos_new, n_build_new, overflow_new, new_history

    def no_rebuild_branch(
        operands: tuple[State, System, NeighborList],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, Any]:
        s, sys, col = operands
        hist = col.history
        if hist is None:
            shape = s.pos_c.shape[:-1] + (col.max_neighbors,)
            hist = sys.force_model.init_history(shape)

        return (
            col.neighbor_list,
            col.old_pos,
            col.n_build_times,
            col.overflow,
            hist,
        )

    return jax.lax.cond(
        should_rebuild > 0,
        rebuild_branch,
        no_rebuild_branch,
        (state, system, collider),
    )


@Collider.register("NeighborList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NeighborList(Collider):
    r"""Implementation of a Verlet neighbor list collider.

    Verlet neighbor lists cache candidate interaction pairs over multiple simulation
    timesteps. This bypasses the need to execute full spatial partitioning queries
    (sorting and slab/cell hashing) at every timestep, dramatically reducing contact
    detection overhead.

    Mathematical Formalism & Rebuild Criteria
    -----------------------------------------
    The neighbor list is constructed with a search radius containing a buffer distance
    known as the ``skin``:

    .. math::
        r_{search} = \text{cutoff} + \text{skin}

    Let :math:`\mathbf{x}_i^0` represent the position of particle :math:`i` at the time
    of the last neighbor list rebuild. At any subsequent timestep, the displacement of
    particle :math:`i` from its reference position is:

    .. math::
        \Delta \mathbf{x}_i = \mathbf{x}_i - \mathbf{x}_i^0

    By the triangle inequality, the change in distance between any two particles
    :math:`i` and :math:`j` since the last rebuild is bounded by:

    .. math::
        |d_{ij} - d_{ij}^0| \le \|\Delta \mathbf{x}_i\| + \|\Delta \mathbf{x}_j\| \le 2 \max_{k} \|\Delta \mathbf{x}_k\|

    To guarantee that no pair of particles can come closer than the interaction range
    :math:`\text{cutoff}` without being captured in the neighbor list, a rebuild is
    triggered as soon as:

    .. math::
        \max_{k} \|\Delta \mathbf{x}_k\| > \frac{\text{skin}}{2}

    Runtime and Cost Analysis
    -------------------------
    The computational cost of simulations using neighbor lists consists of two parts:

    1. **Rebuild Cost**: Occurs occasionally when the maximum displacement threshold is exceeded.
       Any registerable collider (e.g., ``NaiveSimulator``, ``DynamicCellList``, or ``DynamicMultiCellList``)
       can be configured and used to perform spatial queries during this rebuild
       phase. The complexity of the rebuild step is directly determined by the chosen underlying collider
       (e.g., :math:`O(N^2)` for ``NaiveSimulator``, or :math:`O(N \log N)` for ``DynamicCellList``/``DynamicMultiCellList``).
    2. **Step Evaluation Cost**: Occurs at every timestep. We iterate directly over the static
       cached neighbor buffer of size ``max_neighbors``.

       .. math::
           \text{cost}_{step} \approx N \cdot \text{max\_neighbors}

    * **Estimating Buffer Size**:
      The size of the neighbor buffer ``max_neighbors`` is estimated based on the search volume and number density:

      .. math::
          \text{max\_neighbors} \approx \gamma \cdot \rho \cdot V_{search}

      where :math:`\gamma` is a safety factor (default 1.2), :math:`\rho = N / V_{domain} = \phi / \langle V \rangle` is the
      macroscopic number density, and :math:`V_{search}` is the volume of the search sphere of radius :math:`r_{search}`:

      .. math::
          V_{search} = \begin{cases}
              \pi r_{search}^2 & \text{in 2D} \\
              \frac{4}{3}\pi r_{search}^3 & \text{in 3D}
          \end{cases}

      Typically, a skin of :math:`0.1 \text{ to } 0.4` times the particle diameter provides a good balance.

    Constructor Parameters
    ----------------------
    - **cutoff**: The physical contact interaction range. Larger cutoffs increase the search volume exponentially, expanding the neighbor buffer.
    - **skin**: The **absolute** buffer distance added to the cutoff (the same quantity the dataclass field ``skin`` stores). It can alternatively be given to ``Create`` as ``skin_fraction``, a fraction of the cutoff (default `0.05`). Larger skin reduces rebuild frequency but inflates `max_neighbors`, increasing step time and memory.
    - **max_neighbors**: The static neighbor buffer size per particle. If not provided, it is estimated using safety factor and density heuristics. Setting this too small causes list overflows, while setting it too large wastes GPU memory.
    - **number_density**: Macroscopic number density used to estimate neighbor counts when not provided. Default is `1.0`.
    - **safety_factor**: Multiplier applied to the estimated density to account for local fluctuations. Default is `1.2`.
    - **secondary_collider_type**: The identifier of the underlying collider used to execute the spatial queries during rebuilds (e.g. ``"CellList"``, ``"naive"``, or ``"MultiCellList"``). Any registered ``Collider`` subclass in the library can be used for the rebuild phase, allowing the rebuild cost to be optimized based on system characteristics.
    - **secondary_collider_kw**: Keyword args for the underlying collider constructor.

    This collider is suitable for dense assemblies, static packings, slow shear flows, gravity settling, or any low-velocity systems. It is less suitable for high-speed granular flows or high-temperature systems where rapid particle motion triggers frequent neighbor list rebuilds, neutralizing the caching advantage. Furthermore, systems of rigid clumps with large overlaps require allocating larger neighbor buffers to accommodate excluded constituent pairs, which increases the memory footprint and the step traversal cost.

    Temperature & Rebuild Frequency Discussion
    -------------------------------------------
    In particle systems (analogous to molecular dynamics), the "temperature" :math:`T` is proportional to the mean squared velocity (kinetic energy) of the particles:

    .. math::
        \langle v^2 \rangle \sim T \implies v_{rms} \propto \sqrt{T}

    The rebuild criterion is triggered when the maximum particle displacement exceeds half the skin distance:

    .. math::
        \max_k \|\Delta \mathbf{x}_k\| > \frac{\text{skin}}{2}

    Approximating the particle displacement over time as :math:`\|\Delta \mathbf{x}\| \approx v \cdot t`, the average time interval between rebuilds :math:`\tau` can be estimated as:

    .. math::
        \tau \approx \frac{\text{skin}}{2 \cdot v_{rms}} \propto \frac{\text{skin}}{\sqrt{T}}

    As a result, the rebuild frequency (:math:`f_{rebuild} = 1/\tau`) scales as:

    .. math::
        f_{rebuild} \propto \frac{\sqrt{T}}{\text{skin}}

    In high-temperature systems, the rebuild frequency becomes extremely high,
    resulting in frequent executions of the :math:`O(N \log N)` reconstruction.
    When :math:`f_{rebuild}` approaches :math:`1` (rebuilding every step),
    the neighbor list becomes slower than direct spatial partitioning colliders because of the redundant list buffering.

    .. warning::
        **Batching with** ``jax.vmap`` **defeats the Verlet-list caching.**
        The conditional rebuild is implemented with ``jax.lax.cond``. Under
        ``jax.vmap``, JAX lowers ``cond`` to ``select``, which means **both**
        branches are executed for every batch element at every step — i.e. a
        full neighbor-list rebuild happens every timestep for every batched
        environment, silently removing the performance benefit of this
        collider. For batched simulations, prefer using the underlying
        spatial-partitioning collider (e.g. ``"CellList"``) directly.
    """

    secondary_collider: Collider
    """The underlying collider used to build the list via ``create_neighbor_list``."""

    neighbor_list: jax.Array
    """Shape (N, max_neighbors). Contains the IDs of neighboring particles, padded with -1."""

    old_pos: jax.Array
    """Shape (N, dim). Positions of particles at the last build time."""

    n_build_times: jax.Array
    """Counter for how many times the list has been rebuilt."""

    cutoff: jax.Array
    """The interaction radius (force cutoff)."""

    skin: jax.Array
    """
    **Absolute** buffer distance. The list is built with
    ``radius = cutoff + skin`` and rebuilt when
    ``max_displacement > skin / 2``.

    This is the same quantity (and meaning) as the ``skin`` argument of
    :meth:`Create`.
    """

    max_neighbors: int = jax.tree.static()
    """Static buffer size for the neighbor list."""

    history: Any = field(default=None, kw_only=True)
    """Pair-wise history variables for stateful force models."""

    @classmethod
    def Create(
        cls,
        state: State,
        cutoff: float,
        skin: float | None = None,
        skin_fraction: float | None = None,
        max_neighbors: int | None = None,
        number_density: float = 1.0,
        safety_factor: float = 1.2,
        secondary_collider_type: str = "CellList",
        secondary_collider_kw: dict[str, Any] | None = None,
    ) -> Self:
        r"""Creates a NeighborList collider.

        Parameters
        ----------
        state : State
            The initial simulation state used to determine system dimensions and
            particle count.
        cutoff : float
            The physical interaction cutoff radius.
        skin : float, optional
            **Absolute** buffer distance added to the cutoff — the same
            quantity stored in the returned collider's ``skin`` field.
            **Must be > 0.0 for performance.** Mutually exclusive with
            ``skin_fraction``.
        skin_fraction : float, optional
            Buffer expressed as a fraction of ``cutoff`` (the absolute buffer
            distance is ``skin_fraction * cutoff``). Defaults to ``0.05``
            when neither ``skin`` nor ``skin_fraction`` is given.
        max_neighbors : int, optional
            Maximum number of neighbors to store per particle. If not provided,
            it is estimated from the ``number_density``.
        number_density : float, default 1.0
            Number density of the system used to estimate ``max_neighbors`` if
            not explicitly provided.
        safety_factor : float, default 1.2
            Multiplier applied to the estimated number of neighbors to account
            for fluctuations in local density.
        secondary_collider_type : str, default "CellList"
            Registered collider type used internally to build the neighbor lists.
        secondary_collider_kw : dict[str, Any], optional
            Keyword arguments passed to the constructor of the internal collider.
            If None, ``cell_size`` is set to ``cutoff + skin``.

        Returns
        -------
        NeighborList
            A configured NeighborList collider instance.

        """
        if skin is not None and skin_fraction is not None:
            raise ValueError(
                "Pass either `skin` (absolute distance) or `skin_fraction` "
                "(fraction of the cutoff), not both."
            )
        if skin is None:
            skin_fraction = 0.05 if skin_fraction is None else skin_fraction
            skin_val = float(skin_fraction) * cutoff
        else:
            skin_val = float(skin)
        list_cutoff = cutoff + skin_val

        # Facet contacts are keyed on the facet's *primary vertex* (see the
        # warning on SphereFacetSpringForce): a too-small cutoff silently
        # misses contacts whose contact point is in range but whose primary
        # vertex is not. Warn when the cutoff cannot even cover one facet
        # extent plus the contact reach (facet-facet pairs need up to two
        # facet extents plus the contact thicknesses).
        if bool(jnp.any(state.facet_id != -1)):
            iota = jnp.arange(state.N)
            safe_vertices = jnp.where(
                state.facet_id[:, None] != -1, state.facet_vertices, iota[:, None]
            )
            v_pos = state.pos[safe_vertices]
            extent = jnp.linalg.norm(v_pos - v_pos[:, 0:1, :], axis=-1).max(axis=-1)
            extent = jnp.where(state.facet_id != -1, extent, 0.0)
            # state.rad is the physical contact radius/thickness (state._rad
            # would double-count the facet extent: it is the inflated
            # bounding-sphere radius for facet particles).
            min_required = float(jnp.max(extent) + 2.0 * jnp.max(state.rad))
            if cutoff < min_required:
                warnings.warn(
                    f"NeighborList cutoff ({cutoff:g}) is smaller than the "
                    f"largest facet extent plus contact reach ({min_required:g}). "
                    "Facet contacts are keyed on the facet's primary vertex, so "
                    "contacts may be silently missed or applied asymmetrically. "
                    "Use cutoff >= max facet extent + contact thicknesses "
                    "(twice the facet extent for facet-facet contacts).",
                    stacklevel=2,
                )

        # Estimate the system bounding box and actual number density
        max_rad = jnp.max(state._rad)
        pos_min = jnp.min(state.pos, axis=0)
        pos_max = jnp.max(state.pos, axis=0)
        box_size = pos_max - pos_min + 2.0 * max_rad
        box_size = jnp.maximum(box_size, 1.0)
        box_volume = jnp.prod(box_size)
        number_density_est = (state.N / box_volume).item()

        # Combine user-provided number density with estimated number density using the maximum to be safe.
        effective_density = jnp.maximum(number_density, number_density_est).item()

        # Calculate a mathematically rigorous upper bound on the number of neighbors:
        # The neighbor centers must lie within a sphere of radius list_cutoff.
        # Since the particles cannot overlap significantly, their centers are separated by at least 2 * min_rad.
        # Thus, the exclusion spheres of radius min_rad around their centers are disjoint and fit within a sphere
        # of radius list_cutoff + min_rad. We allow up to a 10% overlap tolerance for contacts.
        min_rad = jnp.min(state._rad)
        r_eff = 0.9 * min_rad
        packing_upper_bound = ((list_cutoff + r_eff) / r_eff) ** state.dim
        packing_fraction_limit = 0.91 if state.dim == 2 else 0.74
        max_possible_neighbors = int(
            jnp.ceil(packing_fraction_limit * packing_upper_bound).item()
        )

        # Calculate local packing limit for average typical-sized particles in the system
        mean_rad = jnp.mean(state._rad)
        r_eff_mean = 0.9 * mean_rad
        typical_packing_bound = ((list_cutoff + r_eff_mean) / r_eff_mean) ** state.dim
        typical_max_neighbors = int(jnp.ceil(typical_packing_bound).item())

        user_supplied_max = max_neighbors is not None
        if max_neighbors is None:
            # Estimate neighbors based on volume and density
            nl_volume = (
                jnp.pi
                * list_cutoff**state.dim
                * (1.0 if state.dim == 2 else (4.0 / 3.0))
            )
            max_neighbors_density = int(
                jnp.ceil(safety_factor * nl_volume * effective_density).item()
            )

            # Ensure we can handle local dense clusters of typical particles
            max_neighbors = max(max_neighbors_density, typical_max_neighbors)

        # Ensure max_neighbors does not exceed absolute physical limits
        requested_max_neighbors = max_neighbors
        max_neighbors = min(max_neighbors, max_possible_neighbors)
        max_neighbors = min(max_neighbors, state.N)
        max_neighbors = max(max_neighbors, 0)
        if user_supplied_max and max_neighbors < requested_max_neighbors:
            warnings.warn(
                f"NeighborList max_neighbors={requested_max_neighbors} clamped "
                f"to {max_neighbors} (bounded by N={state.N} and the physical "
                f"packing limit of {max_possible_neighbors} neighbors within "
                "the search radius).",
                stacklevel=2,
            )

        if secondary_collider_kw is None:
            secondary_collider_kw = {}
        else:
            secondary_collider_kw = dict(secondary_collider_kw)

        # Forward the state only to secondary colliders whose Create accepts
        # one (e.g. "naive" takes no state and would warn about the dropped
        # keyword otherwise).
        from inspect import signature

        from ..factory import _normalize_key

        sub_cls = Collider._registry.get(_normalize_key(secondary_collider_type))
        create_fn = getattr(sub_cls, "Create", None)
        if create_fn is not None and "state" in signature(create_fn).parameters:
            secondary_collider_kw["state"] = state
        if (
            _normalize_key(secondary_collider_type) == "celllist"
            and "cell_size" not in secondary_collider_kw
        ):
            # Default the cell size to the full search radius. A
            # user-provided cell_size is respected; the cell list inflates
            # its cells at build time if the requested cutoff exceeds the
            # stencil reach.
            secondary_collider_kw["cell_size"] = list_cutoff

        cl = Collider.create(secondary_collider_type, **secondary_collider_kw)

        # Initialize buffers
        current_pos = state.pos
        dummy_nl = jnp.full((state.N, max_neighbors), -1, dtype=int)

        return cls(
            secondary_collider=cl,
            neighbor_list=dummy_nl,
            old_pos=current_pos,
            n_build_times=jnp.array(0, dtype=int),
            cutoff=jnp.asarray(cutoff, dtype=float),
            skin=jnp.asarray(skin_val, dtype=float),
            overflow=jnp.asarray(False, dtype=bool),
            max_neighbors=int(max_neighbors),
        )

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="NeighborList.create_neighbor_list")
    def create_neighbor_list(
        state: State,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[State, System, jax.Array, jax.Array]:
        r"""Returns the current neighbor list from this collider.

        This method refreshes the cached list when it has not been built yet
        or when any particle has moved farther than half the skin distance
        from the last build position. Otherwise it returns the cached
        ``neighbor_list`` and ``overflow`` flag stored in the collider.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        cutoff : float
            Ignored; the collider's configured cutoff is used.
        max_neighbors : int
            Ignored; the collider's configured buffer size is used.

        Returns
        -------
        Tuple[State, System, jax.Array, jax.Array]
            A tuple containing:

            - state: The simulation state.
            - system: The simulation system.
            - neighbor_list: The cached neighbor list of shape (N, max_neighbors).
            - overflow: Boolean flag indicating if the list overflowed during the
              last build.

        Notes
        -----
        - The returned neighbor indices refer to the internal particle ordering
          established during the most recent rebuild inside ``compute_force``.

        """
        collider = cast(NeighborList, system.collider)

        nl, old_pos, n_build, overflow, history = _check_and_rebuild(
            state, system, collider
        )

        system.collider = replace(
            collider,
            neighbor_list=nl,
            old_pos=old_pos,
            n_build_times=n_build,
            overflow=overflow,
            history=history,
        )
        return state, system, nl, overflow

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="NeighborList._rebuild")
    def _rebuild(
        collider: NeighborList, state: State, system: System
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        r"""Internal method to rebuild the neighbor list using the secondary collider.

        Parameters
        ----------
        collider : NeighborList
            The current collider instance.
        state : State
            The current simulation state.
        system : System
            The simulation system.

        Returns
        -------
        Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
            A tuple containing:
            - New neighbor list indices (pointing to original order)
            - New reference positions for displacement tracking
            - Incremented build counter
            - Overflow flag

        """
        list_cutoff = collider.cutoff + collider.skin

        # Create a view of the system using the inner collider
        system.collider = collider.secondary_collider

        # 1. Get neighbors using the spatial partitioner
        # Returns: Sorted State, ..., Neighbors Indices (pointing to Sorted State)
        (
            _,
            _,
            sorted_nl_indices,
            overflow_flag,
        ) = collider.secondary_collider.create_neighbor_list(
            state, system, list_cutoff, collider.max_neighbors
        )

        return (
            sorted_nl_indices,
            state.pos,
            collider.n_build_times + 1,
            overflow_flag,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="NeighborList.compute_force")
    def compute_force(state: State, system: System) -> tuple[State, System]:
        r"""Computes total forces acting on each particle, rebuilding the neighbor list if necessary.

        This method checks if any particle has moved enough to trigger a rebuild
        (displacement > skin/2). If so, it invokes the internal spatial partitioner
        to refresh the neighbor list. It then sums force contributions using the
        cached list.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated ``State`` object with computed forces
            and the updated ``System`` object (with refreshed collider cache).

        """
        iota = jax.lax.iota(dtype=int, size=state.N)
        collider = cast(NeighborList, system.collider)

        # 1. Check Displacement & Trigger Rebuild
        nl, old_pos, n_build, overflow, history = _check_and_rebuild(
            state, system, collider
        )

        # 2. Compute Forces
        # Pre-calculate contact points in global frame for torque
        pos_p_global = state._pos_p_rot
        pos = state.pos

        def per_particle_force(
            i: jax.Array, pos_pi: jax.Array, neighbors: jax.Array, hist_i: Any
        ) -> tuple[jax.Array, jax.Array, Any]:
            valid = neighbors != -1
            safe_j = jnp.maximum(neighbors, 0)
            valid = valid * valid_interaction_mask(
                state.clump_id[i],
                state.clump_id[safe_j],
                state.bond_id[i],
                state.unique_id[safe_j],
                system.interact_same_bond_id,
            )

            f, t, new_hist_i = system.force_model.force_and_history(
                i, safe_j, pos, state, system, hist_i
            )

            # Mask out invalid/padding forces
            f = jnp.where((valid > 0)[..., None], f, 0.0)
            t = jnp.where((valid > 0)[..., None], t, 0.0)

            f_sum = jnp.sum(f, axis=0)
            t_sum = jnp.sum(t, axis=0) + cross(pos_pi, f_sum)

            return f_sum, t_sum, new_hist_i

        state.force, state.torque, history = jax.vmap(per_particle_force)(
            iota, pos_p_global, nl, history
        )

        # Update collider cache
        system.collider = replace(
            collider,
            neighbor_list=nl,
            old_pos=old_pos,
            n_build_times=n_build,
            overflow=overflow,
            history=history,
        )

        return state, system

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="NeighborList.compute_potential_energy")
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        r"""Computes the potential energy associated with each particle using the cached neighbor list.

        This method iterates over the cached neighbors for each particle and sums
        the potential energy contributions computed by the ``system.force_model``.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System, jax.Array]
            Tuple of (state, system, energy).

        """
        collider = cast(NeighborList, system.collider)

        # Check displacement & trigger rebuild if necessary
        nl, old_pos, n_build, overflow, history = _check_and_rebuild(
            state, system, collider
        )

        iota = jax.lax.iota(dtype=int, size=state.N)

        def per_particle_energy(i: jax.Array) -> jax.Array:
            neighbors = nl[i]

            valid = neighbors != -1
            safe_j = jnp.maximum(neighbors, 0)
            valid = valid * valid_interaction_mask(
                state.clump_id[i],
                state.clump_id[safe_j],
                state.bond_id[i],
                state.unique_id[safe_j],
                system.interact_same_bond_id,
            )
            e = system.force_model.energy(i, safe_j, state.pos, state, system)

            # Sum energies and divide by 2 (double counting in neighbor list)
            e = jnp.where(valid > 0, e, 0.0)
            return 0.5 * jnp.sum(e)

        system.collider = replace(
            collider,
            neighbor_list=nl,
            old_pos=old_pos,
            n_build_times=n_build,
            overflow=overflow,
        )

        energy = jnp.sum(jax.vmap(per_particle_energy)(iota))
        return state, system, energy

    @staticmethod
    @jax.jit(static_argnames=("max_neighbors",), inline=True)
    @partial(jax.named_call, name="NeighborList.create_cross_neighbor_list")
    def create_cross_neighbor_list(
        pos_a: jax.Array,
        pos_b: jax.Array,
        system: System,
        cutoff: float,
        max_neighbors: int,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Build a cross-neighbor list between two sets of positions.

        Delegates to the internal ``secondary_collider``'s
        ``create_cross_neighbor_list`` method.

        Parameters
        ----------
        pos_a : jax.Array
            Query positions, shape ``(N_A, dim)``.
        pos_b : jax.Array
            Database positions, shape ``(N_B, dim)``.
        system : System
            The configuration of the simulation (used for domain displacement).
        cutoff : float
            Search radius.
        max_neighbors : int
            Maximum number of neighbors to store per query point.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            A tuple containing:

            - ``neighbor_list``: Array of shape ``(N_A, max_neighbors)`` containing
              indices into ``pos_b``, padded with ``-1``.
            - ``overflow``: Boolean flag indicating if any query point exceeded
              ``max_neighbors`` neighbors within the cutoff.

        """
        collider = cast(NeighborList, system.collider)
        system.collider = collider.secondary_collider
        return collider.secondary_collider.create_cross_neighbor_list(
            pos_a, pos_b, system, cutoff, max_neighbors
        )
