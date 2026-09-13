# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Neighbor List Collider implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from . import Collider

if TYPE_CHECKING:
    from ..domains import Domain
    from ..state import State
    from ..system import System


@Collider.register("NeighborList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NeighborList(Collider):
    r"""Implementation of a Verlet neighbor list collider.

    Verlet neighbor lists cache candidate interaction pairs over multiple simulation
    timesteps. This removes the need to run full spatial partitioning queries
    (sorting and slab/cell hashing) at every timestep and reduces contact
    detection overhead.

    The cache pools ``N * max_neighbors`` directed pairs. ``max_neighbors`` is
    an average capacity budget, not a per-particle limit. CSR offsets identify
    each particle's pairs. Overflow occurs when the total pair count exceeds
    the pool capacity; an exactly full pool is valid.

    Mathematical Formalism & Rebuild Criteria
    -----------------------------------------
    The neighbor list uses a search radius that includes a buffer distance,
    the ``skin``:

    .. math::
        r_{search} = \text{cutoff} + \text{skin}

    Let :math:`\mathbf{x}_i^0` represent the position of particle :math:`i` at the time
    of the last neighbor list rebuild. At any later timestep, the displacement of
    particle :math:`i` from its reference position is:

    .. math::
        \Delta \mathbf{x}_i = \mathbf{x}_i - \mathbf{x}_i^0

    The triangle inequality bounds the distance change between any two particles
    :math:`i` and :math:`j` since the last rebuild by:

    .. math::
        |d_{ij} - d_{ij}^0| \le \|\Delta \mathbf{x}_i\| + \|\Delta \mathbf{x}_j\| \le 2 \max_{k} \|\Delta \mathbf{x}_k\|

    To make sure the list captures every pair before the pair comes closer than
    the interaction range :math:`\text{cutoff}`, the collider rebuilds the list
    as soon as:

    .. math::
        \max_{k} \|\Delta \mathbf{x}_k\| > \frac{\text{skin}}{2}

    Runtime and Cost Analysis
    -------------------------
    The computational cost of simulations using neighbor lists has two parts:

    1. **Rebuild Cost**: Occurs when the maximum displacement exceeds the threshold.
       You can configure any registerable collider (e.g., ``NaiveSimulator``,
       ``DynamicCellList``, or ``DynamicMultiCellList``) to run the spatial queries
       of this rebuild phase. The chosen underlying collider sets the complexity of
       the rebuild step (e.g., :math:`O(N^2)` for ``NaiveSimulator``, or
       :math:`O(N \log N)` for ``DynamicCellList``/``DynamicMultiCellList``).
    2. **Step Evaluation Cost**: Occurs at every timestep. We iterate over occupied
       rows in the shared cache of capacity ``N * max_neighbors``.

       .. math::
           \text{cost}_{step} \approx N \cdot \text{max\_neighbors}

    * **Estimating Buffer Size**:
      Estimate the neighbor buffer size ``max_neighbors`` from the search volume and the
      number density:

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
    - **cutoff**: The physical contact interaction range. Larger cutoffs increase the search volume
      exponentially and expand the neighbor buffer.
    - **skin**: The **absolute** buffer distance added to the cutoff (the same quantity the dataclass
      field ``skin`` stores). You can also pass it to ``Create`` as ``skin_fraction``, a fraction of the
      cutoff (default `0.05`). Larger skin reduces rebuild frequency but inflates `max_neighbors`, which
      increases step time and memory.
    - **max_neighbors**: The average per-particle capacity budget for the shared pool.
      If not provided, the constructor
      estimates it with safety factor and density heuristics. A value too small causes list overflows. A
      value too large wastes GPU memory.
    - **number_density**: Macroscopic number density for the ``max_neighbors`` estimate. Default is `1.0`.
    - **safety_factor**: Multiplier on the estimated density that accounts for local fluctuations.
      Default is `1.2`.
    - **secondary_collider_type**: The identifier of the underlying collider that runs the spatial queries
      during rebuilds (e.g. ``"CellList"``, ``"naive"``, or ``"MultiCellList"``). You can use any registered
      ``Collider`` subclass for the rebuild phase to optimize the rebuild cost for your system.
    - **secondary_collider_kw**: Keyword args for the underlying collider constructor.

    This collider suits dense assemblies, static packings, slow shear flows, gravity settling, or any low-velocity systems. It suits high-speed granular flows and high-temperature systems less, because rapid particle motion triggers frequent neighbor list rebuilds that cancel the caching advantage. Also, systems of rigid clumps with large overlaps need larger neighbor buffers to hold excluded constituent pairs. This increases the memory footprint and the step traversal cost.

    Temperature & Rebuild Frequency Discussion
    -------------------------------------------
    In particle systems (analogous to molecular dynamics), the "temperature" :math:`T` is proportional to the mean squared velocity (kinetic energy) of the particles:

    .. math::
        \langle v^2 \rangle \sim T \implies v_{rms} \propto \sqrt{T}

    The collider triggers a rebuild when the maximum particle displacement exceeds half the skin distance:

    .. math::
        \max_k \|\Delta \mathbf{x}_k\| > \frac{\text{skin}}{2}

    With the particle displacement over time approximated as :math:`\|\Delta \mathbf{x}\| \approx v \cdot t`, the average time interval between rebuilds :math:`\tau` is:

    .. math::
        \tau \approx \frac{\text{skin}}{2 \cdot v_{rms}} \propto \frac{\text{skin}}{\sqrt{T}}

    As a result, the rebuild frequency (:math:`f_{rebuild} = 1/\tau`) scales as:

    .. math::
        f_{rebuild} \propto \frac{\sqrt{T}}{\text{skin}}

    In high-temperature systems, the rebuild frequency becomes very high and
    causes frequent executions of the :math:`O(N \log N)` reconstruction.
    When :math:`f_{rebuild}` approaches :math:`1` (rebuilding every step),
    the neighbor list becomes slower than direct spatial partitioning colliders because of the redundant list buffering.

    .. warning::
        **Batching with** ``jax.vmap`` **defeats the Verlet-list caching.**
        The conditional rebuild uses ``jax.lax.cond``. Under ``jax.vmap``, JAX
        lowers ``cond`` to ``select``, so **both** branches execute for every
        batch element at every step. A full neighbor-list rebuild then happens
        every timestep for every batched environment, and the collider loses
        its performance benefit. For batched simulations, use the underlying
        spatial-partitioning collider (e.g. ``"CellList"``) directly.
    """

    secondary_collider: Collider
    """The spatial search backend used to construct the packed pair cache."""

    neighbor_list: jax.Array
    """Flat neighbor IDs of shape (N * max_neighbors,), with trailing -1 padding."""

    old_pos: jax.Array
    """Shape (N, dim). Positions of particles at the last build time."""

    n_build_times: jax.Array
    """Counter for how many times the list has been rebuilt."""

    cutoff: jax.Array
    """The interaction radius (force cutoff)."""

    skin: jax.Array
    """
    **Absolute** buffer distance. The collider builds the list with
    ``radius = cutoff + skin`` and rebuilds it when
    ``max_displacement > skin / 2``.

    This is the same quantity (and meaning) as the ``skin`` argument of
    :meth:`Create`.
    """

    max_neighbors: int = jax.tree.static()
    """Average per-particle capacity budget for the shared pair pool."""

    row_offsets: jax.Array = field(default_factory=lambda: jnp.empty((0,), dtype=int))
    """Sparse CSR offsets of length N + 1, clipped to allocated capacity."""

    history: jax.Array = field(default_factory=lambda: jnp.empty((0, 0)), kw_only=True)
    """Pair-wise history variables for stateful force models."""

    @property
    def stateful(self) -> bool:
        return True

    @property
    def supports_history(self) -> bool:
        return True

    def validate_domain(self, domain: Domain) -> None:
        """Validate the domain against the collider that builds this cache."""
        self.secondary_collider.validate_domain(domain)

    def invalidate(self) -> Collider:
        """Return a copy whose neighbor cache rebuilds at the next use."""
        return replace(self, invalidated=jnp.asarray(True))

    metric_snapshot: jax.Array = field(default_factory=lambda: jnp.empty((0,)))
    """Periodic box lengths, strain, and shear axes at the last check."""

    physical_cutoff_snapshot: jax.Array = field(
        default_factory=lambda: jnp.asarray(-1.0)
    )
    """Required physical interaction reach at the last check."""

    skin_snapshot: jax.Array = field(default_factory=lambda: jnp.asarray(-1.0))
    """Neighbor-list skin used for the last build."""

    invalidated: jax.Array = field(default_factory=lambda: jnp.asarray(True))
    """Explicit cache invalidation flag for topology or geometry edits."""

    @classmethod
    def Create(
        cls,
        state: State,
        cutoff: float | jax.Array,
        skin: float | jax.Array | None = None,
        skin_fraction: float | None = None,
        max_neighbors: int | None = None,
        number_density: float = 1.0,
        safety_factor: float = 1.2,
        secondary_collider_type: str = "CellList",
        secondary_collider_kw: dict[str, Any] | None = None,
    ) -> Self:
        r"""Create a NeighborList collider.

        Parameters
        ----------
        state : State
            The initial simulation state. It determines the system dimensions
            and the particle count.
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
            Average capacity budget per particle for the shared pool. If not
            provided, the constructor estimates it from ``number_density``
            and packing limits. Construction inside a JAX trace requires an
            explicit Python integer because buffer widths must remain static.
            The entire pool holds ``N * max_neighbors`` directed pairs and
            individual particles may exceed this number of neighbors. An
            exactly full pool is valid; capacity overflow means the total
            directed-pair count exceeds the pool size.
        number_density : float, default 1.0
            Number density of the system. The constructor uses it to estimate
            ``max_neighbors`` when ``max_neighbors`` is not given.
        safety_factor : float, default 1.2
            Multiplier on the estimated number of neighbors that accounts
            for fluctuations in local density.
        secondary_collider_type : str, default "CellList"
            Registered collider type used internally to build the neighbor lists.
        secondary_collider_kw : dict[str, Any], optional
            Keyword arguments for the constructor of the internal collider.
            If None and the internal collider is a cell list, ``cell_size``
            defaults to ``cutoff + skin``.

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
            skin_val = skin
        cutoff_array = jnp.asarray(cutoff)
        skin_array = jnp.asarray(skin_val)
        if cutoff_array.ndim != 0:
            raise ValueError("cutoff must be a scalar")
        if skin_array.ndim != 0:
            raise ValueError("skin must be a scalar")
        traced_state = isinstance(state.pos_c, jax.core.Tracer)
        traced_cutoff = isinstance(cutoff_array, jax.core.Tracer) or isinstance(
            skin_array, jax.core.Tracer
        )
        if not traced_cutoff:
            if not bool(jnp.isfinite(cutoff_array)) or bool(cutoff_array < 0):
                raise ValueError("cutoff must be finite and non-negative")
            if not bool(jnp.isfinite(skin_array)) or bool(skin_array < 0):
                raise ValueError("skin must be finite and non-negative")
        if max_neighbors is not None and (
            isinstance(max_neighbors, bool)
            or not isinstance(max_neighbors, int)
            or max_neighbors < 0
        ):
            raise ValueError("max_neighbors must be a non-negative Python integer")
        if number_density < 0 or not math.isfinite(number_density):
            raise ValueError("number_density must be finite and non-negative")
        if safety_factor <= 0 or not math.isfinite(safety_factor):
            raise ValueError("safety_factor must be finite and positive")
        list_cutoff = cutoff_array + skin_array

        if max_neighbors is None and (traced_state or traced_cutoff):
            raise ValueError(
                "NeighborList.Create requires explicit static `max_neighbors` "
                "when called while tracing."
            )
        if max_neighbors is None and state.N == 0:
            max_neighbors = 0
        if max_neighbors is None:
            # Estimate capacity only when the caller omits it. An explicit
            # capacity is an exact static buffer width, including widths
            # larger than the current particle count.
            max_rad = jnp.max(state._rad)
            pos_min = jnp.min(state.pos, axis=0)
            pos_max = jnp.max(state.pos, axis=0)
            box_size = jnp.maximum(pos_max - pos_min + 2.0 * max_rad, 1.0)
            number_density_est = (state.N / jnp.prod(box_size)).item()
            effective_density = jnp.maximum(number_density, number_density_est).item()
            mean_rad = jnp.mean(state._rad)
            r_eff_mean = 0.9 * mean_rad
            typical_max_neighbors = int(
                jnp.where(
                    r_eff_mean > 0,
                    jnp.ceil(((list_cutoff + r_eff_mean) / r_eff_mean) ** state.dim),
                    state.N,
                ).item()
            )
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
            max_neighbors = min(max(max_neighbors, 0), state.N)

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
            secondary_collider_kw["cell_size"] = jnp.where(
                list_cutoff > 0, list_cutoff, 1.0
            )

        cl = Collider.create(secondary_collider_type, **secondary_collider_kw)
        max_index = (1 << (63 if jax.config.jax_enable_x64 else 31)) - 1
        if state.N * max_neighbors >= max_index:
            raise ValueError("Pool capacity exceeds the particle-index integer range")
        from .cell_list import DynamicCellList
        from .multi_cell_list import DynamicMultiCellList
        from .naive import NaiveSimulator

        if not isinstance(cl, (DynamicCellList, DynamicMultiCellList, NaiveSimulator)):
            raise ValueError("NeighborList requires CellList, MultiCellList, or naive")

        # Initialize buffers
        current_pos = state.pos
        pair_shape = (state.N * max_neighbors,)
        dummy_nl = jnp.full(pair_shape, -1, dtype=int)

        return cls(
            secondary_collider=cl,
            neighbor_list=dummy_nl,
            old_pos=current_pos,
            n_build_times=jnp.array(0, dtype=int),
            cutoff=jnp.asarray(cutoff, dtype=float),
            skin=jnp.asarray(skin_val, dtype=float),
            overflow=jnp.asarray(False, dtype=bool),
            max_neighbors=int(max_neighbors),
            row_offsets=jnp.zeros((state.N + 1,), dtype=int),
            history=jnp.zeros(pair_shape + (0,), dtype=state.pos.dtype),
            metric_snapshot=jnp.zeros((state.dim + 3,), dtype=state.pos.dtype),
            physical_cutoff_snapshot=jnp.asarray(-1.0, dtype=state.pos.dtype),
            skin_snapshot=jnp.asarray(-1.0, dtype=state.pos.dtype),
            invalidated=jnp.asarray(True),
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
        r"""Build an arbitrary exact-cutoff neighbor query using the secondary collider.

        The query honors its own cutoff and buffer size, independently of the
        force-search configuration. It preserves the cached force neighbors
        and their contact history.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.
        cutoff : float
            Maximum center-to-center distance for this query.
        max_neighbors : int
            Number of neighbor slots per particle. Zero returns an empty array
            and reports overflow if any neighbor would have been returned.

        Returns
        -------
        Tuple[State, System, jax.Array, jax.Array]
            A tuple containing:

            - state: The simulation state.
            - system: The simulation system.
            - neighbor_list: Query neighbors of shape (N, max_neighbors).
            - overflow: Boolean flag. True when this query exceeds its capacity
              or the secondary collider reports invalid search geometry.

        Notes
        -----
        - The returned neighbor indices refer to the particle ordering of the
          returned ``state``.

        """
        if max_neighbors < 0:
            raise ValueError("max_neighbors must be non-negative")
        collider = cast(NeighborList, system.collider)
        inner_system = replace(system, collider=collider.secondary_collider)
        query_width = max_neighbors if max_neighbors > 0 else 1
        state_out, inner_system, nl, overflow = (
            collider.secondary_collider.create_neighbor_list(
                state, inner_system, cutoff, query_width
            )
        )
        if max_neighbors == 0:
            overflow = overflow | jnp.any(nl != -1)
            nl = jnp.empty((state.N, 0), dtype=nl.dtype)
        return state_out, replace(inner_system, collider=collider), nl, overflow

    @staticmethod
    @jax.jit(static_argnames=("advance_history",), inline=True)
    @partial(jax.named_call, name="NeighborList.compute_force")
    def compute_force(
        state: State, system: System, *, advance_history: bool = True
    ) -> tuple[State, System]:
        r"""Compute total forces acting on each particle, rebuilding the neighbor list when necessary.

        This method checks whether any particle has moved enough to trigger a
        rebuild (displacement > skin/2). If so, it calls the internal spatial
        partitioner to refresh the neighbor list. It then sums force
        contributions with the cached list.

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
        from ._neighbor_cache import check_and_rebuild, forces

        system = check_and_rebuild(state, system)
        state.force, state.torque, history = forces(
            state, system, advance_history=advance_history
        )
        return state, replace(
            system,
            collider=replace(cast(NeighborList, system.collider), history=history),
        )

    @staticmethod
    @jax.jit(inline=True)
    def evaluate_force(state: State, system: System) -> tuple[State, System]:
        """Evaluate forces while preserving the current contact history."""
        return NeighborList.compute_force(state, system, advance_history=False)

    @staticmethod
    def get_history(
        state: State, system: System, neighbor_list: jax.Array
    ) -> jax.Array:
        """Map stored contact history onto an explicit neighbor query."""
        collider = cast(NeighborList, system.collider)
        initialized = system.force_model.init_history(neighbor_list.shape, state.dim)
        from ._neighbor_cache import pair_sources, remap_history

        sources = jnp.broadcast_to(jnp.arange(state.N)[:, None], neighbor_list.shape)
        return remap_history(
            collider.history,
            pair_sources(collider),
            collider.neighbor_list,
            sources,
            neighbor_list,
            initialized,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="NeighborList.compute_potential_energy")
    def compute_potential_energy(
        state: State, system: System
    ) -> tuple[State, System, jax.Array]:
        r"""Compute the total potential energy of the system with the cached neighbor list.

        This method iterates over the cached neighbors of each particle and
        sums the potential energy contributions of the ``system.force_model``.

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
        from ._neighbor_cache import check_and_rebuild, energy

        system = check_and_rebuild(state, system)
        return state, system, energy(state, system)

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

        This method delegates to the ``create_cross_neighbor_list`` method of
        the internal ``secondary_collider``.

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
            - ``overflow``: Boolean flag. True when any query point has more
              than ``max_neighbors`` neighbors within the cutoff.

        """
        collider = cast(NeighborList, system.collider)
        inner_system = replace(system, collider=collider.secondary_collider)
        return collider.secondary_collider.create_cross_neighbor_list(
            pos_a, pos_b, inner_system, cutoff, max_neighbors
        )
