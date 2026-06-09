# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Neighbor List Collider implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    - **skin**: The buffer distance. Default is `0.05`. Larger skin reduces rebuild frequency but inflates `max_neighbors`, increasing step time and memory.
    - **max_neighbors**: The static neighbor buffer size per particle. If not provided, it is estimated using safety factor and density heuristics. Setting this too small causes list overflows, while setting it too large wastes GPU memory.
    - **number_density**: Macroscopic number density used to estimate neighbor counts when not provided. Default is `1.0`.
    - **safety_factor**: Multiplier applied to the estimated density to account for local fluctuations. Default is `1.2`.
    - **secondary_collider_type**: The identifier of the underlying collider used to execute the spatial queries during rebuilds (e.g. ``"CellList"``, ``"naive"``, or ``"DynamicMultiCellList"``). Any registered ``Collider`` subclass in the library can be used for the rebuild phase, allowing the rebuild cost to be optimized based on system characteristics.
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
    Buffer distance. The list is built with ``radius = cutoff + skin`` and
    rebuilt when ``max_displacement > skin / 2``.
    """

    max_neighbors: int = jax.tree.static()
    """Static buffer size for the neighbor list."""

    @classmethod
    def Create(
        cls,
        state: State,
        cutoff: float,
        skin: float = 0.05,
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
        skin : float, default 0.05
            The buffer distance added to the cutoff for the neighbor list.
            **Must be > 0.0 for performance.**
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
        skin_val = skin * cutoff
        list_cutoff = cutoff + skin_val

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
        max_neighbors = min(max_neighbors, max_possible_neighbors)
        max_neighbors = min(max_neighbors, state.N)
        max_neighbors = max(max_neighbors, 0)

        if secondary_collider_kw is None:
            secondary_collider_kw = {}
        else:
            secondary_collider_kw = dict(secondary_collider_kw)

        secondary_collider_kw["state"] = state
        if secondary_collider_type.lower() == "celllist" or (
            secondary_collider_type.lower() == "staticcelllist"
            and "cell_size" not in secondary_collider_kw
        ):
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
    @jax.jit(static_argnames=("max_neighbors",))
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

        disp = state.pos - collider.old_pos
        max_disp_sq = jnp.max(norm2(disp))
        trigger_dist_sq = collider.skin**2 / 4
        should_rebuild = (max_disp_sq > trigger_dist_sq) + (collider.n_build_times == 0)

        def rebuild_branch(
            operands: tuple[State, System, NeighborList],
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            s, sys, col = operands
            return col._rebuild(col, s, sys)

        def no_rebuild_branch(
            operands: tuple[State, System, NeighborList],
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            _, _, col = operands
            return (
                state,
                col.neighbor_list,
                col.old_pos,
                col.n_build_times,
                col.overflow,
            )

        state, nl, old_pos, n_build, overflow = jax.lax.cond(
            should_rebuild > 0,
            rebuild_branch,
            no_rebuild_branch,
            (state, system, collider),
        )

        system.collider = replace(
            collider,
            neighbor_list=nl,
            old_pos=old_pos,
            n_build_times=n_build,
            overflow=overflow,
        )
        return state, system, nl, overflow

    @staticmethod
    @partial(jax.named_call, name="NeighborList._rebuild")
    def _rebuild(
        collider: NeighborList, state: State, system: System
    ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
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
        Tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]
            A tuple containing:
            - Unsorted State
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
            sorted_state,
            _,
            sorted_nl_indices,
            overflow_flag,
        ) = collider.secondary_collider.create_neighbor_list(
            state, system, list_cutoff, collider.max_neighbors
        )

        return (
            sorted_state,
            sorted_nl_indices,
            sorted_state.pos,
            collider.n_build_times + 1,
            overflow_flag,
        )

    @staticmethod
    @jax.jit
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
        iota = jax.lax.iota(dtype=int, size=state.N)  # should this be cached?
        collider = cast(NeighborList, system.collider)

        # 1. Check Displacement & Trigger Rebuild
        # disp = system.domain.displacement(state.pos, collider.old_pos, system)
        disp = state.pos - collider.old_pos  # this should not be a periodic distance
        max_disp_sq = jnp.max(norm2(disp))
        trigger_dist_sq = collider.skin**2 / 4

        # Force rebuild if displacement is large OR if this is the first step (count == 0)
        should_rebuild = (max_disp_sq > trigger_dist_sq) + (collider.n_build_times == 0)

        def rebuild_branch(
            operands: tuple[State, System, NeighborList],
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            s, sys, col = operands
            return col._rebuild(col, s, sys)

        def no_rebuild_branch(
            operands: tuple[State, System, NeighborList],
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            _, _, col = operands
            return (
                state,
                col.neighbor_list,
                col.old_pos,
                col.n_build_times,
                col.overflow,
            )

        state, nl, old_pos, n_build, overflow = jax.lax.cond(
            should_rebuild > 0,
            rebuild_branch,
            no_rebuild_branch,
            (state, system, collider),
        )

        # 2. Compute Forces
        # Pre-calculate contact points in global frame for torque
        pos_p_global = state._pos_p_rot
        pos = state.pos

        def per_particle_force(
            i: jax.Array, pos_pi: jax.Array, neighbors: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            # i: ID of the current particle
            # pos_pi: vector from COM to surface for particle i
            # neighbors: array of neighbor IDs

            def per_neighbor_force(j_id: jax.Array) -> tuple[jax.Array, jax.Array]:
                # We mask computations for padding (-1)
                valid = j_id != -1
                safe_j = jnp.maximum(j_id, 0)
                valid = valid * valid_interaction_mask(
                    state.clump_id[i],
                    state.clump_id[safe_j],
                    state.bond_id[i],
                    state.unique_id[safe_j],
                )

                f, t = system.force_model.force(i, safe_j, pos, state, system)
                return f * valid, t * valid

            forces, torques = jax.vmap(per_neighbor_force)(neighbors)

            f_sum = jnp.sum(forces, axis=0)
            # Add contact torque: T_total = Sum(T_ij) + (r_i x F_total)
            t_sum = jnp.sum(torques, axis=0) + cross(pos_pi, f_sum)

            return f_sum, t_sum

        # Vmap over particle IDs [0, 1, ..., N]
        state.force, state.torque = jax.vmap(per_particle_force)(iota, pos_p_global, nl)

        # Update collider cache
        system.collider = replace(
            collider,
            neighbor_list=nl,
            old_pos=old_pos,
            n_build_times=n_build,
            overflow=overflow,
        )

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="NeighborList.compute_potential_energy")
    def compute_potential_energy(state: State, system: System) -> tuple[State, System, jax.Array]:
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
        disp = state.pos - collider.old_pos
        max_disp_sq = jnp.max(norm2(disp))
        trigger_dist_sq = collider.skin**2 / 4

        # Force rebuild if displacement is large OR if this is the first step (count == 0)
        should_rebuild = (max_disp_sq > trigger_dist_sq) + (collider.n_build_times == 0)

        def rebuild_branch(
            operands: tuple[State, System, NeighborList],
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            s, sys, col = operands
            return col._rebuild(col, s, sys)

        def no_rebuild_branch(
            operands: tuple[State, System, NeighborList],
        ) -> tuple[State, jax.Array, jax.Array, jax.Array, jax.Array]:
            _, _, col = operands
            return (
                state,
                col.neighbor_list,
                col.old_pos,
                col.n_build_times,
                col.overflow,
            )

        state, nl, old_pos, n_build, overflow = jax.lax.cond(
            should_rebuild > 0,
            rebuild_branch,
            no_rebuild_branch,
            (state, system, collider),
        )

        iota = jax.lax.iota(dtype=int, size=state.N)

        def per_particle_energy(i: jax.Array) -> jax.Array:
            neighbors = nl[i]

            def per_neighbor_energy(j_id: jax.Array) -> jax.Array:
                valid = j_id != -1
                safe_j = jnp.maximum(j_id, 0)
                valid = valid * valid_interaction_mask(
                    state.clump_id[i],
                    state.clump_id[safe_j],
                    state.bond_id[i],
                    state.unique_id[safe_j],
                )
                e = system.force_model.energy(i, safe_j, state.pos, state, system)
                return e * valid

            # Sum energies and divide by 2 (double counting in neighbor list)
            return 0.5 * jnp.sum(jax.vmap(per_neighbor_energy)(neighbors))

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
    @jax.jit(static_argnames=("max_neighbors",))
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
