# This file is part of the JaxDEM library. For more information and source code,
# visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. Feedback and contributions are welcome.

import jax
import jax.numpy as jnp 

from typing import Optional, Tuple
from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass, field

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .System import System
    from .Domain import Domain

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class Grid(Factory, ABC):
    """
    Abstract base class for grid-based spatial acceleration structures.

    Grids accelerate neighbor search operations in particle simulations by reducing
    computational complexity from O(N^2) to O(N log N) or O(N), depending on
    the implementation and particle distribution.

    This interface enforces a common structure for all grid implementations,
    ensuring they can be used interchangeably in simulation systems.
    """
    periodic: bool = False

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def update(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Update the particle state with grid-based neighbor data.

        This method should update internal structures like particle hashes and sorted arrays.

        Parameters
        ----------
        state : State
            The current state of the simulation (positions, velocities, etc.).
        system : System
            The simulation system which includes the grid and domain information.

        Returns
        -------
        Tuple[State, System]
            The updated state and system with new grid data.
        """

    @classmethod
    @abstractmethod
    @partial(jax.jit, inline=True, static_argnames=('cls'))
    def build(cls, state: 'State', system: 'System') -> 'Grid':
        """
        Construct and initialize a grid object using the given state and system.

        Returns
        -------
        Grid
            A new instance of a Grid implementation.
        """

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def find_neighbors(i: int, state: 'State', system: 'System') -> jnp.ndarray:
        """
        Find potential neighbors for a given particle index.

        Parameters
        ----------
        i : int
            Index of the particle to find neighbors for.
        state : State
            Current simulation state.
        system : System
            Simulation system including grid and domain.

        Returns
        -------
        jnp.ndarray
            Indices of potential neighbor particles.
        """

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
@Grid.register("Igrid")
class ImplicitGrid(Grid):
    """
    Implicit grid implementation using spatial hashing for neighbor search.

    Unlike traditional grids that store particle indices explicitly in spatial bins,
    this approach computes a spatial hash based on each particle's position.
    Neighbor search is performed by checking adjacent spatial hashes in a 
    pre-sorted particle list. This enables efficient contact detection without
    storing large grid arrays.

    This grid works in both periodic and non-periodic domains, but performance
    may degrade if particles exit the simulation domain. For periodic domains,
    the grid assumes the domain dimensions exactly match those of the system. 
    A mismatch may result in incorrect or spurious neighbor detections.

    Example
    -------
    ```python
    # Create the grid using the factory system
    grid = jdem.Grid.create('Igrid')

    # Initialize grid values with the current state and system
    system.grid = system.grid.build(state, system, cell_capacity=4, n_neighbors=1)

    # Optionally estimate reasonable parameters before building:
    cell_capacity, n_neighbors = ImplicitGrid.estimate_ocupancy(state, system, cell_size=cell_size)
    ```

    Notes
    -----
    - `cell_capacity` and `n_neighbors` must be passed as static arguments and cannot be dynamically 
      computed inside a `@jax.jit` context. Estimate them outside of JIT if needed.
    - If `cell_size` is not provided, it is computed based on the maximum particle radius.
    - Rebuild the grid if the domain size, number of particles, or particle radii change significantly.

    Attributes
    ----------
    periodic : bool
        Whether periodic boundary conditions are enabled.
    cell_size : float
        The size of a grid cell, typically computed from particle radii.
    cell_capacity : int
        Estimated maximum number of particles in a single grid cell.
    neighbor_mask : jnp.ndarray
        Relative cell offsets to check for potential neighbors.
    n_cells : jnp.ndarray
        Number of grid cells along each spatial dimension.
    weights : jnp.ndarray
        Integer weights used to compute unique 1D spatial hashes from multidimensional indices.
    """
    periodic: bool = field(default = False, metadata = {'static': True})
    cell_capacity: int = field(default = 4, metadata = {'static': True})
    n_neighbors: int = field(default = 1, metadata = {'static': True})
    cell_size: float = 1.0 
    neighbor_mask: jnp.ndarray = jnp.zeros((8, 3)) 
    n_cells: jnp.ndarray = jnp.ones(3) 
    weights: jnp.ndarray = jnp.ones(3) 

    @classmethod
    @partial(jax.jit, inline=True, static_argnames=('cls', 'cell_capacity', 'n_neighbors'))
    def build(cls, state: 'State', system: 'System', cell_capacity: int = 4, n_neighbors: int = 1, cell_size = None) -> 'ImplicitGrid':
        """
        Construct an instance of the ImplicitGrid using the provided simulation state and system.

        This method computes internal grid parameters such as neighbor offsets, cell dimensions, 
        and spatial hashing weights. It is designed to be compatible with JAX JIT compilation, 
        requiring `cell_capacity` and `n_neighbors` to be passed as static arguments.

        Parameters
        ----------
        state : State
            The current simulation state, including particle positions and radii.
        system : System
            The simulation system, including domain information.
        cell_capacity : int, default=4
            The estimated maximum number of particles in a single grid cell.
            Must be static (known at compile time) when used with `@jax.jit`.
        n_neighbors : int, default=1
            Number of neighboring cells to include in the search radius per dimension.
            Defines the stencil of cells used during neighbor searches.
            Must be static under JIT.
        cell_size : float, optional
            The size of each grid cell. If not provided, it is automatically computed 
            as `2.0 * max(state.rad)` to ensure at least one particle per cell diameter.

        Returns
        -------
        ImplicitGrid
            A fully initialized ImplicitGrid object with precomputed neighbor mask, 
            spatial hash weights, and domain-specific parameters.
        """
        periodic = system.domain.periodic
        cell_capacity = cell_capacity
        n_neighbors = n_neighbors

        if cell_size is None:
            cell_size = 2.0 * jnp.max(state.rad)

        grids = jnp.meshgrid(*[jnp.arange(-n_neighbors, n_neighbors+1)] * state.dim, indexing='ij')  
        neighbor_mask = jnp.stack(grids, axis=-1).reshape(-1, state.dim)

        n_cells = jnp.floor(system.domain.box_size/cell_size).astype(int)
        weights = n_cells.at[0].set(1)
        return ImplicitGrid(
            periodic = periodic, 
            cell_capacity = cell_capacity, 
            n_neighbors = n_neighbors, 
            cell_size = cell_size, 
            neighbor_mask = neighbor_mask, 
            n_cells = n_cells, 
            weights = weights
        )

    @staticmethod
    @partial(jax.jit, inline=True)
    def find_neighbors(i: int, state: 'State', system: 'System') -> jnp.ndarray:
        """
        This method assumes particle arrays in the state are sorted by spatial hash.
        Make sure to call `state, system = system.grid.update(state, system)` before using this.

        Parameters
        ----------
        i : int
            Index of the target particle.
        state : State
        system : System

        Returns
        -------
        jnp.ndarray
            Array of indices of potential neighbors.
        """
        current_cell = ImplicitGrid.get_cell(state.pos[i], system)
        neighbor_hash = ImplicitGrid.get_hash(current_cell + system.grid.neighbor_mask, system)

        def find_in_cell(cell_hash):
            start_idx = jnp.searchsorted(state._hash, cell_hash, side='left', method='scan_unrolled')
            valid_cell = (start_idx < state.N) * (state._hash[start_idx] == cell_hash)
            indices_in_cell = start_idx + jax.lax.iota(dtype=int, size=system.grid.cell_capacity)
            valid_mask = (valid_cell * 
                (indices_in_cell < state.N) *
                (state._hash[indices_in_cell] == cell_hash) *
                (indices_in_cell != i)
            )
            return jnp.where(valid_mask, indices_in_cell, -1)
        return jax.vmap(find_in_cell)(neighbor_hash).flatten()

    @staticmethod
    @partial(jax.jit, inline=True)
    def update(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
        Update the grid structure based on the current state.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            The potentially updated state and system.
        """
        new_hash = system.grid._get_hash_fused(state.pos, system)
        state = system.grid._sort_arrays(new_hash, state)
        return state, system

    @staticmethod
    def estimate_ocupancy(state: 'State', system: 'System', cell_size = None) -> Tuple[int, int]:
        """
        Estimate grid cell occupancy parameters based on particle radius.

        This heuristic assumes spherical or circular particles and calculates an
        appropriate cell capacity and neighbor range for efficient spatial hashing.
        """
        if cell_size is None:
            cell_size = 2.0 * jnp.max(state.rad)

        factor = 4.0/3 if state.dim == 3 else 1.0
        cell_vol = cell_size**state.dim
        sphere_vol = factor*jnp.pi*jnp.min(state.rad)**state.dim
        cell_capacity = jnp.ceil(cell_vol/sphere_vol).astype(int)

        n_neighbors = jnp.maximum(1, jnp.ceil(2*jnp.max(state.rad)/cell_size - 1)).astype(int)

        return cell_capacity, n_neighbors

    @staticmethod
    @partial(jax.jit, inline=True)
    def get_cell(pos: jnp.ndarray, system: 'System') -> jnp.ndarray:
        return jnp.floor(pos / system.grid.cell_size).astype(int)

    @staticmethod
    @partial(jax.jit, inline=True)
    def get_hash(cell: jnp.ndarray, system: 'System') -> int:
        cell -= system.grid.periodic * system.grid.n_cells * jnp.floor(cell / system.grid.n_cells).astype(int)
        return jnp.dot(cell, system.grid.weights)

    @staticmethod
    @partial(jax.jit, inline=True)
    def _get_hash_fused(pos: jnp.ndarray, system: 'System') -> int:
        cell = jnp.floor(pos / system.grid.cell_size).astype(int)
        cell -= system.grid.periodic * system.grid.n_cells * jnp.floor(cell / system.grid.n_cells).astype(int)
        return jnp.dot(cell, system.grid.weights)

    @staticmethod
    @partial(jax.jit, inline=True)
    def _sort_arrays(new_hash: jnp.ndarray, state: 'State') -> 'State':
        sort_id = jnp.argsort(new_hash)
        state._hash = new_hash[sort_id]
        state.rad = state.rad[sort_id]
        state.mass = state.mass[sort_id]

        state.pos = state.pos[sort_id]
        state.vel = state.vel[sort_id]
        return state