# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
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
    Abstract base class for spatial acceleration structures (grids).

    Grids are used to accelerate the process of finding neighboring particles
    for interaction calculations (e.g., force computation), reducing complexity
    from O(N^2) to potentially O(N) or O(N log N) depending on the implementation
    and particle distribution.

    This class defines the common interface that all grid implementations must adhere to,
    allowing them to be used interchangeably within the simulation system.
    """
    periodic: bool = False

    @staticmethod
    @abstractmethod
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

    @staticmethod
    @abstractmethod
    @jax.jit
    def update_values(state: 'State', system: 'System') -> Tuple['State', 'System']:
        """
       Update the grid structure's parameters based on the current state. 
       This is useful when changing the domain dimensions, adding new particles, or modifying their radius.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            The potentially updated state and system.
        """

    @staticmethod
    @abstractmethod
    @partial(jax.jit, inline=True)
    def find_neighbors(i: int, state: 'State', system: 'System') -> jnp.ndarray:
        """
        Finds potential neighbors for particle 'i' by checking adjacent cells.

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


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
@Grid.register("Igrid")
class ImplicitGrid(Grid):
    """
    A grid implementation that uses spatial hashing without storing a persistent grid structure.

    This approach calculates a hash for each particle based on its cell location.
    Neighbor searches involve checking the particle's own cell and neighboring cells by
    calculating their hashes and searching within the sorted particle array.

    This avoids allocating large arrays for cell lists but requires sorting and searching.

    If the number of particles in the simulation changes, the grid doesn't need to be reconstructed. 
    What's important is the distribution of radius sizes. 
    If the grid was created with zero particles in the state, call update_values. 
    Reconstruct the grid if the new particles change the maximum and minimum radii, call update_values the grid.

    For periodic domains, the grid's domain size must match the system's domain. Otherwise, incorrect results can occur.
    If particles leave the simulation domain, the method loses performance, and some contacts may be missed for non-periodic domains.
    If changing the domain dimensions, call update_values the grid.
    
    Attributes
    ----------
    periodic : bool
        Inherited from Domain, indicates if periodic boundary conditions are used.
    cell_size : float
        The size of each cubic/square cell in the implicit grid.
    cell_capacity : int
        The maximum number of particles expected within a single cell.
    neighbor_mask : jnp.ndarray
        A precomputed array of relative cell offsets (e.g., [-1,-1], [-1,0], ..., [1,1])
        representing the cells to check for neighbors around a central cell.
    n_cells : jnp.ndarray
        The number of cells along each dimension within the domain boundaries. Shape (dim,).
    weights : jnp.ndarray
        Weights used to compute a unique scalar hash from a multi-dimensional cell index. Shape (dim,).
    """
    periodic: bool = field(default = False, metadata = {'static': True})
    cell_capacity: int = field(default = 4, metadata = {'static': True})
    n_neighbor: int = field(default = 1, metadata = {'static': True})
    cell_size: float = 1.0 
    neighbor_mask: jnp.ndarray = jnp.zeros((8, 3)) 
    n_cells: jnp.ndarray = jnp.ones(3) 
    weights: jnp.ndarray = jnp.ones(3) 

    @staticmethod
    @partial(jax.jit, inline=True)
    def find_neighbors(i: int, state: 'State', system: 'System') -> jnp.ndarray:
        """
        Finds potential neighbors for particle 'i' by checking adjacent cells.

        Implements the Grid interface `find_neighbors` method. *Assumes particle arrays
        in the state ARE currently sorted by hash*. Call state, sytem = system.grid.update()
        before calling this function.

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
    @partial(jax.jit, static_argnames=('n', 'cell_capacity'))
    def update_values(state: 'State', domain: 'System', n:int = 1, cell_size: float = None, cell_capacity: int = 4):
        """
        Update the grid structure's parameters based on the current state. 
        This is useful when changing the domain dimensions, adding new particles, or modifying their radius.
        
        This overrides user defined parameters and replaces them with the progrm's euristics.

        Parameters
        ----------
        state : State
        system : System

        Returns
        -------
        Tuple[State, System]
            The potentially updated state and system.
        """
        periodic = domain.periodic
        if cell_size is None:
            cell_size = 2.0 * jnp.max(state.rad)

        """
        To fullfill the static condition make a function that returns 
        n and cell_capacity so that the user can input it as a static arg.
        Or make a function to bypass this
        if cell_capacity is None:
            factor = 4.0/3 if state.dim == 3 else 1.0
            cell_vol = cell_size**state.dim
            sphere_vol = factor*jnp.pi*jnp.min(state.rad)**state.dim
            cell_capacity = jnp.ceil(1.2*cell_vol/sphere_vol).astype(int)
        """

        #n = jnp.maximum(1, jnp.ceil(2*jnp.max(state.rad)/cell_size - 1)).astype(int)
        grids = jnp.meshgrid(*[jnp.arange(-n, n+1)]*state.dim, indexing='ij')  
        neighbor_mask = jnp.stack(grids, axis=-1).reshape(-1, state.dim)

        n_cells = jnp.floor(domain.box_size/cell_size).astype(int)
        weights = n_cells.at[0].set(1)

        return {'periodic': periodic, 'cell_capacity': cell_capacity, 'cell_size': cell_size, 'neighbor_mask': neighbor_mask, 'n_cells': n_cells, 'weights': weights}

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
        state.pos = state.pos[sort_id]
        state.vel = state.vel[sort_id]
        state.rad = state.rad[sort_id]
        state.mass = state.mass[sort_id]
        return state