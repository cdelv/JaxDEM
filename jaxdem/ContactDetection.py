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

from .Factory import Factory
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State
    from .Space import Domain

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


@Grid.register("fgrid")
class FreeGrid(Grid):
    def __init__(self, state: 'State', domain: 'Domain', cell_size: Optional[float] = None, cell_capacity: Optional[float] = None):
        self.dim = state.dim
        self.periodic = domain.periodic
        self.sort = True

        self.cell_size = cell_size
        if self.cell_size is None:
            self.cell_size = 2.0 * jnp.max(state.rad)

        self.cell_capacity = cell_capacity
        if self.cell_capacity is None:
            factor = 4.0/3 if self.dim == 3 else 1.0
            cell_vol = self.cell_size**self.dim
            sphere_vol = factor*jnp.pi*jnp.min(state.rad)**self.dim
            self.cell_capacity = jnp.ceil(1.2*cell_vol/sphere_vol).astype(int)

        n = jnp.maximum(1, jnp.ceil(2*jnp.max(state.rad)/self.cell_size - 1)).astype(int)
        grids = jnp.meshgrid(*[jnp.arange(-n, n+1)]*self.dim, indexing='ij')  
        self.neighbor_mask = jnp.stack(grids, axis=-1).reshape(-1, self.dim)

        self.n_cells = jnp.floor(domain.box_size/self.cell_size).astype(int)
        self.weights = self.n_cells.at[0].set(1)

    @staticmethod
    @partial(jax.jit, inline=True)
    def find_neighbors(i: int, state: 'State', system: 'System') -> jnp.ndarray:
        ...

    @staticmethod
    def update_values(state: 'State', system: 'System') -> Tuple['State', 'System']:
        ...

    @staticmethod
    @partial(jax.jit, inline=True)
    def update(state: 'State', system: 'System') -> Tuple['State', 'System']:
        ...

    @staticmethod
    @partial(jax.jit, inline=True)
    def get_cell(state, system):
        cell = jnp.floor(state.pos / system.grid.cell_size).astype(int)
        cell -= system.grid.periodic * system.grid.n_cells * jnp.floor(cell / system.grid.n_cells).astype(int)
        return cell

    @staticmethod
    @partial(jax.jit, inline=True)
    def get_hash(cell: jnp.ndarray, system):
        return jnp.dot(cell, system.grid.weights)

    @staticmethod
    @partial(jax.jit, inline=True)
    def get_hash_fused(state, system):
        cell = FreeGrid.get_cell(state, system)
        return FreeGrid.get_hash(cell, system)

    @staticmethod
    @partial(jax.jit, inline=True)
    def sort_arrays(state, system):
        sort_id = jnp.argsort(state._hash)
        state._hash = state._hash[sort_id]
        state.pos = state.pos[sort_id]
        state.vel = state.vel[sort_id]
        state.rad = state.rad[sort_id]
        state.mass = state.mass[sort_id]
        return state, system