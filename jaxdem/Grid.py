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
    TO DO: define an interface
    """
    pass



def create_neighbor_mask(n):
    cell_mask = []
    for ix in range(-n, n + 1):
        for iy in range(-n, n + 1):
            cell_mask.append([ix, iy])
    return jnp.asarray(cell_mask, dtype=int)

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
            self.cell_capacity = 2*jnp.ceil(cell_vol/sphere_vol).astype(int)

        n = jnp.maximum(1, jnp.ceil(2*jnp.max(state.rad)/self.cell_size - 1)).astype(int)
        grids = jnp.meshgrid(*[jnp.arange(-n, n+1)]*self.dim, indexing='ij')  
        self.neighbor_mask = jnp.stack(grids, axis=-1).reshape(-1, self.dim)

        self.n_cells = jnp.floor(domain.box_size/self.cell_size).astype(int)
        self.weights = self.n_cells.at[0].set(1)

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