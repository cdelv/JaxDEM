# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Naive :math:`O(N^2)` collider implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING
from functools import partial

from . import Collider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@Collider.register("CellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CellList(Collider):
    r"""
    Implementation that computes forces and potential energies using a naive :math:`O(N^2)` all-pairs interaction loop.

    Notes
    -----
    Due to its :math:`O(N^2)` complexity, `NaiveSimulator` is suitable for simulations
    with a relatively small number of particles. For larger systems, a more
    efficient spatial partitioning collider should be used. However, this collider should be the fastest
    option for small systems (:math:`<1k-5k` spheres depending on the GPU).

    Parameters
    ----------
    neighbor_mask : jax.Array
        Mask of the cells to check for contact..
    cell_size : State
        Size of each cell in the cell list. Defaults to the diameter of the smallest particle.
    max_n_contacts : System
        Maximun number of contacts a particle can have.
    """
    neighbor_mask: jax.Array
    cell_size: jax.Array
    max_n_contacts: int = field(metadata={"static": True})

    @staticmethod
    def Create(state: "State", cell_size=None, n_cells=None, max_n_contacts: int = 8):
        if cell_size is None:
            cell_size = 2 * jnp.min(state.rad)
        cell_size = jnp.asarray(cell_size, dtype=float)

        if n_cells is None:
            max_rad = 2 * jnp.max(state.rad)
            n_cells = jnp.round(max_rad / cell_size).astype(int)
        n_cells = jnp.asarray(n_cells, dtype=int)

        ranges = [jnp.arange(-n_cells, n_cells + 1)] * state.dim
        mesh = jnp.meshgrid(*ranges, indexing="ij")
        neighbor_mask = jnp.stack([m.ravel() for m in mesh], axis=1)

        return CellList(
            neighbor_mask=neighbor_mask,
            cell_size=cell_size,
            max_n_contacts=int(max_n_contacts),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="NaiveSimulator.compute_potential_energy")
    def compute_potential_energy(state: "State", system: "System") -> jax.Array:
        r"""
        Computes the potential energy associated with each particle using a naive :math:`O(N^2)` all-pairs loop.

        This method iterates over all particle pairs (i, j) and sums the potential energy
        contributions computed by the ``system.force_model``.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        jax.Array
            One-dimensional array containing the total potential energy contribution for each particle.
        """
        iota = (jax.lax.iota(dtype=int, size=state.N),)

        return jax.vmap(
            lambda i, j, st, sys: jax.vmap(
                sys.force_model.energy, in_axes=(None, 0, None, None)
            )(i, j, st, sys).sum(axis=0),
            in_axes=(0, None, None, None),
        )(iota, iota, state, system)

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="NaiveSimulator.compute_force")
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
        r"""
        Computes the total force acting on each particle using a naive :math:`O(N^2)` all-pairs loop.

        This method sums the force contributions from all particle pairs (i, j)
        as computed by the ``system.force_model`` and updates the particle accelerations.

        Parameters
        ----------
        state : State
            The current state of the simulation.
        system : System
            The configuration of the simulation.

        Returns
        -------
        Tuple[State, System]
            A tuple containing the updated ``State`` object with computed accelerations
            and the unmodified ``System`` object.
        """
        iota = (jax.lax.iota(dtype=int, size=state.N),)

        n_cells = jnp.floor(system.domain.box_size / system.collider.cell_size).astype(
            int
        )
        weights = jnp.concatenate([jnp.array([1]), jnp.cumprod(n_cells[:-1])])

        start_idx = jnp.searchsorted(
            particleHash, cell_hash, side="left", method="scan_unrolled"
        )
        valid_H = (start_idx < particleHash.shape[0]) * (
            particleHash[start_idx] == cell_hash
        )

        def loop_body(k):
            candidate_index = start_idx + k
            valid_candidate = (
                valid_H
                * (candidate_index < particleHash.shape[0])
                * (particleHash[candidate_index] == cell_hash)
            )
            return jax.lax.cond(
                valid_candidate,
                lambda _: force_between(
                    positions[i], positions[candidate_index], 1.0, 10000.0
                ),
                lambda _: jnp.zeros_like(positions[i]),
                operand=None,
            )


__all__ = ["NaiveSimulator"]
