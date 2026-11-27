# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Multi-cell (loose-grid / UGrid) collider — a JAX port of dragon-space's loose/tight grid."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from . import Collider
from ._cell_options import validate_cell_capacity_estimate, validate_cell_options
from .cell_list import _CellListBase, _traverse_pairs_impl

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@jax.jit(inline=True, static_argnames=("pair_fn",))
def _traverse_pairs(
    state: State,
    system: System,
    cell_size: jax.Array,
    neighbor_mask: jax.Array,
    pair_fn: Callable[..., Any],
    init_acc: Any,
) -> tuple[Any, jax.Array]:
    """Fold a per-pair kernel over candidate pairs of the loose-grid partition."""
    return _traverse_pairs_impl(
        state, system, cell_size, neighbor_mask, pair_fn, init_acc, prune_cells=True
    )


@Collider.register("MultiCellList")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DynamicMultiCellList(_CellListBase):
    r"""Loose-cell collider with dynamic or fixed-capacity static traversal.

    A regular grid assigns each particle center to one cell. Sorted cell
    hashes place each cell's members in one contiguous run. Each loose cell
    also stores the union of its members' axis-aligned search boxes.

    A fixed stencil selects candidate cells. A conservative box test skips
    cells whose members cannot contact the query particle. Shear-periodic
    domains skip this box rejection and retain the conservative stencil.

    With ``static=False`` and ``tile_capacity=None``, occupancy sets the dynamic
    traversal bound.
    Explicit environment batches share the longest active bound within each
    bounded query batch. The dynamic path has no per-cell occupancy limit.

    Set ``static=True`` for an automatic fixed tile budget. With a reference
    state, ``System.create`` measures the actual cell work and adds headroom.
    Standalone construction uses a density and packing estimate. Later
    geometry changes or particle clustering can still cause overflow.
    Set ``tile_capacity`` to a nonnegative Python integer for an explicit budget.
    This global budget covers four-candidate tiles across all source particles
    and stencil cells in one environment. Rejected cells allocate no tiles.
    Dense cells can use spare capacity from other cells. The static path has
    no per-cell occupancy limit. Each environment uses the same static budget.

    Set ``clump_aware=True`` to exclude same-clump members before static tile
    allocation. All paths exclude same-clump interactions during evaluation.
    External candidates can still exhaust the global tile budget.

    An overflow invalidates the result. After force or energy evaluation,
    call ``system.check_overflow()``. For neighbor queries, check the returned
    overflow flag. The static path reports tile and hash overflow.

    For a fixed stencil, dynamic storage scales as :math:`O(N)`.
    Static storage scales as :math:`O(N + C)`, where :math:`C` is
    ``tile_capacity``. Sorting costs :math:`O(N \log N)` in both paths.

    The dynamic path supports forward differentiation. Set ``tile_capacity``
    or use a ``NeighborList`` cache for reverse differentiation.
    """

    _prune_cells: ClassVar[bool] = True

    @classmethod
    def Create(
        cls,
        state: State,
        cell_size: ArrayLike | None = None,
        search_range: ArrayLike | None = None,
        box_size: ArrayLike | None = None,
        max_hashes: int | None = None,
        tile_capacity: int | None = None,
        clump_aware: bool = False,
        static: bool = False,
        *,
        _auto_capacity: bool = False,
    ) -> Self:
        """Create a DynamicMultiCellList instance from the reference state."""
        del max_hashes  # deprecated no-op, kept for API compatibility
        validate_cell_options(tile_capacity, clump_aware, static, _auto_capacity)
        validate_cell_capacity_estimate(state, cell_size, tile_capacity, static)
        max_rad = jnp.max(state._rad, initial=0.0)
        if cell_size is None:
            cell_size = jnp.where(max_rad > 0, 2.0 * max_rad, 1.0)
        return cls._build(
            state,
            jnp.asarray(cell_size, dtype=float),
            max_rad,
            search_range,
            box_size,
            tile_capacity,
            clump_aware,
            static,
            _auto_capacity,
        )


__all__ = ["DynamicMultiCellList"]
