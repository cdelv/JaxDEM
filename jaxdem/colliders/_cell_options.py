# SPDX-License-Identifier: BSD-3-Clause
"""Shared options for dynamic and static cell traversal."""

from __future__ import annotations

import jax


def validate_cell_options(tile_capacity: int | None, clump_aware: bool) -> None:
    """Check the fixed tile budget and clump option."""
    if tile_capacity is not None:
        if (
            isinstance(tile_capacity, bool)
            or not isinstance(tile_capacity, int)
            or tile_capacity < 0
        ):
            raise ValueError("tile_capacity must be a non-negative Python integer")
        index_limit = (1 << (63 if jax.config.jax_enable_x64 else 31)) - 1
        if tile_capacity >= index_limit:
            raise ValueError("tile_capacity exceeds the particle-index integer range")
    if not isinstance(clump_aware, bool):
        raise ValueError("clump_aware must be a Python boolean")
