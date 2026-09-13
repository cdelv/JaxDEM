# SPDX-License-Identifier: BSD-3-Clause
"""Derived logical rigid-body topology.

The simulation state keeps particle-shaped rigid-body fields for fast contact
kernels. This module provides a padded, body-indexed view without changing the
serialized :class:`jaxdem.State` pytree.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class BodyTopology:
    """Particle-to-body mapping with at most ``N`` padded body slots."""

    member_to_body: jax.Array
    representative: jax.Array
    valid: jax.Array
    fixed: jax.Array
    member_count: jax.Array

    def gather_members(self, body_values: jax.Array) -> jax.Array:
        """Replicate body-slot values onto their particle members."""
        extra = body_values.ndim - self.member_to_body.ndim
        index = self.member_to_body.reshape(self.member_to_body.shape + (1,) * extra)
        return jnp.take_along_axis(body_values, index, axis=-(extra + 1))

    def gather_representatives(self, member_values: jax.Array) -> jax.Array:
        """Read one value per body from its representative member."""
        extra = member_values.ndim - self.representative.ndim
        index = self.representative.reshape(self.representative.shape + (1,) * extra)
        values = jnp.take_along_axis(member_values, index, axis=-(extra + 1))
        mask = self.valid.reshape(self.valid.shape + (1,) * extra)
        return jnp.where(mask, values, jnp.zeros_like(values))


@jax.jit
def body_topology(clump_id: jax.Array, fixed: jax.Array) -> BodyTopology:
    """Derive a padded logical-body view from particle-shaped state fields."""
    n = clump_id.shape[-1]
    if n == 0:
        empty_int = jnp.zeros(clump_id.shape, dtype=clump_id.dtype)
        empty_bool = jnp.zeros(clump_id.shape, dtype=bool)
        return BodyTopology(
            member_to_body=clump_id,
            representative=empty_int,
            valid=empty_bool,
            fixed=empty_bool,
            member_count=empty_int,
        )
    flat_ids = clump_id.reshape((-1, n))
    flat_fixed = fixed.reshape((-1, n))
    indices = jnp.arange(n, dtype=clump_id.dtype)

    def one(
        ids: jax.Array, member_fixed: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        count = jnp.bincount(ids, length=n)
        representative = jax.ops.segment_min(indices, ids, num_segments=n)
        representative = jnp.where(count > 0, representative, 0)
        body_fixed = jax.ops.segment_max(
            member_fixed.astype(jnp.int32), ids, num_segments=n
        ).astype(bool)
        body_fixed = (count > 0) & body_fixed
        return representative, count > 0, body_fixed, count

    representative, valid, body_fixed, member_count = jax.vmap(one)(
        flat_ids, flat_fixed
    )
    lead = clump_id.shape[:-1]
    return BodyTopology(
        member_to_body=clump_id,
        representative=representative.reshape((*lead, n)),
        valid=valid.reshape((*lead, n)),
        fixed=body_fixed.reshape((*lead, n)),
        member_count=member_count.reshape((*lead, n)),
    )
