# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Access one contact face incident on a physical particle key."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import State


def get_facet_incidence(
    idx: int | jax.Array, state: State, slot: int | jax.Array = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return face membership, vertices, primary key, ID, and active mask.

    Slot zero retains the original facet storage. An ordinary sphere has only
    slot zero active; padded extra slots never create sphere contacts.
    """
    idx = jnp.asarray(idx)
    face = state.facet_id[idx]
    vertices = state.facet_vertices[idx]
    if state.facet_ids_extra.shape[-1]:
        ids = jnp.concatenate((face[..., None], state.facet_ids_extra[idx]), axis=-1)
        faces = jnp.concatenate(
            (vertices[..., None, :], state.facet_vertices_extra[idx]), axis=-2
        )
        face = jnp.take(ids, slot, axis=-1)
        vertices = jnp.take(faces, slot, axis=-2)
    is_facet = face >= 0
    vertices = jnp.where(is_facet[..., None], vertices, idx[..., None])
    active = is_facet | ((state.facet_id[idx] < 0) & (slot == 0))
    primary = idx == jnp.min(vertices, axis=-1)
    return is_facet, vertices, primary, face, active
