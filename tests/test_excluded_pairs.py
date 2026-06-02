# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unit tests to verify pair exclusions in colliders via bond_id."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import jaxdem as jdem

def test_excluded_pairs():
    # 3 spheres in a line: 0, 1, 2
    # positions: [0.0, 0.0], [1.0, 0.0], [2.0, 0.0]
    # radius: 1.1 for all (so all pairs overlap: 0-1, 1-2, 0-2)
    # We define bonds between 0-1 and 1-2.
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        rad=jnp.array([1.1, 1.1, 1.1]),
        bond_id=[[1], [0, 2], [1]],
    )
    
    # We want 0-1 and 1-2 to be excluded because they are connected.
    # But 0-2 should still interact because they are not connected.
    mats = [jdem.Material.create("elastic", young=1000.0, poisson=0.3, density=1.0)]
    matcher = jdem.MaterialMatchmaker.create("linear")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)
    
    for collider_name in ["naive", "CellList", "NeighborList", "SweepAndPrune"]:
        collider_kw = {}
        if "CellList" in collider_name or "SweepAndPrune" in collider_name:
            collider_kw = {"state": state}
        elif collider_name == "NeighborList":
            collider_kw = {
                "state": state,
                "cutoff": 3.0,
                "skin": 0.5,
                "secondary_collider_type": "CellList",
                "secondary_collider_kw": {"state": state},
            }
            
        system = jdem.System.create(
            state.shape,
            dt=0.001,
            collider_type=collider_name,
            collider_kw=collider_kw,
            force_model_type="spring",
            mat_table=mat_table,
        )
        
        # Compute forces
        state_f, _ = system.collider.compute_force(state, system)
        
        # Force on sphere 1 should be exactly 0 (since both 0-1 and 1-2 are excluded)
        assert jnp.allclose(state_f.force[1], 0.0, atol=1e-5)
        
        # Force on sphere 0 and 2 should be non-zero due to interaction between 0 and 2
        assert jnp.abs(state_f.force[0, 0]) > 0.1
        assert jnp.abs(state_f.force[2, 0]) > 0.1
        
        # Force on sphere 0 and 2 should be equal and opposite
        assert jnp.allclose(state_f.force[0], -state_f.force[2], atol=1e-5)
