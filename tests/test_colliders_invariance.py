# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unit tests to verify invariance of positions, velocities, and forces across all periodic colliders."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
import jaxdem as jdem

def sort_state_by_unique_id(state: jdem.State) -> jdem.State:
    idx = jnp.argsort(state.unique_id)
    new_q = jdem.utils.quaternion.Quaternion.create(
        w=state.q.w[idx],
        xyz=state.q.xyz[idx]
    )
    return jdem.State(
        pos_c=state.pos_c[idx],
        pos_p=state.pos_p[idx],
        vel=state.vel[idx],
        force=state.force[idx],
        q=new_q,
        ang_vel=state.ang_vel[idx],
        torque=state.torque[idx],
        rad=state.rad[idx],
        volume=state.volume[idx],
        mass=state.mass[idx],
        inertia=state.inertia[idx],
        clump_id=state.clump_id[idx],
        bond_id=state.bond_id[idx],
        unique_id=state.unique_id[idx],
        mat_id=state.mat_id[idx],
        species_id=state.species_id[idx],
        fixed=state.fixed[idx],
        _pos_p_rot=state._pos_p_rot[idx],
    )

def run_simulation(dim: int, collider_name: str) -> jdem.State:
    # Setup 200 particles
    if dim == 2:
        n_per_axis = (10, 20)
    else:
        n_per_axis = (5, 5, 8)

    spacing = 1.2
    radius = 0.5

    state = jdem.utils.grid_state(
        n_per_axis=n_per_axis,
        spacing=spacing,
        radius=radius,
        vel_range=[-1.0, 1.0],
        seed=101,
    )

    box_size = jnp.array([n * spacing for n in n_per_axis])

    mats = [jdem.Material.create("elastic", young=1000.0, poisson=0.3, density=1.0)]
    matcher = jdem.MaterialMatchmaker.create("linear")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)

    collider_kw = {}
    if collider_name in ("CellList", "StaticCellList"):
        collider_kw = {"state": state}
    elif collider_name in ("MultiCellList", "DynamicMultiCellList"):
        collider_kw = {"state": state}
    elif collider_name == "NeighborList":
        collider_kw = {
            "state": state,
            "cutoff": 1.0,
            "skin": 0.5,
            "secondary_collider_type": "CellList",
            "secondary_collider_kw": {"state": state},
        }

    system = jdem.System.create(
        state.shape,
        dt=0.001,
        collider_type=collider_name,
        collider_kw=collider_kw,
        domain_type="periodic",
        domain_kw={"box_size": box_size},
        force_model_type="spring",
        mat_table=mat_table,
    )

    # Wrap coordinates at start
    state, system = system.domain.apply(state, system)

    # Run simulation for 5000 steps
    state_end, system_end = system.step(state, system, n=1500)
    jax.block_until_ready(state_end)

    # Sort the state back to original order
    return sort_state_by_unique_id(state_end)

@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider_name", [
    "CellList",
    "StaticCellList",
    "NeighborList",
    "MultiCellList",
    "DynamicMultiCellList"
])
def test_collider_invariance(dim: int, collider_name: str):
    # Reference simulation using the naive collider
    ref_state = run_simulation(dim, "naive")

    # Target collider simulation
    target_state = run_simulation(dim, collider_name)

    # Assert positions, velocities and forces are identical to machine precision
    assert jnp.allclose(target_state.pos, ref_state.pos, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(target_state.vel, ref_state.vel, atol=1e-8, rtol=1e-8)
    assert jnp.allclose(target_state.force, ref_state.force, atol=1e-8, rtol=1e-8)
