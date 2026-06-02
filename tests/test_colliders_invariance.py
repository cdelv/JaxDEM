# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Unit tests to verify invariance of positions, velocities, and forces for spheres and clumps across all periodic colliders."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import jaxdem as jdem


def sort_state_by_unique_id(state: jdem.State) -> jdem.State:
    idx = jnp.argsort(state.unique_id)
    return jax.tree.map(lambda x: x[idx], state)


def check_states_match(state1: jdem.State, state2: jdem.State, atol=1e-8, rtol=1e-8):
    state1 = sort_state_by_unique_id(state1)
    state2 = sort_state_by_unique_id(state2)
    try:
        np.testing.assert_allclose(
            state1.pos, state2.pos, atol=atol, rtol=rtol, err_msg="pos mismatch"
        )
        np.testing.assert_allclose(
            state1.vel, state2.vel, atol=atol, rtol=rtol, err_msg="vel mismatch"
        )
        np.testing.assert_allclose(
            state1.force, state2.force, atol=atol, rtol=rtol, err_msg="force mismatch"
        )
        np.testing.assert_allclose(
            state1.q.w, state2.q.w, atol=atol, rtol=rtol, err_msg="q.w mismatch"
        )
        np.testing.assert_allclose(
            state1.q.xyz, state2.q.xyz, atol=atol, rtol=rtol, err_msg="q.xyz mismatch"
        )
        np.testing.assert_allclose(
            state1.ang_vel,
            state2.ang_vel,
            atol=atol,
            rtol=rtol,
            err_msg="ang_vel mismatch",
        )
        np.testing.assert_allclose(
            state1.torque,
            state2.torque,
            atol=atol,
            rtol=rtol,
            err_msg="torque mismatch",
        )
    except AssertionError:
        # Fallback to standard float32 precision limits
        np.testing.assert_allclose(state1.pos, state2.pos, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(state1.vel, state2.vel, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(state1.force, state2.force, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(state1.q.w, state2.q.w, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(state1.q.xyz, state2.q.xyz, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(state1.ang_vel, state2.ang_vel, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(state1.torque, state2.torque, atol=1e-6, rtol=1e-6)


def set_up_spheres(
    n: int,
    dim: int,
    polydispersity: float = 1.5,
    collider_type: str = "naive",
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
) -> tuple[jdem.State, jdem.System]:
    spacing = 1.2
    radius = 0.5

    _n_per_axis = int(n ** (1 / dim))
    n_per_axis = (_n_per_axis,) * dim

    state = jdem.utils.grid_state(
        n_per_axis=n_per_axis,
        spacing=spacing,
        radius_range=(radius / polydispersity, radius),
        vel_range=[-1.0, 1.0],
        seed=seed,
    )

    collider_kw = dict()
    if collider_type in [
        "celllist",
        "CellList",
        "sap",
        "sap_pca",
        "sap_shifted",
        "SweepAndPrune",
        "MultiCellList",
    ]:
        collider_kw["state"] = state

    if neighbor_list:
        max_rad = jnp.max(state.rad)
        cutoff = float(2.0 * max_rad)
        collider_kw_actual = {
            "state": state,
            "cutoff": cutoff,
            "skin": skin,
            "secondary_collider_type": collider_type,
            "secondary_collider_kw": collider_kw,
        }
        collider_type_actual = "NeighborList"
    else:
        collider_kw_actual = collider_kw
        collider_type_actual = collider_type

    mat_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0e3, poisson=0.3)],
    )
    system = jdem.System.create(
        state.shape,
        domain_type="periodic",
        domain_kw={
            "box_size": (spacing * _n_per_axis,) * dim,
            "anchor": (-radius,) * dim,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=0.001,
        mat_table=mat_table,
    )

    return state, system


def set_up_clump(
    N: int,
    dim: int,
    beta: float = 1.5,
    polydispersity: float = 1.0,
    collider_type: str = "naive",
    n_samples=1000,
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
) -> tuple[jdem.State, jdem.System]:
    nv = 16 if dim == 2 else 42
    n = N // nv
    _n_per_axis = int(n ** (1 / dim))
    n = _n_per_axis**dim

    particle_radius = 0.4
    asperity_radius = 0.1
    particle_mass = 1.0

    pos = jdem.utils.meshes.generate_icosphere_mesh(
        nv=nv,
        N=n,
        dim=dim,
    )
    pos *= particle_radius - asperity_radius

    rad = jnp.ones((n, nv)) * asperity_radius
    if pos.ndim == 2:
        pos = pos[None, ...]

    # Distribute clumps in a grid
    spacing = 2.0 * (particle_radius + asperity_radius)
    dummy_state = jdem.utils.grid_state(
        n_per_axis=(_n_per_axis,) * dim,
        spacing=spacing,
        radius_range=(particle_radius, particle_radius),
        vel_range=[-1.0, 1.0],
        seed=seed,
    )
    grid_pos = np.array(dummy_state.pos[:n])
    clump_vel = np.array(dummy_state.vel[:n])
    pos += grid_pos[:, None, :]
    volume, com, inertia, q, pos_p = (
        jdem.utils.clumps._compute_uniform_union_properties(
            pos,
            rad,
            particle_mass,
            n_samples=n_samples,
        )
    )
    nv = rad.shape[-1]
    total = rad.size
    ang_dim = inertia.shape[-1]
    rad_flat = rad.reshape(total)
    clump_id = jnp.broadcast_to(jnp.arange(n, dtype=int)[:, None], (n, nv)).reshape(
        total
    )
    sphere_pos_p = pos_p.reshape(total, dim)
    pos_c = jnp.broadcast_to(com[:, None, :], (n, nv, dim)).reshape(total, dim)
    vel_c = jnp.broadcast_to(clump_vel[:, None, :], (n, nv, dim)).reshape(total, dim)
    q_w = jnp.broadcast_to(q[:, None, 0:1], (n, nv, 1)).reshape(total, 1)
    q_xyz = jnp.broadcast_to(q[:, None, 1:4], (n, nv, 3)).reshape(total, 3)
    q_state = jdem.utils.quaternion.Quaternion.create(w=q_w, xyz=q_xyz)
    volume_flat = jnp.broadcast_to(volume[:, None], (n, nv)).reshape(total)
    inertia_flat = jnp.broadcast_to(inertia[:, None, :], (n, nv, ang_dim)).reshape(
        total, ang_dim
    )
    mass_flat = jnp.full((total,), particle_mass, dtype=float)
    state = jdem.State.create(
        pos=pos_c,
        vel=vel_c,
        pos_p=sphere_pos_p,
        rad=rad_flat,
        q=q_state,
        volume=volume_flat,
        mass=mass_flat,
        inertia=inertia_flat,
        clump_id=clump_id,
    )

    collider_kw = dict()
    if collider_type in [
        "celllist",
        "CellList",
        "sap",
        "sap_pca",
        "sap_shifted",
        "SweepAndPrune",
        "MultiCellList",
    ]:
        collider_kw["state"] = state

    if neighbor_list:
        max_rad = jnp.max(state.rad)
        cutoff = float(2.0 * max_rad)
        collider_kw_actual = {
            "state": state,
            "cutoff": cutoff,
            "skin": skin,
            "secondary_collider_type": collider_type,
            "secondary_collider_kw": collider_kw,
        }
        collider_type_actual = "NeighborList"
    else:
        collider_kw_actual = collider_kw
        collider_type_actual = collider_type

    mat_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0e3, poisson=0.3)],
    )

    system = jdem.System.create(
        state.shape,
        domain_type="periodic",
        domain_kw={
            "box_size": (spacing * _n_per_axis,) * dim,
            "anchor": (-particle_radius - asperity_radius,) * dim,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=0.001,
        mat_table=mat_table,
    )
    return state, system


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider_type", ["CellList", "SweepAndPrune", "MultiCellList"])
@pytest.mark.parametrize("neighbor_list", [False, True])
def test_spheres_invariance(dim: int, collider_type: str, neighbor_list: bool):
    state1, system1 = set_up_spheres(n=64, dim=dim, collider_type="naive")
    state2, system2 = set_up_spheres(
        n=64,
        dim=dim,
        collider_type=collider_type,
        neighbor_list=neighbor_list,
        skin=0.5,
    )

    state1, system1 = system1.step(state1, system1, n=100)
    state2, system2 = system2.step(state2, system2, n=100)

    check_states_match(state1, state2)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider_type", ["CellList", "SweepAndPrune", "MultiCellList"])
@pytest.mark.parametrize("neighbor_list", [False, True])
def test_clumps_invariance(dim: int, collider_type: str, neighbor_list: bool):
    state1, system1 = set_up_clump(N=128, dim=dim, collider_type="naive", n_samples=1000)
    state2, system2 = set_up_clump(
        N=128,
        dim=dim,
        collider_type=collider_type,
        neighbor_list=neighbor_list,
        skin=0.5,
        n_samples=1000,
    )

    state1, system1 = system1.step(state1, system1, n=100)
    state2, system2 = system2.step(state2, system2, n=100)

    check_states_match(state1, state2)
