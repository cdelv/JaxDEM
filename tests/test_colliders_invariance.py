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


def check_states_match(state1: jdem.State, state2: jdem.State, atol=5e-2, rtol=5e-2):
    # Colliders visit pairs in different orders, so summation-order rounding
    # differs between them and is amplified by the chaotic dynamics over the
    # course of a run; bit-exact agreement is not achievable even under x64.
    state1 = sort_state_by_unique_id(state1)
    state2 = sort_state_by_unique_id(state2)
    fields = {
        "pos": (state1.pos, state2.pos),
        "vel": (state1.vel, state2.vel),
        "force": (state1.force, state2.force),
        "q.w": (state1.q.w, state2.q.w),
        "q.xyz": (state1.q.xyz, state2.q.xyz),
        "ang_vel": (state1.ang_vel, state2.ang_vel),
        "torque": (state1.torque, state2.torque),
    }
    for name, (a, b) in fields.items():
        np.testing.assert_allclose(
            a, b, atol=atol, rtol=rtol, err_msg=f"{name} mismatch"
        )


def set_up_spheres(
    dim: int,
    n: int = 64,
    polydispersity: float = 1.0,
    domain_type: str = "periodic",
    collider_type: str = "naive",
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
    dt: float = 0.001,
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
    if collider_type.lower() in ["celllist", "multicelllist"]:
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
        domain_type=domain_type,
        domain_kw={
            "box_size": (spacing * _n_per_axis,) * dim,
            "anchor": (-radius,) * dim,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=dt,
        mat_table=mat_table,
    )

    return state, system


def set_up_clump(
    dim: int,
    N: int = 350,
    beta: float = 1.5,
    polydispersity: float = 1.0,
    collider_type: str = "naive",
    domain_type: str = "periodic",
    n_samples=10000,
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
    dt: float = 0.001,
) -> tuple[jdem.State, jdem.System]:
    if dim == 3:
        N = int(jnp.ceil(N * 1.5))

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
    if collider_type.lower() in ["celllist", "multicelllist"]:
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
        domain_type=domain_type,
        domain_kw={
            "box_size": (spacing * _n_per_axis,) * dim,
            "anchor": (-particle_radius - asperity_radius,) * dim,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=dt,
        mat_table=mat_table,
    )
    return state, system


def set_up_facets_and_spheres(
    dim: int,
    collider_type: str = "naive",
    seed: int = 42,
    neighbor_list: bool = False,
    skin: float = 0.5,
    domain_type: str = "periodic",
    dt: float = 0.001,
) -> tuple[jdem.State, jdem.System]:
    np.random.seed(seed)

    # 1. Create empty state
    state = jdem.State.create()

    # 2. Generate a single large grid to avoid overlaps
    n_facets = 50
    n_spheres = 100
    n_total = n_facets + n_spheres

    grid = jdem.utils.grid_state(
        n_per_axis=(6, 6, 6) if dim == 3 else (13, 13),
        spacing=0.9,
        radius_range=(1.0, 1.0),
        vel_range=[-1.0, 1.0],
        seed=seed,
    )

    L = 0.38
    for i in range(n_facets):
        com = np.array(grid.pos[i])
        vel = jnp.array([grid.vel[i]])

        if dim == 3:
            v1 = com + np.array([L, -L / 2, 0.0])
            v2 = com + np.array([-L, -L / 2, 0.0])
            v3 = com + np.array([0.0, L, 0.0])
            vertices = jnp.stack([v1, v2, v3])
            ang_vel = vel / 2.0
        else:
            v1 = com + np.array([L, -L / 2])
            v2 = com + np.array([-L, -L / 2])
            vertices = jnp.stack([v1, v2])
            ang_vel = jnp.array([vel[0, 0] / 2.0])

        state = jdem.State.add_facet(
            state,
            vertices,
            vel=vel,
            ang_vel=ang_vel,
            mass=jnp.array([1.0]),
            species_id=jnp.array([1]),
            thickness=0.1,
        )

    # 3. Add spheres safely from the remainder of the grid
    pos_s = np.array(grid.pos[n_facets:n_total])
    rad_s = jnp.ones(n_spheres) * 0.2
    mass_s = jnp.ones(n_spheres) * 0.5
    vel_s = np.array(grid.vel[n_facets:n_total])
    species_ids_s = jnp.zeros(n_spheres, dtype=int)

    sphere_state = jdem.State.create(
        pos=pos_s, rad=rad_s, mass=mass_s, vel=vel_s, species_id=species_ids_s
    )
    state = jdem.State.merge(state, sphere_state)

    collider_kw = dict()
    if collider_type.lower() in ["celllist", "multicelllist"]:
        collider_kw["state"] = state

    if neighbor_list:
        max_rad = jnp.max(state.rad)
        # Facet contact pairs are keyed on the facet's *primary vertex* (see
        # the warning on SphereFacetSpringForce): the neighbor cutoff must
        # cover the largest primary-vertex-to-contact-point distance, which
        # for a facet-facet contact is two facet diameters plus thicknesses.
        facet_diameter = 2.0 * L
        cutoff = float(2.0 * facet_diameter + 2.0 * max_rad)
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

    mat = jdem.Material.create(
        "elasticfrict", young=1e3, poisson=0.3, density=2000.0, mu=0.0, e=1.0, mu_r=0.0
    )
    mat_table = jdem.MaterialTable.from_materials([mat])

    # Force router to handle sphere-sphere, sphere-facet, facet-facet
    router = jdem.ForceRouter.from_dict(
        S=2,
        mapping={
            (0, 0): jdem.ForceModel.create("spring"),
            (1, 0): jdem.ForceModel.create("sphere_facet_spring"),
            (1, 1): jdem.ForceModel.create("facet_facet_spring"),
        },
    )

    pos_max = np.max(np.asarray(state.pos), axis=0)
    pos_min = np.min(np.asarray(state.pos), axis=0)
    box_size = tuple(float(x) for x in (pos_max - pos_min + 1.0))
    anchor = tuple(float(x) for x in (pos_min - 0.4))

    system = jdem.System.create(
        state.shape,
        domain_type=domain_type,
        domain_kw={
            "box_size": box_size,
            "anchor": anchor,
        },
        collider_type=collider_type_actual,
        collider_kw=collider_kw_actual,
        dt=dt,
        mat_table=mat_table,
        force_model_type="forcerouter",
        force_model_kw={"table": router.table},
    )

    return state, system


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider_type", ["Naive", "CellList", "MultiCellList"])
@pytest.mark.parametrize("neighbor_list", [False, True])
@pytest.mark.parametrize("polydispersity", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("domain_type", ["periodic", "reflect", "free"])
def test_spheres_invariance(
    dim: int,
    collider_type: str,
    neighbor_list: bool,
    polydispersity: float,
    domain_type: str,
):
    if collider_type.lower() == "naive" and not neighbor_list:
        pytest.skip("Redundant: testing naive against itself")

    state1, system1 = set_up_spheres(
        n=64,
        dim=dim,
        collider_type="naive",
        polydispersity=polydispersity,
        domain_type=domain_type,
        neighbor_list=False,
        skin=0.5,
    )
    state2, system2 = set_up_spheres(
        n=64,
        dim=dim,
        collider_type=collider_type,
        polydispersity=polydispersity,
        domain_type=domain_type,
        neighbor_list=neighbor_list,
        skin=0.5,
    )

    state1, system1 = system1.step(state1, system1, n=200)
    state2, system2 = system2.step(state2, system2, n=200)

    check_states_match(state1, state2)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider_type", ["Naive", "CellList", "MultiCellList"])
@pytest.mark.parametrize("neighbor_list", [False, True])
@pytest.mark.parametrize("domain_type", ["periodic", "reflect", "free"])
def test_clumps_invariance(
    dim: int, collider_type: str, neighbor_list: bool, domain_type: str
):
    if collider_type.lower() == "naive" and not neighbor_list:
        pytest.skip("Redundant: testing naive against itself")

    state1, system1 = set_up_clump(
        N=128,
        dim=dim,
        collider_type="naive",
        neighbor_list=False,
        domain_type=domain_type,
        skin=0.5,
        n_samples=1000,
    )
    state2, system2 = set_up_clump(
        N=128,
        dim=dim,
        collider_type=collider_type,
        neighbor_list=neighbor_list,
        domain_type=domain_type,
        skin=0.5,
        n_samples=1000,
    )

    state1, system1 = system1.step(state1, system1, n=200)
    state2, system2 = system2.step(state2, system2, n=200)

    check_states_match(state1, state2)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider_type", ["Naive", "CellList", "MultiCellList"])
@pytest.mark.parametrize("neighbor_list", [False, True])
@pytest.mark.parametrize("domain_type", ["periodic", "reflect", "free"])
def test_facets_invariance(
    dim: int, collider_type: str, neighbor_list: bool, domain_type: str
):
    if collider_type.lower() == "naive" and not neighbor_list:
        pytest.skip("Redundant: testing naive against itself")

    state1, system1 = set_up_facets_and_spheres(
        dim=dim,
        collider_type="naive",
        neighbor_list=False,
        domain_type=domain_type,
        seed=42,
    )
    state2, system2 = set_up_facets_and_spheres(
        dim=dim,
        collider_type=collider_type,
        neighbor_list=neighbor_list,
        domain_type=domain_type,
        seed=42,
    )

    state1, system1 = system1.step(state1, system1, n=200)
    state2, system2 = system2.step(state2, system2, n=200)

    check_states_match(state1, state2)
