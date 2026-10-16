# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check that positions, velocities, and forces agree across all periodic colliders for spheres, clumps, and facets."""

from __future__ import annotations


import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem


def check_states_match(
    state1: jdem.State,
    state2: jdem.State,
    system1: jdem.System,
    system2: jdem.System,
    atol=1e-7,
    rtol=2e-6,
):
    # Colliders visit pairs in different orders, so summation-order rounding
    # differs between them and is amplified by the chaotic dynamics over the
    # course of a run; bit-exact agreement is not achievable even under x64.
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

    pe1 = jdem.utils.compute_potential_energy(state1, system1)
    pe2 = jdem.utils.compute_potential_energy(state2, system2)
    np.testing.assert_allclose(
        pe1, pe2, atol=atol, rtol=rtol, err_msg="potential energy mismatch"
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
    dim,
    collider_type="naive",
    domain_type="periodic",
    neighbor_list=False,
    skin=0.2,
    dt=0.001,
):
    centers = np.array([[0.5, 0.5], [1.05, 0.65], [0.7, 1.15]])
    centers = np.pad(centers, ((0, 0), (0, dim - 2)), constant_values=0.5)
    offsets = np.tile([[-0.2, 0.0], [0.2, 0.0]], (3, 1))
    offsets = np.pad(offsets, ((0, 0), (0, dim - 2)))
    state = jdem.State.create(
        pos=jnp.asarray(np.repeat(centers, 2, axis=0)),
        pos_p=jnp.asarray(offsets),
        rad=jnp.full(6, 0.35),
        mass=jnp.ones(6),
        inertia=jnp.ones((6, 1 if dim == 2 else 3)),
        clump_id=jnp.repeat(jnp.arange(3), 2),
    )
    options = (
        {
            "cutoff": 0.7,
            "skin": skin,
            "max_neighbors": state.N,
            "secondary_collider_type": collider_type,
        }
        if neighbor_list
        else {}
    )
    table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
    )
    system = jdem.System.create(
        state=state,
        dt=dt,
        mat_table=table,
        collider_type="NeighborList" if neighbor_list else collider_type,
        collider_kw=options,
        domain_type=domain_type,
        domain_kw={"box_size": jnp.full(dim, 3.0)},
    )
    return state, system


def set_up_facets_and_spheres(
    dim,
    collider_type="naive",
    seed=42,
    neighbor_list=False,
    skin=0.2,
    domain_type="periodic",
    dt=0.001,
):
    if dim == 2:
        vertices = jnp.array([[0.4, 1.0], [1.4, 1.0]])
        spheres = jnp.array([[0.8, 1.35], [1.5, 1.1], [0.3, 1.1]])
        shift = jnp.array([0.1, 0.15])
    else:
        vertices = jnp.array([[0.4, 0.4, 1.0], [1.4, 0.4, 1.0], [0.4, 1.4, 1.0]])
        spheres = jnp.array([[0.7, 0.7, 1.3], [1.4, 0.7, 1.1], [0.3, 0.3, 1.1]])
        shift = jnp.array([0.1, 0.1, 0.15])
    state = jdem.State.add_facet(
        jdem.State.create(dim=dim),
        vertices,
        thickness=0.1,
        rigid=True,
        mass=1.0,
        species_id=1,
    )
    state = jdem.State.add_facet(
        state, vertices + shift, thickness=0.1, rigid=False, mass=1.0, species_id=1
    )
    state = jdem.State.merge(
        state,
        jdem.State.create(
            pos=spheres, rad=jnp.full(3, 0.25), species_id=jnp.zeros(3, int)
        ),
    )
    law = jdem.ForceRouter.from_dict(
        2,
        {
            (0, 0): jdem.ForceModel.create("spring"),
            (0, 1): jdem.ForceModel.create("sphere_facet_spring"),
            (1, 1): jdem.ForceModel.create("facet_facet_spring"),
        },
    )
    options = (
        {
            "cutoff": 2.0,
            "skin": skin,
            "max_neighbors": state.N,
            "secondary_collider_type": collider_type,
        }
        if neighbor_list
        else {}
    )
    table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, poisson=0.25, density=1.0)]
    )
    system = jdem.System.create(
        state=state,
        dt=dt,
        force_model=law,
        mat_table=table,
        collider_type="NeighborList" if neighbor_list else collider_type,
        collider_kw=options,
        domain_type=domain_type,
        domain_kw={"box_size": jnp.full(dim, 3.0)},
        seed=seed,
    )
    return state, system


@pytest.mark.parametrize(
    "dim,collider_type,neighbor_list,domain_type",
    [
        (2, "CellList", False, "periodic"),
        (3, "MultiCellList", False, "periodic"),
        (2, "MultiCellList", False, "reflect"),
        (3, "CellList", False, "free"),
        (2, "CellList", True, "free"),
        (3, "MultiCellList", True, "reflect"),
        (2, "Naive", True, "periodic"),
        (3, "CellList", True, "periodic"),
    ],
)
def test_spheres_invariance(
    dim: int,
    collider_type: str,
    neighbor_list: bool,
    domain_type: str,
):
    state1, system1 = set_up_spheres(
        n=64,
        dim=dim,
        collider_type="naive",
        polydispersity=2.5,
        domain_type=domain_type,
        neighbor_list=False,
        skin=0.5,
    )
    state2, system2 = set_up_spheres(
        n=64,
        dim=dim,
        collider_type=collider_type,
        polydispersity=2.5,
        domain_type=domain_type,
        neighbor_list=neighbor_list,
        skin=0.5,
    )

    compare_contact_trajectory(state1, system1, state2, system2)


@pytest.mark.parametrize(
    "dim,collider_type,neighbor_list,domain_type",
    [
        (2, "CellList", False, "periodic"),
        (3, "MultiCellList", False, "periodic"),
        (2, "MultiCellList", False, "reflect"),
        (3, "CellList", False, "free"),
        (2, "CellList", True, "free"),
        (3, "MultiCellList", True, "reflect"),
        (2, "Naive", True, "periodic"),
        (3, "CellList", True, "periodic"),
    ],
)
def test_clumps_invariance(
    dim: int, collider_type: str, neighbor_list: bool, domain_type: str
):
    state1, system1 = set_up_clump(
        dim=dim,
        collider_type="naive",
        neighbor_list=False,
        domain_type=domain_type,
        skin=0.5,
    )
    state2, system2 = set_up_clump(
        dim=dim,
        collider_type=collider_type,
        neighbor_list=neighbor_list,
        domain_type=domain_type,
        skin=0.5,
    )

    compare_contact_trajectory(state1, system1, state2, system2)


@pytest.mark.parametrize(
    "dim,collider_type,neighbor_list,domain_type",
    [
        (2, "CellList", False, "periodic"),
        (3, "MultiCellList", False, "periodic"),
        (2, "MultiCellList", False, "reflect"),
        (3, "CellList", False, "free"),
        (2, "CellList", True, "free"),
        (3, "MultiCellList", True, "reflect"),
        (2, "Naive", True, "periodic"),
        (3, "CellList", True, "periodic"),
    ],
)
def test_facets_invariance(
    dim: int, collider_type: str, neighbor_list: bool, domain_type: str
):
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

    compare_contact_trajectory(state1, system1, state2, system2)


def compare_contact_trajectory(state1, system1, state2, system2):
    from dataclasses import replace

    # Start in contact so a short run exercises force/torque assembly immediately.
    state1 = replace(state1, pos_c=0.7 * state1.pos_c)
    state2 = replace(state2, pos_c=0.7 * state2.pos_c)
    if isinstance(system1.domain, jdem.domains.PeriodicDomain):
        # Shift a whole body, including all vertices, so active contacts use images.
        shift = jnp.zeros(state1.dim).at[0].set(system1.domain.box_size[0])
        members = state1.clump_id == state1.clump_id[0]
        state1 = replace(state1, pos_c=state1.pos_c + members[:, None] * shift)
        state2 = replace(state2, pos_c=state2.pos_c + members[:, None] * shift)
    state1, system1 = jdem.System.initialize(state1, system1)
    state2, system2 = jdem.System.initialize(state2, system2)
    assert jnp.any(jnp.abs(state1.force) > 1e-8)
    check_states_match(state1, state2, system1, system2)
    before = np.asarray(state1.pos).copy()
    state1, system1 = system1.step(state1, system1, n=4)
    state2, system2 = system2.step(state2, system2, n=4)
    check_states_match(state1, state2, system1, system2)
    assert int(system1.step_count) == int(system2.step_count) == 4
    np.testing.assert_allclose(system1.time, 4 * system1.dt, rtol=2e-6)
    np.testing.assert_allclose(system2.time, 4 * system2.dt, rtol=2e-6)
    assert np.max(np.abs(np.asarray(state1.pos) - before)) > 1e-8
    assert not system1.search_overflow and not system2.search_overflow
