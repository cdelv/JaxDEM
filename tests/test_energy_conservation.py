# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Dimensionless analytical trajectories constrain integration, forces and energy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.thermal import compute_energy
from tests.custom_forces import harmonic_trap, harmonic_trap_energy


@pytest.mark.parametrize("integrator,order", [("euler", 1), ("verlet", 2)])
@pytest.mark.parametrize("dim", [2, 3])
def test_harmonic_oscillator_trajectory_and_convergence(integrator, order, dim):
    # m=k=1 sets the time unit; only omega*dt matters.
    x0 = np.arange(1, dim + 1, dtype=float) / dim
    v0 = np.full(dim, 0.2)
    initial_energy = 0.5 * np.sum(x0**2 + v0**2)
    errors = []
    for steps in [16, 32, 64]:
        state = jd.State.create(
            pos=jnp.asarray(x0[None]), vel=jnp.asarray(v0[None]), mass=jnp.ones(1)
        )
        system = jd.System.create(
            state=state,
            dt=1 / steps,
            collider_type="",
            rotation_integrator_type="",
            linear_integrator_type=integrator,
            force_manager_kw={
                "force_functions": [(harmonic_trap, harmonic_trap_energy)]
            },
        )
        state, system = jd.System.initialize(state, system)
        _, _, (trajectory, systems) = jd.System.trajectory_rollout(
            state, system, n=steps
        )
        times = np.arange(1, steps + 1)[:, None] / steps
        exact_x = x0 * np.cos(times) + v0 * np.sin(times)
        exact_v = -x0 * np.sin(times) + v0 * np.cos(times)
        x, v = np.asarray(trajectory.pos[:, 0]), np.asarray(trajectory.vel[:, 0])
        error = np.max(
            np.sqrt(np.sum((x - exact_x) ** 2 + (v - exact_v) ** 2, axis=-1))
        )
        errors.append(error)
        independent_energy = 0.5 * np.sum(x * x + v * v, axis=-1)
        np.testing.assert_allclose(
            jax.vmap(compute_energy)(trajectory, systems),
            independent_energy,
            rtol=2e-6,
            atol=1e-7,
        )
        assert (
            np.max(np.abs(independent_energy - initial_energy)) / initial_energy
            < 2 / steps**order
        )
        assert int(systems.step_count[-1]) == steps
        np.testing.assert_allclose(systems.time[-1], 1.0, rtol=2e-6)
    # Two successive refinements must exhibit the stated order, not just a small energy.
    np.testing.assert_allclose(
        np.log2(np.asarray(errors[:-1]) / errors[1:]), order, atol=0.15
    )


@pytest.mark.parametrize(
    "dim,backend,domain,diameter,masses,stiffness",
    [
        (2, "naive", "free", 1.0, [1.0, 1.0], 1.0),
        (2, "NeighborList", "periodic", 2.0, [1.0, 4.0], 3.0),
        (3, "CellList", "periodic", 0.7, [2.0, 3.0], 0.5),
        (3, "MultiCellList", "leesedwards", 1.2, [5.0, 1.0], 2.0),
    ],
)
def test_elastic_collision_matches_reduced_mass_solution(
    dim, backend, domain, diameter, masses, stiffness
):
    # Fixed omega*dt and incoming_speed/(diameter*omega), with unequal masses.
    masses = np.asarray(masses)
    reduced_mass = np.prod(masses) / masses.sum()
    omega = np.sqrt(stiffness / reduced_mass)
    incoming_speed = 0.2 * diameter * omega
    weights = np.array([-masses[1], masses[0]]) / masses.sum()
    direction = np.array([3.0, 4.0]) if dim == 2 else np.array([2.0, -1.0, 2.0])
    direction /= np.linalg.norm(direction)
    center = np.full(dim, 3.0 * diameter)
    state = jd.State.create(
        pos=jnp.asarray(center + diameter * weights[:, None] * direction),
        vel=jnp.asarray(-incoming_speed * weights[:, None] * direction),
        rad=jnp.asarray(diameter * np.array([0.4, 0.6])),
        mass=jnp.asarray(masses),
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("elastic", young=stiffness, poisson=0.25, density=1.0)]
    )
    contact_time = np.pi / omega
    dt = contact_time / 128
    system = jd.System.create(
        state=state,
        dt=dt,
        mat_table=table,
        collider_type=backend,
        collider_kw={
            "cutoff": diameter,
            "skin": 0.05 * diameter,
            "max_neighbors": 1,
            "secondary_collider_type": "CellList",
        }
        if backend == "NeighborList"
        else {},
        domain_type=domain,
        domain_kw={"box_size": jnp.full(dim, 6.0 * diameter)}
        if domain != "free"
        else {},
        rotation_integrator_type="",
        linear_integrator_type="verlet",
    )
    state, system = jd.System.initialize(state, system)
    _, _, (trajectory, systems) = jd.System.trajectory_rollout(state, system, n=144)
    times = dt * np.arange(1, 145)
    overlap = np.where(
        times <= contact_time,
        incoming_speed / omega * np.sin(omega * times),
        -incoming_speed * (times - contact_time),
    )
    speed = np.where(
        times <= contact_time, -incoming_speed * np.cos(omega * times), incoming_speed
    )
    expected_x = (
        center + (diameter - overlap)[:, None, None] * weights[:, None] * direction
    )
    expected_v = speed[:, None, None] * weights[:, None] * direction
    bound = 2 * (omega * dt) ** 2
    assert (
        np.max(np.abs(np.asarray(trajectory.pos) - expected_x))
        < bound * incoming_speed / omega
    )
    assert (
        np.max(np.abs(np.asarray(trajectory.vel) - expected_v)) < bound * incoming_speed
    )
    np.testing.assert_allclose(
        (np.asarray(trajectory.vel) * masses[None, :, None]).sum(axis=1),
        0.0,
        atol=2e-6 * reduced_mass * incoming_speed,
    )
    np.testing.assert_allclose(
        np.average(np.asarray(trajectory.pos), axis=1, weights=masses),
        np.broadcast_to(center, (144, dim)),
        atol=2e-6 * diameter,
    )
    energy = np.asarray(jax.vmap(compute_energy)(trajectory, systems))
    initial_energy = 0.5 * reduced_mass * incoming_speed**2
    assert np.max(np.abs(energy - initial_energy)) / initial_energy < (omega * dt) ** 2
    assert not np.any(systems.search_overflow)


@pytest.mark.parametrize("dim", [2, 3])
def test_off_center_clump_impact_conserves_energy_and_angular_momentum(dim):
    centers = np.array([[0.0, 0.0], [0.0, 0.0], [0.75, 0.25]])
    offsets = np.array([[-0.3, 0.0], [0.3, 0.0], [0.0, 0.0]])
    velocities = np.array([[0.1, 0.2], [0.1, 0.2], [-0.2, -0.4]])
    if dim == 3:
        centers = np.column_stack((centers, [0.0, 0.0, 0.1]))
        offsets = np.column_stack((offsets, np.zeros(3)))
        velocities = np.column_stack((velocities, [0.05, 0.05, -0.1]))
    spin = np.array([[0.3], [0.3], [-0.1]])
    if dim == 3:
        spin = spin * np.array([[0.3, -0.2, 1.0]])
    moments = np.array([0.5, 0.5, 0.1])[:, None] * np.ones((1, spin.shape[-1]))
    state = jd.State.create(
        pos=jnp.asarray(centers),
        pos_p=jnp.asarray(offsets),
        vel=jnp.asarray(velocities),
        rad=jnp.full(3, 0.4),
        mass=jnp.array([2.0, 2.0, 1.0]),
        inertia=jnp.asarray(moments),
        ang_vel=jnp.asarray(spin),
        clump_id=jnp.array([0, 0, 1]),
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("elastic", young=1.0, density=1.0, poisson=0.25)]
    )
    system = jd.System.create(
        state=state,
        dt=0.002,
        mat_table=table,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
    )
    state, system = jd.System.initialize(state, system)
    assert np.linalg.norm(state.torque[0]) > 0.01
    _, _, (trajectory, systems) = jd.System.trajectory_rollout(state, system, n=64)

    def invariants(snapshot):
        position, velocity, angular = map(
            np.asarray, (snapshot.pos_c, snapshot.vel, snapshot.ang_vel)
        )
        # One representative per body; stored mass and inertia belong to the whole body.
        ids, mass, inertia = [0, 2], np.array([2.0, 1.0]), np.array([0.5, 0.1])
        x, v, w = position[..., ids, :], velocity[..., ids, :], angular[..., ids, :]
        kinetic = 0.5 * np.sum(
            mass * np.sum(v * v, axis=-1) + inertia * np.sum(w * w, axis=-1), axis=-1
        )
        pair_distance = np.linalg.norm(
            np.asarray(snapshot.pos)[..., :2, :]
            - np.asarray(snapshot.pos)[..., 2:3, :],
            axis=-1,
        )
        energy = kinetic + 0.5 * np.sum(
            np.maximum(0.8 - pair_distance, 0.0) ** 2, axis=-1
        )
        orbital = (
            x[..., 0] * v[..., 1] - x[..., 1] * v[..., 0]
            if dim == 2
            else np.cross(x, v)
        )
        momentum = np.sum(mass[:, None] * v, axis=-2)
        angular_momentum = (
            np.sum(mass * orbital + inertia * w[..., 0], axis=-1)
            if dim == 2
            else np.sum(mass[:, None] * orbital + inertia[:, None] * w, axis=-2)
        )
        return energy, momentum, angular_momentum

    energy0, momentum0, angular0 = invariants(state)
    energy, momentum, angular = invariants(trajectory)
    np.testing.assert_allclose(energy, energy0, rtol=2e-5)
    np.testing.assert_allclose(
        momentum, np.broadcast_to(momentum0, momentum.shape), atol=2e-7
    )
    np.testing.assert_allclose(
        angular, np.broadcast_to(angular0, angular.shape), atol=1e-5
    )
    np.testing.assert_allclose(
        jax.vmap(compute_energy)(trajectory, systems), energy, atol=2e-7
    )
    assert np.linalg.norm(np.asarray(trajectory.ang_vel[-1, 0]) - spin[0]) > 1e-3
    np.testing.assert_allclose(
        np.linalg.norm(
            np.asarray(trajectory.pos[:, 1] - trajectory.pos[:, 0]), axis=-1
        ),
        0.6,
        atol=2e-6,
    )


def test_sphere_plane_contact_matches_quarter_period_oscillator():
    # Fixed flat facet, m=k=1, initial overlap .1 and zero velocity.
    vertices = jnp.array([[-2.0, -2.0, 0.0], [4.0, -2.0, 0.0], [-2.0, 4.0, 0.0]])
    state = jd.State.add_facet(
        jd.State.create(dim=3),
        vertices,
        thickness=0.1,
        mass=3.0,
        fixed=True,
        species_id=1,
    )
    state = jd.State.merge(
        state,
        jd.State.create(
            pos=jnp.array([[0.0, 0.0, 0.5]]),
            rad=jnp.array([0.5]),
            species_id=jnp.array([0]),
        ),
    )
    law = jd.ForceRouter.from_dict(
        2, {(0, 1): jd.ForceModel.create("sphere_facet_spring")}
    )
    table = jd.MaterialTable.from_materials(
        [jd.Material.create("elastic", young=1.0, density=1.0, poisson=0.25)]
    )
    dt = np.pi / 256
    system = jd.System.create(
        state=state,
        force_model=law,
        mat_table=table,
        dt=dt,
        linear_integrator_type="verlet",
    )
    state, system = jd.System.initialize(state, system)
    _, _, (trajectory, _) = jd.System.trajectory_rollout(state, system, n=128)
    times = dt * np.arange(1, 129)
    np.testing.assert_allclose(
        trajectory.pos[:, -1, 2], 0.6 - 0.1 * np.cos(times), atol=3e-6
    )
    np.testing.assert_allclose(trajectory.vel[:, -1, 2], 0.1 * np.sin(times), atol=3e-6)
    np.testing.assert_allclose(
        trajectory.pos[:, :3], np.broadcast_to(vertices, (128, 3, 3)), atol=2e-6
    )
