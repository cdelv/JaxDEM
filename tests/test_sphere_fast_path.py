# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Compare managed forces and trajectories with the explicit sphere mode."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces.force_manager import ForceManager


def _particle_force(pos, state, system):
    return -0.17 * pos, jnp.full_like(state.torque, 0.09)


def _com_force(pos, state, system):
    return 0.11 + 0.03 * pos, jnp.full_like(state.torque, -0.04)


def _energy(pos, state, system):
    return 0.05 * jnp.sum(pos**2, axis=-1)


def _state(dim, only_spheres=False):
    pos = jnp.array([[0.0, 0.0], [0.85, 0.0], [1.6, 0.2]])
    if dim == 3:
        pos = jnp.pad(pos, ((0, 0), (0, 1)))
    return jd.State.create(
        pos=pos,
        only_spheres=only_spheres,
        clump_id=jnp.array([2, 0, 1]),
        rad=jnp.full(3, 0.5),
        mass=jnp.array([1.0, 1.5, 0.75]),
        vel=0.02 * jnp.arange(3 * dim).reshape(3, dim),
        ang_vel=jnp.full((3, 1 if dim == 2 else 3), 0.2),
        force=0.1 * jnp.arange(3 * dim).reshape(3, dim),
        torque=jnp.full((3, 1 if dim == 2 else 3), 0.13),
    )


def _system(state, collider=""):
    return jd.System.create(
        state=state,
        dt=1e-4,
        collider_type=collider,
        force_manager_kw={
            "gravity": jnp.arange(state.dim, dtype=float) * -0.3 - 0.1,
            "force_functions": (
                (_particle_force, _energy, False),
                (_com_force, None, True),
            ),
        },
    )


def _loads(state, system):
    force = jnp.arange(state.N * state.dim).reshape(state.N, state.dim) * 0.07
    torque = jnp.full_like(state.torque, 0.08)
    system = ForceManager.add_force(state, system, force)
    system = ForceManager.add_force(state, system, force + 0.2, is_com=True)
    system = ForceManager.add_force_at(state, system, force[1], jnp.array([1]))
    system = ForceManager.add_force_at(
        state, system, force[2], jnp.array([2]), is_com=True
    )
    system = ForceManager.add_torque(state, system, torque)
    return ForceManager.add_torque_at(state, system, torque[0], jnp.array([0]))


def _assert_arrays_close(actual, expected):
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("dim", [2, 3])
def test_managed_loads_energy_and_cleared_buffers_match_general_mode(dim):
    states = [_state(dim, mode) for mode in (False, True)]
    systems = [_loads(state, _system(state)) for state in states]
    energies = [
        ForceManager.compute_potential_energy(state, system)
        for state, system in zip(states, systems)
    ]
    outputs = [
        ForceManager.apply(state, system) for state, system in zip(states, systems)
    ]
    _assert_arrays_close(outputs[1], outputs[0])
    np.testing.assert_allclose(energies[1], energies[0], rtol=2e-6)
    for name in ("external_force", "external_force_com", "external_torque"):
        np.testing.assert_array_equal(getattr(outputs[1][1].force_manager, name), 0.0)
    assert not outputs[0][0].only_spheres
    assert outputs[1][0].only_spheres


@pytest.mark.parametrize("dim", [2, 3])
def test_short_contact_trajectory_matches_general_mode(dim):
    outputs = []
    for mode in (False, True):
        state = _state(dim, mode)
        system = _system(state, "naive")
        initial, system = jd.System.initialize(state, system)
        outputs.append(jd.System.trajectory_rollout(initial, system, n=4, stride=3))
    _assert_arrays_close(outputs[1], outputs[0])


@pytest.mark.parametrize("dim", [2, 3])
def test_batched_force_and_reverse_mode_match(dim):
    derivatives = []
    outputs = []
    for mode in (False, True):
        state = _state(dim, mode)
        system = _system(state)
        batch = jd.State.stack([state, state])
        systems = jd.System.stack([system, system])
        outputs.append(jax.jit(jax.vmap(ForceManager.apply))(batch, systems))

        def objective(x):
            trial = replace(state, pos_c=x)
            updated, _ = ForceManager.apply(trial, system)
            return jnp.sum(updated.force**2) + jnp.sum(updated.torque**2)

        derivatives.append(jax.grad(objective)(state.pos_c))
    _assert_arrays_close(outputs[1], outputs[0])
    np.testing.assert_allclose(derivatives[1], derivatives[0], rtol=2e-6, atol=1e-7)


def _primitives(value):
    graph = getattr(value, "jaxpr", value)
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield equation.primitive.name
            for parameter in equation.params.values():
                yield from _primitives(parameter)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _primitives(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _primitives(item)


@pytest.mark.parametrize(
    "method", ["apply", "add_force", "add_torque", "compute_potential_energy"]
)
def test_sphere_managed_forces_omit_body_scatter_reductions(method):
    found = []
    for mode in (False, True):
        state = _state(3, mode)
        system = _system(state)
        function = getattr(ForceManager, method)
        args = (state, system)
        if method == "add_force":
            args += (jnp.ones_like(state.force),)
        elif method == "add_torque":
            args += (jnp.ones_like(state.torque),)
        found.append(
            any(
                "scatter" in name
                for name in _primitives(jax.make_jaxpr(function)(*args))
            )
        )
    assert found == [True, False]


@pytest.mark.parametrize("mode", [False, True])
def test_empty_state_managed_forces(mode):
    state = jd.State.create(dim=2, only_spheres=mode)
    system = jd.System.create(state=state, collider_type="")
    updated, _ = jd.System.initialize(state, system)
    assert updated.force.shape == (0, 2)
    assert updated.torque.shape == (0, 1)
    assert float(ForceManager.compute_potential_energy(updated, system)) == 0.0


@pytest.mark.parametrize("dim", [2, 3])
def test_friction_history_and_sphere_spin_match_general_mode(dim):
    positions = jnp.zeros((2, dim)).at[1, 0].set(0.8)
    angular_shape = (2, 1 if dim == 2 else 3)
    initial = jd.State.create(
        pos=positions,
        rad=jnp.full(2, 0.5),
        vel=jnp.zeros_like(positions).at[0, 1].set(0.2),
        ang_vel=jnp.full(angular_shape, 0.1),
        mass=jnp.ones(2),
    )
    material = jd.Material.create(
        "elasticfrict",
        density=1.0,
        young=100.0,
        poisson=0.0,
        e=0.9,
        mu=0.5,
        mu_r=0.1,
    )
    table = jd.MaterialTable.from_materials([material])
    outputs = []
    for mode in (False, True):
        state = replace(initial, only_spheres=mode)
        system = jd.System.create(
            state=state,
            dt=1e-4,
            mat_table=table,
            force_model=jd.forces.CundallStrackForce(),
            collider_type="NeighborList",
            collider_kw={
                "max_neighbors": 2,
                "cutoff": 1.0,
                "secondary_collider_type": "naive",
            },
        )
        state, system = jd.System.initialize(state, system)
        outputs.append(jd.System.step(state, system, n=20))
    _assert_arrays_close(outputs[1], outputs[0])
    assert np.any(np.asarray(outputs[1][1].collider.history) != 0)
    assert np.any(np.asarray(outputs[1][0].torque) != 0)
    assert not np.allclose(outputs[1][0].ang_vel, initial.ang_vel)


def test_batched_static_neighbor_prefix_runs_with_sphere_mode():
    positions = jnp.zeros((16, 2)).at[:, 0].set(0.9 * jnp.arange(16))
    state = jd.State.create(pos=positions, rad=jnp.full(16, 0.5), only_spheres=True)
    system = jd.System.create(
        state=state,
        dt=1e-4,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 64,
            "cutoff": 1.0,
            "secondary_collider_type": "CellList",
            "secondary_collider_kw": {
                "static": True,
                "cell_size": 1.1,
                "search_range": 1,
            },
        },
    )
    state, system = jd.System.initialize(
        jd.State.stack([state, state]), jd.System.stack([system, system])
    )
    final, system = jd.System.step(state, system, n=2)
    assert final.only_spheres
    assert np.all(np.isfinite(final.pos))
    system.check_overflow()
