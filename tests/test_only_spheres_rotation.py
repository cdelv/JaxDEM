# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check sphere rotation, thermal statistics, and static specialization."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.utils.quaternion import Quaternion
from jaxdem.utils.thermal import (
    compute_rotational_kinetic_energy_per_particle,
    compute_temperature,
    compute_translational_kinetic_energy_per_particle,
    count_dynamic_dofs,
    scale_to_temperature,
    set_temperature,
)


def _sphere_state(dim: int, only_spheres: bool) -> jdem.State:
    angular_dim = 1 if dim == 2 else 3
    rotvec = jnp.array(
        [[0.2, -0.1, 0.4], [-0.6, 0.3, 0.1], [0.1, 0.4, 0.7], [0.5, 0.2, -0.2]]
    )
    if dim == 2:
        rotvec = rotvec.at[:, :2].set(0.0)
    return jdem.State.create(
        pos=jnp.arange(4 * dim, dtype=float).reshape(4, dim),
        vel=jnp.linspace(-0.7, 0.8, 4 * dim).reshape(4, dim),
        mass=jnp.array([0.2, 0.4, 0.1, 0.3]),
        inertia=jnp.broadcast_to(
            jnp.array([0.3, 0.7, 0.2, 0.9])[:, None], (4, angular_dim)
        ),
        q=Quaternion.from_rotvec(rotvec),
        ang_vel=jnp.linspace(-0.5, 0.8, 4 * angular_dim).reshape(4, angular_dim),
        torque=jnp.linspace(0.4, -0.7, 4 * angular_dim).reshape(4, angular_dim),
        fixed=jnp.array([False, False, True, False]),
        clump_id=jnp.array([2, 0, 3, 1]),
        only_spheres=only_spheres,
    )


def _system(state: jdem.State, rotation: str) -> jdem.System:
    return jdem.System.create(
        state=state,
        dt=0.03,
        collider_type=None,
        linear_integrator_type=None,
        rotation_integrator_type=rotation,
    )


def _jaxpr_call_names(closed_jaxpr: object) -> str:
    names: list[str] = []

    def visit(value: object) -> None:
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                names.append(str(equation.source_info.name_stack))
                names.append(str(equation.params.get("name", "")))
                visit(equation.params)
        elif hasattr(value, "jaxpr"):
            visit(value.jaxpr)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(closed_jaxpr)
    return "\n".join(names)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rotation", ["spiral", "verletspiral"])
def test_sphere_rotation_matches_general_path(dim: int, rotation: str) -> None:
    general = _sphere_state(dim, False)
    specialized = dataclasses.replace(general, only_spheres=True)
    systems = [_system(general, rotation), _system(specialized, rotation)]
    states = [general, specialized]
    for _ in range(12):
        for index, system in enumerate(systems):
            state, system = system.rotation_integrator.step_before_force(
                states[index], system
            )
            state, system = system.rotation_integrator.step_after_force(state, system)
            states[index], systems[index] = state, system

    general, specialized = states
    np.testing.assert_allclose(
        specialized.ang_vel, general.ang_vel, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(specialized.q.w, general.q.w, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(specialized.q.xyz, general.q.xyz, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        specialized.ang_vel[2],
        _sphere_state(dim, True).ang_vel[2],
        rtol=2e-5,
        atol=2e-6,
    )
    assert not np.allclose(specialized.q.xyz[2], _sphere_state(dim, True).q.xyz[2])


@pytest.mark.parametrize("rotation", ["spiral", "verletspiral"])
def test_direct_rotation_hooks_follow_state_mode(rotation: str) -> None:
    general = _sphere_state(3, False)
    specialized = dataclasses.replace(general, only_spheres=True)
    system = _system(general, rotation)
    integrator = system.rotation_integrator
    for hook in (integrator.step_before_force, integrator.step_after_force):
        general, _ = hook(general, system)
        specialized, _ = hook(specialized, system)
    assert general.only_spheres is False
    assert specialized.only_spheres is True
    np.testing.assert_allclose(
        specialized.ang_vel, general.ang_vel, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(specialized.q.w, general.q.w, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(specialized.q.xyz, general.q.xyz, rtol=2e-5, atol=2e-6)


def _axial_torque(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> tuple[jax.Array, jax.Array]:
    return jnp.zeros_like(pos), jnp.broadcast_to(
        jnp.array([0.0, 0.0, 2.0]), state.ang_vel.shape
    )


@pytest.mark.parametrize("rotation", ["spiral", "verletspiral"])
def test_sphere_torque_and_prescribed_spin_keep_physical_orientation(
    rotation: str,
) -> None:
    q0 = Quaternion.from_rotvec(jnp.array([[0.6, -0.2, 0.1], [-0.3, 0.5, 0.2]]))
    initial = jdem.State.create(
        pos=jnp.zeros((2, 3)),
        q=q0,
        ang_vel=jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, -0.4]]),
        inertia=jnp.full((2, 3), 0.5),
        fixed=jnp.array([False, True]),
        only_spheres=True,
    )
    system = jdem.System.create(
        state=initial,
        dt=0.001,
        collider_type=None,
        linear_integrator_type=None,
        rotation_integrator_type=rotation,
        force_manager_kw={"force_functions": [(_axial_torque, None)]},
    )
    state, system = jdem.System.initialize(initial, system)
    final, _ = jdem.System.step(state, system, n=100)
    np.testing.assert_allclose(
        final.ang_vel[:, 2], jnp.array([0.7, -0.4]), rtol=2e-5, atol=2e-6
    )
    expected_q = (
        Quaternion.from_rotvec(jnp.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.04]])) @ q0
    )
    np.testing.assert_allclose(final.q.w, expected_q.w, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(final.q.xyz, expected_q.xyz, rtol=2e-5, atol=2e-6)
    expected_ke = 0.25 * jnp.array([0.7**2, 0.4**2])
    np.testing.assert_allclose(
        compute_rotational_kinetic_energy_per_particle(final),
        expected_ke,
        rtol=3e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize("hook", ["step_before_force", "step_after_force"])
def test_sphere_verlet_rotation_omits_frame_transforms_and_euler_equations(
    hook: str,
) -> None:
    general = _sphere_state(3, False)
    specialized = dataclasses.replace(general, only_spheres=True)
    general_system = _system(general, "verletspiral")
    specialized_system = _system(specialized, "verletspiral")
    method = getattr(general_system.rotation_integrator, hook)
    ordinary_jaxpr = _jaxpr_call_names(jax.make_jaxpr(method)(general, general_system))
    sphere_jaxpr = _jaxpr_call_names(
        jax.make_jaxpr(method)(specialized, specialized_system)
    )
    for call in ("Quaternion.rotate", "spiral.omega_dot"):
        assert call in ordinary_jaxpr
        assert call not in sphere_jaxpr


@pytest.mark.parametrize("dim", [2, 3])
def test_sphere_thermal_reductions_keep_snapshot_axes_and_fixed_counts(
    dim: int,
) -> None:
    state = _sphere_state(dim, True)
    batch = jdem.State.stack([state, dataclasses.replace(state, vel=2.0 * state.vel)])
    trajectory = jax.tree.map(lambda x: jnp.stack([x, x]), batch)
    expected_translational = 0.5 * state.mass * jnp.sum(state.vel**2, axis=-1)
    expected_rotational = 0.5 * jnp.sum(state.inertia * state.ang_vel**2, axis=-1)
    translational = compute_translational_kinetic_energy_per_particle(trajectory)
    rotational = compute_rotational_kinetic_energy_per_particle(trajectory)
    assert translational.shape == (2, 2, 4)
    assert rotational.shape == (2, 2, 4)
    np.testing.assert_allclose(translational[0, 0], expected_translational, rtol=2e-6)
    np.testing.assert_allclose(
        translational[1, 1], 4.0 * expected_translational, rtol=2e-6
    )
    np.testing.assert_allclose(
        rotational, jnp.broadcast_to(expected_rotational, (2, 2, 4)), rtol=2e-6
    )
    n_dof, n_v, n_w = count_dynamic_dofs(trajectory, True, True)
    np.testing.assert_array_equal(n_v, jnp.full((2, 2), 2 * dim))
    np.testing.assert_array_equal(n_w, jnp.full((2, 2), 3 * state.ang_vel.shape[-1]))
    np.testing.assert_array_equal(n_dof, n_v + n_w)


@pytest.mark.parametrize("dim", [2, 3])
def test_sphere_temperature_scaling_and_random_assignment_match_general_path(
    dim: int,
) -> None:
    general = _sphere_state(dim, False)
    specialized = dataclasses.replace(general, only_spheres=True)
    for operation in (scale_to_temperature, set_temperature):
        ordinary = operation(general, 0.6, can_rotate=True, subtract_drift=True)
        sphere = operation(specialized, 0.6, can_rotate=True, subtract_drift=True)
        np.testing.assert_allclose(sphere.vel, ordinary.vel, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            sphere.ang_vel, ordinary.ang_vel, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            compute_temperature(sphere, True, True), 0.6, rtol=2e-5
        )


@pytest.mark.parametrize("dim", [2, 3])
def test_sphere_rescaling_keeps_mass_weighted_drift_and_fixed_velocities(
    dim: int,
) -> None:
    outputs = []
    for only_spheres in (False, True):
        state = _sphere_state(dim, only_spheres)
        integrator = jdem.LinearIntegrator.create(
            "verlet_rescaling", temperature=0.8, can_rotate=True, subtract_drift=True
        )
        system = dataclasses.replace(_system(state, ""), linear_integrator=integrator)
        state, _ = integrator.step_after_force(state, system)
        outputs.append(state)
    general, specialized = outputs
    np.testing.assert_allclose(specialized.vel, general.vel, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        specialized.ang_vel, general.ang_vel, rtol=2e-5, atol=2e-6
    )
    np.testing.assert_allclose(specialized.vel[2], _sphere_state(dim, True).vel[2])
    np.testing.assert_allclose(
        specialized.ang_vel[2], _sphere_state(dim, True).ang_vel[2]
    )
    free_mass = specialized.mass * ~specialized.fixed
    np.testing.assert_allclose(
        jnp.sum(free_mass[:, None] * specialized.vel, axis=0), 0.0, atol=2e-6
    )
    np.testing.assert_allclose(
        compute_temperature(specialized, True, True), 0.8, rtol=2e-5
    )
