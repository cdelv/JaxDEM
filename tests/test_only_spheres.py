# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check static only_spheres configuration, rotation, thermal reductions, and force fast paths."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces.force_manager import ForceManager
from jaxdem.utils.linalg import cross_3X3D_1X2D
from jaxdem.utils.quaternion import Quaternion
from jaxdem.utils.thermal import (
    compute_rotational_kinetic_energy_per_particle,
    compute_temperature,
    compute_translational_kinetic_energy_per_particle,
    count_dynamic_dofs,
    scale_to_temperature,
    set_temperature,
)


def _walk_jaxpr(value, *, primitives=False):
    graph = getattr(value, "jaxpr", value)
    if hasattr(graph, "eqns"):
        for eq in graph.eqns:
            if primitives:
                yield eq.primitive.name
            else:
                yield str(eq.params.get("name", ""))
                yield str(eq.source_info.name_stack)
            for param in eq.params.values():
                yield from _walk_jaxpr(param, primitives=primitives)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_jaxpr(item, primitives=primitives)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_jaxpr(item, primitives=primitives)


def _assert_arrays_close(actual, expected):
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-6)


# ---------------------------------------------------------------------------
# Configuration, pytree metadata, state ops, and geometry cache
# ---------------------------------------------------------------------------


def _config_state(dim=3, *, only_spheres=False):
    positions = jnp.array([[0.0, 0.0], [1.4, 0.1], [2.8, -0.1]])
    if dim == 3:
        positions = jnp.pad(positions, ((0, 0), (0, 1)))
    return jd.State.create(
        pos=positions,
        vel=jnp.full_like(positions, 0.03),
        ang_vel=jnp.full((3, 1 if dim == 2 else 3), 0.2),
        rad=jnp.full(3, 0.5),
        mass=jnp.array([1.0, 1.5, 2.0]),
        clump_id=jnp.array([2, 0, 1]),
        only_spheres=only_spheres,
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_default_vs_explicit_sphere_mode_geometry_and_surface_spin(dim):
    general = _config_state(dim, only_spheres=False)
    assert general.only_spheres is False
    offsets = jnp.zeros_like(general.pos_p).at[0, 0].set(-0.3).at[1, 0].set(0.3)
    rotation = Quaternion.from_rotvec(jnp.tile(jnp.array([0.0, 0.0, 0.6]), (3, 1)))
    clump = replace(general, pos_p=offsets, q=rotation, clump_id=jnp.array([0, 0, 1]))
    expected_offsets = Quaternion.rotate(clump.q, clump.pos_p)
    np.testing.assert_allclose(clump.pos, clump.pos_c + expected_offsets)
    offset = jnp.full(dim, 0.07)
    np.testing.assert_allclose(
        clump.velocity_at(0, offset),
        clump.vel[0] + cross_3X3D_1X2D(clump.ang_vel[0], expected_offsets[0] + offset),
        rtol=2e-6,
        atol=2e-8,
    )

    spheres = _config_state(dim, only_spheres=True)
    assert spheres.only_spheres is True
    np.testing.assert_array_equal(spheres.pos, spheres.pos_c)
    np.testing.assert_array_equal(spheres._pos_p_rot, 0.0)
    surf_offset = jnp.arange(dim, dtype=float) * 0.1 + 0.03
    expected_vel = spheres.vel[1] + cross_3X3D_1X2D(spheres.ang_vel[1], surf_offset)
    np.testing.assert_allclose(
        spheres.velocity_at(1, surf_offset), expected_vel, rtol=2e-6, atol=2e-8
    )
    assert not np.allclose(expected_vel, spheres.vel[1])


def test_mode_is_static_pytree_metadata_and_preserves_boolean_masks():
    general = _config_state()
    spheres = replace(general, only_spheres=True)
    assert jax.tree.structure(spheres) != jax.tree.structure(general)
    assert len(jax.tree.leaves(spheres)) == len(jax.tree.leaves(general))
    assert jax.jit(lambda state: state)(spheres).only_spheres is True
    assert replace(spheres).only_spheres is True
    masks = jax.tree.map(lambda _: True, spheres)
    assert masks.only_spheres is True
    assert all(type(leaf) is bool for leaf in jax.tree.leaves(masks))
    axes = jax.tree.map(lambda batched: 0 if batched else None, masks)
    assert axes.only_spheres is True
    assert all(leaf == 0 for leaf in jax.tree.leaves(axes))


def test_traced_factory_keeps_the_explicit_mode():
    @jax.jit
    def construct(pos, ids):
        state = jd.State.create(pos=pos, clump_id=ids, only_spheres=True)
        return jd.System.initialize(
            state, jd.System.create(state=state, dt=1e-3, collider_type="")
        )

    result, _ = construct(_config_state().pos_c, jnp.array([2, 0, 1]))
    assert result.only_spheres is True
    np.testing.assert_array_equal(result.pos, result.pos_c)


def test_position_cache_omits_quaternion_rotation_for_spheres():
    general = _config_state()
    spheres = _config_state(only_spheres=True)

    def replace_orientation(st):
        st = replace(st, q=Quaternion.conj(st.q))
        return st.pos, st._pos_p_rot

    general_names = tuple(_walk_jaxpr(jax.make_jaxpr(replace_orientation)(general)))
    sphere_names = tuple(_walk_jaxpr(jax.make_jaxpr(replace_orientation)(spheres)))
    assert any(
        name == "rotate" or "Quaternion.rotate" in name for name in general_names
    )
    assert not any(
        name == "rotate" or "Quaternion.rotate" in name for name in sphere_names
    )
    position, cache = jax.jit(replace_orientation)(spheres)
    np.testing.assert_array_equal(position, spheres.pos_c)
    np.testing.assert_array_equal(cache, 0.0)


def test_merge_stack_and_builders_track_sphere_mode():
    for first_mode in (True, False):
        for second_mode in (True, False):
            first = _config_state(only_spheres=first_mode)
            second = _config_state(only_spheres=second_mode)
            expected = first_mode and second_mode
            merged = jd.State.merge(first, second)
            assert merged.only_spheres is expected
            assert merged.N == first.N + second.N
            np.testing.assert_allclose(
                merged.pos, jnp.concatenate([first.pos, second.pos])
            )
            stacked = jd.State.stack([first, second])
            assert stacked.only_spheres is expected
            assert all(st.only_spheres is expected for st in jd.State.unstack(stacked))
            assert first.only_spheres is first_mode
            assert second.only_spheres is second_mode

    for mode in (False, True):
        state = _config_state(only_spheres=mode)
        added = jd.State.add(state, pos=jnp.array([[4.2, 0.0, 0.0]]))
        assert added.only_spheres is mode and added.N == state.N + 1
        np.testing.assert_allclose(added.pos[-1], [4.2, 0.0, 0.0])

    spheres = _config_state(only_spheres=True)
    assert (
        jd.State.add_clump(
            spheres, pos=jnp.array([[4.2, 0.0, 0.0], [5.2, 0.0, 0.0]])
        ).only_spheres
        is False
    )
    assert (
        jd.State.add_contact_facets(spheres, jnp.array([[0, 1, 2]])).only_spheres
        is False
    )
    assert spheres.only_spheres is True


# ---------------------------------------------------------------------------
# Rotation integration and thermal statistics
# ---------------------------------------------------------------------------


def _rot_state(dim: int, only_spheres: bool) -> jd.State:
    angular_dim = 1 if dim == 2 else 3
    rotvec = jnp.array(
        [[0.2, -0.1, 0.4], [-0.6, 0.3, 0.1], [0.1, 0.4, 0.7], [0.5, 0.2, -0.2]]
    )
    if dim == 2:
        rotvec = rotvec.at[:, :2].set(0.0)
    return jd.State.create(
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


def _rot_system(state: jd.State, rotation: str) -> jd.System:
    return jd.System.create(
        state=state,
        dt=0.03,
        collider_type=None,
        linear_integrator_type=None,
        rotation_integrator_type=rotation,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("rotation", ["spiral", "verletspiral"])
def test_sphere_rotation_matches_general_path(dim: int, rotation: str) -> None:
    general = _rot_state(dim, False)
    specialized = replace(general, only_spheres=True)
    systems = [_rot_system(general, rotation), _rot_system(specialized, rotation)]
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
        _rot_state(dim, True).ang_vel[2],
        rtol=2e-5,
        atol=2e-6,
    )
    assert not np.allclose(specialized.q.xyz[2], _rot_state(dim, True).q.xyz[2])


def _axial_torque(pos, state, system):
    return jnp.zeros_like(pos), jnp.broadcast_to(
        jnp.array([0.0, 0.0, 2.0]), state.ang_vel.shape
    )


def test_sphere_torque_and_prescribed_spin_keep_physical_orientation() -> None:
    q0 = Quaternion.from_rotvec(jnp.array([[0.6, -0.2, 0.1], [-0.3, 0.5, 0.2]]))
    initial = jd.State.create(
        pos=jnp.zeros((2, 3)),
        q=q0,
        ang_vel=jnp.array([[0.0, 0.0, 0.3], [0.0, 0.0, -0.4]]),
        inertia=jnp.full((2, 3), 0.5),
        fixed=jnp.array([False, True]),
        only_spheres=True,
    )
    system = jd.System.create(
        state=initial,
        dt=0.001,
        collider_type=None,
        linear_integrator_type=None,
        rotation_integrator_type="verletspiral",
        force_manager_kw={"force_functions": [(_axial_torque, None)]},
    )
    state, system = jd.System.initialize(initial, system)
    final, _ = jd.System.step(state, system, n=100)
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


def test_sphere_verlet_rotation_omits_frame_transforms_and_euler_equations() -> None:
    general = _rot_state(3, False)
    specialized = replace(general, only_spheres=True)
    general_system = _rot_system(general, "verletspiral")
    specialized_system = _rot_system(specialized, "verletspiral")
    for hook in ("step_before_force", "step_after_force"):
        method = getattr(general_system.rotation_integrator, hook)
        ordinary = "\n".join(
            _walk_jaxpr(jax.make_jaxpr(method)(general, general_system))
        )
        sphere = "\n".join(
            _walk_jaxpr(jax.make_jaxpr(method)(specialized, specialized_system))
        )
        for call in ("Quaternion.rotate", "spiral.omega_dot"):
            assert call in ordinary
            assert call not in sphere


def test_sphere_thermal_reductions_scaling_and_rescaling_match_general_path() -> None:
    dim = 3
    general = _rot_state(dim, False)
    specialized = replace(general, only_spheres=True)

    # Multi-axis snapshot reductions & DOF counts.
    batch = jd.State.stack(
        [specialized, replace(specialized, vel=2.0 * specialized.vel)]
    )
    trajectory = jax.tree.map(lambda x: jnp.stack([x, x]), batch)
    expected_trans = 0.5 * specialized.mass * jnp.sum(specialized.vel**2, axis=-1)
    expected_rot = 0.5 * jnp.sum(specialized.inertia * specialized.ang_vel**2, axis=-1)
    trans = compute_translational_kinetic_energy_per_particle(trajectory)
    rot = compute_rotational_kinetic_energy_per_particle(trajectory)
    assert trans.shape == (2, 2, 4) and rot.shape == (2, 2, 4)
    np.testing.assert_allclose(trans[0, 0], expected_trans, rtol=2e-6)
    np.testing.assert_allclose(trans[1, 1], 4.0 * expected_trans, rtol=2e-6)
    np.testing.assert_allclose(
        rot, jnp.broadcast_to(expected_rot, (2, 2, 4)), rtol=2e-6
    )
    n_dof, n_v, n_w = count_dynamic_dofs(trajectory, True, True)
    np.testing.assert_array_equal(n_v, jnp.full((2, 2), 2 * dim))
    np.testing.assert_array_equal(n_w, jnp.full((2, 2), 3 * 3))
    np.testing.assert_array_equal(n_dof, n_v + n_w)

    # Temperature scaling and random velocity assignment.
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

    # Verlet rescaling drift removal and fixed-particle preservation.
    outputs = []
    for st in (general, specialized):
        integrator = jd.LinearIntegrator.create(
            "verlet_rescaling", temperature=0.8, can_rotate=True, subtract_drift=True
        )
        system = replace(_rot_system(st, ""), linear_integrator=integrator)
        rescaled, _ = integrator.step_after_force(st, system)
        outputs.append(rescaled)
    gen_res, sph_res = outputs
    np.testing.assert_allclose(sph_res.vel, gen_res.vel, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(sph_res.ang_vel, gen_res.ang_vel, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(sph_res.vel[2], specialized.vel[2])
    np.testing.assert_allclose(sph_res.ang_vel[2], specialized.ang_vel[2])
    free_mass = sph_res.mass * ~sph_res.fixed
    np.testing.assert_allclose(
        jnp.sum(free_mass[:, None] * sph_res.vel, axis=0), 0.0, atol=2e-6
    )
    np.testing.assert_allclose(
        compute_temperature(sph_res, True, True), 0.8, rtol=2e-5
    )


# ---------------------------------------------------------------------------
# Managed forces, trajectories, friction history, and static neighbor prefix
# ---------------------------------------------------------------------------


def _particle_force(pos, state, system):
    return -0.17 * pos, jnp.full_like(state.torque, 0.09)


def _com_force(pos, state, system):
    return 0.11 + 0.03 * pos, jnp.full_like(state.torque, -0.04)


def _energy(pos, state, system):
    return 0.05 * jnp.sum(pos**2, axis=-1)


def _force_state(dim=3, only_spheres=False):
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


def _force_system(state, collider=""):
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


def test_managed_loads_energy_trajectory_and_ad_match_general_mode():
    states = [_force_state(3, mode) for mode in (False, True)]
    systems = [_loads(state, _force_system(state)) for state in states]
    energies = [
        ForceManager.compute_potential_energy(st, sy)
        for st, sy in zip(states, systems)
    ]
    outputs = [ForceManager.apply(st, sy) for st, sy in zip(states, systems)]
    _assert_arrays_close(outputs[1], outputs[0])
    np.testing.assert_allclose(energies[1], energies[0], rtol=2e-6)
    for name in ("external_force", "external_force_com", "external_torque"):
        np.testing.assert_array_equal(getattr(outputs[1][1].force_manager, name), 0.0)
    assert not outputs[0][0].only_spheres and outputs[1][0].only_spheres

    # Short contact trajectory rollout.
    rollouts = []
    for st in states:
        sy = _force_system(st, "naive")
        init_st, sy = jd.System.initialize(st, sy)
        rollouts.append(jd.System.trajectory_rollout(init_st, sy, n=4, stride=3))
    _assert_arrays_close(rollouts[1], rollouts[0])

    # Batched ForceManager.apply and reverse-mode gradients.
    batched_out, grads = [], []
    for st in states:
        sy = _force_system(st)
        batched_out.append(
            jax.jit(jax.vmap(ForceManager.apply))(
                jd.State.stack([st, st]), jd.System.stack([sy, sy])
            )
        )

        def objective(x, st=st, sy=sy):
            updated, _ = ForceManager.apply(replace(st, pos_c=x), sy)
            return jnp.sum(updated.force**2) + jnp.sum(updated.torque**2)

        grads.append(jax.grad(objective)(st.pos_c))
    _assert_arrays_close(batched_out[1], batched_out[0])
    np.testing.assert_allclose(grads[1], grads[0], rtol=2e-6, atol=1e-7)


def test_sphere_managed_forces_omit_body_scatter_reductions():
    for method in ("apply", "add_force", "add_torque", "compute_potential_energy"):
        found = []
        for mode in (False, True):
            state = _force_state(3, mode)
            system = _force_system(state)
            fn = getattr(ForceManager, method)
            args = (state, system)
            if method == "add_force":
                fn = lambda st, sy, f: ForceManager.add_force(st, sy, f, is_com=True)
                args += (jnp.ones_like(state.force),)
            elif method == "add_torque":
                args += (jnp.ones_like(state.torque),)
            found.append(
                any(
                    "scatter" in name
                    for name in _walk_jaxpr(
                        jax.make_jaxpr(fn)(*args), primitives=True
                    )
                )
            )
        assert found == [True, False], method

    for mode in (False, True):
        state = _force_state(3, mode)
        system = _force_system(state)
        has_scatter = any(
            "scatter" in name
            for name in _walk_jaxpr(
                jax.make_jaxpr(ForceManager.add_force)(
                    state, system, jnp.ones_like(state.force)
                ),
                primitives=True,
            )
        )
        assert not has_scatter


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


def test_batched_static_neighbor_prefix_matches_general_mode():
    positions = jnp.zeros((16, 2)).at[:, 0].set(0.9 * jnp.arange(16))
    results = []
    for mode in (False, True):
        state = jd.State.create(pos=positions, rad=jnp.full(16, 0.5), only_spheres=mode)
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
        batched_state, batched_system = jd.System.initialize(
            jd.State.stack([state, state]), jd.System.stack([system, system])
        )
        final, final_system = jd.System.step(batched_state, batched_system, n=2)
        final_system.check_overflow()
        results.append(final)
    general, spheres = results
    assert spheres.only_spheres
    np.testing.assert_allclose(spheres.pos, general.pos, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(spheres.vel, general.vel, rtol=3e-5, atol=3e-6)
    assert float(jnp.max(jnp.abs(spheres.vel))) > 0.0
