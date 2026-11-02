# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check the static sphere flag and the geometry paths it selects."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils.linalg import cross_3X3D_1X2D
from jaxdem.utils.quaternion import Quaternion


def _state(dim=3, *, only_spheres=False):
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


def _system(state, **kwargs):
    return jd.System.create(state=state, dt=1e-3, collider_type="", **kwargs)


def _jaxpr_names(value):
    graph = getattr(value, "jaxpr", value)
    if hasattr(graph, "eqns"):
        for equation in graph.eqns:
            yield str(equation.params.get("name", ""))
            yield str(equation.source_info.name_stack)
            for parameter in equation.params.values():
                yield from _jaxpr_names(parameter)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _jaxpr_names(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _jaxpr_names(item)


@pytest.mark.parametrize("dim", [2, 3])
def test_default_mode_keeps_rigid_body_geometry(dim):
    state = _state(dim)
    offsets = jnp.zeros_like(state.pos_p).at[0, 0].set(-0.3).at[1, 0].set(0.3)
    rotation = Quaternion.from_rotvec(jnp.tile(jnp.array([0.0, 0.0, 0.6]), (3, 1)))
    state = replace(state, pos_p=offsets, q=rotation, clump_id=jnp.array([0, 0, 1]))
    assert state.only_spheres is False
    expected_offsets = Quaternion.rotate(state.q, state.pos_p)
    np.testing.assert_allclose(state.pos, state.pos_c + expected_offsets)
    offset = jnp.full(dim, 0.07)
    expected = state.vel[0] + cross_3X3D_1X2D(
        state.ang_vel[0], expected_offsets[0] + offset
    )
    np.testing.assert_allclose(
        state.velocity_at(0, offset), expected, rtol=2e-6, atol=2e-8
    )


def test_unique_ids_do_not_enable_sphere_mode_automatically():
    state = jd.State.create(pos=jnp.zeros((3, 3)))
    assert state.only_spheres is False


@pytest.mark.parametrize("dim", [2, 3])
def test_explicit_sphere_mode_uses_centers_and_keeps_surface_spin(dim):
    state = _state(dim, only_spheres=True)
    assert state.only_spheres is True
    np.testing.assert_array_equal(state.pos, state.pos_c)
    np.testing.assert_array_equal(state._pos_p_rot, 0.0)
    offset = jnp.arange(dim, dtype=float) * 0.1 + 0.03
    expected = state.vel[1] + cross_3X3D_1X2D(state.ang_vel[1], offset)
    np.testing.assert_allclose(
        state.velocity_at(1, offset), expected, rtol=2e-6, atol=2e-8
    )
    assert not np.allclose(expected, state.vel[1])


def test_mode_is_static_pytree_metadata():
    general = _state()
    spheres = replace(general, only_spheres=True)
    assert general.only_spheres is False
    assert spheres.only_spheres is True
    assert jax.tree.structure(spheres) != jax.tree.structure(general)
    assert len(jax.tree.leaves(spheres)) == len(jax.tree.leaves(general))
    assert jax.jit(lambda state: state)(spheres).only_spheres is True
    assert replace(spheres).only_spheres is True
    assert jax.tree.map(lambda leaf: leaf, spheres).only_spheres is True


@pytest.mark.parametrize("only_spheres", [False, True])
def test_mode_survives_jit_vmap_and_scan(only_spheres):
    state = _state(only_spheres=only_spheres)
    system = _system(state)
    result, _ = jax.jit(lambda st, sy: jd.System.step(st, sy, n=2))(state, system)
    assert result.only_spheres is only_spheres
    batch = jd.State.stack([state, state])
    systems = jd.System.stack([system, system])
    result, _ = jax.jit(jax.vmap(lambda st, sy: jd.System.step(st, sy, n=2)))(
        batch, systems
    )
    assert result.only_spheres is only_spheres

    def scan_step(carry, _):
        moved = jd.System.step(*carry, n=1)
        return moved, moved[0].pos

    (result, _), positions = jax.jit(
        lambda st, sy: jax.lax.scan(scan_step, (st, sy), None, length=3)
    )(state, system)
    assert result.only_spheres is only_spheres
    np.testing.assert_allclose(positions[-1], result.pos)
    assert state.only_spheres is only_spheres


@pytest.mark.parametrize(
    "entry",
    [
        "initialize",
        "evaluate_forces",
        "step_dynamic",
        "trajectory",
        "minimize",
        "exported_minimize",
    ],
)
def test_system_entry_points_preserve_the_state_mode(entry):
    state = _state(only_spheres=True)
    system = _system(state, minimizer=jd.fire)
    if entry == "trajectory":
        result, _, saved = jd.System.trajectory_rollout(state, system, n=2)
        assert saved[0].only_spheres is True
    elif entry in ("minimize", "exported_minimize"):
        minimize = jd.System.minimize if entry == "minimize" else jd.minimize
        result = minimize(state, system, max_steps=0).state
    elif entry == "step_dynamic":
        result, _ = jd.System.step_dynamic(state, system, n=2)
    else:
        result, _ = getattr(jd.System, entry)(state, system)
    assert result.only_spheres is True
    assert state.only_spheres is True


def test_traced_factory_keeps_the_explicit_mode():
    @jax.jit
    def construct(pos, ids):
        state = jd.State.create(pos=pos, clump_id=ids, only_spheres=True)
        return jd.System.initialize(state, _system(state))

    result, _ = construct(_state().pos_c, jnp.array([2, 0, 1]))
    assert result.only_spheres is True
    np.testing.assert_array_equal(result.pos, result.pos_c)


def test_orientation_assignment_skips_cache_refresh_for_spheres(monkeypatch):
    state = _state(only_spheres=True)
    calls = []
    refresh = jd.State.refresh_rotation_cache
    cache = state._pos_p_rot

    def tracked_refresh(st):
        calls.append(st)
        return refresh(st)

    monkeypatch.setattr(jd.State, "refresh_rotation_cache", tracked_refresh)
    state.q = Quaternion.conj(state.q)
    state.pos_p = jnp.zeros_like(state.pos_p)
    assert calls == []
    assert refresh(state) is state
    assert state._pos_p_rot is cache
    assert state.only_spheres is True
    np.testing.assert_array_equal(state.pos, state.pos_c)


def test_position_cache_omits_quaternion_rotation_for_spheres():
    general = _state()
    spheres = _state(only_spheres=True)

    def replace_orientation(st):
        st = replace(st, q=Quaternion.conj(st.q))
        return st.pos, st._pos_p_rot

    general_names = tuple(_jaxpr_names(jax.make_jaxpr(replace_orientation)(general)))
    sphere_names = tuple(_jaxpr_names(jax.make_jaxpr(replace_orientation)(spheres)))
    assert any(
        name == "rotate" or "Quaternion.rotate" in name for name in general_names
    )
    assert not any(
        name == "rotate" or "Quaternion.rotate" in name for name in sphere_names
    )
    position, cache = jax.jit(replace_orientation)(spheres)
    np.testing.assert_array_equal(position, spheres.pos_c)
    np.testing.assert_array_equal(cache, 0.0)


@pytest.mark.parametrize("only_spheres", [False, True])
def test_geometry_cache_preserves_boolean_pytree_masks(only_spheres):
    state = _state(only_spheres=only_spheres)
    masks = jax.tree.map(lambda _: True, state)
    assert masks.only_spheres is only_spheres
    assert all(type(leaf) is bool for leaf in jax.tree.leaves(masks))
    axes = jax.tree.map(lambda batched: 0 if batched else None, masks)
    assert axes.only_spheres is only_spheres
    assert all(leaf == 0 for leaf in jax.tree.leaves(axes))


@pytest.mark.parametrize(
    "first_mode,second_mode",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_merge_preserves_sphere_mode_only_for_all_sphere_inputs(
    first_mode, second_mode
):
    first = _state(only_spheres=first_mode)
    second = _state(only_spheres=second_mode)
    combined = jd.State.merge(first, second)
    assert combined.only_spheres is (first_mode and second_mode)
    assert combined.N == first.N + second.N
    np.testing.assert_allclose(combined.pos, jnp.concatenate([first.pos, second.pos]))
    assert first.only_spheres is first_mode
    assert second.only_spheres is second_mode


@pytest.mark.parametrize(
    "first_mode,second_mode",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_stack_normalizes_mixed_modes_without_changing_inputs(first_mode, second_mode):
    first = _state(only_spheres=first_mode)
    second = _state(only_spheres=second_mode)
    combined = jd.State.stack([first, second])
    expected_mode = first_mode and second_mode
    assert combined.only_spheres is expected_mode
    np.testing.assert_allclose(combined.pos, jnp.stack([first.pos, second.pos]))
    assert all(
        state.only_spheres is expected_mode for state in jd.State.unstack(combined)
    )
    assert first.only_spheres is first_mode
    assert second.only_spheres is second_mode


@pytest.mark.parametrize("only_spheres", [False, True])
def test_plain_sphere_additions_preserve_the_input_mode(only_spheres):
    state = _state(only_spheres=only_spheres)
    added = jd.State.add(state, pos=jnp.array([[4.2, 0.0, 0.0]]))
    assert added.only_spheres is only_spheres
    assert state.only_spheres is only_spheres
    assert added.N == state.N + 1
    np.testing.assert_allclose(added.pos[:-1], state.pos)
    np.testing.assert_allclose(added.pos[-1], [4.2, 0.0, 0.0])


@pytest.mark.parametrize("operation", ["clump", "facets"])
def test_rigid_and_facet_builders_select_general_mode(operation):
    state = _state(only_spheres=True)
    if operation == "clump":
        result = jd.State.add_clump(
            state, pos=jnp.array([[4.2, 0.0, 0.0], [5.2, 0.0, 0.0]])
        )
    else:
        result = jd.State.add_contact_facets(state, jnp.array([[0, 1, 2]]))
    assert result.only_spheres is False
    assert state.only_spheres is True
