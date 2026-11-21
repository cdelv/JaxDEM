# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Mechanical convergence, overflow guards, and rigid-body minimization."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jaxdem as jd
from jaxdem.minimizers.routines import (
    COLLIDER_OVERFLOW,
    CONVERGED,
    MAX_STEPS,
    NONFINITE,
    TerminationReason,
    convergence_info,
    minimize,
)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _NanEnergySpring(jd.forces.SpringForce):
    @staticmethod
    def energy(i, j, pos, state, system, history):
        return jnp.where(i != j, jnp.nan, 0.0)


@pytest.mark.parametrize("dim", [2, 3])
def test_minimize_known_rigid_pose(dim):
    # A nonsymmetric body has a unique target pose; its exact minimum is zero.
    body = np.array([[1.0, 0.0], [-0.5, 0.6], [-0.5, -0.6]])
    body = np.pad(body, ((0, 0), (0, dim - 2)))
    if dim == 2:
        angle = 0.5
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    else:
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + np.sin(0.5) * skew + (1 - np.cos(0.5)) * skew @ skew
    target = body @ rotation.T + np.arange(dim) * 0.2
    state = jd.State.add_clump(
        jd.State.create(dim=dim), pos=jnp.asarray(body + 1.0), rad=jnp.full(3, 0.1)
    )
    system = jd.System.create(
        state=state,
        collider_type="",
        minimizer=optax.sgd,
        minimizer_kw={"learning_rate": 0.1},
        target_fn=lambda trial, _: (
            0.5 * jnp.sum((trial.pos - jnp.asarray(target)) ** 2)
        ),
    )
    result = minimize(state, system, max_steps=300, force_tol=1e-6, torque_tol=1e-6)
    assert result.info.converged
    np.testing.assert_allclose(result.state.pos, target, atol=3e-6)
    expected_distances = np.linalg.norm(body[:, None] - body[None], axis=-1)
    actual = np.asarray(result.state.pos)
    np.testing.assert_allclose(
        np.linalg.norm(actual[:, None] - actual[None], axis=-1),
        expected_distances,
        atol=2e-6,
    )
    assert result.energy < 1e-10


def pair(dim=2, overlap=1e-7):
    pos = jnp.zeros((2, dim)).at[1, 0].set(1.0 - overlap)
    st = jd.State.create(pos=pos, rad=jnp.full(2, 0.5), mass=jnp.ones(2))
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    sy = jd.System.create(
        st.shape,
        collider_type="naive",
        dt=0.01,
        minimizer=jd.minimizers.fire,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0)},
        mat_table=jd.MaterialTable.from_materials([mat]),
    )
    return st, sy


def _overflow_system(x):
    st = jd.State.create(
        pos=jnp.column_stack((jnp.asarray(x), jnp.zeros(len(x)))),
        rad=jnp.full(len(x), 0.5),
    )
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    sy = jd.System.create(
        state=st,
        dt=0.01,
        minimizer=jd.minimizers.fire,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 10.0)},
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.05,
            "max_neighbors": 1,
            "secondary_collider_type": "naive",
        },
        mat_table=jd.MaterialTable.from_materials([mat]),
    )
    return st, sy


def test_small_energy_does_not_certify_force_balance():
    st, sy = pair(dim=2)
    out = sy.minimize(st, sy, max_steps=3)
    assert int(out[2]) == 3
    assert not out.info.converged
    assert 0.0 <= out[3] <= 1e-14
    assert int(out.info.status) == MAX_STEPS
    assert TerminationReason(int(out.reason)) is TerminationReason.MAX_STEPS


def test_energy_plateau_is_not_convergence():
    st, sy = pair(overlap=0.1)
    sy = replace(sy, minimizer=optax.sgd(0.0))
    out = minimize(st, sy, max_steps=4)
    assert int(out[2]) == 4
    np.testing.assert_allclose(out[3], minimize(st, sy, max_steps=0)[3])
    assert not out.info.converged


@pytest.mark.parametrize("dim", [2, 3])
def test_balanced_force_does_not_hide_a_clump_torque(dim):
    centers = np.zeros((4, dim))
    centers[2, :2] = [0.5, -0.9]
    centers[3, :2] = [-0.5, 0.9]
    st, sy = pair(dim)
    st = jd.State.create(
        pos=jnp.asarray(centers),
        rad=jnp.full(4, 0.5),
        clump_id=jnp.array([0, 0, 1, 2]),
        fixed=jnp.array([False, False, True, True]),
    )
    arms = jnp.zeros((4, dim)).at[0, 0].set(0.5).at[1, 0].set(-0.5)
    st.pos_p = arms
    sy = jd.System.create(
        st.shape,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0)},
        mat_table=sy.mat_table,
        minimizer=jd.minimizers.fire,
    )
    out = minimize(st, sy, max_steps=0, force_tol=1e-10, torque_tol=1e-10)
    assert out.info.force_max < 1e-12
    assert out.info.torque_max > 0.09
    assert not out.info.converged


def test_fixed_reactions_and_contact_free_states():
    st, sy = pair(overlap=0.1)
    st.fixed = jnp.ones(2, dtype=bool)
    out = minimize(st, sy, max_steps=0)
    assert out.info.converged and out.info.force_max == 0
    st, sy = pair(overlap=-0.1)
    out = minimize(st, sy)
    assert int(out[2]) == 0 and int(out.info.status) == CONVERGED
    assert len(out) == 4


def test_nonfinite_is_a_failure():
    st, sy = pair()
    st.pos_c = st.pos_c.at[0, 0].set(jnp.nan)
    out = minimize(st, sy, max_steps=2)
    assert not out.info.converged and int(out.info.status) == NONFINITE


def test_norms_are_per_body_and_rotation_invariant():
    force = jnp.tile(jnp.array([[0.8, 0.8]]), (17, 1))
    torque = jnp.zeros((17, 1))
    info = convergence_info(force, torque, jnp.zeros(17, dtype=bool), 1.0, 1.0)
    assert not info.converged
    assert float(info.force_max) == pytest.approx(np.sqrt(1.28))


def test_jit_vmap_preserves_per_system_convergence():
    st, sy = pair()

    def run(dx):
        trial = replace(st, pos_c=st.pos_c.at[1, 0].set(1.0 + dx))
        return minimize(trial, sy, max_steps=0).info.converged

    np.testing.assert_array_equal(
        jax.jit(jax.vmap(run))(jnp.array([0.1, -0.1])), [True, False]
    )


def test_custom_signed_objective_uses_its_gradient():
    st, sy = pair(overlap=-0.1)
    sy = replace(sy, target_fn=lambda s, y: jnp.sum((s.pos_c - 2.0) ** 2) - 100.0)
    out = minimize(st, sy, max_steps=1)
    assert int(out[2]) == 1 and not out.info.converged
    assert out[3] < 0.0


@pytest.mark.parametrize(
    "optimizer", [jd.minimizers.fire, jd.minimizers.damped_newtonian]
)
def test_physical_relaxation_evaluates_energy_once(monkeypatch, optimizer):
    from jaxdem.minimizers import routines

    max_steps = 3
    st, sy = pair(overlap=0.1)
    sy = replace(sy, minimizer=optimizer(dt=0.01))
    original_energy = routines._evaluate_energy
    original_force = routines._evaluate_readonly_forces
    evaluations = []
    force_evaluations = []

    def counted_energy(state, system):
        state, system, pe = original_energy(state, system)
        jax.debug.callback(lambda e: evaluations.append(float(e)), pe, ordered=True)
        return state, system, pe

    def counted_force(state, system):
        state, system = original_force(state, system)
        jax.debug.callback(
            lambda f: force_evaluations.append(np.asarray(f)), state.force, ordered=True
        )
        return state, system

    routines.minimize.clear_cache()
    monkeypatch.setattr(routines, "_evaluate_energy", counted_energy)
    monkeypatch.setattr(routines, "_evaluate_readonly_forces", counted_force)
    try:
        result = routines.minimize(
            st, sy, max_steps=max_steps, force_tol=0.0, torque_tol=0.0
        )
        jax.block_until_ready(result)
        jax.effects_barrier()
        assert int(result[2]) == max_steps
        assert len(evaluations) == 1
        assert len(force_evaluations) == max_steps + 1
        # With overlap delta = 0.1 + 2*x_left, h = 0.01, and unit masses:
        # FIRE: v_next = v - h*delta; damped Newtonian: v_next = (399*v - 4*delta)/401.
        x, v, overlaps = 0.0, 0.0, [0.1]
        for _ in range(max_steps):
            delta = 0.1 + 2.0 * x
            v = (
                v - 0.01 * delta
                if optimizer is jd.minimizers.fire
                else (399.0 * v - 4.0 * delta) / 401.0
            )
            x += 0.01 * v
            overlaps.append(0.1 + 2.0 * x)
        np.testing.assert_allclose(
            np.asarray(force_evaluations)[..., 0],
            np.array(overlaps)[:, None] * [-1, 1],
            rtol=3e-6,
            atol=2e-7,
        )
        shift = (overlaps[-1] - 0.1) / 2
        np.testing.assert_allclose(
            result[0].pos_c, [[shift, 0], [0.9 - shift, 0]], rtol=3e-6, atol=2e-8
        )
        np.testing.assert_allclose(
            result[3], overlaps[-1] ** 2 / 4, rtol=3e-6, atol=1e-9
        )
    finally:
        routines.minimize.clear_cache()


def test_conjugate_gradient_keeps_objective_evaluations():
    st, sy = pair(overlap=0.1)
    sy = replace(sy, minimizer=jd.minimizers.conjugate_gradient())
    initial_energy = minimize(st, sy, max_steps=0)[3]
    result = minimize(st, sy, max_steps=8)
    assert result.info.finite
    assert 0.0 <= result[3] < initial_energy
    np.testing.assert_allclose(
        result[3], jd.utils.compute_potential_energy(result[0], result[1]) / st.N
    )


def test_force_only_path_rejects_nonfinite_final_energy():
    st, sy = pair(overlap=-0.1)
    sy = replace(sy, force_model=_NanEnergySpring())
    out = minimize(st, sy, max_steps=0)
    assert not out.info.converged and not out.info.finite
    assert int(out.info.status) == NONFINITE


def test_overflow_rejected_even_if_force_check_passes():
    st, sy = _overflow_system([0, 0.2, 0.4])
    out = minimize(st, sy, max_steps=10, force_tol=1e6, torque_tol=1e6)
    assert out.info.finite
    assert not out.info.converged
    assert int(out.info.status) == COLLIDER_OVERFLOW
    assert int(out[2]) == 0


def test_overflow_during_relaxation_stops_at_first_bad_evaluation():
    st, sy = _overflow_system([0, 0.5, 1.6])
    initial = minimize(st, sy, max_steps=0)
    assert not initial[1].collider.overflow
    sy = replace(sy, minimizer=optax.sgd(0.2))
    out = minimize(st, sy, max_steps=10)
    assert int(out[2]) == 1
    assert out[1].collider.overflow
    assert int(out.info.status) == COLLIDER_OVERFLOW
    assert not out.info.converged
