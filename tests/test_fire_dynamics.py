"""FIRE follows global, mass-weighted velocity-Verlet dynamics."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import jaxdem as jd
from jaxdem.minimizers.routines import (
    _rotation_inverse_inertia,
    _rotation_pullback,
    _rotation_residual,
)
from jaxdem.utils import Quaternion
from jaxdem.utils.h5 import _construct_minimizer


def test_fire_mixes_with_one_norm_over_every_degree_of_freedom():
    optimizer = jd.fire(dt=1.0)
    momentum = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    force = jnp.array([[0.0, 1.0], [3.0, 0.0]])
    state = optimizer.init(jnp.zeros((2, 2)))._replace(momentum=momentum)

    update, advanced = optimizer.update(-force, state)

    # Both half-kicks precede the mixing; the norms span both particles.
    p_half = np.asarray(momentum + force)
    ratio = np.sqrt(np.sum(p_half**2) / np.sum(np.asarray(force) ** 2))
    expected = 0.9 * p_half + 0.1 * ratio * np.asarray(force)
    np.testing.assert_allclose(advanced.momentum, expected, rtol=1e-12)
    np.testing.assert_allclose(update, expected, rtol=1e-12)


def test_fire_uphill_step_backtracks_half_a_reduced_step_and_stops():
    optimizer = jd.fire(dt=1.0, alpha_init=0.1)
    state = optimizer.init(jnp.zeros((1, 2)))._replace(
        momentum=jnp.array([[1.0, 0.0]]), alpha=jnp.array(0.05), N_good=jnp.array(7)
    )

    update, advanced = optimizer.update(jnp.array([[1.0, 0.0]]), state)

    # v = 1 - 1/2 opposes F = -1, so dt halves and x moves back by v dt / 2.
    np.testing.assert_allclose(update, [[-0.5 * 0.5 / 2.0, 0.0]], rtol=1e-12)
    np.testing.assert_array_equal(advanced.momentum, 0.0)
    assert float(advanced.dt) == 0.5
    assert float(advanced.alpha) == 0.1
    assert int(advanced.N_good) == 0


def _quadratic_force(stiffness, center):
    def grad(x):
        return jax.tree.map(lambda k, c, xi: k * (xi - c), stiffness, center, x)

    return grad


def test_fire_masses_match_unit_mass_fire_in_mass_weighted_coordinates():
    """FIRE with masses M on U(x) equals unit-mass FIRE on U(M^-1/2 y)."""
    key = jax.random.PRNGKey(3)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    stiffness = {
        "a": jax.random.uniform(k1, (4, 2), minval=0.5, maxval=3.0),
        "b": jax.random.uniform(k2, (4, 1), minval=0.5, maxval=3.0),
    }
    center = {"a": jnp.ones((4, 2)), "b": -jnp.ones((4, 1))}
    mass = {
        "a": jax.random.uniform(k3, (4, 1), minval=0.2, maxval=5.0),
        "b": jax.random.uniform(k4, (4, 1), minval=0.2, maxval=5.0),
    }
    root = jax.tree.map(jnp.sqrt, mass)
    grad_x = _quadratic_force(stiffness, center)

    def grad_y(y):
        x = jax.tree.map(lambda yi, s: yi / s, y, root)
        return jax.tree.map(lambda g, s: g / s, grad_x(x), root)

    def inverse_mass(tree):
        return jax.tree.map(lambda t, m: t / m, tree, mass)

    optimizer = jd.fire(dt=0.1)
    x = jax.tree.map(jnp.zeros_like, stiffness)
    y = jax.tree.map(lambda xi, s: xi * s, x, root)
    sx, sy = optimizer.init(x), optimizer.init(y)

    @jax.jit
    def step(x, y, sx, sy):
        ux, sx = optimizer.update(grad_x(x), sx, x, inverse_mass=inverse_mass)
        uy, sy = optimizer.update(grad_y(y), sy, y)
        return (
            jax.tree.map(jnp.add, x, ux),
            jax.tree.map(jnp.add, y, uy),
            sx,
            sy,
        )

    downhill = uphill = 0
    for _ in range(300):
        previous = float(sx.dt)
        x, y, sx, sy = step(x, y, sx, sy)
        uphill += float(sx.dt) < previous
        downhill += float(sx.dt) >= previous
        np.testing.assert_allclose(
            jnp.concatenate([v.ravel() for v in jax.tree.leaves(x)]),
            jnp.concatenate(
                [
                    (yi / s).ravel()
                    for yi, s in zip(jax.tree.leaves(y), jax.tree.leaves(root))
                ]
            ),
            rtol=1e-9,
            atol=1e-12,
        )
    assert uphill > 0 and downhill > 0
    np.testing.assert_allclose(x["a"], center["a"], atol=1e-6)


def test_rotation_inverse_inertia_applies_the_world_inertia_tensor():
    orientation = Rotation.from_rotvec([0.4, -1.1, 0.7])
    w_x_y_z = orientation.as_quat(scalar_first=True)
    q = Quaternion.create(
        w=jnp.asarray(w_x_y_z[:1])[None, :], xyz=jnp.asarray(w_x_y_z[1:])[None, :]
    )
    inertia = jnp.array([[1.0, 2.5, 7.0]])
    raw = jnp.array([[1.7, 0.3, -0.5, 0.2]])
    torque = jnp.array([[0.3, -1.2, 0.8]])

    rate = _rotation_inverse_inertia(raw, _rotation_pullback(raw, torque), q, inertia)

    matrix = orientation.as_matrix()
    expected = matrix @ np.diag(1.0 / np.asarray(inertia[0])) @ matrix.T @ torque[0]
    omega = 4.0 / jnp.sum(raw**2) * _rotation_residual(raw, rate)
    np.testing.assert_allclose(omega[0], expected, rtol=1e-12)
    np.testing.assert_allclose(jnp.vdot(rate, raw), 0.0, atol=1e-14)

    angle = jnp.array([[0.3]])
    np.testing.assert_allclose(
        _rotation_inverse_inertia(angle, jnp.array([[2.0]]), q, jnp.array([[4.0]])),
        [[0.5]],
    )


def _clump_system(mass_scale, dt):
    body = jnp.array([[1.0, 0.0, 0.0], [-0.5, 0.6, 0.2], [-0.5, -0.6, -0.2]])
    target = jnp.array([[0.3, 0.5, -0.2], [-0.7, 0.1, 0.4], [0.2, -0.3, 0.6]])
    state = jd.State.add_clump(
        jd.State.create(dim=3),
        pos=body,
        rad=jnp.full(3, 0.1),
        mass=2.0 * mass_scale,
        inertia=jnp.array([0.5, 1.5, 3.0]) * mass_scale,
    )
    state = jd.State.merge(
        state,
        jd.State.create(
            pos=jnp.array([[2.0, 2.0, 2.0]]),
            rad=jnp.array([0.1]),
            mass=jnp.array([0.25 * mass_scale]),
        ),
    )
    goal = jnp.concatenate([target, jnp.array([[2.5, 1.0, 2.2]])])

    def force(pos, st, sy):
        return -(pos - goal), jnp.zeros_like(st.torque)

    def energy(pos, st, sy):
        return 0.5 * jnp.sum((pos - goal) ** 2, axis=-1)

    system = jd.System.create(
        state=state,
        collider_type="",
        minimizer=jd.fire,
        minimizer_kw={"dt": dt},
        force_manager_kw={"force_functions": [(force, energy)]},
    )
    return state, system


def test_minimize_moves_bodies_with_their_masses_and_inertia():
    # Scaling every mass and inertia by c^2 and dt by c leaves the iterates
    # unchanged only when FIRE uses the physical masses and inertia.
    results = []
    for scale, dt in ((1.0, 0.05), (9.0, 0.15)):
        state, system = _clump_system(scale, dt)
        results.append(system.minimize(state, system, max_steps=20_000))
    (light, *_), (heavy, *_) = results
    assert results[0].converged and results[1].converged
    assert int(results[0].steps) == int(results[1].steps)
    np.testing.assert_allclose(light.pos, heavy.pos, rtol=1e-8, atol=1e-10)

    state, system = _clump_system(1.0, 0.15)
    unscaled = system.minimize(state, system, max_steps=20_000)
    assert int(unscaled.steps) != int(results[0].steps)


def test_saved_minimizer_arguments_removed_from_fire_are_ignored_on_load():
    with pytest.warns(RuntimeWarning, match="N_bad_max"):
        restored = _construct_minimizer(
            "jaxdem.minimizers.optimizers.fire", {"dt": 0.01, "N_bad_max": 10}
        )
    assert dict(restored.kw) == {"dt": 0.01}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _construct_minimizer("jaxdem.minimizers.optimizers.fire", {"dt": 0.01})
