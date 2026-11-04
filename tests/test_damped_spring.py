# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check normal damping, rigid-body motion, and moving periodic images."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

import jaxdem as jd

_RTOL = 1e-12 if jax.config.jax_enable_x64 else 2e-6
_ATOL = 1e-12 if jax.config.jax_enable_x64 else 2e-6
_SCALES = [1e-200, 1.0, 1e200] if jax.config.jax_enable_x64 else [1e-30, 1.0, 1e30]

# This restitution gives a damping ratio of one half.
HALF_DAMPING_RESTITUTION = float(np.exp(-np.pi / np.sqrt(3.0)))


def _system(state, restitution=(HALF_DAMPING_RESTITUTION,), young=20.0, **options):
    table = jd.MaterialTable.from_materials(
        [
            jd.Material.create(
                "elasticfrict", density=1.0, young=young, poisson=0.25, mu=0.0, e=e
            )
            for e in restitution
        ]
    )
    return jd.System.create(
        state=state,
        force_model_type="damped_spring",
        mat_table=table,
        collider_type="naive",
        **options,
    )


@pytest.mark.parametrize("name", ["dampedspring", "damped_spring", "Damped-Spring"])
def test_factory_names_share_the_canonical_stateless_law(name):
    law = jd.ForceModel.create(name)
    assert isinstance(law, jd.forces.DampedSpringForce)
    assert law.type_name == "dampedspring"
    assert law.history_shape(3) == (0,)
    assert not law.supports_analytical_energy_gradient
    assert law.required_material_properties == ("young_eff", "e")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("normal_speed", [-1.0, 0.0, 1.0, 100.0])
def test_compression_rebound_and_repulsive_clamp(dim, normal_speed):
    state = jd.State.create(
        pos=jnp.zeros((2, dim)).at[1, 0].set(0.75),
        vel=jnp.zeros((2, dim)).at[0, 0].set(-normal_speed).at[0, 1].set(9.0),
        rad=jnp.full(2, 0.5),
        mass=jnp.array([2.0, 3.0]),
    )
    system = _system(state)
    history = system.force_model.init_history((), dim)
    force, torque, updated = system.force_model.force(
        0, 1, state.pos, state, system, history
    )
    # Reduced mass is 6/5. The selected damping ratio gives gamma = sqrt(24).
    expected = np.zeros(dim)
    expected[0] = -max(0.0, 5.0 - np.sqrt(24.0) * normal_speed)
    np.testing.assert_allclose(force, expected, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(torque, np.zeros(1 if dim == 2 else 3))
    np.testing.assert_array_equal(updated, history)
    assert history.shape == (0,)
    np.testing.assert_allclose(
        system.force_model.energy(0, 1, state.pos, state, system, history),
        0.625,
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("restitution", [0.0, 1.0])
@pytest.mark.parametrize("scale", _SCALES)
def test_restitution_limits_and_extreme_finite_scales(restitution, scale):
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.75, 0.0]]),
        vel=jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.full(2, 2.0 * scale),
    )
    system = _system(state, restitution=(restitution,), young=scale)
    force, _, _ = system.force_model.force(0, 1, state.pos, state, system, jnp.zeros(0))
    # Reduced mass and stiffness both equal scale. Critical damping is 2*scale.
    magnitude = (0.25 + (2.0 if restitution == 0.0 else 0.0)) * scale
    np.testing.assert_allclose(force, [-magnitude, 0.0], rtol=_RTOL, atol=0.0)
    assert np.isfinite(force).all()


@pytest.mark.parametrize("distance", [1.0, 1.2])
def test_no_force_without_overlap_or_on_self(distance):
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [distance, 0.0]]),
        vel=jnp.array([[100.0, 0.0], [0.0, 0.0]]),
        rad=jnp.full(2, 0.5),
    )
    system = _system(state)
    for j in (0, 1):
        force, torque, _ = system.force_model.force(
            0, j, state.pos, state, system, jnp.zeros(0)
        )
        np.testing.assert_array_equal(force, 0.0)
        np.testing.assert_array_equal(torque, 0.0)


def test_material_mixing_quasistatic_force_and_elastic_energy():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.75, 0.0]]),
        vel=jnp.array([[1.0, 4.0], [0.0, 0.0]]),
        rad=jnp.full(2, 0.5),
        mass=jnp.array([2.0, 3.0]),
        mat_id=jnp.array([0, 1]),
    )
    system = _system(state, restitution=(1.0, HALF_DAMPING_RESTITUTION))
    law, history = system.force_model, jnp.zeros(0)
    dynamic, _, updated = law.force(
        0, 1, state.pos, state, system, history, advance_history=False
    )
    np.testing.assert_allclose(
        dynamic, [-5.0 - np.sqrt(24.0), 0.0], rtol=_RTOL, atol=_ATOL
    )
    elastic, _, _ = law.force(0, 1, state.pos, state, system, history, quasistatic=True)
    gradient = jax.grad(lambda pos: law.energy(0, 1, pos, state, system, history))(
        state.pos
    )
    np.testing.assert_allclose(elastic, -gradient[0], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(elastic, [-5.0, 0.0], rtol=_RTOL, atol=_ATOL)
    assert (
        np.dot(np.asarray(dynamic - elastic), np.asarray(state.vel[0] - state.vel[1]))
        < 0
    )
    np.testing.assert_array_equal(updated, history)
    resting = replace(state, vel=jnp.zeros_like(state.vel))
    np.testing.assert_array_equal(
        law.energy(0, 1, state.pos, state, system, history),
        law.energy(0, 1, resting.pos, resting, system, history),
    )


def test_rotating_clump_velocity_and_collider_moment():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.0], [0.75, 1.0]]),
        pos_p=jnp.array([[0.0, 1.0], [0.0, -1.0], [0.0, 0.0]]),
        rad=jnp.full(3, 0.5),
        mass=jnp.array([2.0, 2.0, 3.0]),
        inertia=jnp.ones((3, 1)),
        clump_id=jnp.array([0, 0, 1]),
        ang_vel=jnp.array([[-2.0], [-2.0], [0.0]]),
    )
    system = _system(state)
    force, torque, _ = system.force_model.force(
        0, 2, state.pos, state, system, jnp.zeros(0)
    )
    magnitude = 5.0 + 2.0 * np.sqrt(24.0)
    np.testing.assert_allclose(force, [-magnitude, 0.0], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_array_equal(torque, 0.0)
    computed, updated = system.collider.compute_force(state, system)
    np.testing.assert_allclose(computed.force[0], force, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(computed.torque[0], [magnitude], rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.sum(computed.force, axis=0), 0.0, rtol=_RTOL, atol=_ATOL
    )
    assert updated.collider.history.shape == (3, 3, 0)


@pytest.mark.parametrize("domain", ["periodic", "leesedwards"])
def test_moving_image_velocity_contributes_to_normal_damping(domain):
    if domain == "periodic":
        positions = [[0.1, 0.2], [1.85, 0.2]]
        options = dict(box_size=jnp.full(2, 2.0), box_rate=jnp.array([0.3, 0.0]))
        normal, overlap, normal_speed = np.array([1.0, 0.0]), 0.75, 0.3
    else:
        positions = [[0.1, 0.1], [0.2, 1.7]]
        options = dict(
            box_size=jnp.full(2, 2.0),
            box_rate=jnp.array([0.2, -0.1]),
            gamma=0.2,
            gamma_dot=0.3,
            alpha=0,
            beta=1,
        )
        # Image velocity is (-0.58, 0.1), and the normal is (0.6, 0.8).
        normal, overlap, normal_speed = np.array([0.6, 0.8]), 0.5, 0.268
    state = jd.State.create(
        pos=jnp.array(positions), rad=jnp.full(2, 0.5), mass=jnp.array([2.0, 3.0])
    )
    system = _system(state, domain_type=domain, domain_kw=options)
    force, _, _ = system.force_model.force(0, 1, state.pos, state, system, jnp.zeros(0))
    expected = (20.0 * overlap - np.sqrt(24.0) * normal_speed) * normal
    np.testing.assert_allclose(force, expected, rtol=_RTOL, atol=_ATOL)
    elastic, _, _ = system.force_model.force(
        0, 1, state.pos, state, system, jnp.zeros(0), quasistatic=True
    )
    np.testing.assert_allclose(elastic, 20.0 * overlap * normal, rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_coincident_force_and_position_jacobian_are_finite(dim):
    state = jd.State.create(
        pos=jnp.zeros((2, dim)),
        vel=jnp.ones((2, dim)).at[1].set(0.0),
        rad=jnp.full(2, 0.5),
    )
    system = _system(state)

    def evaluate(position):
        return system.force_model.force(0, 1, position, state, system, jnp.zeros(0))[0]

    np.testing.assert_array_equal(evaluate(state.pos), 0.0)
    assert np.isfinite(jax.jacrev(evaluate)(state.pos)).all()


@pytest.mark.parametrize("dim", [2, 3])
def test_neighbor_shapes_and_pair_antisymmetry(dim):
    state = jd.State.create(
        pos=jnp.zeros((3, dim)).at[1, 0].set(0.75).at[2, 1].set(0.8),
        vel=jnp.zeros((3, dim)).at[0, :2].set(jnp.array([1.0, 2.0])),
        rad=jnp.full(3, 0.5),
        mass=jnp.array([2.0, 3.0, 4.0]),
    )
    system = _system(state)
    law = system.force_model
    force, torque, history = law.force(
        0, jnp.arange(3), state.pos, state, system, jnp.zeros((3, 0))
    )
    assert force.shape == (3, dim)
    assert torque.shape == (3, 1 if dim == 2 else 3)
    assert history.shape == (3, 0)
    for j in range(3):
        scalar, _, _ = law.force(0, j, state.pos, state, system, jnp.zeros(0))
        reverse, _, _ = law.force(j, 0, state.pos, state, system, jnp.zeros(0))
        np.testing.assert_allclose(force[j], scalar, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(force[j], -reverse, rtol=_RTOL, atol=_ATOL)
