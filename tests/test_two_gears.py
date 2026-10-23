# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Gear contact mechanics, local sensing, and action-spanning rewards."""

import jax
import jax.numpy as jnp

from dataclasses import replace

import numpy as np
import pytest

from jaxdem.rl.environments import TwoGears
from jaxdem.rl.environments.two_gears import _outer_radius, frictional_floor_force
from jaxdem.rl.env_wrappers import clip_action_env, vectorise_env
from jaxdem.state import State
from jaxdem.system import System
from jaxdem.utils import advance_action


@pytest.mark.parametrize(
    "offset,spin,expected_force,expected_torque",
    [
        ([0.0, -0.9], 1.0, [-80.0, 200.0], -80.0),
        ([0.5, -0.9], 0.0, [0.0, 200.0], 100.0),
    ],
)
def test_floor_contact_adds_the_member_lever_arm_once(
    offset, spin, expected_force, expected_torque
):
    # Only the lower constituent sphere touches the y=1 floor. Its surface
    # arm about the COM is (offset.x, -1), and penetration is 0.001.
    state = State.create(
        pos=jnp.array([[0.0, 1.999], [0.0, 1.999]]),
        pos_p=jnp.array([offset, [0.0, 0.9]]),
        rad=jnp.full(2, 0.1),
        mass=jnp.ones(2),
        inertia=jnp.ones((2, 1)),
        clump_id=jnp.zeros(2, dtype=int),
        ang_vel=jnp.full((2, 1), spin),
    )
    system = System.create(
        state=state,
        collider_type=None,
        force_manager_kw={"force_functions": (frictional_floor_force,)},
    )
    updated, _ = system.force_manager.apply(state, system)
    np.testing.assert_allclose(
        updated.force, np.broadcast_to(expected_force, (2, 2)), atol=1e-8
    )
    np.testing.assert_allclose(updated.torque, expected_torque, atol=1e-8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_gears": 0},
        {"num_gears": True},
        {"num_gears": 2.5},
        {"max_steps": 0},
        {"max_steps": 1.5},
        {"n_lidar_rays": 0},
        {"n_lidar_rays": True},
        {"friction": -0.1},
        {"friction": float("nan")},
        {"attraction_mag": -1.0},
        {"kinetic_energy_coeff": -0.1},
        {"kinetic_energy_coeff": float("inf")},
        {"kinetic_energy_scale": 0.0},
        {"lidar_range": 0.0},
        {"box_size": float("nan")},
        {"box_size": 2 * _outer_radius * 4 - 0.01},
    ],
)
def test_invalid_geometry_and_training_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TwoGears.Create(**kwargs)


@pytest.fixture(scope="module")
def gears():
    env = TwoGears.Create()
    return env.reset(env, jax.random.key(13))


def _synthetic_gears(centers, velocities=None, spins=None, masses=None, inertias=None):
    """Represent each body by two spheres with replicated rigid-body fields."""
    centers = jnp.asarray(centers, dtype=float)
    n = centers.shape[0]
    if velocities is None:
        velocities = jnp.zeros_like(centers)
    if spins is None:
        spins = jnp.zeros((n, 1))
    if masses is None:
        masses = jnp.ones(n)
    if inertias is None:
        inertias = jnp.ones((n, 1))
    return State.create(
        pos=jnp.repeat(centers, 2, axis=0),
        pos_p=jnp.tile(jnp.array([[0.2, 0.0], [-0.2, 0.0]]), (n, 1)),
        rad=jnp.full(2 * n, 0.1),
        vel=jnp.repeat(jnp.asarray(velocities), 2, axis=0),
        ang_vel=jnp.repeat(jnp.asarray(spins), 2, axis=0),
        mass=jnp.repeat(jnp.asarray(masses), 2),
        inertia=jnp.repeat(jnp.asarray(inertias), 2, axis=0),
        clump_id=jnp.repeat(jnp.arange(n), 2),
    )


def test_observation_uses_goals_body_motion_and_center_lidar_with_the_real_floor(gears):
    state = _synthetic_gears(
        [[5.0, 3.0], [7.0, 3.0], [5.0, 7.0]],
        velocities=[[0.5, -0.25], [0.0, 0.0], [0.0, 0.0]],
        spins=[[0.75], [0.0], [0.0]],
    )
    env = replace(
        gears,
        state=state,
        env_params={
            **gears.env_params,
            "objective": jnp.array([[8.0, 7.0], [7.0, 3.0], [5.0, 7.0]]),
        },
    )
    observation = env.observation(env)
    assert observation.shape == (3, 7 + 16)
    assert env.observation_space_size == 7 + 16
    np.testing.assert_allclose(
        observation[0, :7], [-0.6, -0.8, -3.0, -3.0, 0.5, -0.25, 0.75]
    )
    expected = np.zeros(16)
    expected[[0, 4, 8, 12]] = [0.5, 0.8, 0.8, 0.6]
    np.testing.assert_allclose(observation[0, 7:], expected, atol=1e-12)


def test_a_third_gear_changes_perception_even_when_the_nearest_gear_is_unchanged(gears):
    near = _synthetic_gears([[5.0, 3.0], [7.0, 3.0], [5.0, 7.0]])
    far = _synthetic_gears([[5.0, 3.0], [7.0, 3.0], [15.0, 17.0]])
    nearby = gears.observation(replace(gears, state=near))
    distant = gears.observation(replace(gears, state=far))
    np.testing.assert_array_equal(nearby[0, :7], distant[0, :7])
    assert float(nearby[0, 7 + 8]) == pytest.approx(0.8)
    assert float(distant[0, 7 + 8]) == pytest.approx(0.8)
    assert float(nearby[0, 7 + 12]) == pytest.approx(0.6)
    assert float(distant[0, 7 + 12]) == pytest.approx(0.0)


@pytest.mark.parametrize("coefficient", [0.0, 0.2])
def test_reward_matches_quartic_distance_and_body_kinetic_energy(gears, coefficient):
    centers = jnp.array([[5.0, 3.0], [9.0, 3.0], [13.0, 3.0]])
    state = _synthetic_gears(
        centers,
        velocities=[[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]],
        spins=[[2.0], [0.0], [1.0]],
        masses=[1.0, 2.0, 3.0],
        inertias=[[0.5], [0.7], [0.8]],
    )
    scale = 2.0
    offsets = jnp.array(
        [[0.0, 0.0], [2 * _outer_radius, 0.0], [0.0, 4 * _outer_radius]]
    )
    baseline = np.array([0.05, 0.1, 0.0])
    env = replace(
        gears,
        state=state,
        env_params={
            **gears.env_params,
            "objective": centers + offsets,
            "kinetic_energy_coeff": jnp.asarray(coefficient),
            "kinetic_energy_scale": jnp.asarray(scale),
            "prev_potential": jnp.asarray(baseline),
        },
    )
    # Each member stores its body's full mass/inertia: energy must count each
    # body once rather than multiply by its number of constituent spheres.
    energy = np.array([13.5, 0.0, 1.9])
    expected = np.exp(-np.array([0.0, 1.0, 16.0]) - coefficient * energy / scale)
    np.testing.assert_allclose(env.reward(env), expected - baseline, atol=1e-12)
    checkpointed = env.checkpoint(env, jnp.zeros((3, 1)))
    np.testing.assert_allclose(checkpointed.env_params["prev_potential"], expected)
    np.testing.assert_allclose(checkpointed.reward(checkpointed), 0.0, atol=1e-12)


def test_skipped_frames_keep_one_action_start_potential_and_read_the_live_endpoint(
    gears,
):
    state = replace(
        gears.state, vel=jnp.broadcast_to(jnp.array([0.1, 0.0]), gears.state.vel.shape)
    )
    state, system = System.initialize(state, gears.system)
    indices = jnp.array([jnp.argmax(state.clump_id == i) for i in range(3)])
    objective = state.pos_c[indices] + jnp.array([2 * _outer_radius, 0.0])
    env = replace(
        gears,
        state=state,
        system=system,
        env_params={**gears.env_params, "objective": objective},
    )
    env, terminated, truncated = advance_action(
        env, jnp.full((3, 1), 0.3), skip_frames=3
    )
    expected_start = np.full(3, np.exp(-1.0))
    np.testing.assert_allclose(env.env_params["prev_potential"], expected_start)
    distance = np.linalg.norm(np.asarray(env.state.pos_c[indices] - objective), axis=-1)
    expected = np.exp(-((distance / (2 * _outer_radius)) ** 4)) - expected_start
    np.testing.assert_allclose(env.reward(env), expected, atol=1e-12)
    assert np.any(np.abs(expected) > 1e-6)
    assert int(env.system.step_count) == 4
    assert not bool(terminated | truncated)
    frozen, _, _ = advance_action(
        env, jnp.ones((3, 1)), skip_frames=1, terminated=jnp.asarray(True)
    )
    np.testing.assert_array_equal(frozen.state.pos, env.state.pos)
    np.testing.assert_allclose(frozen.reward(frozen), 0.0, atol=1e-12)


def test_requested_torque_is_applied_once_per_gear(gears):
    state = replace(gears.state, pos_c=gears.state.pos_c + jnp.array([0.0, 10.0]))
    state, system = System.initialize(state, gears.system)
    env = replace(
        gears,
        state=state,
        system=system,
        env_params={
            **gears.env_params,
            "friction": jnp.asarray(0.0),
            "attraction_mag": jnp.asarray(0.0),
        },
    )
    action = jnp.array([[0.2], [-0.3], [0.4]])
    stepped = env.step(env, action)
    np.testing.assert_allclose(
        stepped.state.torque, action[stepped.state.clump_id], atol=1e-12
    )


def test_wrapped_batched_reset_clipped_actions_and_truncation_are_compatible():
    env = TwoGears.Create(max_steps=2, n_lidar_rays=8)
    env = vectorise_env(clip_action_env(env, -0.25, 0.25), n=2)
    keys = jax.random.split(jax.random.key(14), 2)
    env = env.reset(env, keys)
    assert env.observation(env).shape == (2, 3, 15)
    np.testing.assert_allclose(env.reward(env), 0.0, atol=1e-12)
    initial_structure = jax.tree.structure(env)
    advanced, terminated, truncated = advance_action(
        env, jnp.full((2, 3, 1), 4.0), skip_frames=3
    )
    np.testing.assert_array_equal(advanced.system.step_count, [2, 2])
    np.testing.assert_array_equal(terminated, [False, False])
    np.testing.assert_array_equal(truncated, [True, True])
    assert np.isfinite(advanced.observation(advanced)).all()
    assert np.isfinite(advanced.reward(advanced)).all()
    reference = env
    for _ in range(2):
        reference = jax.vmap(TwoGears.step)(reference, jnp.full((2, 3, 1), 0.25))
    np.testing.assert_allclose(advanced.state.pos, reference.state.pos, atol=1e-12)
    np.testing.assert_allclose(advanced.state.vel, reference.state.vel, atol=1e-12)
    np.testing.assert_allclose(
        advanced.state.ang_vel, reference.state.ang_vel, atol=1e-12
    )
    reset = advanced.reset_if_done(advanced, truncated, keys)
    assert jax.tree.structure(reset) == initial_structure
    np.testing.assert_array_equal(reset.system.step_count, [0, 0])
    np.testing.assert_allclose(reset.reward(reset), 0.0, atol=1e-12)
