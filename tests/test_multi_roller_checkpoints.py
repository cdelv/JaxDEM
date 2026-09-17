"""MultiRoller shares navigator rewards and single-roller actuator dynamics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.colliders.naive import NaiveSimulator
from jaxdem.forces.spring import SpringForce
from jaxdem.system import System
from jaxdem.utils.environment import advance_action


def moving_rollers(N=2, num_envs=None, action_alpha=0.22):
    env = Environment.create("multiRoller", N=N, action_alpha=action_alpha)
    key = jax.random.key(22)
    if num_envs is not None:
        env = vectorise_env(env, n=num_envs)
        key = jax.random.split(key, num_envs)
    env = env.reset(env, key)
    positions = jnp.stack(
        (jnp.full(N, 10.0), 5.0 + 5.0 * jnp.arange(N), jnp.ones(N)), axis=-1
    )
    env.state.pos_c = jnp.broadcast_to(positions, env.state.pos_c.shape)
    env.env_params["objective"] = env.state.pos_c + jnp.array([2.0, 0.0, 0.0])
    env.state.vel = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), env.state.vel.shape)
    # Leave the reset baseline stale to exercise the action-start checkpoint.
    return env


def distance(env):
    return jnp.linalg.norm(env.state.pos_c - env.env_params["objective"], axis=-1)


def potential(env):
    return jnp.exp(-((distance(env) / (1.5 * env.state.rad)) ** 4))


@pytest.mark.parametrize("skip_frames", [0, 1, 49])
def test_reward_spans_actions_and_physics_steps_preserve_checkpoint(skip_frames):
    initial = moving_rollers()
    action = jnp.array([[0.0, 0.2, 0.0], [0.1, -0.3, 0.0]])
    first, _, _ = advance_action(initial, action, skip_frames=skip_frames)
    second, _, _ = advance_action(first, action, skip_frames=skip_frames)
    whole, _, _ = advance_action(initial, action, skip_frames=2 * skip_frames + 1)
    for before, after in [(initial, first), (first, second)]:
        np.testing.assert_allclose(after.env_params["prev_dist"], distance(before))
        np.testing.assert_allclose(
            after.reward(after), potential(after) - potential(before), atol=1e-12
        )
    np.testing.assert_allclose(second.state.pos_c, whole.state.pos_c)
    np.testing.assert_allclose(
        first.reward(first) + second.reward(second), whole.reward(whole), atol=1e-12
    )
    checkpoint = initial.checkpoint(initial, action)
    raw = jax.lax.fori_loop(
        0, skip_frames + 1, lambda _, e: e.step(e, action), checkpoint
    )
    np.testing.assert_array_equal(raw.env_params["prev_dist"], distance(initial))
    np.testing.assert_allclose(
        raw.observation(raw), first.observation(first), atol=1e-12
    )
    np.testing.assert_allclose(raw.reward(raw), first.reward(first), atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.22, 1.0])
def test_per_agent_filter_reverses_across_checkpoints_and_drives_physics(alpha):
    initial = moving_rollers(action_alpha=alpha)
    reference = moving_rollers(action_alpha=1.0)
    action = jnp.array([[0.1, 0.2, 0.3], [-0.4, 0.5, -0.6]])
    first, _, _ = advance_action(initial, action, skip_frames=3)
    first_fraction = 1 - (1 - alpha) ** 4
    np.testing.assert_allclose(
        first.env_params["applied_action"], first_fraction * action
    )
    checkpoint = first.checkpoint(first, -action)
    np.testing.assert_array_equal(
        checkpoint.env_params["applied_action"], first.env_params["applied_action"]
    )
    final, _, _ = advance_action(checkpoint, -action, skip_frames=5)
    np.testing.assert_allclose(
        final.env_params["applied_action"],
        (-1 + (first_fraction + 1) * (1 - alpha) ** 6) * action,
        atol=1e-14,
    )
    values = [1 - (1 - alpha) ** n for n in range(1, 5)]
    values += [-1 + (first_fraction + 1) * (1 - alpha) ** n for n in range(1, 7)]
    commands = jnp.asarray(values)[:, None, None] * action
    reference, _ = jax.lax.scan(lambda e, a: (e.step(e, a), None), reference, commands)
    for field in ("pos_c", "vel", "ang_vel", "force", "torque"):
        np.testing.assert_allclose(
            getattr(final.state, field), getattr(reference.state, field), atol=1e-12
        )
    reset = final.reset(final, jax.random.key(31))
    np.testing.assert_array_equal(reset.env_params["applied_action"], 0.0)
    np.testing.assert_array_equal(reset.reward(reset), 0.0)


def test_vectorized_episode_boundaries_freeze_each_actuator():
    initial = moving_rollers(num_envs=3)
    initial.env_params["max_steps"] = jnp.array([2, 4, 100])
    action = jnp.ones((3, 2, 3))
    final, terminated, truncated = advance_action(initial, action, skip_frames=4)
    np.testing.assert_array_equal(final.system.step_count, [2, 4, 5])
    np.testing.assert_array_equal(terminated, False)
    np.testing.assert_array_equal(truncated, [True, True, False])
    expected = (1 - 0.78 ** jnp.array([2, 4, 5]))[:, None, None] * action
    np.testing.assert_allclose(final.env_params["applied_action"], expected)
    np.testing.assert_allclose(
        final.reward(final), potential(final) - potential(initial)
    )
    continued, _, _ = advance_action(
        final, -action, skip_frames=4, terminated=terminated, truncated=truncated
    )
    np.testing.assert_array_equal(continued.system.step_count, [2, 4, 10])
    np.testing.assert_array_equal(
        continued.env_params["applied_action"][:2],
        final.env_params["applied_action"][:2],
    )
    np.testing.assert_array_equal(continued.reward(continued)[:2], 0.0)
    np.testing.assert_array_equal(
        continued.observation(continued)[:2], final.observation(final)[:2]
    )


def test_live_reward_uses_radius_scaled_3d_distance_without_velocity_penalty():
    env = moving_rollers()
    env.state.rad = jnp.array([0.5, 2.0])
    objectives = env.state.pos_c
    env.env_params["objective"] = objectives
    env.state.pos_c = objectives.at[:, 2].add(1.5 * env.state.rad)
    env = env.checkpoint(env, jnp.zeros_like(env.state.torque))
    env.state.pos_c = objectives
    np.testing.assert_allclose(env.reward(env), 1 - np.exp(-1))
    env.state.vel = jnp.full_like(env.state.vel, 7.0)
    env.state.ang_vel = jnp.full_like(env.state.ang_vel, -5.0)
    np.testing.assert_allclose(env.reward(env), 1 - np.exp(-1))
    np.testing.assert_array_equal(env.observation(env)[..., :4], 0.0)
    np.testing.assert_array_equal(
        env.observation(env)[..., 4:6], env.state.vel[..., :2]
    )
    np.testing.assert_array_equal(env.observation(env)[..., 6:9], env.state.ang_vel)
    env = env.checkpoint(env, jnp.zeros_like(env.state.torque))
    np.testing.assert_array_equal(env.reward(env), 0.0)
    env.env_params["objective"] = objectives.at[:, 0].add(3 * env.state.rad)
    np.testing.assert_allclose(env.reward(env), np.exp(-16) - 1)
    assert np.all(env.observation(env)[..., 0] != 0.0)
    assert (
        not {"prev_ke", "ke_tau", "ke_gate", "near_goal_bonus", "delta_xy", "lidar"}
        & env.env_params.keys()
    )


def test_live_lidar_and_naive_spring_particle_contacts():
    env = moving_rollers()
    assert isinstance(env.system.collider, NaiveSimulator)
    assert isinstance(env.system.force_model, SpringForce)
    env.state.pos_c = jnp.array([[10.0, 10.0, 1.0], [14.0, 10.0, 1.0]])
    before = env.observation(env)[0, 9:]
    env.state.pos_c = env.state.pos_c.at[1, 0].set(17.0)
    assert not np.allclose(env.observation(env)[0, 9:], before)
    env.state.pos_c = env.state.pos_c.at[1, 0].set(11.9)
    env.state.vel = jnp.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    env.state, env.system = System.initialize(env.state, env.system)
    np.testing.assert_array_equal(env.state.torque, 0.0)
    env = env.checkpoint(env, jnp.zeros_like(env.state.torque))
    result = env.step(env, jnp.zeros_like(env.state.torque))
    assert result.state.vel[0, 0] < 0.0
    assert result.state.vel[1, 0] > 0.0
    assert not bool(result.system.collider.overflow)
    assert not bool(result.system.search_overflow)
    assert np.isfinite(result.state.force).all()
    np.testing.assert_array_equal(
        result.env_params["prev_dist"], env.env_params["prev_dist"]
    )


def test_reward_matches_multi_navigator_at_equal_scaled_distances():
    roller = moving_rollers()
    navigator = Environment.create("multiNavigator", N=2)
    navigator = navigator.reset(navigator, jax.random.key(23))
    for env in (roller, navigator):
        env.state.pos_c = jnp.full_like(env.state.pos_c, 10.0)
        env.env_params["objective"] = env.state.pos_c
        scale = 1.5 if env is roller else 2.0
        env.state.pos_c = env.state.pos_c.at[:, 0].add(scale * jnp.array([0.3, 2.5]))
    roller = roller.checkpoint(roller, jnp.zeros_like(roller.state.torque))
    navigator = navigator.checkpoint(navigator, jnp.zeros_like(navigator.state.force))
    for env in (roller, navigator):
        scale = 1.5 if env is roller else 2.0
        env.state.pos_c = env.state.pos_c.at[:, 0].add(scale * jnp.array([0.2, -0.7]))
    np.testing.assert_allclose(roller.reward(roller), navigator.reward(navigator))


def test_one_agent_matches_single_roller_dynamics_and_observations():
    multi = moving_rollers(N=1)
    single = Environment.create("singleRoller")
    single = single.reset(single, jax.random.key(22))
    single.state = multi.state
    single.env_params["objective"] = multi.env_params["objective"]
    for action in (jnp.array([[0.3, -0.7, 0.1]]), jnp.array([[-0.3, 0.7, -0.1]])):
        single, _, _ = advance_action(single, action, skip_frames=49)
        multi, _, _ = advance_action(multi, action, skip_frames=49)
        for field in ("pos_c", "vel", "ang_vel"):
            np.testing.assert_allclose(
                getattr(multi.state, field), getattr(single.state, field), atol=1e-12
            )
        np.testing.assert_allclose(
            multi.observation(multi)[..., :9], single.observation(single), atol=1e-12
        )


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_invalid_smoothing_fraction(alpha):
    with pytest.raises(ValueError, match="action_alpha"):
        Environment.create("multiRoller", action_alpha=alpha)
