"""Navigator rewards and observations are measured at action boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.environments.multi_navigator import MultiNavigator
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils.environment import advance_action


def moving_env(name, num_envs=None, N=1):
    kwargs = {"N": N} if name == "multiNavigator" else {}
    env = Environment.create(name, max_steps=1000, **kwargs)
    key = jax.random.key(22)
    if num_envs is not None:
        env = vectorise_env(env, n=num_envs)
        key = jax.random.split(key, num_envs)
    env = env.reset(env, key)
    positions = jnp.stack((jnp.full(N, 10.0), 5.0 + 5.0 * jnp.arange(N)), axis=-1)
    env.state.pos_c = jnp.broadcast_to(positions, env.state.pos_c.shape)
    env.env_params["objective"] = env.state.pos_c + jnp.array([2.0, 0.0])
    env.state.vel = jnp.broadcast_to(jnp.array([1.0, 0.0]), env.state.vel.shape)
    # Save only the action-start baseline after arranging the physical state.
    return env.checkpoint(env, jnp.zeros_like(env.state.vel))


def potential(env):
    """Compute each navigator's distance potential directly from physical state."""
    distance = jnp.linalg.norm(env.state.pos_c - env.env_params["objective"], axis=-1)
    if isinstance(env, MultiNavigator):
        return jnp.exp(-((distance / (2 * env.state.rad)) ** 4))
    return jnp.exp(-2 * distance)


@pytest.mark.parametrize("name", ["multiNavigator", "singleNavigator"])
@pytest.mark.parametrize("skip_frames", [0, 1, 50])
def test_reward_spans_complete_actions_and_refreshes_baseline(name, skip_frames):
    initial = moving_env(name)
    action = jnp.array([[0.2, 0.0]])
    first, _, _ = advance_action(initial, action, skip_frames=skip_frames)
    second, _, _ = advance_action(first, action, skip_frames=skip_frames)
    whole, _, _ = advance_action(initial, action, skip_frames=2 * skip_frames + 1)

    for before, after in [(initial, first), (first, second)]:
        np.testing.assert_allclose(
            after.reward(after),
            potential(after) - potential(before),
            rtol=2e-5,
            atol=1e-7,
        )
        np.testing.assert_array_equal(
            after.env_params["prev_dist"],
            jnp.linalg.norm(
                before.state.pos_c - before.env_params["objective"], axis=-1
            ),
        )
    np.testing.assert_allclose(second.state.pos_c, whole.state.pos_c)
    np.testing.assert_allclose(
        first.reward(first) + second.reward(second), whole.reward(whole), atol=1e-7
    )
    assert int(second.system.step_count) == 2 * (1 + skip_frames)


@pytest.mark.parametrize("name", ["multiNavigator", "singleNavigator"])
def test_physics_steps_preserve_history_while_observations_are_live(name):
    env = moving_env(name)
    initial_dist = np.asarray(env.env_params["prev_dist"]).copy()
    observation = np.asarray(env.observation(env)).copy()
    action = jnp.array([[0.7, 0.0]])
    raw = jax.lax.fori_loop(0, 8, lambda _, e: e.step(e, action), env)
    np.testing.assert_array_equal(raw.env_params["prev_dist"], initial_dist)
    assert not np.allclose(raw.observation(raw), observation)
    assert not np.allclose(raw.state.pos_c, env.state.pos_c)
    advanced, _, _ = advance_action(env, action, skip_frames=7)
    np.testing.assert_allclose(raw.observation(raw), advanced.observation(advanced))
    np.testing.assert_allclose(raw.reward(raw), advanced.reward(advanced))
    np.testing.assert_allclose(raw.observation(raw)[..., 4:6], raw.state.vel)

    reset = raw.reset(raw, jax.random.key(31))
    distance = jnp.linalg.norm(
        reset.state.pos_c - reset.env_params["objective"], axis=-1
    )
    np.testing.assert_allclose(reset.env_params["prev_dist"], distance)
    np.testing.assert_array_equal(reset.reward(reset), 0.0)


@pytest.mark.parametrize("name", ["multiNavigator", "singleNavigator"])
def test_vectorized_truncation_uses_each_actual_endpoint(name):
    initial = moving_env(name, num_envs=3, N=3 if name == "multiNavigator" else 1)
    initial.env_params["max_steps"] = jnp.array([2, 4, 100])
    action = jnp.broadcast_to(jnp.array([0.2, 0.0]), (3, initial.max_num_agents, 2))
    final, terminated, truncated = advance_action(initial, action, skip_frames=4)
    np.testing.assert_array_equal(final.system.step_count, [2, 4, 5])
    expected_action = (1 - 0.78 ** jnp.array([2, 4, 5]))[:, None, None] * action
    np.testing.assert_allclose(final.env_params["applied_action"], expected_action)
    np.testing.assert_array_equal(terminated, [False, False, False])
    np.testing.assert_array_equal(truncated, [True, True, False])
    np.testing.assert_allclose(
        final.reward(final), potential(final) - potential(initial), rtol=2e-5, atol=1e-7
    )
    continued, _, _ = advance_action(
        final, action, skip_frames=4, terminated=terminated, truncated=truncated
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


@pytest.mark.parametrize("dim", [2, 3])
def test_single_navigator_distance_reward_and_zero_distance_observation(dim):
    env = Environment.create("singleNavigator", dim=dim)
    env = env.reset(env, jax.random.key(35))
    env.state.pos_c = jnp.full((1, dim), 10.0)
    env.env_params["objective"] = env.state.pos_c.at[0, 0].add(2.0)
    env = env.checkpoint(env, jnp.zeros_like(env.state.vel))
    np.testing.assert_array_equal(env.reward(env), 0.0)

    # Move closer without an endpoint checkpoint; observations and reward are live.
    env.state.pos_c = env.state.pos_c.at[0, 0].add(1.0)
    env.state.vel = jnp.full((1, dim), 7.0)
    np.testing.assert_allclose(env.reward(env), [np.exp(-2.0) - np.exp(-4.0)])
    np.testing.assert_array_equal(env.observation(env)[..., -dim:], env.state.vel)
    # Start a new action and change only velocity: reward depends only on distance.
    env = env.checkpoint(env, jnp.zeros_like(env.state.vel))
    env.state.vel = jnp.full((1, dim), -3.0)
    np.testing.assert_array_equal(env.reward(env), 0.0)
    np.testing.assert_array_equal(env.observation(env)[..., -dim:], env.state.vel)

    env = env.checkpoint(env, jnp.zeros_like(env.state.vel))
    env.state.pos_c = env.state.pos_c.at[0, 0].add(-1.0)
    np.testing.assert_allclose(env.reward(env), [np.exp(-4.0) - np.exp(-2.0)])

    env = env.checkpoint(env, jnp.zeros_like(env.state.vel))
    env.state.pos_c = env.env_params["objective"]
    observation = np.asarray(env.observation(env))
    assert observation.shape == (1, 3 * dim)
    assert np.isfinite(observation).all()
    np.testing.assert_array_equal(observation[..., : 2 * dim], 0.0)
    # Remaining at the objective earns no separate occupancy bonus.
    env = env.checkpoint(env, jnp.zeros_like(env.state.vel))
    np.testing.assert_array_equal(env.reward(env), 0.0)


@pytest.mark.parametrize("name", ["singleNavigator", "multiNavigator"])
def test_target_edit_is_visible_without_checkpoint_and_next_action_rebases(name):
    env = moving_env(name)
    initial_obs = np.asarray(env.observation(env)).copy()
    env.env_params["objective"] = env.state.pos_c + jnp.array([[0.0, 3.0]])
    assert not np.allclose(env.observation(env), initial_obs)
    env, _, _ = advance_action(env, jnp.zeros((1, 2)), skip_frames=0)
    np.testing.assert_array_equal(env.env_params["prev_dist"], [3.0])


@pytest.mark.parametrize("alpha", [0.0, 0.1, 0.22, 1.0])
def test_multi_agent_action_filter_preserves_each_agents_previous_applied_force(alpha):
    env = moving_env("multiNavigator", N=3)
    env.env_params["action_alpha"] = jnp.asarray(alpha)
    action = jnp.array([[1.0, 0.0], [0.0, -0.5], [-0.2, 0.3]])
    first, _, _ = advance_action(env, action, skip_frames=3)
    expected_first = (1 - (1 - alpha) ** 4) * action
    np.testing.assert_allclose(first.env_params["applied_action"], expected_first)
    checkpoint = first.checkpoint(first, -action)
    np.testing.assert_array_equal(
        checkpoint.env_params["applied_action"], first.env_params["applied_action"]
    )
    final, _, _ = advance_action(checkpoint, -action, skip_frames=7)
    expected_final = -action + (expected_first + action) * (1 - alpha) ** 8
    np.testing.assert_allclose(
        final.env_params["applied_action"], expected_final, atol=1e-14
    )
    np.testing.assert_array_equal(
        final.env_params["prev_dist"], checkpoint.env_params["prev_dist"]
    )
    reset = final.reset(final, jax.random.key(51))
    np.testing.assert_array_equal(reset.env_params["applied_action"], 0.0)
    np.testing.assert_array_equal(reset.reward(reset), 0.0)


def test_one_agent_multi_navigator_matches_single_navigator_dynamics():
    single = moving_env("singleNavigator")
    multi = moving_env("multiNavigator")
    # Compare the same physical initial condition in an area away from walls.
    multi.state = single.state
    for action in (jnp.array([[0.7, -0.3]]), jnp.array([[-0.7, 0.3]])):
        single, _, _ = advance_action(single, action, skip_frames=49)
        multi, _, _ = advance_action(multi, action, skip_frames=49)
        np.testing.assert_allclose(multi.state.pos_c, single.state.pos_c, atol=1e-12)
        np.testing.assert_allclose(multi.state.vel, single.state.vel, atol=1e-12)
        np.testing.assert_allclose(
            multi.observation(multi)[..., :6], single.observation(single), atol=1e-12
        )


def test_multi_agent_reward_depends_only_on_each_live_center_distance():
    env = moving_env("multiNavigator", N=3)
    env.env_params["objective"] = env.state.pos_c + jnp.array(
        [[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]]
    )
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    before = potential(env)
    env.state.pos_c = env.state.pos_c.at[1, 0].add(-0.5).at[2, 0].add(1.0)
    expected = jnp.exp(-((jnp.array([0.0, 1.0, 2.0]) / 2) ** 4)) - before
    np.testing.assert_allclose(env.reward(env), expected)
    env.state.vel = jnp.full_like(env.state.vel, 7.0)
    np.testing.assert_allclose(env.reward(env), expected)
    np.testing.assert_array_equal(env.reward(env)[0], 0.0)
    assert np.isfinite(env.observation(env)).all()
    np.testing.assert_array_equal(env.observation(env)[0, :4], 0.0)
    assert (
        not {
            "prev_ke",
            "ke_tau",
            "ke_gate",
            "near_goal_bonus",
            "delta",
            "lidar",
            "prev_neighbor_potential",
            "neighbor_reward_coeff",
        }
        & env.env_params.keys()
    )


def test_multi_navigator_keeps_live_lidar_and_particle_contacts():
    env = moving_env("multiNavigator", N=2)
    env.state.pos_c = jnp.array([[10.0, 10.0], [14.0, 10.0]])
    before = env.observation(env)[0, 6:]
    env.state.pos_c = env.state.pos_c.at[1, 0].set(17.0)
    assert not np.allclose(env.observation(env)[0, 6:], before)

    env.state.pos_c = env.state.pos_c.at[1, 0].set(11.9)
    env.state.vel = jnp.zeros_like(env.state.vel)
    result = env.step(env, jnp.zeros_like(env.state.force))
    assert result.state.vel[0, 0] < 0.0
    assert result.state.vel[1, 0] > 0.0
    np.testing.assert_array_equal(result.env_params["applied_action"], 0.0)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_multi_navigator_rejects_invalid_smoothing_fraction(alpha):
    with pytest.raises(ValueError, match="action_alpha"):
        Environment.create("multiNavigator", action_alpha=alpha)


def test_quartic_reward_scales_with_each_particles_radius():
    env = moving_env("multiNavigator", N=3)
    env.state.rad = jnp.array([0.5, 1.0, 2.0])
    objectives = env.state.pos_c
    env.env_params["objective"] = objectives
    env.state.pos_c = objectives.at[:, 0].add(2 * env.state.rad)
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    np.testing.assert_array_equal(env.reward(env), 0.0)

    env.state.pos_c = objectives
    np.testing.assert_allclose(env.reward(env), 1 - np.exp(-1))
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    env.state.pos_c = objectives.at[:, 0].add(4 * env.state.rad)
    np.testing.assert_allclose(env.reward(env), np.exp(-16) - 1)


def test_quartic_reward_is_flat_near_goal_and_negligible_far_away():
    env = moving_env("multiNavigator")
    objective = env.env_params["objective"]
    env.state.pos_c = objective
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    env.state.pos_c = objective + jnp.array([[0.1, 0.0]])
    np.testing.assert_allclose(env.reward(env), np.expm1(-(0.05**4)), atol=1e-15)
    assert abs(float(env.reward(env)[0])) < 1e-5

    env.state.pos_c = objective + jnp.array([[4.0, 0.0]])
    env = env.checkpoint(env, jnp.zeros_like(env.state.force))
    env.state.pos_c = objective + jnp.array([[6.0, 0.0]])
    np.testing.assert_allclose(env.reward(env), np.exp(-81) - np.exp(-16))
    assert abs(float(env.reward(env)[0])) < 2e-7


def test_other_agents_and_their_objectives_do_not_change_own_reward():
    env = moving_env("multiNavigator", N=3)
    env.state.pos_c = env.state.pos_c.at[1:].add(1.0)
    env.env_params["objective"] = env.env_params["objective"].at[1:].add(2.0)
    np.testing.assert_array_equal(env.reward(env)[0], 0.0)
