"""Shared swarm coverage semantics and navigator geometry."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import SwarmNavigator, SwarmRoller
from jaxdem.rl.environments._swarm import filter_objectives, shared_potential
from jaxdem.utils import advance_action
from jaxdem.state import State
from jaxdem.system import System


def _sensor_env():
    return SimpleNamespace(
        max_num_agents=2,
        n_lidar_rays=4,
        state=SimpleNamespace(rad=jnp.ones(2)),
        env_params={
            "lidar_range": 10.0,
            "sharing_range": 1.5,
            "attraction_coeff": 0.1,
            "attraction_constant": 0.01,
            "attraction_quadratic": 0.02,
            "attraction_decay": 1 / 3,
            "hide_occupied_objectives": True,
        },
    )


@pytest.mark.parametrize("elevation", [None, 4])
def test_shared_goal_value_masks_empty_wall_and_self_returns(elevation):
    env = _sensor_env()
    bins = 4 * (elevation or 1)
    objective = jnp.zeros((2, bins)).at[0, 0].set(7.0)
    empty = jnp.zeros_like(objective)
    own_ids = jnp.broadcast_to(jnp.arange(2)[:, None], objective.shape)
    kwargs = {"n_elevation": elevation}
    baseline = shared_potential(objective, empty, own_ids, env, **kwargs)
    # With one objective at d=3 and no peers, the analytic value has no sharing.
    x, decay = 3.0, 1 / 3
    approach = np.exp(-decay * x) * (
        0.01 / decay + 0.02 * (x * x / decay + 2 * x / decay**2 + 2 / decay**3)
    )
    np.testing.assert_allclose(baseline, [np.exp(-16) + 0.1 * approach, 0.0])
    for ids in [own_ids, jnp.full_like(own_ids, -1), jnp.full_like(own_ids, 2)]:
        peers = empty.at[0, 0].set(7.0)
        np.testing.assert_allclose(
            shared_potential(objective, peers, ids, env, **kwargs), baseline
        )
        np.testing.assert_array_equal(
            filter_objectives(objective, peers, ids, env, **kwargs), objective
        )
    # Even a plausible peer ID must not turn an empty return into a claim.
    ids = own_ids.at[0, 0].set(1)
    np.testing.assert_allclose(
        shared_potential(objective, empty, ids, env, **kwargs), baseline
    )
    peers = empty.at[0, 0].set(7.0)
    np.testing.assert_allclose(
        shared_potential(objective, peers, ids, env, **kwargs), baseline / 2
    )
    assert float(filter_objectives(objective, peers, ids, env, **kwargs)[0, 0]) == 0


def test_spherical_sharing_distinguishes_elevation_at_the_same_azimuth():
    env = _sensor_env()
    objective = jnp.zeros((2, 16)).at[0, 0].set(7.0)
    empty = jnp.zeros_like(objective)
    ids = jnp.broadcast_to(jnp.arange(2)[:, None], objective.shape).at[0, 3].set(1)
    # Bins 0 and 3 have the same azimuth but elevations -67.5 and +67.5 degrees.
    peer = empty.at[0, 3].set(7.0)
    baseline = shared_potential(objective, empty, ids, env, n_elevation=4)
    np.testing.assert_allclose(
        shared_potential(objective, peer, ids, env, n_elevation=4),
        baseline,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        filter_objectives(objective, peer, ids, env, n_elevation=4), objective
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_navigator_reset_and_skipped_actions_preserve_the_reward_baseline(dim):
    env = SwarmNavigator.Create(
        N=2, num_objectives=3, dim=dim, n_lidar_rays=4, n_lidar_elevation=4
    )
    env = env.reset(env, jax.random.key(19))
    bins = 4 if dim == 2 else 16
    assert env.observation(env).shape == (2, dim + 2 * bins)
    assert env.observation_space_size == dim + 2 * bins
    assert env.action_space_size == dim
    np.testing.assert_allclose(env.state.mass, 1.0)
    np.testing.assert_allclose(env.reward(env), 0.0, atol=1e-12)
    assert env.env_params["objective"].shape == (3, dim)
    assert bool(jnp.all(jnp.any((env.state.pos < 0) | (env.state.pos > 20), axis=-1)))
    action = jnp.full((2, dim), 0.2)
    start = env._potential(env)
    env, _, _ = advance_action(env, action, skip_frames=3)
    np.testing.assert_allclose(env.env_params["prev_potential"], start)
    np.testing.assert_allclose(env.reward(env), env._potential(env) - start)
    assert int(env.system.step_count) == 4
    assert np.isfinite(env.observation(env)).all()
    frozen, _, _ = advance_action(
        env, action, skip_frames=2, terminated=jnp.asarray(True)
    )
    np.testing.assert_array_equal(frozen.state.pos, env.state.pos)
    np.testing.assert_allclose(frozen.reward(frozen), 0.0, atol=1e-12)
    reset = frozen.reset(frozen, jax.random.key(20))
    assert jax.tree.structure(reset) == jax.tree.structure(frozen)


@pytest.mark.parametrize("environment", [SwarmNavigator, SwarmRoller])
def test_empty_swarm_objectives_produce_zero_reward(environment):
    kwargs = {"N": 2, "num_objectives": 0, "n_lidar_rays": 4}
    env = environment.Create(**kwargs)
    env = env.reset(env, jax.random.key(2))
    np.testing.assert_array_equal(env.reward(env), jnp.zeros(2))
    assert np.isfinite(env.observation(env)).all()


def test_multiple_goals_use_best_settling_and_total_approach():
    env = _sensor_env()
    objectives = jnp.zeros((2, 4)).at[0, 0].set(9.0).at[0, 1].set(8.0)
    peers = jnp.zeros_like(objectives)
    ids = jnp.broadcast_to(jnp.arange(2)[:, None], peers.shape)
    x = np.array([1.0, 2.0])
    decay = 1 / 3
    approach = np.exp(-decay * x) * (
        0.01 / decay + 0.02 * (x * x / decay + 2 * x / decay**2 + 2 / decay**3)
    )
    expected = np.exp(-((x / 1.5) ** 4)).max() + 0.1 * approach.sum()
    np.testing.assert_allclose(
        shared_potential(objectives, peers, ids, env), [expected, 0.0]
    )


def test_objective_filter_preserves_own_far_and_disabled_goals():
    env = _sensor_env()
    objectives = jnp.array([[9.5, 7.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    ids = jnp.ones_like(objectives, dtype=int)
    np.testing.assert_allclose(
        filter_objectives(objectives, objectives, ids, env),
        [[9.5, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
    )
    env.env_params["hide_occupied_objectives"] = False
    np.testing.assert_array_equal(
        filter_objectives(objectives, objectives, ids, env), objectives
    )


def test_roller_floor_contact_is_unilateral_and_coulomb_limited():
    from jaxdem.rl.environments.single_roller import frictional_wall_force

    state = State.create(
        pos=jnp.array([[0.0, 0.0, 0.9999], [3.0, 0.0, 0.99], [6.0, 0.0, 1.1]]),
        vel=jnp.array([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]),
        rad=jnp.ones(3),
        mass=jnp.ones(3),
    )
    system = System.create(state=state, dt=0.002, collider_type="")
    force, torque = frictional_wall_force(state.pos, state, system)
    # Damping is capped at m/(2*dt)=250. The approaching sphere has
    # N=k*delta-c*v_n=2250; Coulomb friction is .4*N=900.
    np.testing.assert_allclose(
        force, [[0, 0, 0], [-900, 0, 2250], [0, 0, 0]], rtol=2e-5
    )
    np.testing.assert_allclose(torque, [[0, 0, 0], [0, 900, 0], [0, 0, 0]], rtol=2e-5)


def test_roller_action_smoothing_follows_the_known_step_response():
    env = SwarmRoller.Create(N=2, num_objectives=0, n_lidar_rays=4, action_alpha=0.25)
    env = env.reset(env, jax.random.key(9))
    action = jnp.ones((2, *env.action_space_shape))
    for value in (0.25, 0.4375, 0.578125):
        env = env.step(env, action)
        np.testing.assert_allclose(env.env_params["applied_action"], value)
    env = env.reset(env, jax.random.key(10))
    np.testing.assert_array_equal(env.env_params["applied_action"], 0.0)
