"""Single-agent actuators smooth every accepted physics step, across checkpoints."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import vectorise_env
from jaxdem.utils.environment import advance_action


@pytest.mark.parametrize("name, kwargs", [
    ("singleNavigator", {"dim": 2}),
    ("singleNavigator", {"dim": 3}),
    ("SingleRoller", {}),
])
@pytest.mark.parametrize("alpha", [0.22, 0.1, 1.0, 0.0])
def test_step_response_reversal_and_physics(name, kwargs, alpha):
    env = Environment.create(name, action_alpha=alpha, **kwargs)
    reference = Environment.create(name, action_alpha=1.0, **kwargs)
    key = jax.random.key(42)
    env = env.reset(env, key)
    reference = reference.reset(reference, key)
    action = jnp.ones((1, env.action_space_size))
    np.testing.assert_array_equal(env.env_params["applied_action"], 0.0)

    # A held command converges each physics step, then reverses from its
    # actual applied value even when a new reward checkpoint is taken.
    env, _, _ = advance_action(env, action, skip_frames=3)
    first_applied = 1 - (1 - alpha) ** 4
    np.testing.assert_allclose(env.env_params["applied_action"], first_applied)
    checkpoint = env.checkpoint(env, -action)
    np.testing.assert_array_equal(checkpoint.env_params["applied_action"], env.env_params["applied_action"])
    env, _, _ = advance_action(checkpoint, -action, skip_frames=5)
    final_applied = -1 + (first_applied + 1) * (1 - alpha) ** 6
    np.testing.assert_allclose(env.env_params["applied_action"], final_applied, atol=1e-14)
    np.testing.assert_array_equal(env.env_params["prev_dist"], checkpoint.env_params["prev_dist"])

    # Compare actual force/torque dynamics with explicitly supplied analytic
    # commands through an unsmoothed actuator, including the same damping.
    values = [1 - (1 - alpha) ** n for n in range(1, 5)]
    values += [-1 + (first_applied + 1) * (1 - alpha) ** n for n in range(1, 7)]
    commands = jnp.asarray(values)[:, None, None] * action

    @jax.jit
    def replay(e, commands):
        def step(e, command):
            return e.step(e, command), None
        return jax.lax.scan(step, e, commands)[0]

    reference = replay(reference, commands)
    for field in ("pos_c", "vel", "ang_vel", "force", "torque"):
        np.testing.assert_allclose(getattr(env.state, field), getattr(reference.state, field), atol=1e-12)
    reset = env.reset(env, key)
    np.testing.assert_array_equal(reset.env_params["applied_action"], 0.0)
    np.testing.assert_array_equal(reset.env_params["action_alpha"], alpha)


@pytest.mark.parametrize("name", ["singleNavigator", "SingleRoller"])
def test_default_smoothing_stops_at_each_batched_episode_boundary(name):
    env = vectorise_env(Environment.create(name), n=3)
    env = env.reset(env, jax.random.split(jax.random.key(43), 3))
    env.env_params["max_steps"] = jnp.array([2, 4, 100])
    action = jnp.ones((3, 1, env.action_space_size))
    final, terminated, truncated = advance_action(env, action, skip_frames=4)
    expected = jnp.broadcast_to((1 - 0.78 ** jnp.array([2, 4, 5]))[:, None, None], action.shape)
    np.testing.assert_allclose(final.env_params["applied_action"], expected)
    continued, _, _ = advance_action(final, -action, skip_frames=4,
                                    terminated=terminated, truncated=truncated)
    np.testing.assert_array_equal(continued.env_params["applied_action"][:2], final.env_params["applied_action"][:2])
    np.testing.assert_allclose(continued.env_params["applied_action"][2], -1 + (expected[2] + 1) * 0.78 ** 5)


@pytest.mark.parametrize("name", ["singleNavigator", "SingleRoller"])
@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_invalid_smoothing_fraction(name, alpha):
    with pytest.raises(ValueError, match="action_alpha"):
        Environment.create(name, action_alpha=alpha)
