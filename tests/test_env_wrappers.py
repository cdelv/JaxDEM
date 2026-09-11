"""Wrapper composition and batch-one counterexamples."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdem.rl.environments import Environment
from jaxdem.rl.env_wrappers import clip_action_env, is_wrapped, unwrap, vectorise_env


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CounterEnvironment(Environment):
    @staticmethod
    def step(env, action):
        return replace(env, state=replace(env.state, pos_c=env.state.pos_c + action))

    @staticmethod
    def reset(env, key):
        return replace(
            env, state=replace(env.state, pos_c=jnp.zeros_like(env.state.pos_c))
        )

    @staticmethod
    def truncated(env):
        return env.state.pos_c[0, 0] >= 1

    @staticmethod
    def observation(env):
        return env.state.pos_c


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("clip_first", [True, False])
def test_wrappers_compose_and_keep_inherited_methods_scalar(batch_size, clip_first):
    env = CounterEnvironment.Create()
    if clip_first:
        env = vectorise_env(clip_action_env(env), n=batch_size)
    else:
        env = clip_action_env(vectorise_env(env, n=batch_size))
    action = jnp.full((batch_size, 1, 2), 10.0)
    env = jax.jit(lambda e: e.step(e, action))(env)
    np.testing.assert_array_equal(env.observation(env), jnp.ones((batch_size, 1, 2)))
    np.testing.assert_array_equal(env.done(env), jnp.ones(batch_size, dtype=bool))
    env = env.reset_if_done(
        env, env.done(env), jax.random.split(jax.random.key(0), batch_size)
    )
    np.testing.assert_array_equal(env.observation(env), jnp.zeros((batch_size, 1, 2)))
    assert is_wrapped(env)
    base = unwrap(env)
    assert type(base) is CounterEnvironment
    np.testing.assert_array_equal(base.state.pos_c, env.state.pos_c)


def test_equivalent_wrappers_share_compilation_identity():
    env = CounterEnvironment.Create()
    assert type(vectorise_env(env, n=1)) is type(vectorise_env(env, n=2))
    assert type(clip_action_env(env, -2, 2)) is type(clip_action_env(env, -2, 2))
    assert type(clip_action_env(env, -2, 2)) is not type(clip_action_env(env, -1, 1))


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_vectorization_rejects_invalid_batch_sizes(n):
    with pytest.raises(ValueError, match="positive Python integer"):
        vectorise_env(CounterEnvironment.Create(), n=n)
