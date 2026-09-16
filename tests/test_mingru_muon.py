"""Stacked MinGRU updates equal independent Optax Muon layer updates."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from jaxdem.rl.models import MinGRUActorCritic
from jaxdem.rl.trainers.ppo_trainer import _build_optimizer


@pytest.mark.parametrize('use_nnx', [False, True])
def test_stacked_mingru_matches_independent_optax_layers(use_nnx):
    model = MinGRUActorCritic(2, 1, nnx.Rngs(3), hidden_features=4, gru_features=4, num_layers=2)
    params = nnx.state(model, nnx.Param) if use_nnx else {'mingru_kernel': model.mingru_kernel[...]}
    tx = _build_optimizer(optax.contrib.muon, .01, float('inf'), 1)
    state = tx.init(params)
    stacked = model.mingru_kernel[...]
    layer_txs = [optax.contrib.muon(.01, eps=1e-12) for _ in range(2)]
    layer_states = [t.init(p) for t,p in zip(layer_txs, stacked)]
    for step in range(3):
        grads = jax.tree.map(lambda x: jnp.sin(jnp.arange(x.size).reshape(x.shape)+step).astype(x.dtype), params)
        updates, state = tx.update(grads, state, params)
        actual = updates['mingru_kernel'][...] if use_nnx else updates['mingru_kernel']
        kernel_grad = grads['mingru_kernel'][...] if use_nnx else grads['mingru_kernel']
        expected = []
        for i, layer_tx in enumerate(layer_txs):
            update, layer_states[i] = layer_tx.update(kernel_grad[i], layer_states[i], stacked[i])
            expected.append(update)
        np.testing.assert_allclose(actual, jnp.stack(expected), rtol=2e-5, atol=2e-6)
        params = optax.apply_updates(params, updates)
        stacked = stacked + jnp.stack(expected)


def test_muon_user_dimension_override_is_preserved():
    params = {'mingru_kernel': jnp.ones((2, 4, 12))}
    factory = partial(optax.contrib.muon, muon_weight_dimension_numbers=lambda p: jax.tree.map(lambda _: None, p))
    wrapped = _build_optimizer(factory, .01, float('inf'), 1)
    direct = factory(.01, eps=1e-12)
    a, _ = wrapped.update(params, wrapped.init(params), params)
    b, _ = direct.update(params, direct.init(params), params)
    np.testing.assert_allclose(a['mingru_kernel'], b['mingru_kernel'])
