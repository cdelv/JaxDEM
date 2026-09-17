# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""CAPS respects action transforms, recurrent history, and PPO boundaries."""

from dataclasses import replace

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jaxdem.rl.action_spaces import ActionSpace
from jaxdem.rl.models import Model
from jaxdem.rl.trainers import TrajectoryData
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from tests.test_ppo_alignment import _loss_args
from tests.test_rl_action_contracts import ActionLookupEnv

MODELS = ["ActorCritic", "SharedActorCritic", "LSTMActorCritic", "MinGRUActorCritic"]
SPACES = ["Free", "Box", "MaxNorm", "discrete"]


def make_model(name, space, obs_dim=2, action_dim=2):
    kwargs = {}
    if name == "ActorCritic":
        kwargs.update(actor_architecture=[4], critic_architecture=[4])
    elif name == "SharedActorCritic":
        kwargs.update(architecture=[4])
    elif name == "LSTMActorCritic":
        kwargs.update(hidden_features=4, lstm_features=4, remat=True)
    else:
        kwargs.update(hidden_features=3, gru_features=4, num_layers=2)
    if space == "discrete":
        kwargs["discrete"] = True
    elif space == "Box":
        kwargs["action_space"] = ActionSpace.create("Box", x_min=-2.0, x_max=3.0)
    else:
        kwargs["action_space"] = ActionSpace.create(space)
    return Model.create(
        name,
        observation_space_size=obs_dim,
        action_space_size=action_dim,
        key=nnx.Rngs(10),
        **kwargs,
    )


def assert_trees_close(actual, expected, **kwargs):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, **kwargs)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("space", SPACES)
def test_spatial_forward_and_gradients_match_independent_same_history_probes(
    name, space
):
    model = make_model(name, space)
    obs = jax.random.normal(jax.random.key(11), (4, 2, 2))
    perturbed = obs + 0.1 * jax.random.normal(jax.random.key(12), obs.shape)
    done = jnp.array([[False, True], [False, False], [True, False], [False, False]])
    model.reset(obs[0].shape)
    # Exercise nonzero incoming carry and resets within the sequence.
    if name == "LSTMActorCritic":
        model.c[...] = jnp.full_like(model.c[...], 0.2)
        model.h[...] = jnp.full_like(model.h[...], 0.3)
    elif name == "MinGRUActorCritic":
        model.h[...] = jnp.full_like(model.h[...], 0.3)
    graph, state = nnx.split(model)

    def evaluate(state, paired):
        current = nnx.merge(graph, state, copy=True)
        if paired:
            pi, value, noisy = current.policy_with_perturbation(
                obs,
                perturbed,
                initial_carry=current.carry,
                done=done,
            )
            clean = current.policy_output(pi)
        else:
            clean, values, noisy = [], [], []
            for x, x_noisy, boundary in zip(obs, perturbed, done, strict=True):
                # Probe this observation with clean history, then discard its
                # carry. Only the clean observation advances the next step.
                probe = nnx.merge(graph, nnx.state(current), copy=True)
                noisy_pi, _ = probe(x_noisy)
                noisy.append(probe.policy_output(noisy_pi))
                pi, value = current(x)
                clean.append(current.policy_output(pi))
                values.append(value)
                current.reset(x.shape, mask=boundary)
            clean, value, noisy = map(jnp.stack, (clean, values, noisy))
        loss = jnp.sum((noisy - clean) ** 2) + 0.1 * jnp.sum(value**2)
        return loss, (clean, value, noisy)

    with jax.default_matmul_precision("highest"):
        actual = jax.jit(jax.value_and_grad(evaluate, has_aux=True), static_argnums=1)(
            state, True
        )
        expected = jax.value_and_grad(evaluate, has_aux=True)(state, False)
    assert_trees_close(actual, expected, rtol=3e-5, atol=3e-7)
    # Paired evaluation itself must leave persistent rollout state untouched.
    before = nnx.state(model)
    pi, value, noisy = model.policy_with_perturbation(
        obs, obs, initial_carry=model.carry, done=done
    )
    assert_trees_close(noisy, model.policy_output(pi), rtol=3e-5, atol=1e-7)
    assert_trees_close(nnx.state(model), before, rtol=0, atol=0)


class LinearPolicy(Model):
    def __init__(self):
        self.slope = nnx.Param(jnp.array(0.7))

    def __call__(self, obs, sequence=False, **kwargs):
        mean = obs * self.slope[...]
        return distrax.MultivariateNormalDiag(mean, jnp.ones_like(mean)), jnp.zeros(
            (*obs.shape[:-1], 1)
        )


def trajectory(obs, done=None, active=None):
    shape = obs.shape[:-1]
    zeros = jnp.zeros(shape)
    done = jnp.zeros(shape, dtype=bool) if done is None else done
    active = jnp.ones(shape, dtype=bool) if active is None else active
    return TrajectoryData(
        obs=obs,
        action=jnp.zeros_like(obs),
        value=zeros,
        log_prob=zeros,
        ratio=jnp.ones_like(zeros),
        reward=zeros,
        done=done,
        terminated=done,
        truncated=jnp.zeros_like(done),
        agent_mask=active,
        bootstrap_value=zeros,
    )


def loss_args(td):
    args = _loss_args()
    args.update(last_value=jnp.zeros(td.obs.shape[1]), ppo_value_coeff=jnp.array(0.0))
    return args


@pytest.mark.parametrize("empty", [False, True])
def test_temporal_masks_resets_inactive_agents_and_partial_minibatches(empty):
    obs = jnp.array(
        [[[0.0], [0.0]], [[2.0], [100.0]], [[50.0], [200.0]], [[55.0], [300.0]]]
    )
    done = jnp.array([[False, True], [True, False], [False, False], [False, False]])
    active = jnp.array([[True, True], [True, False], [True, True], [True, True]])
    td = trajectory(obs, done, active)
    selected = jnp.array([[True, True], [True, True], [True, False], [False, True]])
    if empty:
        selected = jnp.zeros_like(selected)
    # Only 0 -> 2 and 50 -> 55 are valid selected pairs. The latter uses
    # an unselected successor as context. No cross-reset or inactive pair.
    expected = 0.0 if empty else (2**2 + 5**2) / 2
    model = LinearPolicy()
    (loss, aux), grad = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model,
        td,
        **loss_args(td),
        loss_mask=selected,
        caps_temporal_coeff=0.3,
    )
    np.testing.assert_allclose(aux["caps_temporal_loss"], 0.7**2 * expected)
    np.testing.assert_allclose(loss, 0.3 * 0.7**2 * expected)
    np.testing.assert_allclose(grad.slope[...], 0.3 * 2 * 0.7 * expected)


def test_spatial_loss_and_gradient_use_per_feature_noise_and_selected_observations():
    model = LinearPolicy()
    obs = jnp.zeros((3, 2, 2))
    td = trajectory(obs)
    selected = jnp.array([[True, False], [True, True], [False, False]])
    key = jax.random.key(13)
    sigma = jnp.array([0.2, 0.0])
    noise = jax.random.normal(key, obs.shape) * sigma
    expected = jnp.sum(jnp.where(selected[..., None], noise**2, 0.0)) / selected.sum()
    (loss, aux), grad = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model,
        td,
        **loss_args(td),
        loss_mask=selected,
        caps_spatial_coeff=0.4,
        caps_noise_std=sigma,
        caps_key=key,
    )
    np.testing.assert_allclose(aux["caps_spatial_loss"], 0.7**2 * expected)
    np.testing.assert_allclose(loss, 0.4 * 0.7**2 * expected)
    np.testing.assert_allclose(grad.slope[...], 0.4 * 2 * 0.7 * expected)


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("space", SPACES)
def test_both_caps_terms_train_with_all_models_and_action_spaces(name, space):
    model = make_model(
        name, space, obs_dim=1, action_dim=3 if space == "discrete" else 1
    )
    tr = PPOTrainer.Create(
        ActionLookupEnv.Create(),
        model,
        num_envs=1,
        num_steps_epoch=3,
        num_minibatches=1,
        learning_rate=1e-3,
        caps_temporal_coeff=0.1,
        caps_spatial_coeff=0.2,
        caps_noise_std=[0.1],
    )
    before = tr.graphstate
    tr, _, metrics = tr.epoch(tr, jnp.array(0))
    jax.block_until_ready(metrics)
    assert metrics["caps_temporal_loss"] >= 0
    assert metrics["caps_spatial_loss"] >= 0
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(tr.graphstate))
    )


def test_disabled_caps_does_not_evaluate_policy_outputs_or_spatial_branch():
    class NoCapsPolicy(LinearPolicy):
        @staticmethod
        def policy_output(pi):
            raise AssertionError("disabled CAPS must not transform outputs")

        def policy_with_perturbation(self, *args, **kwargs):
            raise AssertionError("disabled CAPS must not evaluate perturbations")

    td = trajectory(jnp.zeros((1, 1, 1)))
    model = NoCapsPolicy()
    (_, aux), grad = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model, td, **loss_args(td)
    )
    assert aux["caps_temporal_loss"] == aux["caps_spatial_loss"] == 0
    assert np.isfinite(grad.slope[...])


def test_zero_weights_preserve_default_epoch_and_rng():
    model = make_model("SharedActorCritic", "Free", obs_dim=1, action_dim=1)
    tr = PPOTrainer.Create(
        ActionLookupEnv.Create(),
        model,
        num_envs=1,
        num_steps_epoch=2,
        num_minibatches=1,
    )
    explicit = replace(
        tr,
        caps_temporal_coeff=0.0,
        caps_spatial_coeff=0.0,
        caps_noise_std=jnp.array([100.0]),
    )
    a, td_a, metrics_a = tr.epoch(tr, jnp.array(0))
    b, td_b, metrics_b = explicit.epoch(explicit, jnp.array(0))
    np.testing.assert_array_equal(
        jax.random.key_data(a.key), jax.random.key_data(b.key)
    )
    assert_trees_close(
        (a.graphstate, td_a, metrics_a), (b.graphstate, td_b, metrics_b), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"caps_temporal_coeff": -1.0},
        {"caps_spatial_coeff": float("nan")},
        {"caps_temporal_coeff": float("inf")},
        {"caps_noise_std": -0.1},
        {"caps_noise_std": [0.1, 0.2, 0.3]},
        {"caps_noise_std": [[0.1, 0.2]]},
    ],
)
def test_caps_configuration_validation(kwargs):
    model = make_model("SharedActorCritic", "Free")
    with pytest.raises(ValueError, match="caps_"):
        PPOTrainer.Create(ActionLookupEnv.Create(), model, **kwargs)


def test_single_step_rollout_and_zero_noise_have_finite_zero_caps_gradients():
    model = LinearPolicy()
    td = trajectory(jnp.zeros((1, 1, 1)))
    (loss, aux), grad = nnx.value_and_grad(PPOTrainer.loss_fn, has_aux=True)(
        model,
        td,
        **loss_args(td),
        caps_temporal_coeff=1.0,
        caps_spatial_coeff=1.0,
        caps_noise_std=0.0,
        caps_key=jax.random.key(0),
    )
    assert loss == aux["caps_temporal_loss"] == aux["caps_spatial_loss"] == 0
    assert grad.slope[...] == 0
