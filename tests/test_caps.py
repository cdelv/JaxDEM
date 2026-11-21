# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""CAPS respects action transforms, recurrent history, and PPO boundaries."""

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
from tests.test_ppo_objective import _loss_args


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
    elif space == "Blockwise":
        kwargs["action_space"] = ActionSpace.create(
            "Blockwise",
            blocks=(
                (1, ActionSpace.create("MaxNorm", max_norm=1.7)),
                (action_dim - 1, ActionSpace.create("Box", x_min=-2.0, x_max=3.0)),
            ),
        )
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


@pytest.mark.parametrize(
    "name,space",
    [
        ("ActorCritic", "Box"),
        ("SharedActorCritic", "Blockwise"),
        ("LSTMActorCritic", "MaxNorm"),
        ("MinGRUActorCritic", "Free"),
        ("MinGRUActorCritic", "discrete"),
    ],
)
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
    # Only 0 -> 2 and 50 -> 55 are valid selected pairs.
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


def test_enabled_caps_epoch_matches_the_exact_linear_sgd_update(monkeypatch):
    import optax
    from tests.test_ppo_math import TimeLimitCounterEnv

    def sgd(learning_rate, eps):
        return optax.sgd(learning_rate)

    trainer = PPOTrainer.Create(
        TimeLimitCounterEnv.Create(),
        LinearPolicy(),
        num_envs=1,
        num_steps_epoch=3,
        num_minibatches=1,
        optimizer=sgd,
        learning_rate=0.01,
        anneal_learning_rate=False,
        max_grad_norm=float("inf"),
        ppo_value_coeff=0.0,
        ppo_entropy_coeff=0.0,
        caps_temporal_coeff=0.3,
        caps_spatial_coeff=0.4,
        caps_noise_std=0.2,
    )
    td = trajectory(jnp.arange(3.0).reshape(3, 1, 1, 1))
    rollout_key = jax.random.PRNGKey(73)

    def rollout(env, graphdef, graphstate, key, *args, **kwargs):
        return env, graphstate, rollout_key, td

    monkeypatch.setattr(PPOTrainer, "trajectory_rollout", staticmethod(rollout))
    updated, _, metrics = jax.jit(PPOTrainer.epoch.__wrapped__)(trainer, jnp.array(0))
    next_key, noise_root = jax.random.split(rollout_key)
    noise = 0.2 * jax.random.normal(jax.random.fold_in(noise_root, 0), (3, 1, 1))
    mean_square_noise = np.mean(np.asarray(noise) ** 2)
    expected = 0.7 - 0.01 * 2 * 0.7 * (0.3 + 0.4 * mean_square_noise)
    np.testing.assert_allclose(updated.model.slope[...], expected, rtol=2e-6)
    np.testing.assert_allclose(metrics["caps_temporal_loss"], 0.7**2, rtol=2e-6)
    np.testing.assert_allclose(
        metrics["caps_spatial_loss"], 0.7**2 * mean_square_noise, rtol=2e-6
    )
    np.testing.assert_array_equal(updated.key, next_key)


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


def test_single_step_rollout_and_zero_noise_have_finite_zero_caps_gradients():
    from tests.test_ppo_math import BanditModel, TimeLimitCounterEnv

    # Custom models without observation_space_size accept per-feature caps_noise_std.
    trainer = PPOTrainer.Create(
        TimeLimitCounterEnv.Create(),
        BanditModel(),
        num_envs=1,
        num_steps_epoch=2,
        num_minibatches=1,
        caps_spatial_coeff=0.1,
        caps_noise_std=[0.1],
    )
    np.testing.assert_allclose(trainer.caps_noise_std, [0.1])

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
