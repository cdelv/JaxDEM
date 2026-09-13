import math
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
import distrax
import optax
import pytest
from flax import nnx

from jaxdem.rl.trainers import Trainer, TrajectoryData
from jaxdem.rl.trainers.ppo_trainer import (
    PPOTrainer,
    _build_optimizer,
    _epoch_learning_rate_schedule,
    _last_occurrence_indices,
    _priority_probabilities,
)
from jaxdem.rl.environments import Environment
from jaxdem.rl.models import Model
from jaxdem.system import System
from jaxdem.state import State
from jaxdem.rl.action_spaces import ActionSpace
from jaxdem.rl.env_wrappers import vectorise_env


def test_ppo_learning_rate_schedule_uses_epochs():
    schedule = _epoch_learning_rate_schedule(1.0, num_epochs=4, updates_per_epoch=2)

    values = jax.vmap(schedule)(jnp.arange(8))
    assert jnp.allclose(values[::2], values[1::2])
    assert values[0] > values[2] > values[4] > values[6]


def test_ppo_schedule_continues_from_accumulated_optimizer_state():
    def sgd_factory(learning_rate, eps):
        del eps
        return optax.sgd(learning_rate)

    schedule = _epoch_learning_rate_schedule(1.0, num_epochs=4, updates_per_epoch=2)
    tx = _build_optimizer(sgd_factory, schedule, float("inf"), 2)
    params = jnp.array(0.0)
    state = tx.init(params)
    emitted = []
    for _ in range(4):
        update, state = tx.update(jnp.array(1.0), state, params)
        params = optax.apply_updates(params, update)
        emitted.append(update)

    # Continuing with this restored state resumes at the next optimizer update.
    resumed_state = state
    for _ in range(2):
        update, resumed_state = tx.update(jnp.array(1.0), resumed_state, params)
        params = optax.apply_updates(params, update)
        emitted.append(update)

    assert jnp.allclose(jnp.array(emitted[:4]), jnp.array([0.0, -1.0, 0.0, -1.0]))
    assert emitted[4] == 0.0
    assert jnp.allclose(emitted[5], -schedule(jnp.array(2)))


def test_ppo_accumulates_raw_gradient_mean_before_clipping():
    accumulated = _build_optimizer(optax.adam, 1.0, 1.0, 2)
    reference = _build_optimizer(optax.adam, 1.0, 1.0, 1)
    accumulated_params = jnp.zeros(2)
    reference_params = jnp.zeros(2)
    accumulated_state = accumulated.init(accumulated_params)
    reference_state = reference.init(reference_params)

    gradient_groups = (
        (jnp.array([10.0, -8.0]), jnp.array([-8.0, 10.0])),
        (jnp.array([4.0, 0.0]), jnp.array([0.0, 2.0])),
    )
    for gradient_group in gradient_groups:
        first_update, accumulated_state = accumulated.update(
            gradient_group[0], accumulated_state, accumulated_params
        )
        second_update, accumulated_state = accumulated.update(
            gradient_group[1], accumulated_state, accumulated_params
        )
        mean_gradient = (gradient_group[0] + gradient_group[1]) / 2
        reference_update, reference_state = reference.update(
            mean_gradient, reference_state, reference_params
        )

        assert jnp.all(first_update == 0.0)
        assert jnp.allclose(second_update, reference_update)
        accumulated_params = optax.apply_updates(accumulated_params, second_update)
        reference_params = optax.apply_updates(reference_params, reference_update)

    assert jnp.allclose(accumulated_params, reference_params)
    inner_leaves = jax.tree.leaves(accumulated_state.inner_opt_state)
    reference_leaves = jax.tree.leaves(reference_state)
    assert len(inner_leaves) == len(reference_leaves)
    assert all(jnp.allclose(x, y) for x, y in zip(inner_leaves, reference_leaves))


def test_ppo_priority_probabilities_are_normalized():
    uniform = _priority_probabilities(jnp.zeros(4), jnp.array(0.8))
    unequal = _priority_probabilities(jnp.array([0.0, 1.0, 9.0]), jnp.array(1.0))

    assert jnp.allclose(uniform, jnp.full(4, 0.25))
    assert jnp.all(unequal > 0.0)
    assert jnp.allclose(unequal[1:], jnp.array([0.1, 0.9]), atol=1e-6)
    assert jnp.allclose(unequal.sum(), 1.0)

    uniform_weights = jnp.power(4 * uniform, -1.0)
    assert jnp.allclose(uniform_weights, jnp.ones(4))

    values = jnp.array([2.0, 5.0, 11.0])
    correction = jnp.power(3 * unequal, -1.0)
    corrected_expectation = jnp.sum(unequal * correction * values)
    assert jnp.allclose(corrected_expectation, values.mean())


def test_ppo_priority_probabilities_are_stable_in_float32():
    dtype = jnp.float32

    dominant = _priority_probabilities(
        jnp.array([1.0, 1e30], dtype=dtype), jnp.array(2.0, dtype=dtype)
    )
    equal_large = _priority_probabilities(
        jnp.array([2e38, 2e38], dtype=dtype), jnp.array(1.0, dtype=dtype)
    )
    zero_priority = _priority_probabilities(
        jnp.array([0.0, 1.0], dtype=dtype), jnp.array(1.0, dtype=dtype)
    )
    alpha_zero = _priority_probabilities(
        jnp.array([0.0, 1.0, 1e30], dtype=dtype), jnp.array(0.0, dtype=dtype)
    )

    assert dominant.dtype == dtype
    assert dominant[1] > dominant[0] > 0.0
    assert jnp.allclose(dominant.sum(), 1.0)
    assert jnp.all(jnp.isfinite(jnp.power(2 * dominant, -1.0)))
    assert jnp.allclose(equal_large, jnp.array([0.5, 0.5], dtype=dtype))
    assert zero_priority[0] > 0.0
    expected_zero = jnp.array([1e-6, 1.0 + 1e-6], dtype=dtype) / (1.0 + 2e-6)
    assert jnp.allclose(zero_priority, expected_zero)
    assert jnp.allclose(alpha_zero, jnp.full(3, 1.0 / 3.0, dtype=dtype))


def test_ppo_priority_probabilities_are_invariant_to_inactive_padding():
    priority = jnp.array([2.0, 5.0], dtype=jnp.float32)
    base = _priority_probabilities(priority, jnp.array(0.7))
    padded = _priority_probabilities(
        jnp.array([2.0, 5.0, 1e30, 1e30], dtype=jnp.float32),
        jnp.array(0.7),
        jnp.array([True, True, False, False]),
    )

    assert jnp.allclose(padded[:2], base)
    assert jnp.array_equal(padded[2:], jnp.zeros(2))


def test_ppo_duplicate_segment_writeback_is_coherent():
    current = jnp.array([[10.0, 20.0, 30.0]])
    sampled = jnp.array([[2.0, 4.0, 8.0]])
    write_idx = _last_occurrence_indices(jnp.array([1, 1, 2]), size=3)
    result = current.at[:, write_idx].set(sampled, mode="drop")

    assert jnp.array_equal(write_idx, jnp.array([3, 1, 2]))
    assert jnp.allclose(result, jnp.array([[10.0, 4.0, 8.0]]))


def test_gae_analytical():
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.array([[0.5], [0.5]])
    dones = jnp.array([[0.0], [1.0]])
    ratios = jnp.array([[1.0], [1.0]])

    returns, advantages = Trainer.compute_advantages(
        value=values,
        reward=rewards,
        ratio=ratios,
        done=dones,
        advantage_rho_clip=jnp.array(1.0),
        advantage_c_clip=jnp.array(1.0),
        advantage_gamma=jnp.array(1.0),
        advantage_lambda=jnp.array(1.0),
        last_value=jnp.array([0.0]),
        unroll=1,
    )

    expected_advantages = jnp.array([[2.5], [1.5]])
    assert jnp.allclose(advantages, expected_advantages)
    assert jnp.allclose(returns, expected_advantages + values)


def test_truncation_bootstraps_but_termination_does_not():
    common = dict(
        value=jnp.array([[2.0]]),
        reward=jnp.array([[1.0]]),
        ratio=jnp.ones((1, 1)),
        done=jnp.ones((1, 1), dtype=bool),
        advantage_rho_clip=jnp.array(1.0),
        advantage_c_clip=jnp.array(1.0),
        advantage_gamma=jnp.array(1.0),
        advantage_lambda=jnp.array(1.0),
        last_value=jnp.array([99.0]),
        bootstrap_value=jnp.array([[5.0]]),
        unroll=1,
    )
    _, terminal_advantage = Trainer.compute_advantages(
        **common,
        terminated=jnp.ones((1, 1), dtype=bool),
        truncated=jnp.zeros((1, 1), dtype=bool),
    )
    _, truncated_advantage = Trainer.compute_advantages(
        **common,
        terminated=jnp.zeros((1, 1), dtype=bool),
        truncated=jnp.ones((1, 1), dtype=bool),
    )
    assert jnp.allclose(terminal_advantage, jnp.array([[-1.0]]))
    assert jnp.allclose(truncated_advantage, jnp.array([[4.0]]))


def test_inactive_agents_do_not_contribute_advantages_or_sampling_support():
    mask = jnp.array([[True, False], [True, False]])
    _, advantage = Trainer.compute_advantages(
        value=jnp.zeros((2, 2)),
        reward=jnp.array([[1.0, 1e6], [1.0, 1e6]]),
        ratio=jnp.ones((2, 2)),
        done=jnp.zeros((2, 2), dtype=bool),
        advantage_rho_clip=jnp.array(1.0),
        advantage_c_clip=jnp.array(1.0),
        advantage_gamma=jnp.array(1.0),
        advantage_lambda=jnp.array(1.0),
        last_value=jnp.zeros(2),
        agent_mask=mask,
        unroll=1,
    )
    probabilities = _priority_probabilities(
        jnp.sum(jnp.abs(advantage), axis=0), jnp.array(1.0), jnp.any(mask, axis=0)
    )
    assert jnp.allclose(advantage[:, 1], 0.0)
    assert jnp.array_equal(probabilities, jnp.array([1.0, 0.0]))


def test_policy_heads_do_not_create_unused_sigma_parameters():
    from jaxdem.rl.models.mlp import SharedActorCritic

    discrete = SharedActorCritic(
        observation_space_size=2,
        action_space_size=3,
        key=nnx.Rngs(0),
        discrete=True,
    )
    continuous_head = SharedActorCritic(
        observation_space_size=2,
        action_space_size=1,
        key=nnx.Rngs(1),
        actor_sigma_head=True,
    )
    assert not hasattr(discrete, "actor_sigma")
    assert not hasattr(continuous_head.actor_sigma, "log_std")


def test_recurrent_reset_masks_individual_agents():
    from jaxdem.rl.models.lstm import LSTMActorCritic
    from jaxdem.rl.models.mingru import MinGRUActorCritic

    mask = jnp.array([[False, True, False], [True, False, True]])
    models = [
        LSTMActorCritic(2, 1, nnx.Rngs(2), carry_leading_shape=(2, 3)),
        MinGRUActorCritic(2, 1, nnx.Rngs(3), carry_leading_shape=(2, 3)),
    ]
    for model in models:
        assert not hasattr(model, "actor_mu")
        assert not hasattr(model, "critic")
        if hasattr(model, "c"):
            model.c.value = jnp.ones_like(model.c.value)
        model.h.value = jnp.ones_like(model.h.value)
        model.reset((2, 3, 2), mask=mask)
        assert jnp.all(model.h.value[mask] == 0.0)
        assert jnp.all(model.h.value[~mask] == 1.0)


@pytest.mark.parametrize("model_name", ["lstm", "mingru"])
def test_recurrent_replay_matches_online_rollout_across_inactive_gap(model_name):
    # Sequence replay and single-step rollout use different GPU kernels. Highest
    # precision keeps this test focused on boundary placement rather than TF32
    # accumulation differences between those kernels.
    with jax.default_matmul_precision("highest"):
        if model_name == "lstm":
            from jaxdem.rl.models.lstm import LSTMActorCritic

            model_type = LSTMActorCritic
        else:
            from jaxdem.rl.models.mingru import MinGRUActorCritic

            model_type = MinGRUActorCritic

        online = model_type(1, 1, nnx.Rngs(17), carry_leading_shape=(1,))
        replay = model_type(1, 1, nnx.Rngs(17), carry_leading_shape=(1,))
        observations = jnp.array([[[0.4]], [[9.0]], [[0.4]]])
        agent_mask = jnp.array([[True], [False], [True]])
        recorded_done = jnp.array([[True], [False], [False]])
        replay_boundaries = recorded_done | ~agent_mask

        online_values = []
        for observation, boundary in zip(observations, replay_boundaries):
            _, value = online(observation, sequence=False)
            online_values.append(value)
            online.reset(observation.shape, mask=boundary)

        zeros = jnp.zeros((3, 1))
        td = TrajectoryData(
            obs=observations,
            action=jnp.zeros((3, 1, 1)),
            value=zeros,
            log_prob=zeros,
            ratio=jnp.ones_like(zeros),
            reward=zeros,
            done=recorded_done,
            terminated=recorded_done,
            truncated=jnp.zeros_like(recorded_done),
            agent_mask=agent_mask,
            bootstrap_value=zeros,
        )
        _, aux = PPOTrainer.loss_fn(
            replay,
            td,
            returns=zeros,
            advantage=zeros,
            ppo_clip_eps=jnp.array(0.2),
            ppo_value_coeff=jnp.array(0.5),
            ppo_entropy_coeff=jnp.array(0.0),
            initial_carry=replay.carry,
        )
        assert jnp.allclose(aux["value"], jnp.squeeze(jnp.stack(online_values), -1))


def test_clipped_surrogate_objective():
    def clipped_actor_loss(ratio, advantage, eps):
        ratio_bounded = jnp.where(
            advantage >= 0,
            jnp.minimum(ratio, 1.0 + eps),
            jnp.maximum(ratio, 1.0 - eps),
        )
        return -(advantage * ratio_bounded)

    eps = 0.2

    l1 = clipped_actor_loss(jnp.array(1.1), jnp.array(1.0), eps)
    assert jnp.allclose(l1, -1.1)

    l2 = clipped_actor_loss(jnp.array(1.5), jnp.array(1.0), eps)
    assert jnp.allclose(l2, -1.2)

    l3 = clipped_actor_loss(jnp.array(0.5), jnp.array(-1.0), eps)
    assert jnp.allclose(l3, 0.8)


def test_entropy_continuous():
    k = 3
    mu = jnp.zeros(k)
    sigma = jnp.ones(k)

    dist = distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
    entropy = dist.entropy()

    expected_entropy = (k / 2.0) * math.log(2 * math.pi * math.e)
    assert jnp.allclose(entropy, expected_entropy)


def test_gradient_detachment():
    def advantage_fn(values, rewards):
        dones = jnp.zeros_like(rewards)
        ratios = jnp.ones_like(rewards)
        ret, adv = Trainer.compute_advantages(
            value=values,
            reward=rewards,
            ratio=ratios,
            done=dones,
            advantage_rho_clip=jnp.array(1.0),
            advantage_c_clip=jnp.array(1.0),
            advantage_gamma=jnp.array(1.0),
            advantage_lambda=jnp.array(1.0),
            last_value=jnp.array([0.0]),
            unroll=1,
        )
        return ret.sum() + adv.sum()

    values = jnp.array([[0.5], [0.5]])
    rewards = jnp.array([[1.0], [2.0]])

    grads = jax.grad(advantage_fn, argnums=(0, 1))(values, rewards)

    assert jnp.all(grads[0] == 0.0)
    assert jnp.all(grads[1] == 0.0)


@Model.register("BanditModel")
class BanditModel(Model):
    def __init__(
        self, observation_space_size=1, action_space_size=1, action_space=None, key=None
    ):
        self.action_mean = nnx.Param(jnp.zeros((1,)))
        self.value_bias = nnx.Param(jnp.zeros((1,)))

    def __call__(self, x, sequence=False, **kwargs):
        mean = jnp.broadcast_to(self.action_mean[...], (*x.shape[:-1], 1))
        sigma = jnp.broadcast_to(jnp.array([0.1]), (*x.shape[:-1], 1))
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        val = jnp.broadcast_to(self.value_bias[...], (*x.shape[:-1], 1))
        return dist, val


@Environment.register("TimeLimitCounter")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class TimeLimitCounterEnv(Environment):
    @classmethod
    def Create(cls, dim=2):
        state = State.create(pos=jnp.zeros((1, dim)))
        return cls(
            state=state,
            system=System.create(state.shape),
            env_params={"count": jnp.asarray(0)},
        )

    @staticmethod
    def reset(env, key):
        del key
        return replace(env, env_params={"count": jnp.asarray(0)})

    @staticmethod
    def step(env, action):
        del action
        return replace(env, env_params={"count": env.env_params["count"] + 1})

    @staticmethod
    def observation(env):
        return jnp.asarray([[env.env_params["count"]]], dtype=float)

    @staticmethod
    def reward(env):
        return jnp.asarray([env.env_params["count"]], dtype=float)

    @staticmethod
    def truncated(env):
        return env.env_params["count"] >= 1


class ObservationValueModel(BanditModel):
    def __call__(self, x, sequence=False, **kwargs):
        dist, _ = super().__call__(x, sequence=sequence, **kwargs)
        return dist, x[..., :1]


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ResetInactiveEnv(TimeLimitCounterEnv):
    @staticmethod
    def reset(env, key):
        del key
        return replace(env, env_params={"count": jnp.asarray(0), "active": False})

    @staticmethod
    def agent_mask(env):
        return jnp.full((env.max_num_agents,), env.env_params.get("active", True))


def test_ppo_rejects_reset_that_produces_no_active_agents():
    base = TimeLimitCounterEnv.Create()
    env = ResetInactiveEnv(base.state, base.system, base.env_params)
    with pytest.raises(ValueError, match="reset.*at least one active agent"):
        PPOTrainer.Create(env=env, model=ObservationValueModel(), num_envs=1)


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DisappearingAgentEnv(TimeLimitCounterEnv):
    @classmethod
    def Create(cls, dim=2):
        state = State.create(pos=jnp.zeros((2, dim)))
        return cls(
            state=state,
            system=System.create(state.shape),
            env_params={"count": jnp.asarray(0)},
        )

    @staticmethod
    def observation(env):
        return jnp.full((2, 1), env.env_params["count"], dtype=float)

    @staticmethod
    def reward(env):
        return jnp.zeros(2)

    @staticmethod
    def truncated(env):
        return jnp.asarray(False)

    @staticmethod
    def agent_mask(env):
        return jnp.stack((jnp.asarray(True), env.env_params["count"] == 0), axis=0)


def test_disappearing_agent_is_terminal_and_does_not_bootstrap():
    env = vectorise_env(DisappearingAgentEnv.Create(), n=1)
    model = ObservationValueModel()
    graphdef, graphstate = nnx.split((model,))

    (_, _, _), trajectory = Trainer.step(env, graphdef, graphstate, jax.random.key(3))
    assert jnp.array_equal(trajectory.agent_mask[0], jnp.array([True, True]))
    assert jnp.array_equal(trajectory.terminated[0], jnp.array([False, True]))
    assert jnp.array_equal(trajectory.done[0], jnp.array([False, True]))

    _, advantage = Trainer.compute_advantages(
        value=trajectory.value[None, 0],
        reward=trajectory.reward[None, 0],
        ratio=trajectory.ratio[None, 0],
        done=trajectory.done[None, 0],
        terminated=trajectory.terminated[None, 0],
        truncated=trajectory.truncated[None, 0],
        bootstrap_value=trajectory.bootstrap_value[None, 0],
        agent_mask=trajectory.agent_mask[None, 0],
        last_value=jnp.array([100.0, 100.0]),
        advantage_rho_clip=jnp.array(1.0),
        advantage_c_clip=jnp.array(1.0),
        advantage_gamma=jnp.array(1.0),
        advantage_lambda=jnp.array(1.0),
        unroll=1,
    )
    assert advantage[0, 0] == 100.0
    assert advantage[0, 1] == 0.0


def test_trainer_resets_truncation_and_bootstraps_final_observation():
    env = vectorise_env(TimeLimitCounterEnv.Create(), n=2)
    env = env.reset(env, jax.random.split(jax.random.key(0), 2))
    model = ObservationValueModel()
    graphdef, graphstate = nnx.split((model,))

    (env, _, _), trajectory = Trainer.step(
        env, graphdef, graphstate, jax.random.key(1), skip_frames=3
    )
    assert jnp.all(env.env_params["count"] == 0)
    assert jnp.all(trajectory.truncated)
    assert not jnp.any(trajectory.terminated)
    assert jnp.all(trajectory.reward == 1.0)
    assert jnp.all(trajectory.bootstrap_value == 1.0)


@Environment.register("StatelessBandit")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class StatelessBanditEnv(Environment):
    @classmethod
    def Create(cls, dim: int = 2) -> "Environment":
        state = State.create(pos=jnp.zeros((1, 2)))
        system = System.create(state.shape)
        return cls(state=state, system=system, env_params={"reward": jnp.zeros((1,))})

    @staticmethod
    def reset(env, key):
        return env

    @staticmethod
    def step(env, action):
        reward = jnp.where(action[..., 0] > 0, 1.0, -1.0)
        reward = jnp.reshape(reward, (1,))
        from dataclasses import replace

        return replace(env, env_params={"reward": reward})

    @staticmethod
    def reward(env):
        return env.env_params.get("reward", jnp.zeros((env.max_num_agents,)))

    @staticmethod
    def terminated(env):
        return jnp.array(True, dtype=bool)

    @staticmethod
    def observation(env):
        return jnp.zeros((env.max_num_agents, 1))

    @property
    def action_space_size(self):
        return 1

    @property
    def action_space_shape(self):
        return (1,)

    @property
    def observation_space_size(self):
        return 1


def test_stateless_bandit():
    env = StatelessBanditEnv.Create()
    model = BanditModel(
        observation_space_size=1,
        action_space_size=1,
        action_space=ActionSpace.create("maxNorm", max_norm=1.0),
    )

    tr = PPOTrainer.Create(
        env=env,
        model=model,
        key=jax.random.PRNGKey(0),
        num_epochs=100,
        num_envs=128,
        num_minibatches=1,
        num_steps_epoch=2,
        optimizer=optax.adam,
        learning_rate=0.1,
    )

    tr = tr.train(tr, verbose=False, log=False)

    model, optimizer = nnx.merge(tr.graphdef, tr.graphstate)

    dist, val = model(jnp.zeros((1, 1)))

    # Verify the agent learned to output a positive mean action
    assert dist.mean() > 0.5
    # Verify the value function predicts the expected return (which should be ~1.0)
    assert val.mean() > 0.5


@Environment.register("DeterministicCorridor")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class DeterministicCorridorEnv(Environment):
    @classmethod
    def Create(cls, dim: int = 2) -> "Environment":
        state = State.create(pos=jnp.zeros((1, 2)))
        system = System.create(state.shape)
        return cls(
            state=state,
            system=system,
            env_params={
                "reward": jnp.zeros((1,)),
                "done": jnp.array(False, dtype=bool),
            },
        )

    @staticmethod
    def reset(env, key):
        from dataclasses import replace

        new_pos = jnp.zeros_like(env.state.pos_c)
        new_state = replace(env.state, pos_c=new_pos, pos_p=new_pos)
        return replace(
            env,
            state=new_state,
            env_params={
                "reward": jnp.zeros((1,)),
                "done": jnp.array(False, dtype=bool),
            },
        )

    @staticmethod
    def step(env, action):
        pos_x = env.state.pos_c[..., 0] + action[..., 0]
        pos_y = env.state.pos_c[..., 1]
        new_pos = jnp.stack([pos_x, pos_y], axis=-1)
        from dataclasses import replace

        new_state = replace(env.state, pos_c=new_pos, pos_p=new_pos)

        reached_goal = pos_x >= 5.0
        reward = jnp.where(reached_goal, 10.0, -0.1)
        reward = jnp.reshape(reward, (env.max_num_agents,))

        done = jnp.reshape(reached_goal, ())

        from dataclasses import replace

        return replace(
            env, state=new_state, env_params={"reward": reward, "done": done}
        )

    @staticmethod
    def reward(env):
        return env.env_params.get("reward", jnp.zeros((env.max_num_agents,)))

    @staticmethod
    def terminated(env):
        d = env.env_params.get("done", jnp.array(False, dtype=bool))
        return jnp.reshape(d, ())

    @staticmethod
    def observation(env):
        return env.state.pos_c[..., 0:1]

    @property
    def action_space_size(self):
        return 1


def test_deterministic_corridor():
    from jaxdem.rl.models.mlp import SharedActorCritic

    env = DeterministicCorridorEnv.Create()
    model = SharedActorCritic(
        observation_space_size=1,
        action_space_size=1,
        action_space=ActionSpace.create("maxNorm", max_norm=2.0),
        key=nnx.Rngs(0),
    )

    tr = PPOTrainer.Create(
        env=env,
        model=model,
        key=jax.random.PRNGKey(0),
        num_epochs=300,
        num_envs=256,
        num_minibatches=4,
        num_steps_epoch=8,
        optimizer=optax.adam,
        learning_rate=3e-3,
        advantage_gamma=0.99,
    )

    tr = tr.train(tr, verbose=False, log=False)

    model, optimizer = nnx.merge(tr.graphdef, tr.graphstate)
    dist, val = model(jnp.zeros((1, 1)))
    mean = dist.distribution.mean()

    # Should strongly prefer moving right
    assert mean[0, 0] > 0.55
