# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Interface for defining reinforcement learning model trainers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from ...factory import Factory
from ...utils.environment import _mask_agent_actions, advance_action

if TYPE_CHECKING:
    from ..environments import Environment


@jax.tree_util.register_dataclass
@dataclass(slots=True, kw_only=True)
class TrajectoryData:
    """Container for rollout data (single step or stacked across time)."""

    obs: jax.Array
    """
    Observations.
    """

    action: jax.Array
    """
    Actions sampled from the policy; inactive agent entries are zeroed.
    """

    value: jax.Array
    r"""
    Immutable behavior-policy value estimates :math:`V_b(s_t)` used as the
    PPO value-clipping reference. Current learner values are computed separately.
    """

    log_prob: jax.Array
    r"""
    Behavior-policy log-probabilities :math:`\log \pi_b(a_t \mid s_t)` at collection time.
    """

    ratio: jax.Array
    r"""
    Collection-time probability ratio (one). Retained for trajectory consumers;
    PPO computes current importance ratios locally without overwriting this field.
    """

    reward: jax.Array
    r"""
    Post-action rewards :math:`r_t` during collection. PPO epoch output
    contains DRIP-transformed rewards when that option is enabled.
    """

    done: jax.Array
    """
    Episode-boundary flags (``terminated | truncated``).
    """

    terminated: jax.Array
    """Terminal flags; these suppress value bootstrapping."""

    truncated: jax.Array
    """Time-limit/external boundary flags; these retain bootstrapping unless
    the same transition is also terminal."""

    agent_mask: jax.Array
    """Boolean flags selecting active agents in padded environments."""

    bootstrap_value: jax.Array
    """Collection-time value of the final pre-reset observation for truncations."""

    latent_action: jax.Array | None = None
    """Pre-bijector sample for transformed continuous policies, otherwise None."""

    latent_log_prob: jax.Array | None = None
    """Behavior base-distribution log-probability of ``latent_action``.

    Ratios for a fixed bijector are evaluated in base coordinates to avoid
    inverting saturated actions. ``log_prob`` still records transformed density.
    """


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Trainer(Factory, ABC):
    """Base class for reinforcement learning trainers.

    This class holds the environment and model state (Flax NNX GraphDef/GraphState).
    It provides rollout utilities (:meth:`step`, :meth:`trajectory_rollout`) and
    a general advantage method (:meth:`compute_advantages`).
    Subclasses must implement algorithm-specific training logic in :meth:`epoch`.

    Example:
    --------
    To define a custom trainer, inherit from :class:`Trainer` and implement its abstract methods:

    >>> @Trainer.register("myCustomTrainer")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomTrainer(Trainer):
            ...

    """

    env: Environment
    """
    Environment object.
    """

    graphdef: nnx.GraphDef[Any]
    """
    Static graph definition of the model/optimizer.
    """

    graphstate: nnx.GraphState
    """
    Mutable state (parameters, optimizer state, RNGs, etc.).
    """

    key: ArrayLike
    """
    PRNGKey used to sample actions and for other stochastic operations.
    """

    advantage_gamma: jax.Array
    r"""
    Discount factor :math:`\gamma \in [0, 1]`.
    """

    advantage_lambda: jax.Array
    r"""
    Generalized Advantage Estimation parameter :math:`\lambda \in [0, 1]`.
    """

    advantage_rho_clip: jax.Array
    r"""
    V-trace :math:`\bar{\rho}` (importance weight clip for the TD term).
    """

    advantage_c_clip: jax.Array
    r"""
    V-trace :math:`\bar{c}` (importance weight clip for the recursion/trace term).
    """

    @property
    def model(self) -> Any:
        """Return the live model rebuilt from (graphdef, graphstate)."""
        model, *_ = nnx.merge(self.graphdef, self.graphstate)
        return model

    @staticmethod
    @jax.jit(inline=True, static_argnames=("skip_frames",))
    @partial(jax.named_call, name="Trainer.step")
    def step(
        env: Environment,
        graphdef: nnx.GraphDef[Any],
        graphstate: nnx.GraphState,
        key: jax.Array,
        skip_frames: int = 0,
    ) -> tuple[tuple[Environment, nnx.GraphState, jax.Array], TrajectoryData]:
        """Take one environment step and record a single-step trajectory.
        Repeats the action for ``skip_frames`` extra frames when set.

        Parameters
        ----------
        env : Environment
            The (vectorized) environment to step.
        graphdef : nnx.GraphDef
            Python part of the nnx model.
        graphstate : nnx.GraphState
            State of the nnx model.
        key : jax.Array
            Jax random key.
        skip_frames : int
            Number of additional frames to repeat the action.

        Returns
        -------
        Tuple[Tuple[Environment, nnx.GraphState, jax.Array], TrajectoryData]
            Updated state and the new single-step trajectory.
            Trajectory data is shaped (N_envs, N_agents, ...).

        Finished environments are reset before the next transition. Frame
        skipping stops each environment at its first boundary. Truncations
        retain a value estimate of their final observation for bootstrapping.
        ``advance_action`` calls ``checkpoint(env, action)`` once with the
        masked policy action before its physics loop. Action wrappers apply
        the same transformation at the checkpoint and at each physics step.
        Reward and truncation bootstrap observations read the live endpoint
        before reset, with no endpoint checkpoint.
        Intermediate rewards are not accumulated.

        """
        key, subkey, reset_root = jax.random.split(key, 3)
        model, *rest = nnx.merge(graphdef, graphstate)

        obs = env.observation(env)  # shape: (N_envs, N_agents, *)
        agent_mask = env.agent_mask(env)
        pi, value = model(obs, sequence=False)
        from ..action_spaces import Transformed

        latent_action = latent_log_prob = None
        if isinstance(pi, Transformed):
            action, log_prob, latent_action, latent_log_prob = (
                pi.sample_and_log_prob_with_latent(seed=subkey)
            )
        else:
            action, log_prob = pi.sample_and_log_prob(seed=subkey)
        action = _mask_agent_actions(action, agent_mask)

        env, terminated, truncated = advance_action(
            env, action, skip_frames=skip_frames
        )
        done = terminated | truncated
        # Consume the live endpoint against the saved action-start baseline.
        reward = env.reward(env)
        next_agent_mask = env.agent_mask(env)
        agent_terminated = agent_mask & ~next_agent_mask
        done_agents = jnp.broadcast_to(done[..., None], reward.shape) | agent_terminated
        terminated_agents = (
            jnp.broadcast_to(terminated[..., None], reward.shape) | agent_terminated
        )

        def evaluate_truncation(_: None) -> jax.Array:
            # Bootstrap evaluation must not mutate the live rollout carry.
            # Copy variables here so they belong to this conditional trace.
            boot_model, *_ = nnx.merge(
                graphdef, nnx.state((model, *rest)), copy=True
            )
            boot_model.eval()
            _, bootstrap = boot_model(env.observation(env), sequence=False)
            return jnp.squeeze(bootstrap, -1)

        bootstrap_value = jax.lax.cond(
            jnp.any(truncated),
            evaluate_truncation,
            lambda _: jnp.zeros_like(reward),
            operand=None,
        )

        # Shape -> (N_envs, N_agents, *)
        traj = TrajectoryData(
            obs=obs,
            action=action,
            value=jnp.squeeze(value, -1),
            log_prob=log_prob,
            ratio=jnp.ones_like(log_prob),
            reward=reward,
            done=done_agents,
            terminated=terminated_agents,
            truncated=jnp.broadcast_to(truncated[..., None], reward.shape),
            agent_mask=jnp.broadcast_to(agent_mask, reward.shape),
            bootstrap_value=jnp.broadcast_to(bootstrap_value, reward.shape),
            latent_action=latent_action,
            latent_log_prob=latent_log_prob,
        )

        # Reset recurrent state and environments before the next transition.
        carry_reset = done[..., None] | ~agent_mask | ~next_agent_mask
        model.reset(shape=obs.shape, mask=carry_reset)

        env = jax.lax.cond(
            jnp.any(done),
            lambda current: current.reset_if_done(
                current, done, jax.random.split(reset_root, current.num_envs)
            ),
            lambda current: current,
            env,
        )

        graphstate = nnx.state((model, *rest))
        return (env, graphstate, key), traj

    @staticmethod
    @jax.jit(inline=True, static_argnames=("num_steps_epoch", "unroll", "skip_frames"))
    @partial(jax.named_call, name="Trainer.trajectory_rollout")
    def trajectory_rollout(
        env: Environment,
        graphdef: nnx.GraphDef[Any],
        graphstate: nnx.GraphState,
        key: jax.Array,
        num_steps_epoch: int,
        unroll: int = 8,
        skip_frames: int = 0,
    ) -> tuple[Environment, nnx.GraphState, jax.Array, TrajectoryData]:
        r"""Roll out :math:`T = \text{num\_steps\_epoch}` policy decisions using :func:`jax.lax.scan`.

        Parameters
        ----------
        env : Environment
            The (vectorized) environment to roll out.
        graphdef : nnx.GraphDef
            Python part of the nnx model.
        graphstate : nnx.GraphState
            State of the nnx model.
        key : jax.Array
            Jax random key.
        num_steps_epoch : int
            Number of policy decisions to roll out per agent slot.
        unroll : int
            Number of loop iterations to unroll for compilation speed.
        skip_frames : int
            Number of additional physics frames requested per policy action.

        Returns
        -------
        Tuple[Environment, nnx.GraphState, jax.Array, TrajectoryData]
            The final environment, graph state, and PRNG key, plus a
            :class:`TrajectoryData` instance whose fields are stacked
            along time (leading dimension :math:`T = \text{num_steps_epoch}`).

        """
        model, *rest = nnx.merge(graphdef, graphstate)
        model.eval()
        graphstate = nnx.state((model, *rest))

        @partial(jax.named_call, name="Trainer.rollout_body")
        def body(
            carry: tuple[Environment, nnx.GraphState, jax.Array], _: None
        ) -> tuple[tuple[Environment, nnx.GraphState, jax.Array], TrajectoryData]:
            env, graphstate, key = carry
            carry, traj = Trainer.step(env, graphdef, graphstate, key, skip_frames)
            return carry, traj

        (env, graphstate, key), trajectory = jax.lax.scan(
            body,
            (env, graphstate, key),
            xs=None,
            length=num_steps_epoch,
            unroll=unroll,
        )

        return env, graphstate, key, trajectory

    @staticmethod
    @jax.jit(inline=True, static_argnames=("unroll",))
    @partial(jax.named_call, name="Trainer.compute_advantages")
    def compute_advantages(
        value: jax.Array,
        reward: jax.Array,
        ratio: jax.Array,
        done: jax.Array,
        advantage_rho_clip: jax.Array,
        advantage_c_clip: jax.Array,
        advantage_gamma: jax.Array,
        advantage_lambda: jax.Array,
        last_value: jax.Array | None = None,
        terminated: jax.Array | None = None,
        truncated: jax.Array | None = None,
        bootstrap_value: jax.Array | None = None,
        agent_mask: jax.Array | None = None,
        unroll: int = 8,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Return detached targets and GAE/V-trace-style advantages.

        All transition arrays have leading time dimension :math:`T`.
        For active-agent indicator :math:`m_t`, terminal flag :math:`z_t`,
        and episode-boundary flag :math:`d_t`, the recurrence is

        .. math::

            \widehat\rho_t &= \min(\mathrm{ratio}_t, \bar\rho), \qquad
            \widehat c_t = \min(\mathrm{ratio}_t, \bar c),\\
            \delta_t &= m_t\widehat\rho_t
                [r_t+\gamma(1-z_t)B_t-V_t],\\
            A_t &= \delta_t+\gamma\lambda m_t(1-d_t)\widehat c_t A_{t+1},
                \qquad A_T=0,\\
            R_t &= V_t + m_t A_t.

        Here :math:`B_t` is ``bootstrap_value[t]`` at a truncation,
        ``last_value`` at the final non-truncated transition, and
        ``value[t+1]`` otherwise. A terminal flag suppresses bootstrapping
        even if the same transition is also truncated. Inputs must be finite,
        including inactive padding; multiplication by zero does not mask NaNs.

        Unit ratios and unit clipping caps give ordinary GAE. Otherwise this
        is a clipped-importance lambda trace. PPO uses the resulting detached
        :math:`A_t` in its clipped actor objective, rather than the separate
        policy-gradient advantage from the IMPALA algorithm.

        Parameters
        ----------
        value, reward, ratio : jax.Array
            Per-transition values, post-action rewards, and importance ratios.
            PPO supplies its current learner values and, with ``vtrace=True``,
            current-to-behavior policy ratios. Otherwise it supplies unit
            ratios and unit caps, regardless of the configured trace caps.
        done : jax.Array
            Episode boundaries, consistent with ``terminated | truncated``.
            These cut the trace after the current transition.
        advantage_rho_clip, advantage_c_clip : jax.Array
            Upper caps on the TD residual and trace importance weights.
        advantage_gamma, advantage_lambda : jax.Array
            Discount and trace decay per recorded policy transition.
        last_value : jax.Array | None
            Bootstrap on the post-rollout observation, without a time axis.
            If omitted, ``value[-1]`` is reused; this is generally biased at
            a continuing horizon boundary. PPO provides a collection-time
            estimate and holds it fixed during minibatch updates.
        terminated, truncated : jax.Array | None
            Supply both to distinguish terminal from truncated transitions.
            If ``terminated`` is omitted it defaults to ``done``; if
            ``truncated`` is omitted it defaults to false. Passing only
            ``truncated`` therefore does not enable time-limit bootstrapping.
        bootstrap_value : jax.Array | None
            Values of final pre-reset observations at truncations, zero if
            omitted. PPO records these during collection.
        agent_mask : jax.Array | None
            Active-agent mask, all true if omitted. Inactive entries have
            zero advantage and return target equal to their input value.
        unroll : int
            Unroll factor for the reverse scan.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            ``(returns, advantages)``, both with gradients stopped and the
            same shape as ``value``.

        References
        ----------
        - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016.
        - Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, 2018.

        """
        if last_value is None:
            # Backwards-compatible fallback: biased bootstrap with the last
            # transition's own value V(s_{T-1}) instead of V(s_T).
            last_value = value[-1]
        gae0 = jnp.zeros_like(last_value)

        if terminated is None:
            terminated = done
        if truncated is None:
            truncated = jnp.zeros_like(done, dtype=bool)
        if bootstrap_value is None:
            bootstrap_value = jnp.zeros_like(value)
        if agent_mask is None:
            agent_mask = jnp.ones_like(done, dtype=bool)
        active = agent_mask.astype(value.dtype)
        not_done = (1.0 - done.astype(value.dtype)) * active
        not_terminated = (1.0 - terminated.astype(value.dtype)) * active
        rho = jnp.minimum(ratio, advantage_rho_clip)
        c = jnp.minimum(ratio, advantage_c_clip)

        next_value = jnp.concatenate((value[1:], last_value[None]), axis=0)
        next_value = jnp.where(truncated, bootstrap_value, next_value)
        delta = (
            rho
            * (reward + advantage_gamma * not_terminated * next_value - value)
            * active
        )
        gae_coeff = advantage_gamma * advantage_lambda * not_done * c

        @partial(jax.named_call, name="Trainer.calculate_advantage")
        def calculate_advantage(
            gae: jax.Array,
            xs: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            delta_t, coefficient = xs
            gae = delta_t + coefficient * gae
            return gae, gae

        _, advantage = jax.lax.scan(
            calculate_advantage,
            gae0,
            xs=(delta, gae_coeff),
            reverse=True,
            unroll=unroll,
        )
        advantage = advantage * active
        returns = jnp.where(agent_mask, advantage + value, value)
        return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(advantage)

    @staticmethod
    @abstractmethod
    @jax.jit(inline=True)
    def epoch(tr: Trainer, epoch: ArrayLike) -> Any:
        """Run one training epoch.

        Subclasses implement this method with their algorithm-specific logic.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train(tr: Trainer, *args: Any, **kwargs: Any) -> Any:
        """Training loop.

        Subclasses implement this method with their algorithm-specific logic.
        """
        raise NotImplementedError


from .ppo_trainer import PPOTrainer

__all__ = ["PPOTrainer"]
