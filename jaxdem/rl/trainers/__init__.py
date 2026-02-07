# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning model trainers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import TYPE_CHECKING, Tuple, Any, Sequence

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from flax import nnx

from ...factory import Factory

if TYPE_CHECKING:
    from ..environments import Environment


@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class TrajectoryData:
    """
    Container for rollout data (single step or stacked across time).
    """

    obs: jax.Array
    """
    Observations.
    """

    action: jax.Array
    """
    Actions sampled from the policy.
    """

    value: jax.Array
    r"""
    Baseline value estimates :math:`V(s_t)`.
    """

    log_prob: jax.Array
    r"""
    Behavior-policy log-probabilities :math:`\log \pi_b(a_t \mid s_t)` at collection time.
    """

    ratio: jax.Array
    r"""
    Ratio between behavior-policy probabilities :math:`\exp\big( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_\text{old}}(a_t \mid s_t) \big)`.
    """

    reward: jax.Array
    r"""
    Immediate rewards :math:`r_t`.
    """

    done: jax.Array
    """
    Episode-termination flags (boolean).
    """


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Trainer(Factory, ABC):
    """
    Base class for reinforcement learning trainers.

    This class holds the environment and model state (Flax NNX GraphDef/GraphState).
    It provides rollout utilities (:meth:`step`, :meth:`trajectory_rollout`) and
    a general advantage computation method (:meth:`compute_advantages`).
    Subclasses must implement algorithm-specific training logic in :meth:`epoch`.

    Example
    -------
    To define a custom trainer, inherit from :class:`Trainer` and implement its abstract methods:

    >>> @Trainer.register("myCustomTrainer")
    >>> @jax.tree_util.register_dataclass
    >>> @dataclass(slots=True)
    >>> class MyCustomTrainer(Trainer):
            ...
    """

    env: "Environment"
    """
    Environment object.
    """

    graphdef: nnx.GraphDef
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
    @jax.jit
    @partial(jax.named_call, name="Trainer.step")
    def step(
        env: "Environment",
        graphdef: nnx.GraphDef,
        graphstate: nnx.GraphState,
        key: jax.Array,
    ) -> Tuple[Tuple["Environment", nnx.GraphState, jax.Array], "TrajectoryData"]:
        """
        Take one environment step and record a single-step trajectory.

        Parameters
        ----------
        env : Environment
            The trainer carrying model state.
        graphdef : nnx.GraphDef
            Python part of the nnx model
        graphstate : nnx.GraphState
            State of the nnx model
        key : jax.Array
            Jax random key

        Returns
        -------
        Tuple[Tuple[Environment, nnx.GraphState, jax.Array], TrajectoryData]
            Updated state and the new single-step trajectory.
            Trajectory data is shaped (N_envs, N_agents, ...).
        """
        key, subkey = jax.random.split(key)
        model, *rest = nnx.merge(graphdef, graphstate)

        obs = env.observation(env)  # shape: (N_envs, N_agents, *)
        pi, value = model(obs, sequence=False)
        action, log_prob = pi.sample_and_log_prob(seed=subkey)
        env = env.step(env, action)
        reward = env.reward(env)
        done = env.done(env)

        # Shape -> (N_agents, *)
        traj = TrajectoryData(
            obs=obs,
            action=action,
            value=jnp.squeeze(value, -1),
            log_prob=log_prob,
            ratio=jnp.ones_like(log_prob),
            reward=reward,
            done=jnp.broadcast_to(done[..., None], reward.shape),
        )

        graphstate = nnx.state((model, *rest))
        return (env, graphstate, key), traj

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("num_steps_epoch", "unroll"),
    )
    @partial(jax.named_call, name="Trainer.trajectory_rollout")
    def trajectory_rollout(
        env: "Environment",
        graphdef: nnx.GraphDef,
        graphstate: nnx.GraphState,
        key: jax.Array,
        num_steps_epoch: int,
        unroll: int = 8,
    ) -> Tuple["Environment", nnx.GraphState, jax.Array, "TrajectoryData"]:
        r"""
        Roll out :math:`T = \text{num_steps_epoch}` environment steps using :func:`jax.lax.scan`.

        Parameters
        ----------
        env : Environment
            The trainer carrying model state.
        graphdef : nnx.GraphDef
            Python part of the nnx model
        graphstate : nnx.GraphState
            State of the nnx model
        key : jax.Array
            Jax random key
        num_steps_epoch : int
            Number of steps to roll out.
        unroll : int
            Number of loop iterations to unroll for compilation speed.

        Returns
        -------
        Tuple[Environment, nnx.GraphState, jax.Array, TrajectoryData]
            The final trainer and a :class:`TrajectoryData` instance whose fields are stacked
            along time (leading dimension :math:`T = \text{num_steps_epoch}`).
        """

        model, *rest = nnx.merge(graphdef, graphstate)
        model.eval()
        graphstate = nnx.state((model, *rest))

        def body(
            carry: Tuple["Environment", nnx.GraphState, jax.Array], _: None
        ) -> Tuple[Tuple["Environment", nnx.GraphState, jax.Array], "TrajectoryData"]:
            env, graphstate, key = carry
            carry, traj = Trainer.step(env, graphdef, graphstate, key)
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
    @partial(jax.jit, static_argnames=("unroll",))
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
        unroll: int = 8,
    ) -> Tuple[jax.Array, jax.Array]:
        r"""
        Compute V-trace/GAE advantages and return targets.

        Given a policy :math:`\pi`, define per-step importance ratios and clipped versions:

        .. math::

            \rho_t = \exp\big( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_\text{old}}(a_t \mid s_t) \big)

        and their clipped versions :math:`\hat{\rho}, \hat{c}`:

        .. math::

            \hat{\rho}_t = \min(\rho_t, \bar{\rho}), \quad
            \hat{c}_t = \min(\rho_t, \bar{c}).

        We form a TD-like residual with an off-policy correction:

        .. math::

            \delta_t = \hat{\rho}_t \, r_t + \gamma V(s_{t+1})(1 - \text{done}_t) - V(s_t)

        and propagate a GAE-style trace using :math:`\hat{c}_t`:

        .. math::

            A_t = \delta_t + \gamma \lambda (1 - \text{done}_t) \hat{c}_t A_{t+1}

        Finally, the return targets are:

        .. math::

            \text{returns}_t = A_t + V(s_t)

        Notes
        -----
            When :math:`\pi_\theta = \pi_{\theta_\text{old}}` (i.e. ``ratio==1``) and
            :math:`\bar{\rho} = \bar{c} = 1`, this function reduces to standard GAE.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            Computed advantage and returns.

        References
        ----------
        - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016
        - Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, 2018
        """
        last_value = value[-1]
        gae0 = jnp.zeros_like(last_value)

        def calculate_advantage(
            gae_and_next_value: Tuple[jax.Array, jax.Array],
            xs: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
            gae, next_value = gae_and_next_value
            value, reward, ratio, done = xs
            rho = jnp.minimum(ratio, advantage_rho_clip)
            c = jnp.minimum(ratio, advantage_c_clip)

            delta = rho * (reward + advantage_gamma * next_value * (1 - done) - value)
            gae = delta + advantage_gamma * advantage_lambda * (1 - done) * c * gae
            return (gae, value), gae

        _, advantage = jax.lax.scan(
            calculate_advantage,
            (gae0, last_value),
            xs=(value, reward, ratio, done),
            reverse=True,
            unroll=unroll,
        )
        returns = advantage + value
        return jax.lax.stop_gradient(returns), jax.lax.stop_gradient(advantage)

    @staticmethod
    @abstractmethod
    @jax.jit
    def epoch(tr: "Trainer", epoch: ArrayLike) -> Any:
        """
        Run one training epoch.

        Subclasses must implement this with their algorithm-specific logic.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def train(tr: "Trainer", *args: Any, **kwargs: Any) -> Any:
        """
        Training loop

        Subclasses must implement this with their algorithm-specific logic.
        """
        raise NotImplementedError


from .PPOtrainer import PPOTrainer

__all__ = ["PPOTrainer"]
