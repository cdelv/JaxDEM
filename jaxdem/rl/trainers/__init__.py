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
from dataclasses import dataclass, replace
from functools import partial

from flax import nnx

from ...factory import Factory

if TYPE_CHECKING:
    from ..environments import Environment


@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
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

    reward: jax.Array
    r"""
    Immediate rewards :math:`r_t`.
    """

    done: jax.Array
    """
    Episode-termination flags (boolean).
    """

    value: jax.Array
    r"""
    Baseline value estimates :math:`V(s_t)`.
    """

    log_prob: jax.Array
    r"""
    Behavior-policy log-probabilities :math:`\log \pi_b(a_t \mid s_t)` at collection time.
    """

    new_log_prob: jax.Array
    r"""
    Target-policy log-probabilities :math:`\log \pi(a_t \mid s_t)` after policy update.

    Fill with ``log_prob`` during on-policy collection; must be recomputed after updates.
    """

    advantage: jax.Array
    r"""
    Advantages :math:`A_t`.
    """

    returns: jax.Array
    """
    Return targets (e.g., GAE or V-trace targets).
    """


@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
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
    >>> @dataclass(slots=True, frozen=True)
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
    def model(self):
        """Return the live model rebuilt from (graphdef, graphstate)."""
        model, *rest = nnx.merge(self.graphdef, self.graphstate)
        return model

    @property
    def optimizer(self):
        """Return the optimizer rebuilt from (graphdef, graphstate)."""
        model, optimizer, *rest = nnx.merge(self.graphdef, self.graphstate)
        return optimizer

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="Trainer.step")
    def step(tr: "Trainer") -> Tuple["Trainer", "TrajectoryData"]:
        """
        Take one environment step and record a single-step trajectory.

        Returns
        -------
        (Trainer, TrajectoryData)
            Updated trainer and the new single-step trajectory record.
            Trajectory data shape: (N_envs, N_agents, *)
        """
        key, subkey = jax.random.split(tr.key)
        tr = replace(tr, key=key)
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.eval()

        obs = tr.env.observation(tr.env)
        pi, value = model(obs, sequence=False)
        action, log_prob = pi.sample_and_log_prob(seed=subkey)
        tr = replace(tr, env=tr.env.step(tr.env, action))
        reward = tr.env.reward(tr.env)
        done = tr.env.done(tr.env)

        # new_log_prob, advantage, and returns need to be computed later
        # Shape -> (N_agents, *)
        traj = TrajectoryData(
            obs=obs,
            action=action,
            reward=reward,
            done=jnp.broadcast_to(done[..., None], reward.shape),
            value=jnp.squeeze(value, -1),
            log_prob=log_prob,
            new_log_prob=log_prob,
            advantage=log_prob,
            returns=log_prob,
        )

        tr = replace(tr, graphstate=nnx.state((model, *rest)))
        return tr, traj

    @staticmethod
    @partial(jax.named_call, name="Trainer.reset_model")
    def reset_model(
        tr: "Trainer",
        shape: Sequence[int] | None = None,
        mask: jax.Array | None = None,
    ) -> "Trainer":
        """
        Reset a model's persistent recurrent state (e.g., LSTM carry) for all
        environments/agents and persist the mutation back into the trainer.

        Parameters
        ----------
        tr : Trainer
            Trainer carrying the environment and NNX graph state. The target carry
            shape is inferred as ``(tr.num_envs, tr.env.max_num_agents)`` if not specified.
        mask : jax.Array, optional
            Boolean mask selecting which (env, agent) entries to reset. A value of
            ``True`` resets that entry. The mask may be shape
            ``(num_envs, num_agents)`` or any shape broadcastable to it. If
            ``None``, all entries are reset.

        Returns
        -------
        Trainer
            A new trainer with the updated ``graphstate``.
        """
        ...

    @staticmethod
    @partial(jax.jit, static_argnames=("num_steps_epoch", "unroll"))
    @partial(jax.named_call, name="Trainer.trajectory_rollout")
    def trajectory_rollout(
        tr: "Trainer", num_steps_epoch: int, unroll: int = 8
    ) -> Tuple["Trainer", "TrajectoryData"]:
        r"""
        Roll out :math:`T = \text{num_steps_epoch}` environment steps using :func:`jax.lax.scan`.

        Parameters
        ----------
        tr : Trainer
            The trainer carrying model state.
        num_steps_epoch : int
            Number of steps to roll out.
        unroll : int
            Number of loop iterations to unroll for compilation speed.

        Returns
        -------
        (Trainer, TrajectoryData)
            The final trainer and a :class:`TrajectoryData` instance whose fields are stacked
            along time (leading dimension :math:`T = \text{num_steps_epoch}`).
        """
        return jax.lax.scan(
            lambda tr, _: Trainer.step(tr),
            tr,
            None,
            length=num_steps_epoch,
            unroll=unroll,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("unroll",))
    @partial(jax.named_call, name="Trainer.compute_advantages")
    def compute_advantages(
        td: "TrajectoryData",
        advantage_rho_clip: jax.Array,
        advantage_c_clip: jax.Array,
        advantage_gamma: jax.Array,
        advantage_lambda: jax.Array,
        unroll: int = 8,
    ) -> "TrajectoryData":
        r"""
        Compute advantages and return targets with V-trace-style off-policy
        correction or generalized advantage estimation (GAE).

        Let the behavior policy be :math:`\pi_b` and the target policy be :math:`\pi`.
        Define importance ratios per step:

        .. math::

            \rho_t = \exp\big( \log \pi(a_t \mid s_t) - \log \pi_b(a_t \mid s_t) \big)

        and their clipped versions :math:`\bar{\rho}, \bar{c}`:

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
        • When :math:`\pi_b = \pi` (i.e. ``TrajectoryData.log_prob == TrajectoryData.new_log_prob``) and
          :math:`\bar{\rho} = \bar{c} = 1`, this function reduces to standard GAE.

        Returns
        -------
        (TrajectoryData)
            :class:`TrajectoryData` with new ``advantage`` and ``returns``.

        References
        ----------
        - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016
        - Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, 2018
        """
        last_value = td.value[-1]
        gae0 = jnp.zeros_like(last_value)

        def calculate_advantage(gae_and_next_value: Tuple, td: TrajectoryData) -> Tuple:
            gae, next_value = gae_and_next_value

            ratio = jnp.exp(td.new_log_prob - td.log_prob)
            rho = jnp.minimum(ratio, advantage_rho_clip)
            c = jnp.minimum(ratio, advantage_c_clip)

            delta = (
                rho * td.reward
                + advantage_gamma * next_value * (1 - td.done)
                - td.value
            )
            gae = delta + advantage_gamma * advantage_lambda * (1 - td.done) * c * gae

            return (gae, td.value), gae

        _, adv = jax.lax.scan(
            calculate_advantage, (gae0, last_value), td, reverse=True, unroll=unroll
        )
        return replace(td, advantage=adv, returns=adv + td.value)

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
    def train(tr) -> Any:
        """
        Training loop
        """
        raise NotImplementedError


from .PPOtrainer import PPOTrainer

__all__ = ["PPOTrainer"]
