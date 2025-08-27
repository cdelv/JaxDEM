# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning model trainers.
"""
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from flax import nnx
import distrax

from ..factory import Factory


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
    """
    Immediate rewards :math:`r_t`.
    """

    done: jax.Array
    """
    Episode-termination flags (boolean).
    """

    value: jax.Array
    """
    Baseline value estimates :math:`V(s_t)`.
    """

    log_prob: jax.Array
    """
    Behavior-policy log-probabilities :math:`\log \pi_b(a_t \mid s_t)` at collection time.
    """

    new_log_prob: jax.Array
    """
    Target-policy log-probabilities :math:`\log \pi(a_t \mid s_t)` after policy update.

    Fill with ``log_prob`` during on-policy collection; must be recomputed after updates.
    """

    advantage: jax.Array
    """
    Advantages :math:`A_t`.
    """

    returns: jax.Array
    """
    Return targets (e.g., GAE or V-trace targets).
    """


@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
class Trainer:
    """
    Base class for reinforcement learning trainers.

    This class holds the environment and model state (Flax NNX GraphDef/GraphState).
    It provides rollout utilities (:meth:`step`, :meth:`trajectory_rollout`) and
    a general advantage computation method (:meth:`compute_advantages`).
    Subclasses must implement algorithm-specific training logic in :meth:`epoch`.
    """

    env: "rl.Environment"
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
    def eval(tr: "Trainer", input_data: jax.Array) -> Tuple["Trainer", Any]:
        """
        Run a forward pass in evaluation mode.

        Parameters
        ----------
        tr : Trainer
            The trainer carrying model state.
        input_data : jax.Array
            Model input batch.

        Returns
        -------
        (Trainer, Any)
            Updated trainer (with any mutated module state persisted) and the model output.
        """
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.eval()
        result = model(input_data)
        tr.graphstate = nnx.state((model, *rest))
        return tr, result

    @staticmethod
    @jax.jit
    def step(tr: "Trainer") -> Tuple["Trainer", "TrajectoryData"]:
        """
        Take one environment step and record a single-step trajectory.

        Returns
        -------
        (Trainer, TrajectoryData)
            Updated trainer and the new single-step trajectory record.
        """
        tr.key, subkey = jax.random.split(tr.key)
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.eval()

        obs = tr.env.observation(tr.env)
        pi, value = model(obs)
        action, log_prob = pi.sample_and_log_prob(seed=subkey)
        tr.env = tr.env.step(tr.env, action)
        reward = tr.env.reward(tr.env)
        done = tr.env.done(tr.env)

        # new_log_prob, advantage, and returns need to be computed later
        traj = TrajectoryData(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            value=jnp.squeeze(value, -1),
            log_prob=log_prob,
            new_log_prob=log_prob,
            advantage=log_prob,
            returns=log_prob,
        )

        tr.graphstate = nnx.state((model, *rest))
        return tr, traj

    @staticmethod
    @partial(jax.jit, static_argnames=("num_steps_epoch", "unroll"))
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
    @jax.jit
    def compute_advantages(
        tr: "Trainer", td: "TrajectoryData"
    ) -> Tuple["Trainer", "TrajectoryData"]:
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
        (Trainer, TrajectoryData)
            :class:`Trainer` and :class:`TrajectoryData` with ``advantage`` and ``returns`` filled.

        References
        ----------
        - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016
        - Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, 2018
        """
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.eval()
        pi, value = model(td.obs[-1])
        last_value = jnp.squeeze(value, -1)
        gae0 = jnp.zeros_like(last_value)

        def calculate_advantage(
            gae_and_next_value: jax.Array, td: TrajectoryData
        ) -> Tuple:
            ratio = jnp.exp(td.new_log_prob - td.log_prob)
            rho = jnp.minimum(ratio, tr.advantage_rho_clip)
            c = jnp.minimum(ratio, tr.advantage_c_clip)

            gae, next_value = gae_and_next_value
            delta = (
                rho * td.reward
                + tr.advantage_gamma * next_value * (1 - td.done)
                - td.value
            )
            gae = (
                delta
                + tr.advantage_gamma * tr.advantage_lambda * (1 - td.done) * c * gae
            )

            return (gae, td.value), gae

        _, td.advantage = jax.lax.scan(
            calculate_advantage, (gae0, last_value), td, reverse=True, unroll=8
        )
        td.returns = td.advantage + td.value

        tr.graphstate = nnx.state((model, *rest))
        return tr, td

    @staticmethod
    @abstractmethod
    @jax.jit
    def epoch(tr: "Trainer", epoch: ArrayLike) -> Any:
        """
        Run one training epoch.

        Subclasses must implement this with their algorithm-specific logic.
        """
        raise NotImplementedError
