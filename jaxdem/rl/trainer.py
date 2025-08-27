# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning model trainers.
"""
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    Immediate rewards `r_t`.
    """

    done: jax.Array
    """
    Episode-termination flags (boolean).
    """

    value: jax.Array
    """
    Baseline value estimates `V(s_t)`.
    """

    log_prob: jax.Array
    """
    Behavior-policy log π_b(a_t | s_t) at collection time.
    """

    new_log_prob: jax.Array
    """
    Target-policy log π(a_t | s_t) after policy update. Fill with `log_prob` during on-policy collection; Needs to be computed after updates.
    """

    advantage: jax.Array
    """
    Advantages `A_t`.
    """

    returns: jax.Array
    """
    Return targets (e.g., V-trace/GAE targets).
    """


@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True)
class Trainer:
    """
    Base class for reinforcement learning trainers.

    This class holds the environment and model state (Flax NNX GraphDef/GraphState).
    It provides rollouts (`step`, `trajectory_rollout`), compute advantages base methods.
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
    PRNGKey used to sample actions and for other stochastic ops.
    """

    advantage_gamma: jax.Array
    """
    Discount factor γ ∈ [0, 1].
    """

    advantage_lambda: jax.Array
    """
    General advantage estimation parameter λ ∈ [0, 1]
    """

    advantage_rho_clip: jax.Array
    """
    V-trace ρ̄ (importance weight clip for the TD term).
    """

    advantage_c_clip: jax.Array
    """
    V-trace c̄ (importance weight clip for the recursion/trace term).
    """

    @property
    def model(self):
        """Return the live model rebuilt from (graphdef, graphstate)."""
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        return model

    @property
    def optimizer(self):
        """Return the optimizer rebuilt from (graphdef, graphstate)."""
        model, optimizer, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        return optimizer

    @staticmethod
    @jax.jit
    def eval(tr: "Trainer", input_data: jax.Array) -> jax.Array:
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
        """
        Roll out `num_steps_epoch` environment steps using `jax.lax.scan`.


        Parameters
        ----------
        tr : Trainer
            The trainer carrying model state.
        num_steps_epoch : int
            Number of steps ro rollout
        unroll: int
            Number of loops to unroll

        Returns
        -------
        (Trainer, TrajectoryData)
            The final trainer and a `TrajectoryData` whose fields are stacked
            along time (leading dimension `T = num_steps_epoch`)
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
        """
        Compute advantages and return targets with V-trace-style off-policy
        correction or general advantage estimation recursion.

        Let the behavior policy be π_b and the target policy be π. Define
        importance ratios per step
            ρ_t = exp( log π(a_t|s_t) - log π_b(a_t|s_t) )
        and their clipped versions ρ̄, c̄:
            ρ̂_t = min(ρ_t, ρ̄),   ĉ_t = min(ρ_t, c̄).

        We form a TD-like residual with an off-policy correction:
            δ_t = ρ̂_t * r_t + γ * V_{t+1} * (1 - done_t) - V_t

        and propagate a GAE-style trace using ĉ_t:
            A_t = δ_t + γ * λ * (1 - done_t) * ĉ_t * A_{t+1}

        Finally:
            returns_t = A_t + V_t

        Notes
        -----
        • When π_b == π (td.log_prob == td.new_log_prob) and ρ̄, c̄ = 1, the above reduces to standard GAE.

        Returns
        -------
        (Trainer, TrajectoryData)
            Trainer and TrajectoryData with `advantage` and `returns` filled.

        References
        ----------
            - High-Dimensional Continuous Control Using Generalized Advantage Estimation: Schulman et al., 2015/2016
            - IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures: Espeholt et al., 2018
        """
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.eval()
        obs = tr.env.observation(tr.env)
        pi, value = model(td.obs[-1])
        last_value = jnp.squeeze(value, -1)
        gae0 = jnp.zeros_like(last_value)

        @jax.jit
        def calculate_advantage(
            gae_and_next_value: jax.Array, td: TrajectoryData
        ) -> Tuple:
            """
            GAE(t) = delta(t)
            GAE(t - 1) = gamma * lambda * GAE(t) + delta(t - 1)
            """
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

    @abstractmethod
    @staticmethod
    @jax.jit
    def epoch(tr: "Trainer", epoch: ArrayLike) -> Any:
        """
        Run one training 'epoch'.

        Subclasses must implement this with their algorithm-specific logic.
        """
        raise NotImplementedError
