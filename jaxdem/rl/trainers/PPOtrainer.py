# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of PPO algorithm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import TYPE_CHECKING, Tuple, Optional, Sequence, cast

try:
    # Python 3.11+
    from typing import Self
except ImportError:
    from typing_extensions import Self

from dataclasses import dataclass, field, replace
from functools import partial
import time
import datetime

from flax import nnx
from flax.metrics import tensorboard
import optax
from tqdm.auto import trange

from . import Trainer, TrajectoryData
from ..envWrappers import clip_action_env, vectorise_env

if TYPE_CHECKING:
    from ..environments import Environment
    from ..models import Model


@Trainer.register("PPO")
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class PPOTrainer(Trainer):
    r"""
    Proximal Policy Optimization (PPO) trainer in `PufferLib <https://github.com/PufferAI/PufferLib>`_ style.

    This trainer implements the PPO algorithm with
    clipped surrogate objectives, value-function loss, entropy regularization,
    and importance-sampling reweighting.

    **Loss function**

    Given a trajectory batch with actions :math:`a_t`, states :math:`s_t`,
    rewards :math:`r_t`, advantages :math:`A_t`, and old log-probabilities
    :math:`\log \pi_{\theta_\text{old}}(a_t \mid s_t)`, we define:

    - **Probability ratio**:

      .. math::

          r_t(\theta) = \exp\big( \log \pi_\theta(a_t \mid s_t) -
                                  \log \pi_{\theta_\text{old}}(a_t \mid s_t) \big)

    - **Clipped policy loss**:

      .. math::

          L^{\text{policy}}(\theta) =
              - \mathbb{E}_t \Big[ \min\big( r_t(\theta) A_t,\;
                                             \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \Big]

      where :math:`\epsilon` is the PPO clipping parameter.

    - **Value-function loss (with clipping)**:

      .. math::

          L^{\text{value}}(\theta) =
              \tfrac{1}{2} \mathbb{E}_t \Big[ \max\big( (V_\theta(s_t) - R_t)^2,\;
                                                       (\text{clip}(V_\theta(s_t), V_{\theta_\text{old}}(s_t) - \epsilon,
                                                                    V_{\theta_\text{old}}(s_t) + \epsilon) - R_t)^2 \big) \Big]

      where :math:`R_t = A_t + r_t` are return targets.

    - **Entropy bonus**:

      .. math::

          L^{\text{entropy}}(\theta) = \mathbb{E}_t \big[ \mathcal{H}[\pi_\theta(\cdot \mid s_t)] \big]

      which encourages exploration.

    - **Total loss**:

      .. math::

          L(\theta) = L^{\text{policy}}(\theta)
                      + c_v L^{\text{value}}(\theta)
                      - c_e L^{\text{entropy}}(\theta)

      where :math:`c_v` and :math:`c_e` are coefficients for the value and entropy terms.

    **Prioritized minibatch sampling and importance weighting**

    This trainer uses a prioritized categorical distribution over environments to
    form minibatches. For each environment index :math:`i \in \{1,\dots,N\}`,
    we define a *priority* from the trajectory advantages:

    .. math::

        \tilde{p}_i \;=\; \Big\| A_{\cdot,i} \Big\|_1^{\,\alpha}
        \quad\text{with}\quad
        \Big\| A_{\cdot,i} \Big\|_1 \;=\; \sum_{t=1}^{T} \big|A_{t,i}\big|,

    where :math:`\alpha \ge 0` (:attr:`importance_sampling_alpha`) controls the
    strength of prioritization. We then form a categorical sampling distribution

    .. math::

        P(i) \;=\; \frac{\tilde{p}_i}{\sum_{k=1}^{N} \tilde{p}_k},

    and sample indices :math:`\{i\}` to create each minibatch
    (:func:`jax.random.choice` with probabilities :math:`P(i)`).
    This mirrors Prioritized Experience Replay (PER), where :math:`\tilde{p}` is
    derived from TD-error magnitude; here we use the per-environment advantage
    magnitude as a proxy for learning progress. This design is also inspired by
    recent large-scale self-play systems for autonomous driving.

    To correct sampling bias we apply PER-style importance weights
    (:attr:`importance_sampling_beta` with optional linear annealing):

    .. math::

        w_i(\beta_t) \;=\; \Big(N \, P(i)\Big)^{-\beta_t},
        \qquad \beta_t \in [0,1].

    In classical PER, :math:`w_i` is often normalized by :math:`\max_j w_j` to keep
    the scale bounded; in this implementation we omit that normalization and use
    :math:`w_i` directly. The minibatch advantages are standardized and *reweighted*
    with these IS weights before the PPO loss:

    .. math::

        \hat{A}_{t,i}
        \;=\;
        w_i(\beta_t)\;
        \frac{A_{t,i} - \mu_{\text{mb}}(A)}{\sigma_{\text{mb}}(A)+\varepsilon}.

    **Off-policy correction of advantages (V-trace)**

    After sampling, we recompute log-probabilities under the *current* policy
    (:code:`td.new_log_prob = pi.log_prob(td.action)`) and compute
    targets/advantages with a V-trace–style off-policy correction in
    :meth:`compute_advantages`.

    ---
    **References**

    - Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
    - Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, ICML 2018.
    - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016.
    - Schaul et al., *Prioritized Experience Replay*, ICLR 2016.
    - Cusumano-Towner et al., *Robust Autonomy Emerges from Self-Play*, ICML 2025.
    """

    ppo_clip_eps: jax.Array
    r"""
    PPO clipping parameter :math:`\epsilon` used for both the policy ratio clip
    and (value-function clip.
    """

    ppo_value_coeff: jax.Array
    r"""
    Coefficient :math:`c_v` scaling the value-function loss term in the total loss.
    """

    ppo_entropy_coeff: jax.Array
    r"""
    Coefficient :math:`c_e` scaling the entropy bonus (encourages exploration).
    """

    importance_sampling_alpha: jax.Array
    r"""
    Prioritization strength :math:`\alpha \ge 0` for minibatch sampling;
    higher values put more probability mass on envs with larger advantages.
    """

    importance_sampling_beta: jax.Array
    r"""
    Initial PER importance-weight exponent :math:`\beta \in [0,1]` used in
    :math:`w_i(\beta) = (N P(i))^{-\beta}`; compensates sampling bias.
    """

    anneal_importance_sampling_beta: jax.Array
    r"""
    If nonzero/True, linearly anneals :math:`\beta` toward 1 across training
    (more correction later in training).
    """

    num_envs: int = field(default=4096, metadata={"static": True})
    r"""
    Number of vectorized environments :math:`N` running in parallel.
    """

    num_epochs: int = field(default=2000, metadata={"static": True})
    """
    Number of PPO training epochs (outer loop count).
    """

    num_steps_epoch: int = field(default=128, metadata={"static": True})
    r"""
    Rollout horizon :math:`T` per epoch; total collected steps = :math:`N \times T`.
    """

    num_minibatches: int = field(default=8, metadata={"static": True})
    """
    Number of minibatches per epoch used for PPO updates.
    """

    minibatch_size: int = field(default=512, metadata={"static": True})
    r"""
    Minibatch size (number of env indices sampled per update); typically
    :math:`N / \text{num_minibatches}`.
    """

    num_segments: int = field(default=4096, metadata={"static": True})
    r"""
    Number of vectorized environments times max number of agents.
    """

    @classmethod
    def Create(
        cls,
        env: "Environment",
        model: "Model",
        key: ArrayLike = jax.random.key(1),
        learning_rate: float = 2e-2,
        max_grad_norm: float = 0.4,
        ppo_clip_eps: float = 0.2,
        ppo_value_coeff: float = 0.5,
        ppo_entropy_coeff: float = 0.03,
        importance_sampling_alpha: float = 0.4,
        importance_sampling_beta: float = 0.1,
        advantage_gamma: float = 0.99,
        advantage_lambda: float = 0.95,
        advantage_rho_clip: float = 0.5,
        advantage_c_clip: float = 0.5,
        num_envs: int = 2048,
        num_epochs: int = 1000,
        num_steps_epoch: int = 64,
        num_minibatches: int = 10,
        minibatch_size: Optional[int] = None,
        accumulate_n_gradients: int = 1,  # only use for memory savings, bad performance
        clip_actions: bool = False,
        clip_range: Tuple[float, float] = (-0.2, 0.2),
        anneal_learning_rate: bool = True,
        learning_rate_decay_exponent: float = 2.0,
        learning_rate_decay_min_fraction: float = 0.01,
        anneal_importance_sampling_beta: bool = True,
        optimizer=optax.contrib.muon,
    ) -> Self:
        key, subkeys = jax.random.split(key)
        subkeys = jax.random.split(subkeys, num_envs)

        num_epochs = int(num_epochs)
        if anneal_learning_rate:
            schedule = optax.cosine_decay_schedule(
                init_value=float(learning_rate),
                alpha=float(learning_rate_decay_min_fraction),
                decay_steps=int(num_epochs),
                exponent=float(learning_rate_decay_exponent),
            )
        else:
            schedule = jnp.asarray(learning_rate, dtype=float)

        tx = optax.chain(
            optax.clip_by_global_norm(float(max_grad_norm)),
            optimizer(schedule, eps=1e-5),
            optax.apply_every(int(accumulate_n_gradients)),
        )

        metrics = nnx.MultiMetric(
            score=nnx.metrics.Average(argname="score"),
            loss=nnx.metrics.Average(argname="loss"),
            actor_loss=nnx.metrics.Average(argname="actor_loss"),
            value_loss=nnx.metrics.Average(argname="value_loss"),
            entropy=nnx.metrics.Average(argname="entropy"),
            approx_KL=nnx.metrics.Average(argname="approx_KL"),
            returns=nnx.metrics.Average(argname="returns"),
            ratio=nnx.metrics.Average(argname="ratio"),
            policy_std=nnx.metrics.Average(argname="policy_std"),
            explained_variance=nnx.metrics.Average(argname="explained_variance"),
            grad_norm=nnx.metrics.Average(argname="grad_norm"),
        )

        graphdef, graphstate = nnx.split(
            (model, nnx.Optimizer(model, tx, wrt=nnx.Param), metrics)
        )

        num_envs = int(num_envs)
        env = jax.vmap(lambda _: env)(jnp.arange(num_envs))
        if clip_actions:
            min_val, max_val = clip_range
            env = clip_action_env(env, min_val=min_val, max_val=max_val)
        env = vectorise_env(env)
        env = env.reset(env, subkeys)

        num_segments = int(num_envs * env.max_num_agents)
        num_minibatches = int(num_minibatches)
        if minibatch_size is None:
            minibatch_size = num_segments // num_minibatches
        minibatch_size = int(minibatch_size)

        assert (
            minibatch_size <= num_segments
        ), f"minibatch_size = {minibatch_size} is larger than num_envs * max_num_agents={num_segments}."

        model, optimizer, *rest = nnx.merge(graphdef, graphstate)
        model.reset(shape=(num_envs, env.max_num_agents))
        graphstate = nnx.state((model, optimizer, *rest))

        return cls(
            key=key,
            env=env,
            graphdef=graphdef,
            graphstate=graphstate,
            advantage_gamma=jnp.asarray(advantage_gamma, dtype=float),
            advantage_lambda=jnp.asarray(advantage_lambda, dtype=float),
            advantage_rho_clip=jnp.asarray(advantage_rho_clip, dtype=float),
            advantage_c_clip=jnp.asarray(advantage_c_clip, dtype=float),
            ppo_clip_eps=jnp.asarray(ppo_clip_eps, dtype=float),
            ppo_value_coeff=jnp.asarray(ppo_value_coeff, dtype=float),
            ppo_entropy_coeff=jnp.asarray(ppo_entropy_coeff, dtype=float),
            importance_sampling_alpha=jnp.asarray(
                importance_sampling_alpha, dtype=float
            ),
            importance_sampling_beta=jnp.asarray(importance_sampling_beta, dtype=float),
            anneal_importance_sampling_beta=jnp.asarray(
                anneal_importance_sampling_beta, dtype=float
            ),
            num_envs=int(num_envs),
            num_epochs=int(num_epochs),
            num_steps_epoch=int(num_steps_epoch),
            num_minibatches=int(num_minibatches),
            minibatch_size=int(minibatch_size),
            num_segments=int(num_segments),
        )

    @staticmethod
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
        tr = cast("PPOTrainer", tr)
        if shape is None:
            shape = (tr.num_envs, tr.env.max_num_agents)

        model, optimizer, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.reset(shape=shape, mask=mask)
        return replace(
            tr,
            graphstate=nnx.state((model, optimizer, *rest)),
        )

    @staticmethod
    def train(tr: "PPOTrainer", verbose=True):
        metrics_history = []
        tr, _ = tr.epoch(tr, jnp.asarray(0))
        it = trange(1, tr.num_epochs) if verbose else range(1, tr.num_epochs)
        start_time = time.perf_counter()

        log_folder = "runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tensorboard.SummaryWriter(log_folder)

        for epoch in it:
            model, optimizer, metrics, *rest = nnx.merge(tr.graphdef, tr.graphstate)
            metrics.reset()
            tr = replace(tr, graphstate=nnx.state((model, optimizer, metrics, *rest)))
            tr, _ = tr.epoch(tr, jnp.asarray(epoch))
            model, optimizer, metrics, *rest = nnx.merge(tr.graphdef, tr.graphstate)
            data = metrics.compute()

            data["elapsed"] = time.perf_counter() - start_time
            steps_done = (epoch + 1) * tr.num_envs * tr.num_steps_epoch
            data["steps_per_sec"] = steps_done / data["elapsed"]

            if verbose:
                it.set_postfix(
                    {
                        "steps/s": f"{data['steps_per_sec']:.2e}",
                        "avg_score": f"{data['score']:.2f}",
                    }
                )

                for k, v in data.items():
                    writer.scalar(k, v, step=int(epoch))
                writer.flush()

        print(
            f"steps/s: {data['steps_per_sec']:.2e}, final avg_score: {data['score']:.2f}"
        )
        return tr, metrics_history

    @staticmethod
    @nnx.jit(donate_argnames=("td", "seg_w"))
    def loss_fn(
        model: "Model",
        td: "TrajectoryData",
        seg_w: jax.Array,
        advantage_rho_clip: jax.Array,
        advantage_c_clip: jax.Array,
        advantage_gamma: jax.Array,
        advantage_lambda: jax.Array,
        ppo_clip_eps: jax.Array,
        ppo_value_coeff: jax.Array,
        ppo_entropy_coeff: jax.Array,
        entropy_key: jax.Array,
    ):
        """
        Compute the PPO minibatch loss.

        Parameters
        ----------
        model : Model
            Live model rebuilt by NNX for this step.
        td : TrajectoryData
            Time-stacked trajectory mini batch (e.g., shape ``[T, B, ...]``).
        seg_w : jax.Array
            Weights for advantage normalization.

        Returns
        -------
        jax.Array
            Scalar loss to be minimized.

        See Also
        --------
        Main docs: PPO trainer overview and equations.
        """
        # 1) Fordward pass
        old_value = td.value
        model.eval()
        pi, value = model(td.obs)
        td = replace(
            td,
            new_log_prob=pi.log_prob(td.action),
            value=jnp.squeeze(value, -1),
        )

        # 2) Recompute advantages and normalize
        td = PPOTrainer.compute_advantages(
            td,
            advantage_rho_clip,
            advantage_c_clip,
            advantage_gamma,
            advantage_lambda,
        )
        td.advantage = (
            (td.advantage - td.advantage.mean()) / (td.advantage.std() + 1e-8) * seg_w
        )
        td.advantage = jax.lax.stop_gradient(td.advantage)  # for policy loss
        td.returns = jax.lax.stop_gradient(td.returns)  # for value loss

        # 3) Value loss (clipped)
        value_pred_clipped = old_value + (td.value - old_value).clip(
            -ppo_clip_eps, ppo_clip_eps
        )
        value_losses = jnp.square(td.value - td.returns)
        value_losses_clipped = jnp.square(value_pred_clipped - td.returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # 4) Policy loss (clipped)
        log_ratio = td.new_log_prob - td.log_prob
        ratio = jnp.exp(log_ratio)
        actor_loss = -jnp.minimum(
            td.advantage * ratio,
            td.advantage * ratio.clip(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps),
        ).mean()

        # 5) Estimate Entropy (Entropy is not available for ditributions transformed by bijectors with non-constant Jacobian determinant)
        # H[π]=E_{a∼π}[−log π(a)]≈−1/K ∑_{k=1}^{K} log π(a^k)
        # entropy_loss = pi.entropy().mean()
        K = 2
        _, sample_logp = pi.sample_and_log_prob(seed=entropy_key, sample_shape=(K,))
        entropy = -jnp.mean(sample_logp, axis=0).mean()

        total_loss = (
            actor_loss + ppo_value_coeff * value_loss - ppo_entropy_coeff * entropy
        )

        # ----- diagnostics -----
        approx_kl = jnp.mean((jnp.exp(log_ratio) - 1.0) - log_ratio)
        explained_var = 1.0 - jnp.var(td.returns - td.value) / (
            jnp.var(td.returns) + 1e-8
        )
        policy_std = jnp.mean(jnp.exp(model.log_std.value))

        aux = {
            # losses
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            # policy diagnostics
            "approx_KL": approx_kl,
            "ratio": ratio,
            "policy_std": policy_std,
            # value diagnostics
            "explained_variance": explained_var,
            "returns": td.returns,
            "score": td.reward,
        }

        return total_loss, aux

    @staticmethod
    @jax.jit
    def epoch(tr: "PPOTrainer", epoch: ArrayLike):
        r"""
        Run one PPO training epoch.

        Returns
        -------
        (PPOTrainer, TrajectoryData)
            Updated trainer state and the most recent time-stacked rollout.
        """
        beta_t = tr.importance_sampling_beta + tr.anneal_importance_sampling_beta * (
            1.0 - tr.importance_sampling_beta
        ) * (epoch / tr.num_epochs)

        key, sample_key, reset_key, entropy_keys_key = jax.random.split(tr.key, 4)
        subkeys = jax.random.split(reset_key, tr.num_envs)
        entropy_keys = jax.random.split(entropy_keys_key, tr.num_minibatches)

        # 0) Reset the environment and LSTM carry
        model, optimizer, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.reset(
            shape=(tr.num_envs, tr.env.max_num_agents), mask=tr.env.done(tr.env)
        )
        tr = replace(
            tr,
            key=key,
            env=jax.vmap(tr.env.reset_if_done)(tr.env, tr.env.done(tr.env), subkeys),
            graphstate=nnx.state((model, optimizer, *rest)),
        )

        # 1) Gather data -> shape: (time, num_envs, num_agents, *)
        tr, td = tr.trajectory_rollout(tr, tr.num_steps_epoch)

        # Reshape data (time, num_envs, num_agents, *) -> (time, num_envs*num_agents, *)
        td = jax.tree_util.tree_map(
            lambda x: x.reshape((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:]), td
        )

        # 2) Compute advantages
        td = tr.compute_advantages(
            td,
            tr.advantage_rho_clip,
            tr.advantage_c_clip,
            tr.advantage_gamma,
            tr.advantage_lambda,
        )

        # 3) Importance sampling
        prio_weights = jnp.nan_to_num(
            jnp.power(jnp.abs(td.advantage).sum(axis=0), tr.importance_sampling_alpha),
            False,
            0.0,
            0.0,
        )
        prio_probs = prio_weights / (prio_weights.sum() + 1.0e-8)
        idxs = jax.random.choice(
            sample_key,
            a=tr.num_segments,
            p=prio_probs,
            shape=(tr.num_minibatches, tr.minibatch_size),
        )

        @partial(jax.jit, donate_argnames=("idx",))
        def train_batch(carry: Tuple, idx: jax.Array) -> Tuple:
            # 4.0) Unpack model
            tr, td, weights = carry
            model, optimizer, metrics, *rest = nnx.merge(tr.graphdef, tr.graphstate)
            idx, entropy_key = idx

            # 4.1) Importance sampling
            mb_td = jax.tree_util.tree_map(lambda x: jnp.take(x, idx, axis=1), td)
            seg_w = jnp.power(
                tr.num_segments * jnp.take(weights[None, :], idx, axis=1), -beta_t
            )

            # 4.2) Compute gradients
            (loss, aux), grads = nnx.value_and_grad(tr.loss_fn, has_aux=True)(
                model,
                mb_td,
                seg_w,
                tr.advantage_rho_clip,
                tr.advantage_c_clip,
                tr.advantage_gamma,
                tr.advantage_lambda,
                tr.ppo_clip_eps,
                tr.ppo_value_coeff,
                tr.ppo_entropy_coeff,
                entropy_key,
            )

            # 4.3) Train model
            model.train()
            optimizer.update(model, grads)

            # 4.4) Log metrics
            metrics.update(
                loss=loss,
                grad_norm=optax.global_norm(grads),
                **aux,
            )

            # 4.5) Return updated model
            tr = replace(tr, graphstate=nnx.state((model, optimizer, metrics, *rest)))
            return (tr, td, weights), loss

        # 4) Loop over mini batches
        (tr, td, prio_probs), loss = jax.lax.scan(
            train_batch, (tr, td, prio_probs), xs=(idxs, entropy_keys), unroll=4
        )

        return tr, td
