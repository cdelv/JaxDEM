# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Implementation of PPO algorithm.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import TYPE_CHECKING, Tuple, Optional

try:
    # Python 3.11+
    from typing import Self
except ImportError:
    from typing_extensions import Self

from dataclasses import dataclass, field
from functools import partial
import time
import datetime
from pathlib import Path
import json

from flax import nnx
from flax.metrics import tensorboard
import optax
from tqdm.auto import trange

from . import Trainer, TrajectoryData
from ..envWrappers import clip_action_env, vectorise_env

if TYPE_CHECKING:
    from ..environments import Environment
    from ..models import Model


def _hparam_dict_from_tr(tr):
    return {
        "algo": "PPO",
        "num_envs": int(tr.env.num_envs),
        "max_num_agents": int(tr.env.max_num_agents),
        "num_steps_epoch": int(tr.num_steps_epoch),
        "num_minibatches": int(tr.num_minibatches),
        "minibatch_size": int(tr.minibatch_size),
        "num_epochs": int(tr.num_epochs),
        "gamma": float(tr.advantage_gamma),
        "gae_lambda": float(tr.advantage_lambda),
        "rho_clip": float(tr.advantage_rho_clip),
        "c_clip": float(tr.advantage_c_clip),
        "ppo_clip_eps": float(tr.ppo_clip_eps),
        "value_coeff": float(tr.ppo_value_coeff),
        "entropy_coeff": float(tr.ppo_entropy_coeff),
        "is_alpha": float(tr.importance_sampling_alpha),
        "is_beta0": float(tr.importance_sampling_beta),
        "is_beta_anneal": bool(tr.anneal_importance_sampling_beta),
        # optionally optimizer info if accessible
    }


def _log_hparams_fallback(writer, tr, step=0):
    hp = _hparam_dict_from_tr(tr)
    writer.text("hparams/json", json.dumps(hp, indent=2), step=step)


@Trainer.register("PPO")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
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

    num_epochs: int
    """
    Number of PPO training epochs (outer loop count).
    """

    stop_at_epoch: int
    """
    Number of PPO training epochs (outer loop count).
    """

    num_steps_epoch: int = field(metadata={"static": True})
    r"""
    Rollout horizon :math:`T` per epoch; total collected steps = :math:`N \times T`.
    """

    num_minibatches: int = field(metadata={"static": True})
    """
    Number of minibatches per epoch used for PPO updates.
    """

    minibatch_size: int = field(metadata={"static": True})
    r"""
    Minibatch size (number of env indices sampled per update); typically
    :math:`N / \text{num_minibatches}`.
    """

    @classmethod
    @partial(jax.named_call, name="PPOTrainer.Create")
    def Create(
        cls,
        env: "Environment",
        model: "Model",
        seed: Optional[int] = None,
        key: ArrayLike = jax.random.key(1),
        # Learning
        optimizer=optax.contrib.muon,
        learning_rate: float = 0.015,
        anneal_learning_rate: bool = True,
        max_grad_norm: float = 1.5,
        accumulate_n_gradients: int = 1,
        # PPO parameters
        ppo_clip_eps: float = 0.2,
        ppo_value_coeff: float = 2.0,
        ppo_entropy_coeff: float = 0.001,
        # Advantage parameters
        advantage_gamma: float = 0.99,
        advantage_lambda: float = 0.95,
        advantage_rho_clip: float = 1.0,
        advantage_c_clip: float = 1.0,
        # PER parameters
        importance_sampling_alpha: float = 0.8,
        importance_sampling_beta: float = 0.2,
        anneal_importance_sampling_beta: bool = True,
        # Batches
        num_envs: int = 1024,
        num_steps_epoch: int = 64,
        num_minibatches: int = 4,
        minibatch_size: Optional[int] = None,
        # Iterations
        num_epochs: int = 1000,
        total_timesteps: Optional[int] = None,
        stop_at_epoch: Optional[int] = None,
        # Env wrappers
        clip_actions: bool = False,
        clip_range: Tuple[float, float] = (-0.2, 0.2),
    ) -> Self:
        # --- RNG split ---
        if seed is not None:
            key = jax.random.key(int(seed))
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, int(num_envs))

        # --- Vectorize envs before sizing math ---
        num_envs = int(num_envs)
        env = jax.vmap(lambda _: env)(jnp.arange(num_envs))
        if clip_actions:
            min_val, max_val = clip_range
            env = clip_action_env(env, min_val=float(min_val), max_val=float(max_val))
        env = vectorise_env(env)
        env = env.reset(env, subkeys)

        # --- Derived sizes ---
        num_steps_epoch = int(num_steps_epoch)
        num_segments = int(num_envs * env.max_num_agents)
        total_steps_per_epoch = int(num_segments * num_steps_epoch)
        num_minibatches = int(num_minibatches)

        if minibatch_size is not None:
            minibatch_size = int(minibatch_size)
            num_minibatches = total_steps_per_epoch // minibatch_size
        else:
            minibatch_size = total_steps_per_epoch // num_minibatches

        assert (
            num_minibatches % int(accumulate_n_gradients) == 0
        ), f"num_minibatches={num_minibatches} must be divisible by accumulate_n_gradients={accumulate_n_gradients}"

        assert (
            1 <= minibatch_size <= total_steps_per_epoch
        ), f"minibatch_size={minibatch_size} must be in [1, {total_steps_per_epoch}]"

        assert (
            total_steps_per_epoch % minibatch_size == 0
        ), f"total_steps_per_epoch={total_steps_per_epoch} must be divisible by minibatch_size={minibatch_size}"

        # --- Epoch count ---
        if total_timesteps is not None:
            total_timesteps = int(total_timesteps)
            assert (
                total_timesteps % total_steps_per_epoch == 0
            ), f"total_timesteps={total_timesteps} must be divisible by total_steps_per_epoch=num_envs * env.max_num_agents * num_steps_epoch={total_steps_per_epoch}"
            num_epochs = total_timesteps // total_steps_per_epoch
        num_epochs = int(num_epochs)

        # --- Stop-at-epoch ---
        if stop_at_epoch is None:
            stop_at_epoch = num_epochs
        stop_at_epoch = int(stop_at_epoch)
        assert (
            1 <= stop_at_epoch <= num_epochs
        ), f"stop_at_epoch={stop_at_epoch} must be in [1, num_epochs={num_epochs}]"

        # --- Optimizer and metrics ---
        if anneal_learning_rate:
            schedule = optax.cosine_decay_schedule(
                init_value=float(learning_rate),
                decay_steps=num_epochs,
            )
        else:
            schedule = float(learning_rate)

        tx = optax.chain(
            optax.clip_by_global_norm(float(max_grad_norm)),
            optimizer(schedule, eps=1e-12),
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
            explained_variance=nnx.metrics.Average(argname="explained_variance"),
            grad_norm=nnx.metrics.Average(argname="grad_norm"),
        )
        metrics.reset()

        graphdef, graphstate = nnx.split(
            (model, nnx.Optimizer(model, tx, wrt=nnx.Param), metrics)
        )

        # --- Reset model carry with correct batch shape ---
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
            num_epochs=num_epochs,
            stop_at_epoch=stop_at_epoch,
            num_steps_epoch=num_steps_epoch,
            num_minibatches=num_minibatches,
            minibatch_size=minibatch_size,
        )

    @staticmethod
    @partial(jax.named_call, name="PPOTrainer.one_epoch")
    def one_epoch(tr: "PPOTrainer", epoch):
        tr, td = tr.epoch(tr, epoch)
        model, optimizer, metrics, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        data = metrics.compute()
        metrics.reset()
        tr.graphstate = nnx.state((model, optimizer, metrics, *rest))
        return tr, td, data

    @staticmethod
    def train(
        tr: "PPOTrainer",
        verbose: bool = True,
        log: bool = True,
        directory: Path | str = "runs",
        save_every: int = 2,
        start_epoch: int = 0,
    ):
        total_epochs = int(tr.stop_at_epoch)
        start_epoch = int(start_epoch)
        save_every = int(save_every)

        writer: tensorboard.SummaryWriter | None = None
        directory = Path(directory)
        log_folder = directory / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if log:
            directory.mkdir(parents=True, exist_ok=True)
            writer = tensorboard.SummaryWriter(log_folder)
            _log_hparams_fallback(writer, tr, step=0)

        # warmup JIT
        tr, td, data = tr.one_epoch(tr, jnp.asarray(0, dtype=int))

        if writer is not None:
            for k, v in data.items():
                writer.scalar(k, v, step=0)
            writer.flush()

        start_time = time.perf_counter()
        it = (
            trange(start_epoch + 1, total_epochs)
            if verbose
            else range(start_epoch + 1, total_epochs)
        )
        for epoch in it:
            tr, td, data = tr.one_epoch(tr, jnp.asarray(epoch, dtype=int))

            elapsed = time.perf_counter() - start_time
            steps_done = (
                (epoch - start_epoch)
                * tr.env.max_num_agents
                * tr.env.num_envs
                * tr.num_steps_epoch
            )

            data["elapsed"] = elapsed
            data["steps_per_sec"] = steps_done / max(elapsed, 1e-9)

            if epoch % save_every == 0:
                if verbose:
                    postfix = {
                        "steps/s": f"{data['steps_per_sec']:.2e}",
                        "avg_score": f"{data['score']:.2f}",
                    }
                    set_postfix = getattr(it, "set_postfix", None)
                    if set_postfix:
                        set_postfix(postfix)

                if writer is not None:
                    for k, v in data.items():
                        writer.scalar(k, v, step=epoch)
                    writer.flush()

        print(
            f"steps/s: {data['steps_per_sec']:.2e}, final avg_score: {data['score']:.2f}"
        )
        if writer is not None:
            writer.close()

        return tr

    @staticmethod
    @nnx.jit
    @partial(jax.named_call, name="PPOTrainer.loss_fn")
    def loss_fn(
        model: "Model",
        td: "TrajectoryData",  # [T, M, ...] minibatch view
        returns: jax.Array,
        advantage: jax.Array,
        ppo_clip_eps: jax.Array,
        ppo_value_coeff: jax.Array,
        ppo_entropy_coeff: jax.Array,
    ):
        # 1) Forward.
        old_value = td.value
        pi, td.value = model(td.obs)
        new_log_prob = pi.log_prob(td.action)
        td.value = jnp.squeeze(td.value, -1)
        log_ratio = new_log_prob - td.log_prob
        td.ratio = jnp.exp(log_ratio)

        # 2) Value loss (clipped).
        value_pred_clipped = old_value + (td.value - old_value).clip(
            -ppo_clip_eps, ppo_clip_eps
        )
        value_losses = jnp.square(td.value - returns)
        value_losses_clipped = jnp.square(value_pred_clipped - returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # 3) Policy loss (clipped).
        actor_loss = jnp.maximum(
            -advantage * td.ratio,
            -advantage * td.ratio.clip(1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps),
        ).mean()

        # 4) Entropy.
        entropy = -(jnp.exp(new_log_prob) * new_log_prob).sum(axis=0).mean()
        # entropy = pi.entropy().mean()

        # 5) Total Loss.
        total_loss = (
            actor_loss + ppo_value_coeff * value_loss - ppo_entropy_coeff * entropy
        )

        # 6) Diagnostics.
        approx_kl = jax.lax.stop_gradient(jnp.mean((td.ratio - 1.0) - log_ratio))
        explained_var = jax.lax.stop_gradient(
            1.0 - jnp.var(returns - td.value) / (jnp.var(returns) + 1e-8)
        )
        aux = {
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_KL": approx_kl,
            "explained_variance": explained_var,
            "ratio": jax.lax.stop_gradient(td.ratio),
            "value": jax.lax.stop_gradient(td.value),
            "returns": returns.mean(),
            "score": td.reward.mean(),
        }
        return total_loss, aux

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="PPOTrainer.epoch")
    def epoch(tr: "PPOTrainer", epoch: ArrayLike):
        beta_t = tr.importance_sampling_beta + tr.anneal_importance_sampling_beta * (
            1.0 - tr.importance_sampling_beta
        ) * (epoch / tr.num_epochs)
        # 0) Split PRNG keys and reset environments where done == True.
        reset_root, rollout_key, mb_root = jax.random.split(tr.key, 3)
        subkeys = jax.random.split(reset_root, tr.env.num_envs)
        tr.env = jax.vmap(tr.env.reset_if_done)(tr.env, tr.env.done(tr.env), subkeys)

        # 1) Roll out trajectories; td has shape [T, E, A, ...].
        tr.env, tr.graphstate, rollout_key, td = tr.trajectory_rollout(
            tr.env, tr.graphdef, tr.graphstate, rollout_key, tr.num_steps_epoch
        )

        # 2) Flatten the agent axis to get [T, S, ...].
        td = jax.tree_util.tree_map(
            lambda x: x.reshape((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:]),
            td,
        )
        T, S = td.value.shape[:2]

        @partial(jax.named_call, name="PPOTrainer.train_batch")
        def train_batch(carry, _):
            # 3.0) Unpack carry and model, then split keys.
            graphdef, graphstate, td, key = carry
            key, samp_key = jax.random.split(key)
            model, optimizer, metrics, *rest = nnx.merge(graphdef, graphstate)

            # 3.1) Compute advantages.
            returns, advantage = tr.compute_advantages(
                td.value,
                td.reward,
                td.ratio,
                td.done,
                tr.advantage_rho_clip,
                tr.advantage_c_clip,
                tr.advantage_gamma,
                tr.advantage_lambda,
            )

            # 3.2) Compute PER sampling probabilities.
            prio_p = jnp.sum(jnp.abs(advantage), axis=0)
            prio_w = jnp.nan_to_num(
                jnp.power(prio_p, tr.importance_sampling_alpha), False, 0.0, 0.0, 0.0
            )
            prio_p = (prio_w + 1e-6) / (prio_w.sum() + 1e-6)

            # Sample segment indices.
            idx = jax.random.choice(
                samp_key,
                a=S,
                shape=(tr.minibatch_size // T,),
                p=prio_p,
            )  # [M]

            # Importance weights: (S * p[idx])^{-beta}, shape [M]; broadcast to [T, M].
            seg_w = jnp.power(S * prio_p[idx], -beta_t)  # [M]

            # 3.3) Normalize and slice advantages.
            adv = jnp.take(advantage, idx, axis=1)
            adv = seg_w * (adv - adv.mean()) / (adv.std() + 1e-8)

            # 3.4) Slice trajectory data to [T, M].
            mb_td = jax.tree_util.tree_map(lambda x: jnp.take(x, idx, axis=1), td)

            # 3.5) Compute loss and gradients.
            model.eval()
            (loss, aux), grads = nnx.value_and_grad(tr.loss_fn, has_aux=True)(
                model,
                mb_td,
                jnp.take(returns, idx, axis=1),
                adv,
                tr.ppo_clip_eps,
                tr.ppo_value_coeff,
                tr.ppo_entropy_coeff,
            )

            # 3.6) Apply optimizer step.
            model.train()
            optimizer.update(model, grads)

            # Write back value and ratio to global buffers.
            td.value = td.value.at[:, idx].set(aux["value"])
            td.ratio = td.ratio.at[:, idx].set(aux["ratio"])

            # 3.7) Log metrics.
            metrics.update(
                loss=loss,
                actor_loss=aux["actor_loss"],
                value_loss=aux["value_loss"],
                entropy=aux["entropy"],
                approx_KL=aux["approx_KL"],
                explained_variance=aux["explained_variance"],
                grad_norm=optax.global_norm(grads),
                ratio=aux["ratio"].mean(),
                returns=aux["returns"],
                score=aux["score"],
            )

            graphstate = nnx.state((model, optimizer, metrics, *rest))
            return (graphdef, graphstate, td, key), loss

        # 3) Scan over minibatches.
        (tr.graphdef, tr.graphstate, td, tr.key), _ = jax.lax.scan(
            train_batch,
            (tr.graphdef, tr.graphstate, td, mb_root),
            xs=None,
            length=tr.num_minibatches,
            unroll=2,
        )

        return tr, td
