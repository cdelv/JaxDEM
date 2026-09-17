# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Implementation of the PPO algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

try:
    # Python 3.11+
    from typing import Self
except ImportError:
    from typing_extensions import Self

import datetime
import json
import math
import time
from dataclasses import dataclass, field
from functools import partial
from numbers import Integral
from pathlib import Path

import optax  # type: ignore[import-untyped]
from flax import nnx
from tensorboardX import SummaryWriter  # type: ignore[import-untyped]
from tqdm.auto import trange  # type: ignore[import-untyped]

from ..env_wrappers import clip_action_env, vectorise_env
from . import Trainer, TrajectoryData

if TYPE_CHECKING:
    from ..environments import Environment
    from ..models import Model


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    """Return an integer argument without silently accepting bools or fractions."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _epoch_learning_rate_schedule(
    learning_rate: float, num_epochs: int, updates_per_epoch: int
) -> Any:
    """Return cosine decay indexed by groups of planned optimizer updates."""
    epoch_schedule = optax.cosine_decay_schedule(
        init_value=float(learning_rate), decay_steps=int(num_epochs)
    )
    return lambda update: epoch_schedule(update // int(updates_per_epoch))


def _minibatch_selection(
    horizon: int, num_segments: int, batch_size: int, batch_index: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Select cyclic transitions and retain their full sequence context."""
    total = horizon * num_segments
    start = (batch_index * batch_size) % total
    # Aligned batches contain only full sequences. Otherwise include enough
    # sequences for a partial first and last sequence, without duplicate agents.
    count = (
        batch_size // horizon
        if batch_size % horizon == 0
        else min(num_segments, (batch_size + 2 * horizon - 2) // horizon)
    )
    indices = (start // horizon + jnp.arange(count)) % num_segments
    positions = indices[None, :] * horizon + jnp.arange(horizon)[:, None]
    loss_mask = (positions - start) % total < batch_size
    return indices, loss_mask


def _build_optimizer(
    optimizer: Any,
    schedule: Any,
    max_grad_norm: float,
    accumulate_n_gradients: int,
) -> Any:
    """Clip averaged raw gradients, then apply the requested Optax optimizer.

    Default Muon treats stacked MinGRU layers as independent matrices while
    retaining Optax's usual routing elsewhere. Explicit dimension overrides
    supplied through a partial factory take precedence.
    """
    factory = optimizer
    configured_dims = None
    while isinstance(factory, partial):
        configured_dims = factory.keywords.get(
            "muon_weight_dimension_numbers", configured_dims
        )
        factory = factory.func
    kwargs = {}
    if factory is optax.contrib.muon and configured_dims is None:
        kwargs["muon_weight_dimension_numbers"] = _muon_dimensions
    inner_tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optimizer(schedule, eps=1e-12, **kwargs),
    )
    if accumulate_n_gradients == 1:
        return inner_tx
    return optax.MultiSteps(
        inner_tx,
        every_k_schedule=accumulate_n_gradients,
        use_grad_mean=True,
    )


def _muon_dimensions(params: Any) -> Any:
    """Keep Optax's matrix routing, with independent stacked MinGRU layers."""

    def dimensions(path: Any, value: jax.Array) -> Any:
        if value.ndim == 3 and any(
            getattr(entry, "key", None) == "mingru_kernel" for entry in path
        ):
            # [layers, input, output]: unspecified axis 0 is a batch axis.
            return optax.contrib.MuonDimensionNumbers(1, 2)
        return optax.contrib.MuonDimensionNumbers() if value.ndim == 2 else None

    return jax.tree_util.tree_map_with_path(dimensions, params)


def _hparam_dict_from_tr(tr: PPOTrainer) -> dict[str, Any]:
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
        "drip_decay": float(tr.drip_decay),
        "ppo_clip_eps": float(tr.ppo_clip_eps),
        "value_coeff": float(tr.ppo_value_coeff),
        "entropy_coeff": float(tr.ppo_entropy_coeff),
        "vtrace": bool(tr.vtrace),
        "caps_temporal_coeff": tr.caps_temporal_coeff,
        "caps_spatial_coeff": tr.caps_spatial_coeff,
        "caps_noise_std": jax.device_get(tr.caps_noise_std).tolist(),
        # optionally optimizer info if accessible
    }


def _log_hparams_fallback(writer: Any, tr: PPOTrainer, step: int = 0) -> None:
    hp = _hparam_dict_from_tr(tr)
    writer.add_text("hparams/json", json.dumps(hp, indent=2), global_step=step)


@Trainer.register("PPO")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PPOTrainer(Trainer):
    r"""Proximal Policy Optimization (PPO) trainer in `PufferLib <https://github.com/PufferAI/PufferLib>`_ style.

    This trainer implements the PPO algorithm with
    clipped surrogate objectives, value-function loss, entropy regularization,
    and sequential minibatches of fixed-horizon agent segments. A segment may
    contain episode boundaries; it need not contain a complete episode.

    **Loss function**

    Given a trajectory batch with actions :math:`a_t`, states :math:`s_t`,
    rewards :math:`r_t`, advantages :math:`A_t`, and old log-probabilities
    :math:`\log \pi_{\theta_\text{old}}(a_t \mid s_t)`, we define:

    - **Probability ratio**:

      .. math::

          \rho_t(\theta) = \exp\big( \log \pi_\theta(a_t \mid s_t) -
                                  \log \pi_{\theta_\text{old}}(a_t \mid s_t) \big)

    - **Clipped policy loss**:

      .. math::

          L^{\text{policy}}(\theta) =
              - \mathbb{E}_t \Big[ \min\big( \rho_t(\theta) A_t,\;
                                             \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big) \Big]

      where :math:`\epsilon` is the PPO clipping parameter and :math:`A_t`
      is detached. Throughout these losses, :math:`\mathbb{E}_t[f_t]` means
      :math:`\sum_t m_t f_t / \max(1, \sum_t m_t)` over all time/agent
      entries in the minibatch, with active-agent indicator :math:`m_t`.

    - **Value-function loss (with clipping)**:

      .. math::

          L^{\text{value}}(\theta) =
              \tfrac{1}{2} \mathbb{E}_t \Big[ \max\big( (V_\theta(s_t) - R_t)^2,\;
                                                       (\text{clip}(V_\theta(s_t), V_{\theta_\text{old}}(s_t) - \epsilon,
                                                                    V_{\theta_\text{old}}(s_t) + \epsilon) - R_t)^2 \big) \Big]

      where :math:`R_t = \operatorname{stop\_gradient}(A_t + V_\theta(s_t))`
      are detached return targets.
      The clipping reference remains the immutable rollout value.

    - **Entropy bonus**:

      .. math::

          L^{\text{entropy}}(\theta) = \mathbb{E}_t \big[ \mathcal{H}[\pi_\theta(\cdot \mid s_t)] \big]

      which encourages exploration.

    - **Total loss**:

      .. math::

          L(\theta) = L^{\text{policy}}(\theta)
                      + c_v L^{\text{value}}(\theta)
                      - c_e L^{\text{entropy}}(\theta)
                      + \lambda_T L_T(\theta) + \lambda_S L_S(\theta)

      where :math:`c_v` and :math:`c_e` are coefficients for the value and entropy terms.

    **Optional CAPS regularization**

    Both coefficients default to zero, removing CAPS computations from the
    compiled update. We use a squared-Euclidean variant of CAPS:

    .. math::

        L_T = \mathbb{E}\!\left[\|u_\theta(o_{t+1},h_{t+1})
              - u_\theta(o_t,h_t)\|_2^2\right],\qquad
        L_S = \mathbb{E}\!\left[\|u_\theta(o_t+\epsilon,h_t)
              - u_\theta(o_t,h_t)\|_2^2\right],
        \quad \epsilon\sim\mathcal{N}(0,\operatorname{diag}(\sigma^2)).

    Continuous outputs are the Gaussian mean transformed through the action
    bijector; categorical outputs are probability vectors. No exploration
    samples or environment rewards enter these penalties. Distances are in
    action units and are not automatically normalized. Temporal pairs use
    consecutive policy decisions from the same active agent and episode;
    rollout-end pairs are omitted. Spatial comparisons share clean incoming
    recurrent state. Both branches remain differentiable.

    References
    ----------
    Mysore, S., Mabsout, B., Mancuso, R., and Saenko, K. (2021).
    *Regularizing Action Policies for Smooth Control with Reinforcement
    Learning*. ICRA. https://arxiv.org/abs/2012.06644.
    The paper uses unsquared Euclidean distances; this implementation uses
    squared distances with finite gradients at identical outputs.

    **Minibatch updates**

    Transitions are selected in contiguous agent-major blocks, cycling through
    the rollout when ``num_minibatches`` exceeds one pass. Full fixed-horizon
    sequences provide context, with a loss mask for partial sequences. Each update runs the
    current model before computing detached advantages and returns. Policy
    advantages are used directly, without normalization or sampling weights.
    Rollout values and behavior log-probabilities remain fixed for PPO clipping.

    With ``vtrace=False`` (default), targets use ordinary GAE. With
    ``vtrace=True``, a V-trace-style lambda trace uses the clipped
    current importance ratios in :meth:`Trainer.compute_advantages`. This
    still uses the PPO actor objective, not the IMPALA actor objective. Final-horizon and
    truncation bootstraps retain their collection-time estimates; episode
    boundary handling is independent of this choice.

    **Distributed Reward Information Processing (DRIP)**

    Before target calculation, rewards are replaced by the backward sum

    .. math::

        \widetilde r_t = r_t + d(1-\mathrm{done}_t)\widetilde r_{t+1},
        \qquad \widetilde r_T = 0.

    Here :math:`d` is ``drip_decay`` and ``done`` includes termination and
    truncation. This is an unnormalized sum of future rewards within the
    current rollout and episode. It changes reward scale and the learning
    objective; it does not conserve total reward or remove horizon dependence.
    Use :math:`0 < d \leq 1` to enable it; ``0.0`` leaves rewards unchanged.
    Rewards are not clipped or normalized by this trainer.

    ---
    **References**

    - Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
    - Espeholt et al., *IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures*, ICML 2018.
    - Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2015/2016.
    """

    drip_decay: jax.Array
    r"""
    Decay factor :math:`\lambda_{DRIP}` for Distributed Reward Information Processing (DRIP).
    Adds decayed future rewards backward within each rollout and episode.
    Set to 0.0 to disable (default).
    """

    ppo_clip_eps: jax.Array
    r"""
    PPO clipping parameter :math:`\epsilon` used for both the policy ratio clip
    and the value-function clip.
    """

    ppo_value_coeff: jax.Array
    r"""
    Coefficient :math:`c_v` scaling the value-function loss term in the total loss.
    """

    ppo_entropy_coeff: jax.Array
    r"""
    Coefficient :math:`c_e` scaling the entropy bonus (encourages exploration).
    """

    vtrace: jax.Array
    """Use current policy ratios in advantage estimation; otherwise use GAE."""

    num_epochs: int
    """
    Planned number of rollout-and-update iterations, also used by the LR schedule.
    This is not the number of replay passes over one rollout.
    """

    stop_at_epoch: int
    """
    Stop after this epoch. Must satisfy 1 ≤ stop_at_epoch ≤ num_epochs.
    """

    num_steps_epoch: int = jax.tree.static()
    r"""
    Rollout horizon :math:`T` in policy decisions per outer iteration. The
    rollout contains :math:`ST` slots, where
    :math:`S=\text{num\_envs}\times\text{max\_num\_agents}`, including padding.
    """

    num_minibatches: int = jax.tree.static()
    """
    Number of sequential minibatch iterations per rollout. With an explicit
    minibatch size, iterations beyond a complete pass replay from the start.
    Inactive-only blocks do not advance optimizer state or accumulation.
    """

    minibatch_size: int = jax.tree.static()
    r"""
    Number of transition slots selected per minibatch, including inactive slots.
    Must be between ``num_steps_epoch`` and
    ``num_envs * max_num_agents * num_steps_epoch``. Selection wraps at the
    rollout end. Full sequences are evaluated for recurrent context and targets;
    only selected transitions contribute to the loss. If omitted, defaults to
    rollout size divided exactly by ``num_minibatches``.
    """

    skip_frames: int = jax.tree.static()
    """
    Number of additional physics frames requested for each policy action.
    State advances by at most ``1 + skip_frames`` frames, ending at a boundary;
    intermediate physics-frame rewards are not accumulated.
    """

    caps_temporal_coeff: float = jax.tree.static(default=0.0)
    """Nonnegative temporal CAPS weight. Static: changing it recompiles updates."""

    caps_spatial_coeff: float = jax.tree.static(default=0.0)
    """Nonnegative spatial CAPS weight. Zero skips perturbations and extra forwards."""

    caps_noise_std: jax.Array = field(default_factory=lambda: jnp.asarray(0.05))
    """Spatial noise standard deviation in observation units, scalar or per feature."""

    @classmethod
    @partial(jax.named_call, name="PPOTrainer.Create")
    def Create(
        cls,
        env: Environment,
        model: Model,
        seed: int | None = None,
        key: ArrayLike | None = None,
        # Learning
        optimizer: Any = optax.contrib.muon,
        learning_rate: float = 1e-2,
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
        vtrace: bool = False,
        # DRIP parameters
        drip_decay: float = 0.0,
        # Batches
        num_envs: int = 64,
        num_steps_epoch: int = 64,
        num_minibatches: int = 4,
        minibatch_size: int | None = None,
        skip_frames: int = 0,
        # Iterations
        num_epochs: int = 1000,
        total_timesteps: int | None = None,
        stop_at_epoch: int | None = None,
        # Env wrappers
        clip_actions: bool = False,
        clip_range: tuple[float, float] = (-0.2, 0.2),
        # Optional action-policy smoothness
        caps_temporal_coeff: float = 0.0,
        caps_spatial_coeff: float = 0.0,
        caps_noise_std: ArrayLike = 0.05,
    ) -> Self:
        r"""Construct a PPO trainer from an environment and a model.

        Vectorizes the environment, builds the optimizer chain, and
        initializes the model carry. When enabled, learning-rate annealing is
        cosine decay over the planned optimizer updates in ``num_epochs``.
        With no inactive-only blocks it remains constant within an epoch;
        skipped empty blocks also pause the optimizer's schedule counter.
        Gradient accumulation averages raw minibatch gradients before clipping
        and the stateful optimizer. See the class-level field docstrings for
        parameter descriptions. Minibatch size counts transition slots;
        set ``minibatch_size`` explicitly to control replay independently of
        ``num_minibatches``. The former PER constructor arguments were removed.

        Parameters
        ----------
        env : Environment
            A *single* (non-vectorised) environment instance.
        model : Model
            An actor–critic model whose ``observation_space_size`` and
            ``action_space_size`` match ``env``.
        seed : int, optional
            Integer seed used for random number generation. Used only when
            ``key`` is not provided.
        key : jax.Array, optional
            PRNG key. When provided, it takes precedence over ``seed``
            (same rule as :meth:`jaxdem.System.create`).
        caps_temporal_coeff, caps_spatial_coeff : float
            Nonnegative CAPS loss weights, disabled by default. These are
            static configuration: changing either recompiles training.
        caps_noise_std : ArrayLike
            Nonnegative finite Gaussian perturbation standard deviation,
            scalar or shape ``(observation_space_size,)``. Expressed in the
            model's input units; use zero for features that must not vary.
            See the class docstring for the CAPS equations and paper citation.

        Returns
        -------
        PPOTrainer
            Ready-to-train trainer instance.

        """
        num_envs = _require_int("num_envs", num_envs, minimum=1)
        num_steps_epoch = _require_int("num_steps_epoch", num_steps_epoch, minimum=1)
        num_minibatches = _require_int("num_minibatches", num_minibatches, minimum=1)
        accumulate_n_gradients = _require_int(
            "accumulate_n_gradients", accumulate_n_gradients, minimum=1
        )
        skip_frames = _require_int("skip_frames", skip_frames, minimum=0)
        if minibatch_size is not None:
            minibatch_size = _require_int("minibatch_size", minibatch_size, minimum=1)
        if total_timesteps is None:
            num_epochs = _require_int("num_epochs", num_epochs, minimum=1)
        else:
            total_timesteps = _require_int(
                "total_timesteps", total_timesteps, minimum=1
            )
        if stop_at_epoch is not None:
            stop_at_epoch = _require_int("stop_at_epoch", stop_at_epoch, minimum=1)
        if num_minibatches % accumulate_n_gradients != 0:
            raise ValueError(
                f"num_minibatches={num_minibatches} must be divisible by "
                f"accumulate_n_gradients={accumulate_n_gradients}"
            )

        for name, coefficient in (
            ("caps_temporal_coeff", caps_temporal_coeff),
            ("caps_spatial_coeff", caps_spatial_coeff),
        ):
            if not math.isfinite(coefficient) or coefficient < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        caps_noise_std = jnp.asarray(caps_noise_std, dtype=float)
        if not bool(jnp.all(jnp.isfinite(caps_noise_std) & (caps_noise_std >= 0))):
            raise ValueError("caps_noise_std must be finite and nonnegative")
        if caps_noise_std.ndim != 0 and caps_noise_std.shape != (
            model.observation_space_size,
        ):
            raise ValueError(
                "caps_noise_std must be scalar or have one entry per observation feature"
            )

        # --- RNG split ---
        initial_agent_mask = jnp.asarray(env.agent_mask(env), dtype=bool)
        if initial_agent_mask.shape != (env.max_num_agents,):
            raise ValueError(
                "Environment.agent_mask() must have shape (max_num_agents,)."
            )
        if not bool(jnp.any(initial_agent_mask)):
            raise ValueError("Environment.agent_mask() must select at least one agent.")
        if key is None:
            key = jax.random.key(int(seed) if seed is not None else 1)
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, num_envs)

        # --- Vectorize envs before sizing math ---
        if clip_actions:
            if getattr(model, "discrete", False):
                raise ValueError(
                    "clip_actions is only supported for continuous policies"
                )
            min_val, max_val = clip_range
            env = clip_action_env(env, min_val=float(min_val), max_val=float(max_val))
        env = vectorise_env(env, n=num_envs)
        env = env.reset(env, subkeys)
        reset_agent_mask = jnp.asarray(env.agent_mask(env), dtype=bool)
        if not bool(jnp.all(jnp.any(reset_agent_mask, axis=-1))):
            raise ValueError(
                "Environment.reset() must leave at least one active agent per environment."
            )

        # --- Derived sizes ---
        num_segments = int(num_envs * env.max_num_agents)
        total_steps_per_epoch = int(num_segments * num_steps_epoch)

        if minibatch_size is None:
            if total_steps_per_epoch % num_minibatches != 0:
                raise ValueError(
                    "With minibatch_size=None, rollout size = "
                    f"num_envs ({num_envs}) * num_steps_epoch ({num_steps_epoch}) "
                    f"* env.max_num_agents ({env.max_num_agents}) = "
                    f"{total_steps_per_epoch} transitions must be divisible "
                    f"by num_minibatches={num_minibatches}. "
                    "Supply an explicit minibatch_size in "
                    f"[{num_steps_epoch}, {total_steps_per_epoch}], or choose "
                    "num_minibatches that divides the rollout size and is at "
                    f"most {num_segments}."
                )
            minibatch_size = total_steps_per_epoch // num_minibatches
        if not num_steps_epoch <= minibatch_size <= total_steps_per_epoch:
            raise ValueError(
                f"minibatch_size={minibatch_size} must be in "
                f"[{num_steps_epoch}, {total_steps_per_epoch}] transitions. "
                f"Minimum = num_steps_epoch ({num_steps_epoch}); maximum = "
                f"num_envs ({num_envs}) * num_steps_epoch ({num_steps_epoch}) "
                f"* env.max_num_agents ({env.max_num_agents}) = "
                f"{total_steps_per_epoch}. When minibatch_size=None, it is "
                f"computed as {total_steps_per_epoch} / "
                f"num_minibatches ({num_minibatches})."
            )
        # --- Epoch count ---
        if total_timesteps is not None:
            if total_timesteps % total_steps_per_epoch != 0:
                raise ValueError(
                    f"total_timesteps={total_timesteps} must be divisible by "
                    "total_steps_per_epoch=num_envs * env.max_num_agents * "
                    f"num_steps_epoch={total_steps_per_epoch}"
                )
            num_epochs = total_timesteps // total_steps_per_epoch

        # --- Stop-at-epoch ---
        if stop_at_epoch is None:
            stop_at_epoch = num_epochs
        if stop_at_epoch > num_epochs:
            raise ValueError(
                f"stop_at_epoch={stop_at_epoch} must be in [1, num_epochs={num_epochs}]"
            )

        # --- Optimizer ---
        updates_per_epoch = num_minibatches // accumulate_n_gradients
        if anneal_learning_rate:
            schedule = _epoch_learning_rate_schedule(
                learning_rate, num_epochs, updates_per_epoch
            )
        else:
            schedule = float(learning_rate)

        tx = _build_optimizer(
            optimizer,
            schedule,
            float(max_grad_norm),
            accumulate_n_gradients,
        )

        graphdef, graphstate = nnx.split(
            (model, nnx.Optimizer(model, tx, wrt=nnx.Param))
        )

        # --- Reset model carry with correct batch shape ---
        model, optimizer = nnx.merge(graphdef, graphstate)
        model.reset(shape=(num_envs, env.max_num_agents, 1))

        graphstate = nnx.state((model, optimizer))

        return cls(
            key=key,
            env=env,
            graphdef=graphdef,
            graphstate=graphstate,
            advantage_gamma=jnp.asarray(advantage_gamma, dtype=float),
            advantage_lambda=jnp.asarray(advantage_lambda, dtype=float),
            advantage_rho_clip=jnp.asarray(advantage_rho_clip, dtype=float),
            advantage_c_clip=jnp.asarray(advantage_c_clip, dtype=float),
            drip_decay=jnp.asarray(drip_decay, dtype=float),
            ppo_clip_eps=jnp.asarray(ppo_clip_eps, dtype=float),
            ppo_value_coeff=jnp.asarray(ppo_value_coeff, dtype=float),
            ppo_entropy_coeff=jnp.asarray(ppo_entropy_coeff, dtype=float),
            vtrace=jnp.asarray(vtrace, dtype=bool),
            num_epochs=num_epochs,
            stop_at_epoch=stop_at_epoch,
            num_steps_epoch=num_steps_epoch,
            num_minibatches=num_minibatches,
            minibatch_size=minibatch_size,
            skip_frames=skip_frames,
            caps_temporal_coeff=float(caps_temporal_coeff),
            caps_spatial_coeff=float(caps_spatial_coeff),
            caps_noise_std=caps_noise_std,
        )

    @staticmethod
    def train(
        tr: Trainer,
        verbose: bool = True,
        log: bool = True,
        directory: Path | str = "runs",
        save_every: int = 2,
        start_epoch: int = 0,
        debug_overflow_checks: bool = False,
        **kwargs: Any,
    ) -> PPOTrainer:
        """Run the full PPO training loop.

        Parameters
        ----------
        tr : Trainer
            Trainer instance (will be cast to :class:`PPOTrainer`).
        verbose : bool
            If ``True``, display a ``tqdm`` progress bar.
        log : bool
            If ``True``, write TensorBoard scalars to *directory*.
        directory : Path | str
            Root directory for TensorBoard logs.
        save_every : int
            Positive integer interval for syncing metrics and logging iterations
            when verbose output or logging is enabled. Does not save checkpoints.
        start_epoch : int
            Resume epoch counter for logging and rollout numbering. Exact
            learning-rate and momentum continuation also requires the restored
            trainer ``graphstate``; changing this label alone does not restore
            optimizer state. Exact trajectory continuation also requires the
            environment, PRNG key, and recurrent carry.
            Must lie in ``[0, stop_at_epoch]``. At ``stop_at_epoch``, returns
            the trainer unchanged without opening a writer or running an epoch.
        debug_overflow_checks : bool
            If ``True``, check the collider overflow flag after *every* epoch.
            This forces a host synchronization per epoch, which defeats async
            dispatch, so it is off by default. The flag is always checked
            once at the end of training.

        Returns
        -------
        PPOTrainer
            Trainer with updated parameters after training.

        """
        _ = kwargs
        tr_typed = cast("PPOTrainer", tr)
        total_epochs = _require_int(
            "trainer.stop_at_epoch", tr_typed.stop_at_epoch, minimum=1
        )
        start_epoch = _require_int("start_epoch", start_epoch, minimum=0)
        save_every = _require_int("save_every", save_every, minimum=1)
        if start_epoch > total_epochs:
            raise ValueError(
                f"start_epoch={start_epoch} must not exceed "
                f"stop_at_epoch={total_epochs}"
            )
        if start_epoch == total_epochs:
            return tr_typed

        writer: Any = None
        directory = Path(directory)
        log_folder = directory / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        if log:
            directory.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_folder)
            if writer:
                _log_hparams_fallback(writer, tr_typed, step=0)

        # Precompute steps-per-epoch from Python-side constants (no JAX sync).
        steps_per_epoch = (
            int(tr_typed.env.max_num_agents)
            * int(tr_typed.env.num_envs)
            * int(tr_typed.num_steps_epoch)
            * (1 + int(tr_typed.skip_frames))
        )

        # Warmup JIT (first call traces + compiles). Runs epoch ``start_epoch``
        # so a resumed run executes every epoch index exactly once.
        tr_typed, _td, data = tr_typed.epoch(
            tr_typed, jnp.asarray(start_epoch, dtype=int)
        )
        if debug_overflow_checks and jnp.any(tr_typed.env.system.collider.overflow):
            print("Warning: overflow detected in collider")

        if writer is not None:
            data_np = jax.device_get(data)
            for k, v in data_np.items():
                writer.add_scalar(k, float(v), global_step=start_epoch)
            writer.flush()

        start_time = time.perf_counter()

        it = (
            trange(start_epoch + 1, total_epochs)
            if verbose
            else range(start_epoch + 1, total_epochs)
        )
        for epoch in it:
            # Dispatch is async — returns immediately with futures.
            tr_typed, _td, data = tr_typed.epoch(
                tr_typed, jnp.asarray(epoch, dtype=int)
            )
            # NOTE: collider overflow is only checked here when explicitly
            # requested -- `jnp.any` forces a host sync every epoch, defeating
            # the async dispatch noted above. It is always checked once after
            # the loop.
            if debug_overflow_checks:
                if jnp.any(tr_typed.env.system.collider.overflow):
                    print("Warning: overflow detected in collider")

            if (epoch % save_every == 0 or epoch == total_epochs - 1) and (
                verbose or log
            ):
                # Single sync point: pull all metric scalars at once.
                data_np = jax.device_get(data)

                elapsed = time.perf_counter() - start_time
                sps = (epoch - start_epoch) * steps_per_epoch / max(elapsed, 1e-9)

                if verbose:
                    set_postfix = getattr(it, "set_postfix", None)
                    if set_postfix:
                        set_postfix(
                            {
                                "steps/s": f"{sps:.2e}",
                                "avg_score": f"{float(data_np['score']):.2f}",
                            }
                        )

                if log and writer is not None:
                    for k, v in data_np.items():
                        writer.add_scalar(k, float(v), global_step=epoch)
                    writer.add_scalar("elapsed", elapsed, global_step=epoch)
                    writer.add_scalar("steps_per_sec", sps, global_step=epoch)
                    writer.flush()

        # Final summary (syncs once).
        data_np = jax.device_get(data)
        elapsed = time.perf_counter() - start_time
        sps = (
            max(total_epochs - 1 - start_epoch, 1)
            * steps_per_epoch
            / max(elapsed, 1e-9)
        )
        print(f"steps/s: {sps:.2e}, final avg_score: {float(data_np['score']):.2f}")
        # Single end-of-training overflow check (one host sync total).
        if jnp.any(tr_typed.env.system.collider.overflow):
            print("Warning: overflow detected in collider")
        if writer is not None:
            writer.close()

        return tr_typed

    @staticmethod
    @partial(jax.named_call, name="PPOTrainer.loss_fn")
    def loss_fn(
        model: Model,
        td: TrajectoryData,  # [T, M, ...] minibatch view
        ppo_clip_eps: jax.Array,
        ppo_value_coeff: jax.Array,
        ppo_entropy_coeff: jax.Array,
        advantage_gamma: jax.Array,
        advantage_lambda: jax.Array,
        advantage_rho_clip: jax.Array,
        advantage_c_clip: jax.Array,
        last_value: jax.Array,
        vtrace: jax.Array,
        initial_carry: Any | None = None,
        loss_mask: jax.Array | None = None,
        caps_temporal_coeff: float = 0.0,
        caps_spatial_coeff: float = 0.0,
        caps_noise_std: ArrayLike = 0.05,
        caps_key: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        r"""Compute the clipped PPO loss for a minibatch.

        Evaluate the current policy, form detached GAE/V-trace targets, and
        compute the composite loss. ``td.value`` and ``td.log_prob`` are frozen
        behavior-policy data. This function does not mutate the trajectory.
        Boundary bootstrap estimates come from the rollout. Transformed
        continuous policies use stored latent actions and base log-probabilities
        for the ratio; this assumes the bijector is fixed during replay.

        Parameters
        ----------
        model : Model
            Actor–critic model (called with ``sequence=True``).
        td : TrajectoryData
            Minibatch trajectory slice ``[T, M, ...]``.
        loss_mask : jax.Array, optional
            Selected transitions, shape ``[T, M]``. Only these contribute to
            losses and metrics; all active context still participates in RNN
            evaluation and GAE/V-trace target calculation.
        last_value : jax.Array
            Rollout-end bootstrap values, shape ``[M]``.
        vtrace : jax.Array
            Whether to use current importance ratios in target calculation.
        ppo_clip_eps : jax.Array
            Clipping parameter :math:`\epsilon`.
        ppo_value_coeff : jax.Array
            Value-loss coefficient :math:`c_v`.
        ppo_entropy_coeff : jax.Array
            Entropy-bonus coefficient :math:`c_e`.
        caps_temporal_coeff, caps_spatial_coeff : float
            Static weights of the squared-distance CAPS losses. Both zero
            preserves the original forward pass without extra RNG use.
        caps_noise_std : ArrayLike
            Scalar or per-feature perturbation standard deviation.
        caps_key : jax.Array, optional
            Required when spatial CAPS is enabled; fresh for each update.
            For the method and citation, see Mysore et al. (ICRA 2021),
            https://arxiv.org/abs/2012.06644, and the class docstring.

        Returns
        -------
        tuple[jax.Array, dict[str, jax.Array]]
            Scalar total loss and diagnostics. ``ratio``, ``value``,
            ``target_values``, and ``advantages`` are detached ``[T, M]``
            arrays; the other entries are scalar minibatch statistics.

        """
        # Current predictions are shared by target calculation and the loss.
        perturbed_output = None
        if caps_spatial_coeff > 0.0:
            if caps_key is None:
                raise ValueError("caps_key is required when spatial CAPS is enabled")
            noise = jax.random.normal(caps_key, td.obs.shape, dtype=td.obs.dtype)
            perturbed_obs = td.obs + noise * jnp.asarray(caps_noise_std, td.obs.dtype)
            pi, value, perturbed_output = model.policy_with_perturbation(
                td.obs,
                perturbed_obs,
                initial_carry=initial_carry,
                done=td.done | ~td.agent_mask,
            )
        else:
            pi, value = model(
                td.obs,
                sequence=True,
                initial_carry=initial_carry,
                done=td.done | ~td.agent_mask,
            )
        value = jnp.squeeze(value, -1)
        if td.latent_action is not None:
            from ..action_spaces import Transformed

            if not isinstance(pi, Transformed) or td.latent_log_prob is None:
                raise ValueError(
                    "Latent actions require a transformed policy and base log-probabilities"
                )
            log_ratio = pi.distribution.log_prob(td.latent_action) - td.latent_log_prob
        else:
            log_ratio = pi.log_prob(td.action) - td.log_prob
        ratio = jnp.exp(log_ratio)
        returns, advantage = Trainer.compute_advantages(
            value=value,
            reward=td.reward,
            ratio=jnp.where(vtrace, ratio, jnp.ones_like(ratio)),
            done=td.done,
            advantage_rho_clip=jnp.where(vtrace, advantage_rho_clip, 1.0),
            advantage_c_clip=jnp.where(vtrace, advantage_c_clip, 1.0),
            advantage_gamma=advantage_gamma,
            advantage_lambda=advantage_lambda,
            last_value=last_value,
            terminated=td.terminated,
            truncated=td.truncated,
            bootstrap_value=td.bootstrap_value,
            agent_mask=td.agent_mask,
        )
        selected = td.agent_mask
        if loss_mask is not None:
            selected = selected & loss_mask
        mask = selected.astype(value.dtype)
        count = jnp.maximum(mask.sum(), 1.0)

        def masked_mean(x: jax.Array) -> jax.Array:
            return jnp.sum(jnp.where(selected, x, 0.0)) / count

        # 2) Value loss (clipped).
        value_pred_clipped = td.value + (value - td.value).clip(
            -ppo_clip_eps, ppo_clip_eps
        )
        v_diff = jnp.abs(value - returns)
        v_clip_diff = jnp.abs(value_pred_clipped - returns)
        value_loss = 0.5 * masked_mean(jnp.square(jnp.maximum(v_diff, v_clip_diff)))

        # 3) Policy loss (clipped).
        ratio_bounded = jnp.where(
            advantage >= 0,
            jnp.minimum(ratio, 1.0 + ppo_clip_eps),
            jnp.maximum(ratio, 1.0 - ppo_clip_eps),
        )
        actor_loss = -masked_mean(advantage * ratio_bounded)

        # 4) Entropy (analytic or quadrature-based, depending on the distribution).
        entropy = masked_mean(pi.entropy())

        # 5) Total Loss.
        total_loss = (
            actor_loss + ppo_value_coeff * value_loss - ppo_entropy_coeff * entropy
        )
        temporal_loss = jnp.zeros((), dtype=value.dtype)
        spatial_loss = jnp.zeros((), dtype=value.dtype)
        if caps_temporal_coeff > 0.0 or caps_spatial_coeff > 0.0:
            output = model.policy_output(pi)
            if caps_temporal_coeff > 0.0:
                # Select the first transition; its successor may be context
                # outside this minibatch's loss mask, but not another episode.
                pairs = selected[:-1] & td.agent_mask[1:] & ~td.done[:-1]
                delta = jnp.where(pairs[..., None], output[1:] - output[:-1], 0.0)
                temporal_loss = jnp.sum(jnp.square(delta)) / jnp.maximum(pairs.sum(), 1)
                total_loss = total_loss + caps_temporal_coeff * temporal_loss
            if caps_spatial_coeff > 0.0:
                assert perturbed_output is not None
                delta = jnp.where(selected[..., None], perturbed_output - output, 0.0)
                spatial_loss = jnp.sum(jnp.square(delta)) / count
                total_loss = total_loss + caps_spatial_coeff * spatial_loss

        # 6) Diagnostics.
        approx_kl = jax.lax.stop_gradient(0.5 * masked_mean(jnp.square(log_ratio)))
        returns_mean = masked_mean(returns)
        return_variance = masked_mean(jnp.square(returns - returns_mean))
        residual = returns - value
        residual_mean = masked_mean(residual)
        residual_variance = masked_mean(jnp.square(residual - residual_mean))
        explained_var = jax.lax.stop_gradient(
            1.0 - residual_variance / jnp.maximum(return_variance, 1e-8)
        )

        aux = {
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "caps_temporal_loss": temporal_loss,
            "caps_spatial_loss": spatial_loss,
            "approx_KL": approx_kl,
            "explained_variance": explained_var,
            "ratio": jax.lax.stop_gradient(ratio),
            "value": jax.lax.stop_gradient(value),
            "target_values": returns,
            "advantages": advantage,
            "returns": masked_mean(returns),
            "score": masked_mean(td.reward),
        }
        return total_loss, aux

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="PPOTrainer.epoch")
    def epoch(
        tr: PPOTrainer, epoch: ArrayLike
    ) -> tuple[PPOTrainer, TrajectoryData, dict[str, jax.Array]]:
        r"""Collect a rollout and train sequential minibatches with fresh targets.

        Steps:
        0. Save initial recurrent carry (resets occur inside each rollout step).
        1. Collect a trajectory of length ``num_steps_epoch``.
        2. Flatten the agent axis and apply DRIP if enabled.
        3. Visit contiguous segment blocks, wrapping for additional passes.
        4. Run the current model, calculate targets, and update parameters.
        Rollout values and log-probabilities remain unchanged.

        Parameters
        ----------
        tr : PPOTrainer
            Current trainer state.
        epoch : ArrayLike
            Zero-based outer iteration index (retained for caller compatibility).

        Returns
        -------
        Tuple[PPOTrainer, TrajectoryData, dict[str, jax.Array]]
            Updated trainer, trajectory data shaped ``[T, S, ...]`` with
            DRIP-transformed rewards, and metrics averaged equally over
            nonempty minibatches. An all-empty rollout yields zero metrics.
            Parameter updates occur according to gradient accumulation;
            empty minibatches do not advance the optimizer.


        """
        del epoch
        # Trainer.step advances the PRNG key and resets episode boundaries.
        model, _ = nnx.merge(tr.graphdef, tr.graphstate)
        initial_carry = model.carry

        # 1) Roll out trajectories; td has shape [T, E, A, ...].
        tr.env, tr.graphstate, tr.key, td = tr.trajectory_rollout(
            tr.env,
            tr.graphdef,
            tr.graphstate,
            tr.key,
            tr.num_steps_epoch,
            skip_frames=tr.skip_frames,
        )

        # 1.5) Bootstrap value V(s_T): one extra critic pass on the
        # post-rollout observation. The merged model is discarded afterwards so
        # this pass does not advance the persistent recurrent carry.
        boot_model, *_ = nnx.merge(tr.graphdef, tr.graphstate, copy=True)
        boot_model.eval()
        _, last_value = boot_model(tr.env.observation(tr.env), sequence=False)
        last_value = jax.lax.stop_gradient(jnp.squeeze(last_value, -1))  # [E, A]

        # 2) Flatten the agent axis to get [T, S, ...].
        td = jax.tree.map(
            lambda x: x.reshape((x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])),
            td,
        )
        T, S = td.value.shape[:2]
        last_value = last_value.reshape(S)  # [S]

        initial_carry = jax.tree.map(
            lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
            initial_carry,
        )

        # --- DRIP (Distributed Reward Information Processing) ---
        @jax.jit(inline=True)
        @partial(jax.named_call, name="PPOTrainer.apply_drip")
        def apply_drip(
            rewards: jax.Array, dones: jax.Array, decay: jax.Array
        ) -> jax.Array:
            decay_not_dones = decay * (1.0 - dones.astype(rewards.dtype))

            def drip_step(
                carry: jax.Array, xs: tuple[jax.Array, jax.Array]
            ) -> tuple[jax.Array, jax.Array]:
                r_t, decay_not_done_t = xs
                drip_val = r_t + carry * decay_not_done_t
                return drip_val, drip_val

            # Scan backwards over the trajectory
            _, dripped_rewards = jax.lax.scan(
                drip_step,
                jnp.zeros_like(rewards[-1]),
                (rewards, decay_not_dones),
                reverse=True,
                unroll=8,
            )
            return dripped_rewards

        # Efficiently apply the recursive DRIP backward pass
        td.reward = apply_drip(td.reward, td.done, tr.drip_decay)
        # ------------------------------------------------------

        caps_key = None
        if tr.caps_spatial_coeff > 0.0:
            tr.key, caps_key = jax.random.split(tr.key)

        @partial(jax.named_call, name="PPOTrainer.train_batch")
        def train_batch(
            graphstate: nnx.GraphState, batch_index: jax.Array
        ) -> tuple[nnx.GraphState, dict[str, jax.Array]]:
            indices, loss_mask = _minibatch_selection(
                T, S, tr.minibatch_size, batch_index
            )
            mb_td = jax.tree.map(
                lambda x: jnp.take(x, indices, axis=1),
                td,
            )
            mb_initial_carry = jax.tree.map(
                lambda x: jnp.take(x, indices, axis=0),
                initial_carry,
            )
            mb_last_value = jnp.take(last_value, indices, axis=0)
            model, optimizer = nnx.merge(tr.graphdef, graphstate)
            model.eval()
            (loss, aux), grads = nnx.value_and_grad(tr.loss_fn, has_aux=True)(
                model,
                mb_td,
                tr.ppo_clip_eps,
                tr.ppo_value_coeff,
                tr.ppo_entropy_coeff,
                tr.advantage_gamma,
                tr.advantage_lambda,
                tr.advantage_rho_clip,
                tr.advantage_c_clip,
                mb_last_value,
                tr.vtrace,
                initial_carry=mb_initial_carry,
                loss_mask=loss_mask,
                caps_temporal_coeff=tr.caps_temporal_coeff,
                caps_spatial_coeff=tr.caps_spatial_coeff,
                caps_noise_std=tr.caps_noise_std,
                caps_key=(
                    None
                    if caps_key is None
                    else jax.random.fold_in(caps_key, batch_index)
                ),
            )
            model.train()
            selected = mb_td.agent_mask & loss_mask

            def apply_update(state: nnx.GraphState) -> nnx.GraphState:
                current_model, current_optimizer = nnx.merge(tr.graphdef, state)
                current_optimizer.update(current_model, grads)
                return nnx.state((current_model, current_optimizer))

            # Sequential blocks may consist entirely of inactive padding.
            # Even zero gradients would otherwise advance optimizer momentum.
            graphstate = jax.lax.cond(
                jnp.any(selected),
                apply_update,
                lambda state: state,
                nnx.state((model, optimizer)),
            )

            # No target cache or rollout-value writeback: every next minibatch
            # computes its targets from its own current forward pass.
            mb_metrics = {
                "_has_samples": jnp.any(selected),
                "loss": loss,
                "actor_loss": aux["actor_loss"],
                "value_loss": aux["value_loss"],
                "entropy": aux["entropy"],
                "caps_temporal_loss": aux["caps_temporal_loss"],
                "caps_spatial_loss": aux["caps_spatial_loss"],
                "approx_KL": aux["approx_KL"],
                "explained_variance": aux["explained_variance"],
                "grad_norm": optax.tree.norm(grads),
                "ratio": jnp.sum(jnp.where(selected, aux["ratio"], 0.0))
                / jnp.maximum(jnp.sum(selected), 1),
                "returns": aux["returns"],
                "score": aux["score"],
            }
            return graphstate, mb_metrics

        tr.graphstate, epoch_metrics = jax.lax.scan(
            train_batch,
            tr.graphstate,
            xs=jnp.arange(tr.num_minibatches),
            unroll=tr.num_minibatches,
        )
        has_samples = epoch_metrics.pop("_has_samples")
        count = jnp.maximum(jnp.sum(has_samples), 1)
        data = jax.tree.map(
            lambda x: jnp.sum(jnp.where(has_samples, x, 0.0)) / count,
            epoch_metrics,
        )
        return tr, td, data


__all__ = ["PPOTrainer"]
