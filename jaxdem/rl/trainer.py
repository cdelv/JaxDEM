# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Interface for defining reinforcement learning model trainers.
"""
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Tuple, Any, Optional

try:
    # Python 3.11+
    from typing import Self
except ImportError:
    from typing_extensions import Self

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import partial

from flax import nnx
import optax

from ..factory import Factory
from .envWrapper import clip_action_env, vectorise_env
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import Environment
    from .model import Model


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
    r"""
    Behavior-policy log-probabilities :math:`\log \pi_b(a_t \mid s_t)` at collection time.
    """

    new_log_prob: jax.Array
    r"""
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
@dataclass(kw_only=True, slots=True, frozen=True)
class Trainer(Factory, ABC):
    """
    Base class for reinforcement learning trainers.

    This class holds the environment and model state (Flax NNX GraphDef/GraphState).
    It provides rollout utilities (:meth:`step`, :meth:`trajectory_rollout`) and
    a general advantage computation method (:meth:`compute_advantages`).
    Subclasses must implement algorithm-specific training logic in :meth:`epoch`.
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
        tr = replace(tr, graphstate=nnx.state((model, *rest)))
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
            Trajectory data shape: (N_envs, N_agents, *)
        """
        key, subkey = jax.random.split(tr.key)
        tr = replace(tr, key=key)
        model, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        model.eval()

        obs = tr.env.observation(tr.env)
        pi, value = model(obs)
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
    @partial(jax.jit, static_argnames=("unroll",))
    def compute_advantages(
        tr: "Trainer", td: "TrajectoryData", unroll: int = 8
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

        def calculate_advantage(gae_and_next_value: Tuple, td: TrajectoryData) -> Tuple:
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
            calculate_advantage, (gae0, last_value), td, reverse=True, unroll=unroll
        )
        td.returns = td.advantage + td.value

        tr = replace(tr, graphstate=nnx.state((model, *rest)))
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

    @staticmethod
    @abstractmethod
    def train(tr) -> Any:
        """
        Training loop
        """
        raise NotImplementedError


@Trainer.register("PPO")
@jax.tree_util.register_dataclass
@dataclass(kw_only=True, slots=True, frozen=True)
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

      where :math:`R_t` are return targets.

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
        clip_actions: bool = True,
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

        graphdef, graphstate = nnx.split(
            (model, nnx.Optimizer(model, tx, wrt=nnx.Param))
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
    def train(tr: "PPOTrainer", verbose=True):
        import time
        from tqdm import trange

        loss_history = jnp.zeros(tr.num_epochs)
        tr, td = tr.epoch(tr, jnp.asarray(0))
        start_time = time.perf_counter()

        it = trange(1, tr.num_epochs) if verbose else range(1, tr.num_epochs)
        steps_per_sec = 0.0
        avg_score = 0.0

        for epoch in it:
            tr, td = tr.epoch(tr, jnp.asarray(epoch))

            elapsed = time.perf_counter() - start_time
            steps_done = (epoch + 1) * tr.num_envs * tr.num_steps_epoch
            steps_per_sec = steps_done / elapsed
            avg_score = jnp.mean(td.reward)

            loss_history = loss_history.at[epoch].set(avg_score)

            if verbose:
                it.set_postfix(
                    {
                        "steps/s": f"{steps_per_sec:.2e}",
                        "avg_score": f"{avg_score:.2f}",
                    }
                )

        print(f"steps/s: {steps_per_sec:.2e}, final avg_score: {avg_score:.2f}")
        return tr, loss_history

    @staticmethod
    @nnx.jit(donate_argnums=(2, 3))
    def loss_fn(
        model: "Model", tr: "PPOTrainer", td: "TrajectoryData", seg_w: jax.Array
    ):
        """
        Compute the PPO minibatch loss.

        Parameters
        ----------
        model : Model
            Live model rebuilt by NNX for this step.
        tr : PPOTrainer
            Trainer holding hyperparameters and state.
        td : TrajectoryData
            Time-stacked trajectory mini batch (e.g., shape ``[T, B, ...]``).
        seg_w : jax.Array
            Importance weights (broadcastable over ``td.advantage``), derived from
            prioritized sampling probabilities.

        Returns
        -------
        jax.Array
            Scalar loss to be minimized.

        See Also
        --------
        Main docs: PPO trainer overview and equations.
        """
        # 1) Get new values and log_probs
        model.eval()
        pi, new_value = model(td.obs)
        new_value = jnp.squeeze(new_value, -1)
        td.new_log_prob = pi.log_prob(td.action)

        # 2) Normalize advantage and perform v-trace of-policy correction
        tr, td = tr.compute_advantages(tr, td)
        td.advantage = (
            seg_w * (td.advantage - td.advantage.mean()) / (td.advantage.std() + 1e-8)
        )

        # 3) Clipped value loss
        value_pred_clipped = td.value + (new_value - td.value).clip(
            -tr.ppo_clip_eps, tr.ppo_clip_eps
        )
        value_losses = jnp.square(new_value - td.returns)
        value_losses_clipped = jnp.square(value_pred_clipped - td.returns)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # 4) Clipped policy objective
        ratio = jnp.exp(td.new_log_prob - td.log_prob)
        loss_actor1 = td.advantage * ratio
        loss_actor2 = td.advantage * ratio.clip(
            1.0 - tr.ppo_clip_eps, 1.0 + tr.ppo_clip_eps
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

        # 5) Entropy loss
        entropy_loss = pi.entropy().mean()

        # 6) Total loss
        total_loss = (
            loss_actor
            + tr.ppo_value_coeff * value_loss
            - tr.ppo_entropy_coeff * entropy_loss
        )

        return total_loss

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

        key, sample_key, reset_key = jax.random.split(tr.key, 3)
        tr = replace(tr, key=key)
        subkeys = jax.random.split(sample_key, tr.num_envs)

        # 0) Reset the environment
        tr = replace(
            tr, env=jax.vmap(tr.env.reset_if_done)(tr.env, tr.env.done(tr.env), subkeys)
        )

        # 1) Gather data -> shape: (time, num_envs, num_agents, *)
        tr, td = tr.trajectory_rollout(tr, tr.num_steps_epoch)

        # Reshape data (time, num_envs, num_agents, *) -> (time, num_envs*num_agents, *)
        td = jax.tree_util.tree_map(
            lambda x: x.reshape((x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:]), td
        )

        # 2) Compute advantages
        tr, td = tr.compute_advantages(tr, td)

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
            model, optimizer, *rest = nnx.merge(tr.graphdef, tr.graphstate)

            # 4.1) Importance sampling
            mb_td = jax.tree_util.tree_map(lambda x: jnp.take(x, idx, axis=1), td)
            seg_w = jnp.power(
                tr.num_segments * jnp.take(weights[None, :], idx, axis=1), -beta_t
            )

            # 4.2) Compute gradients
            loss, grads = nnx.value_and_grad(tr.loss_fn)(model, tr, mb_td, seg_w)

            # 4.3) Train model
            model.train()
            optimizer.update(model, grads)

            # 4.4) Return updated model
            tr = replace(tr, graphstate=nnx.state((model, optimizer, *rest)))
            return (tr, td, weights), loss

        # 4) Loop over mini batches
        (tr, td, prio_probs), loss = jax.lax.scan(
            train_batch, (tr, td, prio_probs), xs=idxs, unroll=4
        )

        return tr, td
