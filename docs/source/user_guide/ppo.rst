PPO trainer semantics
=========================

:class:`~jaxdem.rl.trainers.PPOTrainer` collects one rollout, then visits
contiguous minibatches of fixed-horizon agent segments. Its update order follows
the native CUDA learner in the local PufferLib reference. Reward processing,
boundary bootstrapping, action distributions, and optimizer details still differ.

Rollouts and minibatches
----------------------------

Let :math:`T` be ``num_steps_epoch``, :math:`S` be
``num_envs * max_num_agents``, :math:`B` be ``minibatch_size``, and
:math:`K` be ``num_minibatches``. A rollout stores :math:`ST` transition slots,
including inactive padding. Each segment contains :math:`T` policy decisions
for one agent slot; it can span multiple episodes.

A minibatch contains :math:`M=B/T` segments. The constructor requires
:math:`T\leq B\leq ST`, :math:`B\bmod T=0`, and :math:`ST\bmod B=0`.
Minibatch :math:`k`, indexed from zero, starts at

.. math::

    j_k=(kM)\bmod S

and selects segments :math:`j_k,\ldots,j_k+M-1`. There is no shuffle,
priority sampling, or sampling-weight correction. One pass takes :math:`S/M`
minibatches. With explicit :math:`B`, :math:`K` may specify a partial pass or
multiple passes; each new rollout starts selection at segment zero. Without
explicit :math:`B`, the trainer requires :math:`S\bmod K=0` and chooses
:math:`B=ST/K`, giving exactly one pass.

``num_epochs`` counts rollout-and-update iterations, not replay passes.
``total_timesteps``, when provided, must be divisible by :math:`ST` and sets
``num_epochs = total_timesteps / (S*T)``. This budget includes padding and
excludes additional physics frames. ``stop_at_epoch`` limits execution without
changing the planned learning-rate schedule.

Each minibatch follows this order:

#. Select observations, actions, behavior log-probabilities, behavior values,
   boundary flags, masks, saved initial carry, and boundary bootstrap estimates.
#. Evaluate the current model on the selected sequence.
#. Calculate detached advantages and return targets using those predictions.
#. Differentiate the PPO loss against the same current predictions.
#. Submit gradients to the optimizer, unless the minibatch has no active entries.

There is no initial full-rollout target cache or post-update target writeback.
On a later visit, predictions and targets are calculated again using the then
current parameters. Behavior log-probabilities and value-clipping references
remain fixed. For transformed continuous policies, collection also stores the
pre-transform sample and its base log-probability. PPO evaluates ratios in those
base coordinates: the fixed bijector Jacobian cancels, avoiding inversion of
saturated floating-point actions. The stored transformed log-probability is
retained for diagnostics. With gradient accumulation, a gradient submission need not change
parameters immediately.

Boundary signals and indexing
---------------------------------

For JaxDEM, row :math:`t` contains the observation and action at the start of a
transition, followed by the reward and boundary flags produced by that action.
Environment subclasses provide ``terminated(env)`` and ``truncated(env)``;
``done`` is their logical OR. Both default to false. Overriding only ``done``
is insufficient: the trainer reads the two separate causes during physics
advancement. Environment ``step`` must leave the final state available; the
trainer reads reward and truncation bootstrap before resetting it.

.. list-table:: JaxDEM boundary handling
   :header-rows: 1
   :widths: 27 25 24 24

   * - Transition
     - Bootstrap in TD residual
     - Continue advantage trace?
     - Reset before next observation?
   * - Continuing
     - Next value
     - Yes, within this rollout
     - No
   * - Terminated
     - None
     - No
     - Environment and carry
   * - Truncated only
     - Final pre-reset observation value
     - No
     - Environment and carry
   * - Both flags true
     - None; termination takes precedence
     - No
     - Environment and carry
   * - Continuing at rollout end
     - Extra post-rollout observation value
     - No; trace ends at horizon
     - No

A disappearing agent receives a terminal flag on its last active transition,
without resetting the whole environment unless the environment also ended.
Inactive entries do not contribute to losses or advantages. Carry is cleared
for episode boundaries and inactive/disappearing slots. Data must remain finite
even in padding: zero multiplication does not remove NaNs.

LSTM sequence replay clears carry after ``done[t]``. MinGRU implements the same
semantics by shifting that mask before its recurrence. The trainer additionally
passes ``~agent_mask`` as a recurrent reset condition. Initial sequence carry is
the saved rollout-start carry; learning does not backpropagate into earlier
rollouts or overwrite the live rollout carry. Extra bootstrap forwards use a
copied model.

The inspected native PufferLib trainer has only a ``terminals`` buffer. Its
stored reward and terminal at row :math:`t+1` describe action :math:`t`, and it
resets carry before observation :math:`t+1`. Thus the one-row shift is consistent
with JaxDEM's post-action indexing for true terminals. However, PufferLib uses
that single mask both to suppress value bootstrap and to stop the trace, with
no separate truncation/final-observation channel in the learner.

Timeout behavior therefore depends on the PufferLib environment. In the local
reference, Clifford sets ``terminals = terminated | truncated``; Cartpole sets
``terminals = terminated`` but resets for either cause. The latter leaves a
pure-timeout reset unmarked to the learner, which can bootstrap from the new
reset observation and carry its trace/recurrent state across that reset.
Neither reproduces JaxDEM's separate truncation treatment.

PufferLib also leaves its final stored row's advantage at zero and uses its
current value as that row's return target. JaxDEM instead has the reward from
its final action plus an extra boundary bootstrap, and trains that transition.
A rollout horizon is not an episode termination or truncation in JaxDEM.

Advantages and return targets
---------------------------------

Let :math:`V_t` and :math:`\pi_\theta` be current minibatch predictions, and
:math:`V_{b,t}` and :math:`\pi_b` be their frozen behavior-policy counterparts.
For active indicator :math:`m_t`, terminal flag :math:`z_t`, truncation flag
:math:`u_t`, and :math:`d_t=z_t\lor u_t`, define

.. math::

    q_t = \exp[\log\pi_\theta(a_t\mid o_t)-\log\pi_b(a_t\mid o_t)].

For recurrent models these predictions also depend on the replayed history and
saved initial carry. With ``vtrace=False``, the target recurrence uses
:math:`\widehat\rho_t=\widehat c_t=1`, giving GAE. With ``vtrace=True``, it uses
:math:`\widehat\rho_t=\min(q_t,\bar\rho)` and
:math:`\widehat c_t=\min(q_t,\bar c)`, where the caps are
``advantage_rho_clip`` and ``advantage_c_clip``. The PPO actor ratio is
:math:`q_t` in both modes. This option is a V-trace-style lambda trace inside
PPO, not the full IMPALA actor update.

Define the next-value estimate

.. math::

    B_t = \begin{cases}
        V_{b}(o^{\mathrm{final}}_t), & u_t=1,\\
        V_b(o_T), & t=T-1\text{ and }u_t=0,\\
        V_{t+1}, & \text{otherwise}.
    \end{cases}

The two boundary estimates are recorded before optimization and remain fixed
throughout this rollout's updates. At a terminal boundary their value is
irrelevant because the bootstrap multiplier is zero. Then, with :math:`A_T=0`,

.. math::

    \delta_t &= m_t\widehat\rho_t
        [\widetilde r_t+\gamma(1-z_t)B_t-V_t],\\
    A_t &= \delta_t+\gamma\lambda m_t(1-d_t)\widehat c_t A_{t+1},\\
    R_t &= \operatorname{stop\_gradient}(V_t+m_t A_t),\\
    \widehat A_t &= \operatorname{stop\_gradient}(A_t).

The helper returns ``(returns, advantages)`` in that order. Targets have no
gradient, even though current predictions were used to calculate them. Inactive
entries return zero advantage and a detached target equal to their input value.
There is no advantage centering or variance normalization.

Loss
--------

For any per-entry quantity :math:`f_t`, let
:math:`\langle f\rangle_m=\sum_t m_t f_t/\max(1,\sum_t m_t)`, summing across
both time and agent axes. With ``ppo_clip_eps`` :math:`\epsilon`,

.. math::

    L_\pi &= -\left\langle\min\left(q_t\widehat A_t,
        \operatorname{clip}(q_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right\rangle_m,\\
    V_t^{\mathrm{clip}} &= V_{b,t}+
        \operatorname{clip}(V_t-V_{b,t},-\epsilon,\epsilon),\\
    L_V &= \frac12\left\langle\max\left((V_t-R_t)^2,
        (V_t^{\mathrm{clip}}-R_t)^2\right)\right\rangle_m,\\
    L &= L_\pi+c_v L_V-c_e\langle H(\pi_\theta)\rangle_m.

The same epsilon controls the policy ratio and value clip, though these have
different units. Value clipping stays centered on :math:`V_b` after every
parameter update. Continuous policies constrained by bijectors use differential
entropy of the transformed distribution; BoxSpace and MaxNormSpace approximate
the log-Jacobian expectation by finite Gauss--Hermite quadrature.

Rewards and frame repetition
--------------------------------

JaxDEM does not clip or normalize rewards in this trainer. PufferLib's native
learner clips each reward to :math:`[-1,1]`; it does not map the observed minimum
and maximum to that interval.

For ``drip_decay`` :math:`d`, JaxDEM transforms the collected rewards before
learning using

.. math::

    \widetilde r_t=r_t+d(1-d_t)\widetilde r_{t+1},\qquad
    \widetilde r_T=0.

This backward sum is unnormalized and ends at both episode and rollout
boundaries. At :math:`d=0` it is the identity. For example, :math:`[0,1]`
becomes :math:`[d,1]` if there is no boundary between the two rewards. Thus
DRIP changes reward scale and targets; it does not merely move reward in time.

``skip_frames=k`` requests :math:`1+k` physics frames per action. The fixed
scan evaluates that many steps, but discards state updates after each
environment's first episode boundary. Reward is read once from the live
endpoint; intermediate physics-frame rewards are not summed. Inactive actions are zeroed without changing their dtype; categorical action
indices stay integers. Continuous action clipping rejects categorical inputs.
The evaluation utilities use the same action masking and physics boundary
rules. Splitting a rollout into recording chunks preserves its action/key
sequence. Discount and
trace decay are per recorded policy transition, regardless of its actual
physics duration.

Optimization, schedules, and metrics
----------------------------------------

With ``accumulate_n_gradients=g``, Optax averages :math:`g` nonempty minibatch
gradients, clips their global norm, and then applies the stateful optimizer.
The constructor requires :math:`K\bmod g=0`. Empty minibatches do not advance
momentum, accumulation, or optimizer schedule counters. If empties interrupt a
group, pending gradients may carry into the next rollout. Each nonempty
minibatch has equal accumulation weight, even when active counts differ.

The default optimizer is ``optax.contrib.muon``. Its behavior depends on the
Optax version and parameter shapes; it is not identical to PufferLib's custom
CUDA Muon. In the inspected Optax 0.2.8 default configuration, only rank-two
parameters use Muon without explicit configuration. JaxDEM supplies matrix-axis
specifications for ``mingru_kernel``: axis 0 batches layers and axes 1/2 are the
input/output matrix axes. Each layer therefore receives an independent Optax
Muon update. Other parameters keep Optax's default routing. This also applies
to partial Muon factories unless they provide an explicit dimension override.
The dependency floor is Optax 0.2.8, the version used for these checks.

For initial learning rate :math:`\eta_0`, planned iteration count :math:`N`,
optimizer-update index :math:`q`, and :math:`U=K/g`, enabled annealing uses

.. math::

    e(q)=\min(\lfloor q/U\rfloor,N),\qquad
    \eta(q)=\frac{\eta_0}{2}[1+\cos(\pi e(q)/N)].

Without empty minibatches this is constant within each outer iteration. Empty
batches delay the schedule relative to the outer iteration counter. Normal
execution ends at iteration :math:`N-1`, before the mathematical zero-rate
endpoint. The entropy coefficient stays constant.

Epoch metrics average each nonempty minibatch's statistic equally, including
repeated visits. ``score`` is the mean training reward after DRIP, not episode
return; it need not be the mean of all collected active transitions.
``approx_KL`` is :math:`\tfrac12\langle(\log q)^2\rangle_m` and does not stop
training. ``steps_per_sec`` uses nominal :math:`ST(1+k)` slots per iteration,
including padding and discarded physics updates. It is not measured active
transition throughput.

``save_every`` controls metric synchronization/logging, not checkpoint saving.
``start_epoch`` supplies the outer-loop starting index; it does not restore
optimizer counters. Exact continuation requires the full trainer state,
including parameters, optimizer state, recurrent carry, environment, and key.
The first iteration is real training used for compilation warmup. Without a
synchronization before timing, its execution may overlap the reported timing
interval; single-iteration throughput is not a reliable benchmark.

Reproducibility on GPU
----------------------

Seed both model initialization (``nnx.Rngs``) and the trainer's ``key``.
Collection splits the trainer key into the next key, an action-sampling key,
and an environment-reset root. Evaluation helpers return an advanced key;
thread it into subsequent calls. Sampling during evaluation is reproducible
when the policy state, inputs, and random keys are identical; sampling alone
does not explain differences between identically seeded runs.

Fixed seeds do not guarantee bitwise-identical GPU training. XLA can select
different floating-point kernels during compilation, and some GPU operations
have nondeterministic execution order. Small numerical differences in an
update can grow as later policy actions, trajectories, and updates diverge.
For a reproducibility diagnostic, set the following before importing JAX or
initializing its backend (preserve any other required ``XLA_FLAGS``):

.. code-block:: bash

   XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0" python intro_to_rl2.py

These settings restrict kernel selection and disable live autotuning. They
can reduce performance, and unsupported operations may fail compilation.
JaxDEM does not enable them globally. See the
`OpenXLA GPU determinism guide <https://openxla.org/xla/determinism>`_.
Keep hardware, dependency versions, configuration, and starting state fixed
when comparing runs; these flags are not a cross-platform reproducibility
guarantee. A useful diagnostic compares initialization, the first rollout,
post-update parameters, and final evaluation separately, including the keys
at each stage.
