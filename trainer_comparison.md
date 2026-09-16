# JaxDEM and PufferLib trainer comparison

This report describes the current JaxDEM working tree after the trainer alignment changes. The reference is the **native CUDA learner in this local PufferLib checkout**, not an older Python PPO/PER implementation. Optimizer details below use the locally installed Optax 0.2.8. Conclusions about PufferLib environments are specific to the inspected implementations.

The six requested update stages now follow the same order. That establishes structural alignment, not numerical equivalence: transition indexing, horizon treatment, truncation handling, reward processing, policy distributions, optimizer updates, and defaults still differ. The full JaxDEM equations are in the [PPO semantics guide](/home/wind/Documents/JaxDEM/docs/source/user_guide/ppo.rst).

## 1. The six update stages

| Stage | PufferLib native learner | Current JaxDEM |
|---|---|---|
| Initial targets | No initial full-rollout advantage cache | No initial target cache; records behavior values/log-probabilities and boundary bootstraps |
| Minibatch selection | Contiguous blocks of fixed-horizon agent segments; wraps for replay | Same block order; explicit size and divisibility checks |
| Current predictions | Forward the current learner before target calculation | One current sequence forward shared by targets and loss |
| Targets used by loss | Compute advantages/returns from that forward's values and optionally ratios | Same freshness for interior predictions; final-horizon and truncation bootstraps remain collection-time estimates |
| Policy advantages | Raw advantages, without standardization or PER weights | Raw detached advantages, without standardization or PER weights |
| After updating parameters | Keep rollout references fixed; next minibatch recomputes predictions/targets | Same; no value/ratio/target writeback. Accumulation can defer the actual parameter update |

The earlier moving value-clipping reference, stale target-cache consumption, and silently rounded minibatch-size issues have been addressed. Prioritized sampling, its constructor arguments, and advantage normalization were removed. `vtrace=False` is now the default. Those earlier findings should not be read as current defects.

Sources: [PufferLib minibatch loop](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:1460), [JaxDEM loss](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:676), [JaxDEM epoch](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:817).

## 2. Minibatch arithmetic and coverage

Let `S = num_envs * max_num_agents`, `T = num_steps_epoch`, `B = minibatch_size`, `M = B/T`, and `K = num_minibatches`. JaxDEM visits segments starting at `(k*M) % S`. It requires `T <= B <= S*T`, `B % T == 0`, and `(S*T) % B == 0`, so each contiguous block fits without crossing the segment-array end.

Without an explicit `B`, JaxDEM requires `S % K == 0` and chooses `B = S*T/K`: one full pass. With explicit `B`, `K` is independent of the number of blocks. A partial pass repeatedly omits later blocks because each new rollout starts at block zero; a noninteger number of passes gives earlier blocks extra visits. Nominal replay is `K*B/(S*T)`, counting padded slots. PufferLib derives its update count from `replay_ratio * S*T / B`.

A segment is a horizon for one agent slot, not necessarily a complete episode. Recurrent carry is selected with the same segment indices as observations. JaxDEM excludes inactive entries from loss means and skips optimizer submission for an entirely inactive block. Metrics average nonempty minibatches equally, rather than weighting all active transitions globally.

Sources: [JaxDEM configuration](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:310), [JaxDEM epoch](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:817), [PufferLib minibatch loop](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:1460).

## 3. Done, terminated, and truncated: only partially equivalent

For a JaxDEM transition `(observation[t], action[t])`, `reward[t]` and the three boundary flags describe the outcome of that action. The trainer derives `done = terminated | truncated`. A disappearing agent additionally receives `done=True` and `terminated=True` on its last active transition.

PufferLib copies the current environment reward/terminal buffer before choosing its next action. Its `reward[t+1]` and `terminals[t+1]` therefore describe `action[t]`. Its advantage kernel uses precisely that one-row shift. This is a consistent alternative indexing convention, not an off-by-one defect.

| Event | JaxDEM | PufferLib native learner |
|---|---|---|
| True terminal | Suppress bootstrap, stop trace, reset environment and carry before the next observation | A set terminal mask suppresses bootstrap and trace; resets carry before the associated reset observation. Environment performs its own reset |
| Pure truncation | Bootstrap final pre-reset observation; stop trace; reset environment and carry | No separate truncation or final-observation channel in the inspected learner |
| Both causes true | Terminal wins: no bootstrap | Cannot distinguish causes once encoded in the single mask |
| Continuing rollout boundary | Extra post-rollout bootstrap; train last transition; preserve carry | Final stored advantage is zero; last stored current value supplies the preceding transition's bootstrap |
| Inactive/disappearing agent | Dedicated agent masks and per-agent terminal handling | Legal-action masks are a different concept; no equivalent agent-mask behavior in this inspected loss path |

**PufferLib's timeout semantics depend on the environment.** Clifford sets `terminals = terminated | truncated`: a timeout stops the trace and carry but also loses bootstrapping. Cartpole sets `terminals = terminated` and resets on either cause: a pure timeout is invisible to the learner, allowing bootstrapping from the reset observation and continuation of trace/carry across that reset. Therefore it would be inaccurate to say PufferLib consistently treats every truncation as terminal, or that JaxDEM already matches it exactly.

For true terminals, JaxDEM's post-transition carry reset and PufferLib's pre-observation reset refer to the same boundary. JaxDEM LSTM replay resets after `done[t]`; MinGRU shifts this post-transition mask into the recurrence at `t+1`. Both JaxDEM models also cut carry for inactive slots. Saved initial carry is fixed across replay and may come from older parameters; neither trainer backpropagates into earlier rollouts.

**Assessment:** retain JaxDEM's explicit distinction. Its truncation path reads the final observation before resetting, evaluates a copied model with the pre-reset recurrent history, and stops the trace so the next episode's rewards cannot enter the previous episode's target. Source inspection found no corresponding reset-position or bootstrap-selection error in this path. Runtime parity has not been tested.

Custom JaxDEM environments must implement `terminated` and/or `truncated`; overriding only `done` does not signal a boundary to the physics advancement loop. Their `step` must not autoreset before the trainer reads final reward and bootstrap observations. Built-in navigator, roller, gears, and Granulabot time limits use `truncated`.

Sources: [JaxDEM collection/reset](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/__init__.py:162), [JaxDEM advantage recurrence](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/__init__.py:346), [JaxDEM physics boundary handling](/home/wind/Documents/JaxDEM/jaxdem/utils/environment.py:36), [JaxDEM environment signals](/home/wind/Documents/JaxDEM/jaxdem/rl/environments/__init__.py:247), [LSTM replay reset](/home/wind/Documents/JaxDEM/jaxdem/rl/models/lstm.py:301), [MinGRU replay reset](/home/wind/Documents/JaxDEM/jaxdem/rl/models/mingru.py:230), [PufferLib collection/reset](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:784), [PufferLib advantage kernel](/home/wind/Documents/JaxDEM/PufferLib/src/algo.cu:1698), [PufferLib agent interface](/home/wind/Documents/JaxDEM/PufferLib/src/pufferenv.h:24), [PufferLib Cartpole boundary](/home/wind/Documents/JaxDEM/PufferLib/ocean/cartpole/cartpole.h:185), [PufferLib Clifford boundary](/home/wind/Documents/JaxDEM/PufferLib/ocean/clifford/clifford.h:483).

## 4. Targets and the final horizon row

For active entries, JaxDEM uses the recurrence

```text
ratio_t = exp(current_logprob_t - behavior_logprob_t)
rho_t = min(ratio_t, rho_cap)       # both weights are 1 when vtrace=False
c_t   = min(ratio_t, c_cap)
delta_t = rho_t * (reward_t + gamma*(1-terminated_t)*bootstrap_t - current_value_t)
A_t = delta_t + gamma*lambda*(1-done_t)*c_t*A_(t+1)
A_T = 0
R_t = stop_gradient(A_t + current_value_t)
actor_advantage_t = stop_gradient(A_t)
```

At a truncation, `bootstrap_t` is the stored final pre-reset value. At the final continuing row it is the stored post-rollout value. Otherwise it is `current_value[t+1]`. Inactive entries have zero advantage and target equal to their detached current value; all loss means mask them out. Thus “fresh targets” does not imply that boundary critic estimates are refreshed after each update.

PufferLib uses a related clipped-importance lambda recurrence, but with its shifted reward/done indexing and one mask for both bootstrap and trace. With V-trace disabled it supplies unit ratios; its configured caps still apply, whereas JaxDEM explicitly forces both weights to one. With the default caps of one these choices agree. Neither optional trace makes the PPO actor objective identical to the canonical IMPALA actor objective.

PufferLib sets the last stored advantage to zero and its return to the last current value. It still includes that row in entropy and loss denominators. The clipped value-loss branch can also report a positive loss there if the value moved outside the behavior-value clip, even though the target equals the current prediction. JaxDEM records the final action's outcome and trains that transition with an additional bootstrap pass. Exact fixture comparisons must align indexing and account for this extra learning contribution.

Sources: [JaxDEM advantage recurrence](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/__init__.py:346), [JaxDEM loss](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:676), [PufferLib advantage kernel](/home/wind/Documents/JaxDEM/PufferLib/src/algo.cu:1698), [PufferLib frozen value buffer](/home/wind/Documents/JaxDEM/PufferLib/src/algo.cu:1222).

## 5. PPO loss and reward scale

Both use the clipped policy surrogate with raw advantages and a maximum of unclipped/clipped squared critic errors. JaxDEM's critic clip stays centered on the immutable rollout value. Returns and advantages are detached, including their dependence on current ratios and values. JaxDEM uses one `ppo_clip_eps` for both actor and critic; PufferLib exposes separate policy and value clip settings.

PufferLib clips rewards pointwise to `[-1, 1]`. It does **not** rescale the observed reward minimum and maximum. For example, `[-5, 0.2, 3]` becomes `[-1, 0.2, 1]`. JaxDEM's trainer leaves environment reward magnitudes unchanged unless DRIP is enabled. This affects actor advantage scale as well as value targets and the effective strictness of the absolute value clip.

DRIP replaces each reward with `r_t + decay*(1-done_t)*dripped_reward[t+1]`, starting from zero beyond the horizon. It cuts at termination and truncation. This is an unnormalized backward sum: `[0, 1]` becomes `[decay, 1]`. With gamma=lambda=1 and zero values, the first return becomes `1+decay` instead of 1. DRIP changes the objective and depends on the rollout cut; it is not reward-conserving smoothing. Default decay zero is the identity.

JaxDEM's frame repetition reads one endpoint reward rather than summing physics-substep rewards. This fits action-start-to-end potential-difference rewards, but a per-physics-step reward needs a different environment accounting scheme. Gamma and lambda are per policy decision, including a shortened transition at a boundary.

Sources: [PufferLib reward clipping](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:1487), [JaxDEM epoch](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:817), [JaxDEM loss](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:676), [JaxDEM physics boundary handling](/home/wind/Documents/JaxDEM/jaxdem/utils/environment.py:36).

## 6. Remaining action/distribution issues

**Categorical dtype promotion is fixed.** Collection and environment utilities now zero inactive actions with dtype-preserving zeros. Continuous clipping rejects categorical inputs instead of silently converting category indices.

**BoxSpace inverse clipping and PPO likelihood replay are fixed.** The inverse now clips only at the nearest representable points inside `(-1, 1)`, independently of the output margin. Since even that cannot recover a latent from a fully saturated action, transformed-policy collection also stores the original latent and its base-distribution log-probability. PPO computes ratios directly in base coordinates, where the fixed Jacobian cancels. Collection re-evaluates density at the representable latent instead of using an optimized sampler's pre-rounding noise. Generic externally constructed trajectories without latent fields still use action-space likelihoods; saturation may make that fallback inaccurate.

For an exact invertible, parameter-independent transform, its log-Jacobian cancels in a same-action policy ratio, but transformed entropy still differs from base Gaussian entropy. JaxDEM's BoxSpace and MaxNormSpace use finite quadrature for the entropy correction, not an exact analytic integral. PufferLib's inspected continuous loss uses unsquashed Gaussian likelihood and analytic Gaussian entropy. Saturation, action clipping in environments, and inverse regularization must be controlled before assuming parity.

Sources: [JaxDEM collection/reset](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/__init__.py:162), [BoxSpace inverse](/home/wind/Documents/JaxDEM/jaxdem/rl/action_spaces/box_space.py:140), [JaxDEM transformed entropy](/home/wind/Documents/JaxDEM/jaxdem/rl/action_spaces/__init__.py:53), [PufferLib Gaussian likelihood](/home/wind/Documents/JaxDEM/PufferLib/src/algo.cu:1376).

## 7. Muon is not the same optimizer

| Detail | JaxDEM default, Optax 0.2.8 | PufferLib native Muon |
|---|---|---|
| Matrix routing | Rank two plus explicitly configured per-layer MinGRU | Rank at least two, reshaped for projection |
| Other parameters | Adam-family fallback | Nesterov update without matrix projection |
| Momentum | EMA with bias-corrected Nesterov construction | `m = mu*m + g`, then `u = g + mu*m` |
| Momentum coefficient | 0.95 default | 0.95 base configuration |
| Newton–Schulz projection | Five iterations, repeated `(3.4445, -4.7750, 2.0315)` | Five iterations with different coefficient triples |
| Epsilon | Trainer passes `eps=1e-12` | Gradient clip `1e-6`, projection normalization floor `1e-7` |
| Weight decay | Zero by default | Zero in the shown update call |

JaxDEM's stacked MinGRU kernel is `[layers, H, 3H]`. The trainer now supplies explicit Optax dimension specifications: each layer is an independent matrix with input/output axes 1/2 and batch axis 0. This fixes the earlier Adam-family routing while retaining Optax Muon. Multi-step checks compare these updates against separate Optax Muon instances for each layer, including actual NNX parameter trees. User-supplied dimension overrides are preserved. PufferLib still uses its own per-layer `[3H, H]` matrices and custom optimizer; the other optimizer differences in this section remain.

Both log-standard-deviation parameters are shaped `[1, action_dim]`. PufferLib's aspect scale for that shape is one; Optax's default input/output-axis convention gives `sqrt(action_dim)`. Ordinary dense kernels have opposite storage orientations and corresponding dimension conventions, so the identical log-std layout is a separate concern. Matching the optimizer name and learning rate does not match the update.

Sources: [JaxDEM schedule/optimizer](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:50), [JaxDEM MinGRU kernel](/home/wind/Documents/JaxDEM/jaxdem/rl/models/mingru.py:119), [PufferLib Muon](/home/wind/Documents/JaxDEM/PufferLib/src/algo.cu:1052), [installed Optax Muon](/home/wind/.python3/lib64/python3.14/site-packages/optax/contrib/_muon.py:519).

## 8. Accumulation, schedules, and defaults

JaxDEM averages raw gradients before global clipping and the stateful optimizer. With accumulation factor `g`, the constructor requires `K % g == 0`. Empty blocks are skipped entirely; a partial accumulation group can consequently cross a rollout boundary. Differing active counts make equal minibatch weighting different from one global active-transition mean. PufferLib applies an optimizer update per minibatch; optional NCCL reduction averages across ranks, not successive minibatches.

JaxDEM's cosine schedule uses optimizer-update count `q`, grouped into `U=K/g` planned updates per outer iteration: `lr(q)=lr0/2*(1+cos(pi*min(floor(q/U),N)/N))`. Empty blocks pause that counter. Without them it is constant within each outer iteration. PufferLib uses an outer-epoch cosine schedule with a configurable minimum-rate ratio and optional entropy annealing. JaxDEM has a zero LR floor and constant entropy coefficient. Normal runs end before the mathematical cosine endpoint.

The following JaxDEM sizes assume one agent per environment, no padding, and no accumulation.

| Default | JaxDEM | PufferLib base config |
|---|---:|---:|
| Agent slots / horizon | 1,024 / 64 | 4,096 / 64 |
| Minibatch transitions | 16,384 | 8,192 |
| Updates per rollout | 4 | 32 |
| Nominal replay | 1 | 1 |
| Learning rate | 0.01 | 0.015 |
| Gamma / lambda | 0.99 / 0.95 | 0.995 / 0.90 |
| Policy / value clip | 0.2 / same setting | 0.2 / separate 0.2 setting |
| Value / entropy coefficient | 2 / 0.001 | 2 / 0.001 |
| Max gradient norm | 1.5 | 1.5 |
| Optional trace correction | Off | Off |
| Advantage normalization / PER | Neither | Neither in inspected learner |
| Actor–learner overlap | No | Enabled |

Sources: [JaxDEM configuration](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:310), [JaxDEM schedule/optimizer](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:50), [PufferLib base training config](/home/wind/Documents/JaxDEM/PufferLib/config/default.ini:79), [PufferLib minibatch loop](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:1460).

## 9. Execution and diagnostics

PufferLib's default two-slot pipeline overlaps next-rollout collection with learning and snapshots actor weights at collection start. Behavior can therefore lag the learner before the first minibatch update. JaxDEM's rollout/update sequence has parameter dependencies between iterations; asynchronous Python dispatch does not make it an asynchronous actor–learner algorithm.

PufferLib uses preallocated buffers, CUDA graphs, fused kernels, and float32 master weights with configurable execution precision. JaxDEM uses NNX autodiff and JAX scans, with extra bootstrap forwards and transformed-entropy quadrature where configured. Source structure alone does not establish a measured speed ranking.

JaxDEM reports `approx_KL = 0.5*mean(log_ratio^2)` over active entries. PufferLib reports `mean(ratio-1-log_ratio)` and `mean(-log_ratio)` and a clip fraction. The first two estimators agree to second order around ratio one, but need not agree for larger changes. Neither inspected loop uses KL to stop updates.

JaxDEM `score` averages training rewards after DRIP over each selected active minibatch, then averages minibatches equally. Replay coverage, active counts, and DRIP can change this score independently of episode performance. `steps_per_sec` uses nominal `S*T*(1+skip_frames)`, including padding and requested physics repeats whose state updates may be discarded at boundaries. Constructor `total_timesteps` instead counts `S*T` slots without repeats.

`save_every` controls logging/synchronization, not checkpoint saving. Exact JaxDEM continuation requires model, optimizer, carry, environment, and PRNG state; `start_epoch` alone only sets the loop index. PufferLib's shown binary checkpoint saves model weights rather than the full learner state. JaxDEM warmup timing may include pending device work when logging is disabled, and a single-iteration run does not provide reliable throughput.

Sources: [PufferLib overlapping rollout/update loop](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:3084), [PufferLib checkpoint](/home/wind/Documents/JaxDEM/PufferLib/src/pufferl.cu:1722), [JaxDEM training/logging](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:519), [JaxDEM loss](/home/wind/Documents/JaxDEM/jaxdem/rl/trainers/ppo_trainer.py:676), [PufferLib loss diagnostics](/home/wind/Documents/JaxDEM/PufferLib/src/algo.cu:1599).

## 10. Evidence and remaining scope

The original doc review used source inspection only. The subsequent user-authorized numerical checks now include actual float32 PufferLib CUDA advantage and PPO-loss kernels, extracted without rewriting their computations and compiled in a temporary harness. The checks run on the RTX 5070 Ti against JaxDEM on CUDA. See [the parity results](/home/wind/Documents/JaxDEM/rl_parity_report.md) for conditions, errors, commands, and coverage.

Numerical agreement is checked by component, not claimed for the whole algorithm. Native custom-Muon parity is deliberately not required: JaxDEM retains Optax Muon. Truncation bootstrapping, the extra horizon transition, reward preprocessing, transformed entropy, defaults, and asynchronous collection still differ. Environment checks cover checkpoint timing, boundary states, integer/continuous actions, inactive masks, and evaluation chunking. Long learning curves and end-to-end throughput remain outside these focused checks.
