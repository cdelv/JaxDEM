# Component parity and RL correctness checks

The native comparisons use PufferLib's actual CUDA functions extracted from the local `PufferLib/src/algo.cu` and `pufferl.cu`. The harness compiles only the selected functions into a temporary shared library; it does not reimplement their math or build/run the complete trainer. Both these kernels and JaxDEM run on the NVIDIA GeForce RTX 5070 Ti Laptop GPU, driver 615.71.09. Compilation uses the installed GCC 15 because the default host GCC is newer than this CUDA toolkit supports.

## Numerical parity

| Component | Conditions aligned | Result |
|---|---|---|
| GAE advantages and returns | Float32; same gamma/lambda; rewards already clipped; Puffer reward/done indexing shifted; only its first `T-1` transitions compared | Passed with and without terminal boundaries; maximum advantage absolute difference `0` |
| V-trace-style advantages and returns | Same alignment and unit caps; ratios both above and below one | Passed with and without terminal boundaries; maximum advantage absolute difference `3.73e-9` |
| Categorical PPO loss and gradient | One three-class head; same detached targets, behavior values/log-probabilities, clipping and entropy coefficients | Passed; maximum prediction-gradient absolute difference `4.47e-8` |
| Gaussian PPO loss and gradient | One continuous action dimension; unsquashed Gaussian; mean/logstd within Puffer's guard ranges; same targets and coefficients | Passed; maximum prediction-gradient absolute difference `1.91e-6` |
| Sequential replay/update order | Analytical SGD fixture with distinct rewards per contiguous block and a second pass | Passed; final value `0.87785`, with frozen behavior values and fresh advantages on revisit |
| MinGRU optimizer routing | Three successive updates; stacked kernel compared with independent Optax Muon instances per layer; plain and NNX parameter trees | Passed; tolerances `rtol=2e-5`, `atol=2e-6` |

Advantage/return assertions use `rtol=atol=3e-6`. Loss assertions use the same tolerance; gradient assertions use `rtol=4e-5`, `atol=4e-6`. Fixtures include both advantage signs and clipped/unclipped policy and value branches, away from exact nondifferentiable clipping thresholds. The Gaussian check also compares the summed log-standard-deviation gradient.

These are component comparisons, not proof of whole-trainer equivalence. The native fixtures intentionally exclude truncation, inactive padding, transformed entropy, reduced precision, and actor/learner overlap. Separate JaxDEM contract checks cover the first three where applicable. PufferLib custom-Muon equality is deliberately not asserted: JaxDEM retains Optax Muon, including its momentum, projection coefficients, bias treatment, and scaling conventions.

## Focused correctness results

The GPU chunks cover the three Optax routing cases, categorical/continuous
MLP and MinGRU training, saturated Box/MaxNorm likelihood replay, sequential
PPO alignment, and environment/wrapper contracts. All selected cases passed
after fixing the issues described below. A final targeted utility chunk passed
39 cases; constructor validation and bijector checks passed 45 cases on CPU.
The earlier target/recurrent math chunk passed 29 cases on CPU.

Both real-physics checkpoint tests also passed on GPU: Granulabot and TwoGears.
TwoGears produced a `6.78e-21` observation difference around zero between direct
and utility-wrapped physics loops; the test now uses `atol=1e-12` for that
comparison instead of requiring zero absolute error. No production physics
change was needed for that roundoff.

## Fixes covered by the checks

- **Recurrent full resets:** LSTM clears both hidden and cell carry with `zeros_like`; MinGRU does the same for hidden carry. NaNs and positive/negative infinities are cleared while preserving shape and dtype. The two new full-reset regression cases and the existing masked-reset case passed on CPU.
- **MinGRU:** default/partial Optax Muon factories receive explicit matrix axes for `mingru_kernel[layers, input, output]`. Axis zero batches independent layer projections. Other parameters retain Optax's routing; explicit user dimension overrides take precedence. The dependency floor is Optax 0.2.8. Older optimizer states with the MinGRU kernel in the Adam branch have a different structure and should not be reused as if unchanged.
- **Discrete actions:** inactive masking preserves integer dtype, including environments that directly index an action lookup table. Small PPO epochs cover categorical and continuous policies with both MLP and MinGRU models and inactive slots.
- **Continuous clipping:** integer inputs are rejected; inactive actions stay zero even when the clipping interval excludes zero. Checkpoint and physics see consistent transformed actions.
- **BoxSpace:** inverse clipping uses the nearest representable interior coordinate, rather than the output safety margin. Round trips beyond the former clipping threshold are checked.
- **Saturated policies:** collection stores latent samples and evaluates behavior density at those representable samples. PPO computes ratios using the stored latent/base likelihood, avoiding the lossy inverse. BoxSpace and MaxNormSpace are checked in float32/float64 with large means, unchanged and changed policies, and finite/correct gradients. This assumes a fixed bijector throughout replay.
- **Environment utilities:** tests cover scalar/vectorized evaluation, action-start checkpoints, frame repetition, mixed terminal/truncated/continuing batch members, final pre-reset bootstraps, inactive/disappearing agents, invalid counts, and zero-length evaluation. Recording chunks preserve the action/key sequence of the equivalent unchunked rollout. Clipping captures its own wrapper layer's mask method, so both clipping-before-vectorization and vectorization-before-clipping work.

## Reproduction

Native CUDA component checks (requires the local PufferLib checkout, CUDA toolkit and GPU access):

```bash
PUFFERLIB_CUDA_PARITY=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m pytest -q -s --disable-warnings tests/test_puffer_cuda_parity.py
```

Focused trainer, optimizer, action and environment checks:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -m pytest -q --disable-warnings \
  tests/test_mingru_muon.py tests/test_rl_action_contracts.py \
  tests/test_ppo_alignment.py tests/test_action_checkpoints.py \
  tests/test_env_wrappers.py -k 'not gear_step'
```

`JAX_PLATFORMS=cpu` can run the JaxDEM checks independently of CUDA. The native parity file skips unless `PUFFERLIB_CUDA_PARITY=1` is explicitly set. Compiler selection honors `CUDAHOSTCXX`, otherwise uses `g++-15` when present.

Long learning-curve and throughput experiments are outside this check. Full numerical identity is neither expected nor required for the retained truncation/horizon rules, reward processing, bounded-action entropy, and Optax optimizer.

## Seeded GPU reproducibility investigation

A separate diagnostic reproduced the configuration in
`/home/wind/Documents/experiments/intro_to_rl2.py`: SingleNavigator, LSTM with
128 encoder and 256 recurrent features, MaxNorm actions, model seed 1, trainer
seed 6, 32 environments, 100 actions per rollout, 50 skipped physics frames,
240 training iterations and 4,000 evaluation actions. Visualization was
omitted. The diagnostic executed the same epoch updates in an instrumented
loop, recording states instead of using the progress/logging wrapper. This
is a reproducibility check, not a multi-seed convergence benchmark.

On the same RTX 5070 Ti Laptop GPU with JAX/jaxlib 0.11.0, Flax 0.12.8,
Optax 0.2.8 and Distrax 0.1.9, two fresh default processes showed:

- Identical initial model, optimizer, environment and PRNG state.
- Identical first rollout, including observations, actions, latent samples,
  values, log-probabilities, rewards and boundary flags.
- Different post-update parameters after the first epoch; the largest model
  parameter difference was approximately `6.18e-6`.
- Identical trainer keys at every recorded stage, through final evaluation.
- Final success counts of 29/32 and 19/32.

Repeating the first epoch from the same starting state with the same compiled
executable in the second process gave identical results. This points toward
compilation/kernel-selection differences, rather than incorrect key handling;
the precise divergent kernel was not isolated. The first update's small
numerical differences subsequently grow through the training feedback loop.
XLA documents both compilation-time autotuning and execution-time GPU
nondeterminism in its [determinism guide](https://openxla.org/xla/determinism).

Two further fresh processes were run with:

```bash
XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0'
```

Both obtained 32/32 successes. Every recorded array matched bit-for-bit:
initial state; first-epoch replay; epochs 0, 1, 39 and 239; final trainer
state; and final evaluation state. This includes parameters, optimizer state,
recurrent carry, sampled trajectories, metrics, environments and PRNG keys
where recorded. This establishes repeatability for this configuration on
this hardware/software stack, not across arbitrary platforms or versions.
The flags are a diagnostic/reproducibility option and can trade throughput
for repeatability; they were not installed as global library defaults.

The temporary diagnostic harness and snapshots are under
`/tmp/jaxdem_rng_audit/`; the original experiment script was not modified.

## Proposed metrics

These are recommendations, not changes made in this patch:

- `rollout_reward_mean`: raw active transition reward before DRIP/replay, alongside the existing training-sample score.
- Completed-episode return, length, and completion count: accumulated across rollout boundaries, with termination/truncation counts reported separately. Avoid reporting an absent completed episode as a zero return.
- Policy clip fraction and the Puffer-compatible `mean(ratio - 1 - log_ratio)` estimate, while retaining or clearly renaming the existing squared-log-ratio estimate.
- Active policy transitions, requested physics frames, and accepted physics advances as separate counters/rates. The current nominal frame-slot rate includes padding and discarded updates.
- Actual learning rate, optimizer-update count, and skipped-empty-minibatch count, especially when gradient accumulation is enabled.
