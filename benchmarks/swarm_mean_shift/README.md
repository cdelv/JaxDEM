# Swarm navigator: local vacancy and reward experiments

This isolated benchmark adapts vacancy-aware exploration from Sun et al.,
[Mean-shift exploration in shape assembly of robot swarms](https://www.nature.com/articles/s41467-023-39251-5),
*Nature Communications* 14, 3476 (2023). It tests learned force-control policies
on discrete targets; it does not reproduce the paper's full shape-assembly
controller, shape negotiation, or robot experiments. Production environments and
`/home/wind/Documents/experiments/intro_to_rl2.py` are left unchanged.

## Protocol

The baseline is the current production `SwarmNavigator`, with 64 agents and
64 goals in 2D. Physics, reset distribution, sensor range, action bounds, model
architecture, PPO settings, and training budget are shared across variants.
Settings follow `intro_to_rl2.py`, replacing its selected environment with the
swarm navigator:

- 32 parallel environments; 100 policy decisions per epoch; 120 training epochs.
- Each action advances 50 physics frames (`skip_frames=49`); episodes last
  100,000 physics frames, or 2,000 policy decisions.
- MinGRU actor–critic: 64 encoder features, 64 GRU features, default four layers,
  independent learned action standard deviation, and maximum action norm 2.
- PPO: learning rate `1e-3`, eight minibatches, default Muon optimizer and other
  trainer defaults. Cosine decay retains the source's **2,160 planned epochs**;
  stopping at 120 does not compress that schedule.
- CAPS temporal/spatial coefficients `1e-4`/`1e-3`; observation noise std `0.01`.

## Variants

The first screen compares these seven variants:

| Name | Reward | Observation |
| --- | --- | --- |
| `baseline` | Existing potential difference | Existing 26 features |
| `simple` | Rational state score | Existing 26 features |
| `compact` | Existing potential difference | 18 features from existing raw scans |
| `combined` | Rational state score | Same compact 18 features |
| `vacancy_lidar` | Existing potential difference | 26 features; remove occupied goals before target bin reduction |
| `paper_compact` | Existing potential difference | 18 features; raised-cosine mean shift over locally vacant goals |
| `paper_combined` | Rational state score | Same paper-inspired 18 features |

The rational reward is `max_k K(d_ik)/(1 + sum_j K(d_jk))`, with
`K(d)=1/(1+(d/(1.5*radius))**2)`. Distances and peer claims come from the same
raw angular scans as the baseline. This replaces a potential **difference** with
a positive **state score**, changing the objective and reward scale; it is not
merely an algebraic simplification or a reward proposed by the paper.

The compact features are velocity (2), vacancy-weighted displacement (2), vacancy
mass (1), own-goal proximity (1), and existing peer/wall LiDAR (12). Opposing
directions can cancel; mass distinguishes cancellation from an empty scan.
`vacancy_lidar` and `paper_*` use richer sensing of all goals and peers strictly
within sensor range, without angular occlusion before feature construction.
They remain local, but are not sensor-equivalent to the baseline. The paper
features exclude goals covered by any visible agent, including self, from their
exploration vector; a separate own-goal feature retains settling information.

Optional `exact_compact`/`exact_combined` use soft rational sharing over all local
detections, with baseline/rational reward respectively. `vacancy_combined` pairs
vacancy LiDAR with rational reward. These are available outside the first screen.
All rewards and policy observations use local information; shared simulator
coverage is used only for evaluation.

## Run and inspect

From the repository root, with a CUDA-enabled JAX installation:

```bash
JAX_PLATFORMS=cuda python -m benchmarks.swarm_mean_shift.run \
  --variants baseline simple compact combined vacancy_lidar paper_compact paper_combined \
  --seeds 0 --epochs 120 --eval-at 20 60 120 --output /tmp/swarm_screen
python -m benchmarks.swarm_mean_shift.summarize --input /tmp/swarm_screen
```

Evaluation defaults to 32 held-out episodes of 2,000 stochastic policy decisions.
`--action-mode deterministic` evaluates transformed policy means separately.
Training seed `s` uses model seed `s+1` and trainer seed `s+6`, preserving the
source's seeds 1/6 for `s=0`. Evaluation reset/action seeds are fixed at
10,000/20,000 across variants and training seeds. Repeat with `--seeds 0 1 2`
in a fresh output directory to estimate variation across independent training
runs; 32 evaluation episodes are **not** 32 training seeds.

Coverage counts each goal once if any agent center is within that agent's radius,
using all simulator positions rather than LiDAR. Report endpoint coverage,
coverage averaged over action endpoints 1–2,000, and the final 200-action average.
Speed, nearest-agent distance, and overlapping-agent fraction help interpret
settling; overlap is a contact proxy. Nonfinite metrics/states/parameters and
detected collider overflow invalidate a run.

Each variant/seed directory contains configuration and source hashes, training
and evaluation JSONL, per-action evaluation NPZ arrays, and an atomic final
`result.json`. Checkpoints include model/optimizer state and full trainer leaves;
`checkpoint_readme.txt` explains reconstruction with matching code/configuration.
The summarizer writes `aggregate.json`, `summary.csv`, `coverage_curves.png`, and
`learning_curves.png`, reporting uncertainty across training seeds and matched
candidate-minus-baseline differences. It ignores incomplete runs.

## Follow-ups and independent confirmation

`compare.py` also accepts `rational_delta` (one-scale rational potential
difference), `rational_scaled` (0.01 times rational state reward), and
`paper_augmented` (the original 26 features plus vacancy drift/mass, 29 total).
`combined_followup.py` accepts `rational_delta_vacancy`, combining the simpler
potential difference with vacancy-aware 26-feature LiDAR. These wrappers leave
the original runner and screening variants unchanged.

```bash
JAX_PLATFORMS=cuda python -m benchmarks.swarm_mean_shift.compare \
  --variants baseline vacancy_lidar rational_delta --seeds 0 1 2 3 4 \
  --epochs 120 --output /tmp/swarm_replicated
JAX_PLATFORMS=cuda python -m benchmarks.swarm_mean_shift.evaluate_checkpoints \
  --input /tmp/swarm_replicated --output /tmp/swarm_confirmation \
  --variants baseline vacancy_lidar rational_delta --seeds 0 1 2 3 4 \
  --eval-envs 64 --eval-reset-seed 30000 --eval-action-seed 40000
python -m benchmarks.swarm_mean_shift.summarize --input /tmp/swarm_confirmation
```

Confirmation restores each complete local checkpoint, validates source hashes
and leaf schemas, and evaluates without training on new layouts and action
randomness. Its results stay separate from the scenes used during screening.

## Doubled-spacing robustness

`spacing_followup.py` generates the original seeded geometry first, then scales
all objectives, agent starts, and domain coordinates by two. Agent radii,
sensing range, reward lengths, actions, and physics remain fixed. The baseline
and `rational_delta` both retain 26 observations. Episodes double to 4,000
actions, training doubles to 240 epochs, and the planned LR schedule doubles
to 4,320 epochs, preserving six episode cycles per environment and relative
schedule progress. All rewards and observations remain local.

Use the frozen runtime from the [completed spacing experiment](../results/swarm-mean-shift/2026-10-20-spacing-2x/README.md)
to reproduce the original radial-tanh action mapping. From that runtime:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.swarm_mean_shift.spacing_followup \
  --seeds 0 1 2 3 4 --checkpoint-at-120 --output /tmp/swarm_spacing
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.swarm_mean_shift.spacing_evaluate \
  --runtime-root "$PWD" --input /tmp/swarm_spacing \
  --output /tmp/swarm_spacing/confirmation
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.swarm_mean_shift.spacing_long_evaluate \
  --runtime-root "$PWD" --input /tmp/swarm_spacing \
  --output /tmp/swarm_spacing/extended_confirmation
```

The primary score averages the final 10% of each episode, preserving the
original experiment's fraction of settling time. Confirmation uses 64 new
layouts and new action randomness. `spacing_long_evaluate.py` evaluates the
same trained models for 8,000 actions with a real 400,000-physics-step limit;
it does not train further. `spacing_zero_shot.py` separately restores original
120-epoch models before changing evaluation geometry and time, so transfer
is not confused with learning directly in the enlarged environment.

From the repository root, `python -m benchmarks.swarm_mean_shift.spacing_summarize
--input <collection>` creates summaries and plots with uncertainty across
training seeds and matched candidate-minus-baseline intervals. The completed
test found the baseline stronger after fresh training at doubled spacing;
the original simpler-reward policies nevertheless transferred better without
updates. No production environment or original experiment script was modified.

## Conservative baseline changes

`conservative_variants.py` keeps the baseline's quartic settling, soft peer
sharing, potential-difference timing, and 26 observation outputs. It tests a
smaller factorial comparison at original (`_s1`) and exact doubled (`_s2`)
spacing:

| Family | Reward change | Observation change |
| --- | --- | --- |
| `baseline` | Original reward | Original observations |
| `cosine` | Replace only broad approach with the paper's raised-cosine kernel at sensor range | Original observations |
| `vacancy` | Original reward | Filter occupied targets before nearest-per-bin reduction |
| `cosine_vacancy` | Cosine approach | Vacancy target LiDAR |

The cosine approach uses `0.1 * sum(w * (1 + cos(pi*d/L))/2)` for raw visible
goals, retaining the existing attraction weight and sharing scale. It removes
the constant, quadratic, and decay shape controls from this candidate's score.
This scalar potential is an adaptation of the paper's exploration kernel;
it does not implement its mean-shift velocity controller. The vacancy candidate
requires richer all-local sensing, despite keeping the same observation size.

These comparisons train new matched baselines with the **current action map**.
Use the frozen runtime in the
[conservative experiment](../results/swarm-mean-shift/2026-10-20-conservative/)
to preserve that mapping. Example training commands from that runtime:

```bash
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.swarm_mean_shift.conservative_compare \
  --variants baseline_s1 cosine_s1 vacancy_s1 cosine_vacancy_s1 \
  --seeds 0 1 2 --output /tmp/swarm_conservative
JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m benchmarks.swarm_mean_shift.conservative_compare \
  --variants baseline_s2 cosine_s2 vacancy_s2 cosine_vacancy_s2 \
  --seeds 0 1 2 --epochs 240 --schedule-epochs 4320 \
  --max-physics-steps 200000 --eval-at 40 120 240 --checkpoint-at 120 \
  --output /tmp/swarm_conservative
```

`conservative_evaluate.py` restores selected checkpoints for 64 fresh layouts
(reset/action seeds 70000/80000). `conservative_long_evaluate.py` doubles their
real evaluation time limits without further training or geometry changes.
`conservative_summarize.py --input <collection>` writes summaries and plots
with seed-level uncertainty, matching each candidate to its same-seed,
same-spacing baseline. Summarize training screens and confirmation collections
separately. All helpers preserve source hashes and refuse checkpoint overwrite.

The corresponding production options are available without changing defaults:

```python
from jaxdem.rl.environments import SwarmNavigator

env = SwarmNavigator.Create(approach_kernel="cosine")
env = SwarmNavigator.Create(vacancy_aware_lidar=True)
```

`approach_kernel="legacy"` and `vacancy_aware_lidar=False` preserve the original
baseline. Cosine uses only `attraction_coeff` and `sharing_range` among the five
continuous reward controls; the three legacy approach-shape parameters remain
accepted for compatibility and have no effect in that mode. Vacancy sensing
keeps the original reward. Its initial gains at the original density reversed
at doubled spacing, so it remains a density-dependent experiment. The cosine
option is a simplification with a measured performance tradeoff, rather than
an established performance improvement. Consult the completed conservative
report before choosing an option. Both options are tested in 2-D and 3-D, while
the performance benchmarks are 2-D at N=64.
