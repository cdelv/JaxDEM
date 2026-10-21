"""Generate the final report after all selected evaluations are complete."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parent
PRIMARY = "coverage_final_10_percent_mean"
COLLECTIONS = ("initial_screening", "confirmation", "extended_confirmation")
data = {name: json.loads((ROOT/name/"aggregate_conservative.json").read_text()) for name in COLLECTIONS}
if [data[name]["completed_runs"] for name in COLLECTIONS] != [24, 20, 20]:
    raise ValueError("Final report requires all 24 initial, 20 fresh, and 20 extended evaluations")
if json.loads((ROOT/"qa/final_conservative_audit.json").read_text())["status"] != "passed":
    raise ValueError("Final data audit must pass before completing this report")


def group(collection, variant):
    return data[collection]["variants"][variant]


def measurement(collection, variant, metric=PRIMARY):
    x = group(collection, variant)["metrics"][metric]
    return f"{100*x['mean']:.2f} ± {100*x['std']:.2f}%"


def delta(collection, variant, interval=True):
    x = group(collection, variant)["paired_vs_baseline"]["metrics"][PRIMARY]
    result = f"{100*x['mean']:+.2f} pp"
    if interval:
        result += f" [{100*x['ci95'][0]:+.2f}, {100*x['ci95'][1]:+.2f}]"
    return result


main_rows = []
for collection, horizon in (("confirmation", "Training episode time"), ("extended_confirmation", "Twice training episode time")):
    for scale in (1, 2):
        actions = (2000 if scale == 1 else 4000) * (2 if collection == "extended_confirmation" else 1)
        main_rows.append(f"| {scale}× spacing, {actions:,} actions / {actions/10:.0f} s | {measurement(collection, f'baseline_s{scale}')} | {measurement(collection, f'cosine_s{scale}')} | {delta(collection, f'cosine_s{scale}')} |")

screen_rows = []
for name, label in (("baseline", "Baseline"), ("cosine", "Cosine approach"), ("vacancy", "Vacancy LiDAR"), ("cosine_vacancy", "Cosine + vacancy")):
    d1 = "—" if name == "baseline" else delta("initial_screening", f"{name}_s1", False)
    d2 = "—" if name == "baseline" else delta("initial_screening", f"{name}_s2", False)
    screen_rows.append(f"| {label} | {measurement('initial_screening', f'{name}_s1')} | {d1} | {measurement('initial_screening', f'{name}_s2')} | {d2} |")

for source, target in (("confirmation", "confirmation.png"), ("extended_confirmation", "longer_time.png")):
    shutil.copy2(ROOT/source/"comparison_conservative.png", ROOT/target)

text = """# Conservative improvements to the current swarm navigator

The tested reward simplification keeps the baseline's settling, soft peer sharing, and action-to-action potential differences, and replaces **only its broad approach term** with the paper's raised-cosine kernel. This reduces the active continuous reward controls from five to two and preserves the original 26 observations. It is available as an opt-in; the existing default remains unchanged. The results below quantify the performance tradeoff rather than claiming an established improvement or statistical equivalence.

Vacancy-aware goal sensing improves the original-density screen but loses heavily at doubled spacing. Adding the cosine reward to that sensing change does not resolve its spacing sensitivity. It is therefore an experimental, density-dependent option rather than the recommended general baseline.

## Fresh-layout comparison

All values are mean ± sample SD across **five training seeds**, each evaluated on **64 fresh layouts** with stochastic policy actions. Coverage counts each distinct objective once if an agent center is within its radius. The primary measure averages the final 10% of action endpoints: 200/400 actions at the original training horizons and 400/800 in the extensions. Paired intervals are two-sided 95% Student-t intervals across trained-model differences, with positive values favoring cosine. Evaluation episodes are averaged within each model, not treated as independent training seeds.

| Condition | Current baseline | Cosine approach | Paired difference [95% interval] |
| --- | ---: | ---: | ---: |
""" + "\n".join(main_rows) + r"""

![Five-seed fresh-layout comparison](confirmation.png)

The longer-time rows evaluate the **same trained checkpoints**, without further updates. Original spacing receives 4,000 actions / 400 simulated seconds. Doubled spacing receives 8,000 actions / 800 seconds, four times the original experiment's episode duration. The evaluation environment's actual truncation limit is extended; this is not repeated scoring after the shorter episode has frozen.

![Longer-time comparison](longer_time.png)

Main and extended evaluations compile different rollout lengths. They share starting layouts, checkpoints and random keys, but small floating-point differences grow into different trajectories, so their prefixes are not bit-identical. Treat these as separate finite-horizon evaluations. Any direct measurement of added-time progress should compare action 2,000→4,000 or 4,000→8,000 within the same extended rollout. See the [numerical-drift check](qa/extended_prefix_numeric_drift.json).

Five seeds give limited precision. Candidate selection used an exploratory screen, so these intervals are descriptive and do not establish formal noninferiority. Complete endpoint, episode-average, speed, overlap, per-seed and per-action measurements remain in the linked summaries and archives.

## What changes in the reward

For valid raw goal returns, use

\[
H(d)=\frac{1+\cos(\pi d/L)}2,\qquad
\Phi_i=\max_k w_{ik}\exp[-(d_{ik}/(1.5a_i))^4]
+\eta\sum_k w_{ik}H(d_{ik}),\qquad
r_i=\Phi_i(s_{t+1})-\Phi_i(s_t).
\]

The existing soft peer claim weight `w`, settling term, attraction weight `eta=0.1`, checkpoint timing, sensing range `L=16`, and physical dynamics are retained. `attraction_constant`, `attraction_quadratic`, and `attraction_decay` no longer affect the cosine score. The remaining continuous reward controls are `attraction_coeff` and `sharing_range`; the existing sensor range supplies the approach length scale. The three old shape controls remain accepted for legacy API compatibility and are still used by the default legacy kernel.

The [paper's Methods, Eq. 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC10264375/) use a raised-cosine weight for exploration over nearby vacant cells, alongside a separate entering controller. Here that kernel is adapted into a scalar approach potential for learned force control. It is not the paper's normalized mean-shift velocity command. Preserving separate settling and broad approach avoids repeating the earlier one-kernel replacement that failed at doubled spacing.

![Deterministic approach profiles](reward_profiles.png)

The cosine potential and its distance derivative vanish at sensor range. The old approach potential is nonzero there, so a new visible goal causes a finite approach-value jump. Other sources of discontinuity, including bin changes, remain; the entire reward is not claimed to be smooth.

## Initial factorial screen

Every row below uses the same **three paired seeds**, before the selected baseline/cosine configurations were extended to five. It keeps the baseline reward versus cosine and the original observations versus vacancy sensing separate. Screening uses 32 held-out layouts, reset seed 10000 and action seed 20000. Fresh confirmation uses separate seeds 70000/80000 and 64 layouts, with no tuning on those results.

| Configuration | Original spacing | Paired gap | Doubled spacing | Paired gap |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(screen_rows) + """

Vacancy LiDAR removes goals occupied by locally visible peers **before** nearest-per-bin reduction, preserving the observer's own covered goal. This can reveal a farther vacant goal hidden behind an occupied one in a bin. It retains 26 outputs but requires richer all-local detections than the original angular scans. It does not simplify sensing requirements. The doubled-spacing losses rule out treating it as a universal improvement.

## Matching distance, time and training

All experiments use N=64 and 64 goals, unit radii, LiDAR range 16, maximum action norm 2, and timestep 0.002 seconds. Each policy action advances 50 physics steps. The original seeded layout is generated first, then objective positions, agent starts, and domain coordinates are multiplied by one or two. Thus every objective separation, including jitter, doubles exactly up to floating-point rounding; sensing range and capture radius stay fixed.

Original spacing uses 120 epochs, 2,000-action episodes and a 2,160-epoch planned LR schedule. Doubled spacing uses 240 epochs, 4,000-action episodes and a 4,320-epoch schedule. With 32 environments and 100 actions per epoch, both conditions cover six episode cycles per environment, and the relative LR-schedule phase is preserved. MinGRU, PPO, CAPS and other settings match the source experiment. All training rewards and policy observations remain local; global coverage is used only for evaluation.

These experiments use a frozen copy of the **current Gaussian-CDF action mapping**, with new matched baselines. They are separate from the earlier radial-tanh comparisons; their numerical results must not be combined as if the action mapping were identical.

## Using the options

```python
from jaxdem.rl.environments import SwarmNavigator

# Reward simplification: same observations and physical dynamics.
env = SwarmNavigator.Create(approach_kernel="cosine")

# Experimental sensing option; its gains did not survive doubled spacing.
env = SwarmNavigator.Create(vacancy_aware_lidar=True)
```

The default is `approach_kernel="legacy", vacancy_aware_lidar=False`. Both options work in 2-D and 3-D; performance experiments here are 2-D only. `hide_occupied_objectives=False` disables either observation occupancy mask. Production implementations are checked against the experimental formulas and observations at both spacings. No changes were made to the original `intro_to_rl2.py` script or the current action mapping.

## Recovery, validation and saved artifacts

The app interruption stopped GPU jobs. All completed artifacts were preserved; two partial attempts were archived and restarted with identical seeds and settings. After the CUDA driver update, these restarts did not replay bit for bit: small early floating-point differences grew during training. One already-completed original-spacing control was therefore preserved separately and rerun so the primary five-seed comparison has matching driver cohorts within each pair (seeds 0–2 before the update, 3–4 afterward). All fresh and longer-time evaluations use the restored current driver. The initial doubled-spacing combined screen spans driver cohorts and remains exploratory. The cause of arithmetic drift was not isolated; it must not be described as an exact replay.

The final analysis contains 32 complete training runs and 5,760 training epochs, plus 20 fresh and 20 extended evaluations. Archived interrupted attempts and the earlier completed control are excluded from these counts. Relevant production tests: **42 CPU tests passed**, and the **15 new cosine tests also passed in float32**. Default legacy arithmetic and observations are checked against the frozen original source. The production options were added after the experiment runtime was frozen; tests establish their equivalence to the tested experimental variants, while source hashes keep GPU restoration isolated.

See [protocol](protocol.json), [selection decision](selection.json), [runtime manifest](runtime_manifest.json), [recovery record](recovery.json), [recovery-prefix audit](qa/recovery_prefix_audit.json), [final audit](qa/final_conservative_audit.json), and [production checks](qa/production_cosine_checks.json). Summaries: [initial three-seed screen](initial_screening/summary_conservative.csv), [five-seed fresh evaluation](confirmation/summary_conservative.csv), [longer-time evaluation](extended_confirmation/summary_conservative.csv), and [all final training runs](summary_conservative.csv).

Each training run retains configuration, source hashes, full trainer/model/optimizer checkpoints, contiguous training logs and evaluation arrays. Evaluation-only directories reference their source checkpoints. Frozen reproduction commands and helper descriptions are in the [benchmark guide](../../../swarm_mean_shift/README.md). Regenerate each collection with `conservative_summarize --input <collection>`, then run `python report_generation.py` from this results directory.
"""
lead = (
    "The five-seed fresh-layout comparison did not show a performance improvement. "
    f"At original spacing, baseline coverage was **{measurement('confirmation', 'baseline_s1')}** "
    f"versus **{measurement('confirmation', 'cosine_s1')}** for cosine. "
    f"At doubled spacing, it was **{measurement('confirmation', 'baseline_s2')}** "
    f"versus **{measurement('confirmation', 'cosine_s2')}**. "
    "Keep the current baseline as the performance default; use the cosine option when exploring the tradeoff between fewer controls and coverage."
)
heading = "# Conservative improvements to the current swarm navigator\n\n"
text = text.replace(heading, heading + lead + "\n\n", 1)
(ROOT/"README.md").write_text(text)
completion = {
    "status": "complete", "training_runs": 32, "analyzed_training_epochs": 5760,
    "initial_screen_runs": 24, "confirmation_evaluations": 20, "extended_evaluations": 20,
    "N": 64, "training_seed_counts": {"baseline": 5, "cosine": 5, "vacancy": 3, "cosine_vacancy": 3},
    "production_defaults_changed": False, "production_optins": ["approach_kernel=cosine", "vacancy_aware_lidar=True"],
    "report": "README.md", "audit": "qa/final_conservative_audit.json",
    "comparisons": {name: {key: value for key, value in data[name]["variants"].items()} for name in COLLECTIONS},
}
(ROOT/"completion.json").write_text(json.dumps(completion, indent=2) + "\n")
(ROOT/"progress.json").write_text(json.dumps({"phase": "complete", "training_runs_completed": 32, "confirmation_complete": 20, "extended_complete": 20, "report_pending": False}, indent=2) + "\n")
print(ROOT/"README.md")
