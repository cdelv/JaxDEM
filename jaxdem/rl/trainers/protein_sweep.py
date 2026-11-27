# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""PufferLib Protein Pareto-frontier hyperparameter optimizer for JaxDEM RL.

Pure-Python wrapper around ``pufferlib.sweep.Protein`` (dual Gaussian
Processes for score and log-cost, Pareto-frontier centered search, pinball-loss
quantile early stopping, and logistic failure classification) that sweeps
hyperparameters on any JaxDEM :class:`Environment` using :class:`PufferTrainer`,
JaxDEM's :class:`PPOTrainer`, or a custom evaluation callback.
"""

from __future__ import annotations

import configparser
import json
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from ...utils.environment import advance_action
from ..action_spaces import ActionSpace
from ..env_wrappers import vectorise_env
from ..environments import Environment
from ..models import Model
from . import Trainer
from .ppo_trainer import PPOTrainer
from .puffer_trainer import _PUFFERLIB_INSTALL_HINT, PufferTrainer

SpaceDistribution = Literal[
    "uniform",
    "int_uniform",
    "uniform_pow2",
    "log_normal",
    "logit_normal",
]

_VALID_DISTRIBUTIONS: frozenset[str] = frozenset(
    {"uniform", "int_uniform", "uniform_pow2", "log_normal", "logit_normal"}
)
_INT_DISTRIBUTIONS: frozenset[str] = frozenset({"int_uniform", "uniform_pow2"})


@dataclass(frozen=True, slots=True)
class SweepParam:
    """Single hyperparameter search dimension for Protein.

    Matches PufferLib's ``[sweep.<section>.<key>]`` specification.
    """

    name: str
    distribution: SpaceDistribution
    min: float
    max: float
    default: float
    scale: float | Literal["auto", "time"] = "auto"

    def __post_init__(self) -> None:
        if self.distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution {self.distribution!r}; "
                f"expected one of {sorted(_VALID_DISTRIBUTIONS)}"
            )
        if not (self.min < self.max):
            raise ValueError(
                f"{self.name}: min ({self.min}) must be strictly less than max ({self.max})"
            )
        if not (self.min <= self.default <= self.max):
            raise ValueError(
                f"{self.name}: default ({self.default}) must lie in [{self.min}, {self.max}]"
            )
        if self.distribution in ("log_normal", "uniform_pow2") and self.min <= 0.0:
            raise ValueError(
                f"{self.name}: distribution={self.distribution!r} requires min > 0"
            )
        if self.distribution == "logit_normal" and not (0.0 < self.min < self.max < 1.0):
            raise ValueError(
                f"{self.name}: distribution='logit_normal' requires 0 < min < max < 1"
            )

    @property
    def is_integer(self) -> bool:
        return self.distribution in _INT_DISTRIBUTIONS

    def resolved_scale(self) -> float:
        if self.scale == "auto":
            return 0.5
        if self.scale == "time":
            if self.min <= 0.0 or self.max <= 0.0:
                raise ValueError(f"{self.name}: scale='time' requires positive min/max")
            return float(1.0 / (math.log2(self.max) - math.log2(self.min)))
        return float(self.scale)

    def cast_value(self, value: float) -> float | int:
        return int(round(value)) if self.is_integer else float(value)

    @staticmethod
    def from_puffer_ini(
        ini_path: Path | str,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] = (),
    ) -> list[SweepParam]:
        """Parse ``[sweep.<section>.<key>]`` sections from a PufferLib ``.ini`` file."""
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read(str(ini_path), encoding="utf-8")
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude)

        params: list[SweepParam] = []
        for section in cfg.sections():
            if not section.startswith("sweep."):
                continue
            name = section[len("sweep.") :]
            if include_set is not None and name not in include_set:
                continue
            if name in exclude_set:
                continue
            sec, _, key = name.rpartition(".")
            if not sec or not key:
                raise ValueError(f"Expected [sweep.<section>.<key>], got [{section}]")
            dist_raw = cfg.get(section, "distribution").strip()
            min_v = float(cfg.get(section, "min").replace("_", ""))
            max_v = float(cfg.get(section, "max").replace("_", ""))
            scale_raw = cfg.get(section, "scale", fallback="auto").strip()
            scale_val: float | Literal["auto", "time"]
            if scale_raw == "auto":
                scale_val = "auto"
            elif scale_raw == "time":
                scale_val = "time"
            else:
                scale_val = float(scale_raw.replace("_", ""))

            if cfg.has_option(sec, key):
                default_v = float(cfg.get(sec, key).replace("_", ""))
                default_v = min(max(default_v, min_v), max_v)
            else:
                default_v = 0.5 * (min_v + max_v)

            params.append(
                SweepParam(
                    name=name,
                    distribution=dist_raw,  # type: ignore[arg-type]
                    min=min_v,
                    max=max_v,
                    default=default_v,
                    scale=scale_val,
                )
            )
        return params


@dataclass(frozen=True, slots=True)
class SuggestionInfo:
    """Diagnostics returned by Protein on each suggestion."""

    is_random: bool
    predicted_score: float
    predicted_cost: float
    rating: float
    n_pareto: int
    n_gp_obs: int
    n_candidates: int
    score_loss: float
    cost_loss: float


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """Single non-dominated point on the score-vs-cost Pareto frontier."""

    score: float
    cost: float
    params: dict[str, float | int]
    norm_params: tuple[float, ...]
    obs_index: int


@dataclass(slots=True)
class SweepTrial:
    """Record of one completed (or failed/early-stopped) sweep trial."""

    run_id: int
    params: dict[str, float | int]
    norm_params: tuple[float, ...]
    score: float
    cost: float
    steps: int
    is_failure: bool = False
    early_stopped: bool = False
    error: str | None = None
    suggestion: SuggestionInfo | None = None
    curve_scores: list[float] = field(default_factory=list)
    curve_costs: list[float] = field(default_factory=list)
    curve_steps: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SweepResult:
    """Full output of a :class:`ProteinSweep` run."""

    goal: Literal["maximize", "minimize"]
    metric: str
    trials: list[SweepTrial]
    pareto_frontier: list[ParetoPoint]
    pruned_pareto_frontier: list[ParetoPoint]
    elapsed_s: float

    @property
    def best_trial(self) -> SweepTrial | None:
        valid = [t for t in self.trials if not t.is_failure and math.isfinite(t.score)]
        if not valid:
            return None
        if self.goal == "minimize":
            return min(valid, key=lambda t: t.score)
        return max(valid, key=lambda t: t.score)

    def summary_table(self, *, pruned: bool = False) -> str:
        """Format the Pareto frontier as a compact text table."""
        pts = self.pruned_pareto_frontier if pruned else self.pareto_frontier
        if not pts:
            return "(empty Pareto frontier)"
        keys = list(pts[0].params.keys())
        header = ["cost_s", self.metric, *keys]
        rows = [
            [
                f"{p.cost:.2f}",
                f"{p.score:.4f}",
                *[
                    str(p.params[k])
                    if isinstance(p.params[k], int)
                    else f"{p.params[k]:.4g}"
                    for k in keys
                ],
            ]
            for p in pts
        ]
        widths = [
            max(len(header[c]), *(len(r[c]) for r in rows)) for c in range(len(header))
        ]
        lines = [
            "  ".join(header[c].rjust(widths[c]) for c in range(len(header))),
            "  ".join("-" * widths[c] for c in range(len(header))),
        ]
        for r in rows:
            lines.append("  ".join(r[c].rjust(widths[c]) for c in range(len(header))))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "metric": self.metric,
            "elapsed_s": self.elapsed_s,
            "pareto_frontier": [
                {"score": p.score, "cost": p.cost, "params": dict(p.params)}
                for p in self.pareto_frontier
            ],
            "pruned_pareto_frontier": [
                {"score": p.score, "cost": p.cost, "params": dict(p.params)}
                for p in self.pruned_pareto_frontier
            ],
            "trials": [
                {
                    "run_id": t.run_id,
                    "params": dict(t.params),
                    "score": t.score if math.isfinite(t.score) else None,
                    "cost": t.cost,
                    "steps": t.steps,
                    "is_failure": t.is_failure,
                    "early_stopped": t.early_stopped,
                    "error": t.error,
                    "curve_scores": t.curve_scores,
                    "curve_costs": t.curve_costs,
                    "curve_steps": t.curve_steps,
                }
                for t in self.trials
            ],
        }

    def save_json(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out


def _build_puffer_sweep_config(
    params: Sequence[SweepParam],
    *,
    metric_distribution: Literal["linear", "logit"],
    use_success_prob: bool,
    prune_pareto: bool,
    max_suggestion_cost: float,
    early_stop_quantile: float,
    use_gpu: bool,
) -> dict[str, Any]:
    """Translate a :class:`SweepParam` list into a ``pufferlib.sweep`` nested config."""
    cfg: dict[str, Any] = {
        "method": "Protein",
        "metric": "score",
        "metric_distribution": (
            "percentile" if metric_distribution == "logit" else "linear"
        ),
        # Protein's pareto_points and RobustLogCostModel assume higher score is better;
        # goal='minimize' is handled consistently by sign-flipping in ProteinOptimizer.
        "goal": "maximize",
        "downsample": 1 if use_success_prob else 5,
        "prune_pareto": bool(prune_pareto),
        "max_suggestion_cost": float(max_suggestion_cost),
        "early_stop_quantile": float(early_stop_quantile),
        "use_gpu": bool(use_gpu),
    }
    for p in params:
        parts = p.name.split(".")
        node = cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {
            "distribution": p.distribution,
            "min": float(p.min),
            "max": float(p.max),
            "scale": p.resolved_scale(),
        }
    return cfg


def _flat_to_nested(params: Mapping[str, float | int]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, val in params.items():
        parts = key.split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return nested


def _nested_to_flat(nested: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in nested.items():
        full = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            flat.update(_nested_to_flat(v, full))
        else:
            flat[full] = v
    return flat


class ProteinOptimizer:
    """Pure-Python ask/tell wrapper around ``pufferlib.sweep.Protein``."""

    def __init__(
        self,
        params: Sequence[SweepParam],
        *,
        cost_param: str | None = "train.total_timesteps",
        goal: Literal["maximize", "minimize"] = "maximize",
        metric_distribution: Literal["linear", "logit"] = "linear",
        num_random_samples: int = 10,
        suggestions_per_pareto: int = 256,
        gp_training_iter: int = 50,
        gp_learning_rate: float = 0.001,
        optimizer_reset_frequency: int = 50,
        gp_max_obs: int = 750,
        infer_batch_size: int = 4096,
        use_success_prob: bool = False,
        prune_pareto: bool = True,
        global_search_scale: float = 1.0,
        max_suggestion_cost: float = 3600.0,
        expansion_rate: float = 0.1,
        cost_random_suggestion: float = -0.8,
        early_stop_quantile: float = 0.3,
        top_k: int = 5,
        use_gpu: bool = True,
    ) -> None:
        if not params:
            raise ValueError("ProteinOptimizer requires at least one SweepParam.")
        if goal not in ("maximize", "minimize"):
            raise ValueError(f"goal must be 'maximize' or 'minimize', got {goal!r}")
        if metric_distribution not in ("linear", "logit"):
            raise ValueError(
                f"metric_distribution must be 'linear' or 'logit', got {metric_distribution!r}"
            )
        if goal == "minimize" and metric_distribution == "logit":
            raise ValueError(
                "metric_distribution='logit' is only valid with goal='maximize'"
            )

        self.params = tuple(params)
        self._name_to_idx = {p.name: i for i, p in enumerate(self.params)}
        if len(self._name_to_idx) != len(self.params):
            raise ValueError("Duplicate SweepParam names are not allowed.")

        self.goal: Literal["maximize", "minimize"] = goal
        self._sign = -1.0 if goal == "minimize" else 1.0
        self.cost_idx = (
            self._name_to_idx[cost_param]
            if cost_param is not None and cost_param in self._name_to_idx
            else -1
        )
        self._closed = False

        try:
            import pufferlib.sweep as puf_sweep  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_PUFFERLIB_INSTALL_HINT) from exc

        self._sweep_mod: Any = puf_sweep
        sweep_cfg = _build_puffer_sweep_config(
            self.params,
            metric_distribution=metric_distribution,
            use_success_prob=use_success_prob,
            prune_pareto=prune_pareto,
            max_suggestion_cost=max_suggestion_cost,
            early_stop_quantile=early_stop_quantile,
            use_gpu=use_gpu,
        )
        puf_cost_key = cost_param.replace(".", "/") if cost_param else "__none__"
        self._protein: Any = puf_sweep.Protein(
            sweep_cfg,
            max_suggestion_cost=float(max_suggestion_cost),
            num_random_samples=int(num_random_samples),
            num_keep_top_obs=int(top_k),
            global_search_scale=float(global_search_scale),
            suggestions_per_pareto=int(suggestions_per_pareto),
            expansion_rate=float(expansion_rate),
            gp_training_iter=int(gp_training_iter),
            gp_learning_rate=float(gp_learning_rate),
            gp_max_obs=int(gp_max_obs),
            infer_batch_size=int(infer_batch_size),
            optimizer_reset_frequency=int(optimizer_reset_frequency),
            use_gpu=bool(use_gpu),
            cost_param=puf_cost_key,
            prune_pareto=bool(prune_pareto),
        )
        if self.cost_idx >= 0:
            self._protein.cost_random_suggestion = float(cost_random_suggestion)
        flat_spaces = self._protein.hyperparameters.flat_spaces
        self._spaces: list[Any] = [
            flat_spaces[p.name.replace(".", "/")] for p in self.params
        ]

    @property
    def dim(self) -> int:
        return len(self.params)

    def normalize_value(self, name_or_idx: str | int, value: float) -> float:
        assert not self._closed, "ProteinOptimizer has been closed."
        idx = (
            self._name_to_idx[name_or_idx]
            if isinstance(name_or_idx, str)
            else name_or_idx
        )
        return float(self._spaces[idx].normalize(float(value)))

    def unnormalize_value(self, name_or_idx: str | int, norm: float) -> float | int:
        assert not self._closed, "ProteinOptimizer has been closed."
        idx = (
            self._name_to_idx[name_or_idx]
            if isinstance(name_or_idx, str)
            else name_or_idx
        )
        raw = float(self._spaces[idx].unnormalize(float(norm)))
        return self.params[idx].cast_value(raw)

    def default_point(self) -> tuple[tuple[float, ...], dict[str, float | int]]:
        """Return ``(norm_params, raw_params)`` for the configured defaults."""
        norm = tuple(
            self.normalize_value(i, p.default) for i, p in enumerate(self.params)
        )
        raw = self.unnormalize(norm)
        return norm, raw

    def normalize(self, params: Mapping[str, float | int]) -> tuple[float, ...]:
        return tuple(
            self.normalize_value(i, float(params[p.name]))
            for i, p in enumerate(self.params)
        )

    def unnormalize(self, norm: Sequence[float]) -> dict[str, float | int]:
        return {
            p.name: self.unnormalize_value(i, float(norm[i]))
            for i, p in enumerate(self.params)
        }

    def suggest(
        self, *, fixed_cost: float | None = None
    ) -> tuple[tuple[float, ...], dict[str, float | int], SuggestionInfo]:
        """Ask Protein for the next hyperparameter configuration to evaluate."""
        assert not self._closed, "ProteinOptimizer has been closed."
        fill = _flat_to_nested({p.name: p.default for p in self.params})
        sugg_nested, py_info = self._protein.suggest(
            fill,
            fixed_total_timesteps=(
                float(fixed_cost) if fixed_cost is not None else None
            ),
        )
        flat = _nested_to_flat(sugg_nested)
        if fixed_cost is not None and self.cost_idx >= 0:
            flat[self.params[self.cost_idx].name] = float(fixed_cost)
        raw = {p.name: p.cast_value(float(flat[p.name])) for p in self.params}
        norm = self.normalize(raw)
        is_rand = bool(
            self._protein.suggestion_idx <= self._protein.num_random_samples
            or not py_info
        )
        info_map = py_info if isinstance(py_info, Mapping) else {}
        pred_s = float(info_map.get("score", float("nan")))
        info = SuggestionInfo(
            is_random=is_rand,
            predicted_score=self._sign * pred_s if math.isfinite(pred_s) else 0.0,
            predicted_cost=float(info_map.get("cost", 0.0)),
            rating=float(info_map.get("rating", 0.0)),
            n_pareto=len(self.pareto_frontier(pruned=False)),
            n_gp_obs=len(self._protein.success_observations),
            n_candidates=int(info_map.get("n_candidates", 0)),
            score_loss=float(info_map.get("score_loss", 0.0)),
            cost_loss=float(info_map.get("cost_loss", 0.0)),
        )
        return norm, raw, info

    def observe(
        self,
        params: Sequence[float] | Mapping[str, float | int],
        score: float,
        cost: float,
        *,
        is_failure: bool = False,
    ) -> None:
        """Tell Protein the outcome ``(score, cost)`` of a trial or curve point."""
        assert not self._closed, "ProteinOptimizer has been closed."
        if isinstance(params, Mapping):
            raw_nested = _flat_to_nested(
                {p.name: p.cast_value(float(params[p.name])) for p in self.params}
            )
        else:
            norm = tuple(float(x) for x in params)
            if len(norm) != self.dim:
                raise ValueError(
                    f"Expected {self.dim} normalized params, got {len(norm)}"
                )
            raw_nested = _flat_to_nested(self.unnormalize(norm))

        internal_score = float("nan") if is_failure else self._sign * float(score)
        safe_cost = max(float(cost), 1e-6)
        self._protein.observe(
            raw_nested, internal_score, safe_cost, is_failure=bool(is_failure)
        )

    def should_stop(self, score: float, cost: float) -> bool:
        """Query Protein's pinball-loss quantile cost model for early stopping."""
        assert not self._closed, "ProteinOptimizer has been closed."
        if not math.isfinite(score):
            return True
        return bool(
            self._protein.should_stop(
                self._sign * float(score), max(float(cost), 1e-6)
            )
        )

    @property
    def counts(self) -> tuple[int, int, bool]:
        """Return ``(num_success_obs, num_failure_obs, early_stop_model_fitted)``."""
        assert not self._closed, "ProteinOptimizer has been closed."
        model = getattr(self._protein, "stop_threshold_model", None)
        fitted = bool(
            getattr(model, "is_fitted", getattr(model, "params", None) is not None)
        )
        return (
            len(self._protein.success_observations),
            len(self._protein.failure_observations),
            fitted,
        )

    def pareto_frontier(self, *, pruned: bool = False) -> list[ParetoPoint]:
        """Return the current score-vs-cost Pareto frontier sorted by ascending cost."""
        assert not self._closed, "ProteinOptimizer has been closed."
        obs = self._protein.success_observations
        if not obs:
            return []
        front, idxs = self._sweep_mod.pareto_points(obs)
        if pruned and hasattr(self._sweep_mod, "prune_pareto_front"):
            id_to_idx = {
                id(entry): int(idx) for idx, entry in zip(idxs, front, strict=False)
            }
            front = self._sweep_mod.prune_pareto_front(front)
            idxs = [id_to_idx.get(id(entry), i) for i, entry in enumerate(front)]
        points: list[ParetoPoint] = []
        for obs_idx, entry in zip(idxs, front, strict=False):
            norm = tuple(float(x) for x in entry["input"])
            points.append(
                ParetoPoint(
                    score=self._sign * float(entry["output"]),
                    cost=float(entry["cost"]),
                    params=self.unnormalize(norm),
                    norm_params=norm,
                    obs_index=int(obs_idx),
                )
            )
        return points

    def close(self) -> None:
        self._closed = True
        self._protein = None

    def __enter__(self) -> ProteinOptimizer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def default_sweep_params(
    *,
    min_timesteps: int = 50_000,
    max_timesteps: int = 2_000_000,
    default_timesteps: int = 200_000,
    min_num_envs: int = 32,
    max_num_envs: int = 256,
    default_num_envs: int = 64,
    include_architecture: bool = True,
    include_num_envs: bool = False,
) -> list[SweepParam]:
    """Sensible default Protein sweep space for JaxDEM RL environments."""
    params: list[SweepParam] = [
        SweepParam(
            "train.total_timesteps",
            "log_normal",
            float(min_timesteps),
            float(max_timesteps),
            float(default_timesteps),
            scale="time",
        ),
        SweepParam("train.learning_rate", "log_normal", 3e-4, 1e-1, 1.5e-2, scale=0.5),
        SweepParam("train.ent_coef", "log_normal", 1e-5, 5e-2, 1e-3),
        SweepParam("train.gamma", "logit_normal", 0.85, 0.999, 0.99),
        SweepParam("train.gae_lambda", "logit_normal", 0.5, 0.99, 0.95),
        SweepParam("train.clip_coef", "uniform", 0.1, 0.5, 0.2),
        SweepParam("train.vf_coef", "uniform", 0.25, 4.0, 2.0),
        SweepParam("train.horizon", "uniform_pow2", 16.0, 128.0, 32.0),
        SweepParam("train.replay_ratio", "uniform", 0.5, 4.0, 1.0),
    ]
    if include_architecture:
        params.extend(
            [
                SweepParam("policy.hidden_size", "uniform_pow2", 32.0, 256.0, 64.0),
                SweepParam("policy.num_layers", "int_uniform", 1.0, 4.0, 2.0),
            ]
        )
    if include_num_envs:
        params.append(
            SweepParam(
                "vec.num_envs",
                "uniform_pow2",
                float(min_num_envs),
                float(max_num_envs),
                float(default_num_envs),
            )
        )
    return params


def make_ppo_eval_fn(
    env: Environment,
    *,
    metric_fn: Callable[[Environment, jax.Array], jax.Array] | None = None,
    eval_n: int = 32,
    skip_frames: int = 0,
    max_steps: int | None = None,
    seed: int = 101,
) -> Callable[[Any], float]:
    """Build a JIT-compiled deterministic full-episode evaluator for :class:`PPOTrainer`.

    Parameters
    ----------
    env : Environment
        Single (unvectorized) JaxDEM environment.
    metric_fn : callable, optional
        Pure-JAX function ``(final_env, rewards) -> scalar`` evaluated inside
        ``jax.jit`` after one deterministic episode of ``ep_decisions`` steps on
        ``eval_n`` fixed held-out reset states. ``final_env`` is the live
        vectorized end-of-episode environment (before reset) and ``rewards`` has
        shape ``(ep_decisions, eval_n, max_num_agents)``. When omitted, defaults
        to the mean cumulative episode return ``jnp.mean(jnp.sum(rewards, axis=0))``.
    eval_n : int
        Number of fixed held-out environments to average over.
    skip_frames : int
        Action repeat count matching training.
    max_steps : int, optional
        Episode length in physics steps. Inferred from
        ``env.env_params["max_steps"]`` when omitted.
    seed : int
        Fixed seed for held-out initial states.

    """
    if max_steps is None:
        raw_max = env.env_params.get("max_steps")
        if raw_max is None:
            raise ValueError(
                "max_steps could not be inferred from env.env_params; "
                "pass eval_max_steps explicitly."
            )
        max_steps = int(jnp.asarray(raw_max).item())
    ep_decisions = max(1, int(max_steps) // (1 + int(skip_frames)))

    eval_envs = vectorise_env(env, n=int(eval_n))
    eval_envs = eval_envs.reset(
        eval_envs, jax.random.split(jax.random.key(int(seed)), int(eval_n))
    )

    @jax.jit
    def _eval_impl(graphdef: Any, graphstate: Any) -> jax.Array:
        model, *_ = nnx.merge(graphdef, graphstate, copy=True)
        model.eval()
        obs0 = eval_envs.observation(eval_envs)
        model.reset(shape=obs0.shape)
        m_def, m_state = nnx.split(model)

        def body(
            carry: tuple[Environment, Any], _: None
        ) -> tuple[tuple[Environment, Any], jax.Array]:
            curr_env, curr_state = carry
            m = nnx.merge(m_def, curr_state)
            obs = curr_env.observation(curr_env)
            pi, _ = m(obs, sequence=False)
            act = m.policy_output(pi)
            curr_env, _, _ = advance_action(curr_env, act, skip_frames=skip_frames)
            rew = curr_env.reward(curr_env)
            return (curr_env, nnx.state(m)), rew

        (final_env, _), rewards = jax.lax.scan(
            body, (eval_envs, m_state), xs=None, length=ep_decisions
        )
        if metric_fn is not None:
            return jnp.asarray(metric_fn(final_env, rewards), dtype=jnp.float32)
        return jnp.mean(jnp.sum(rewards, axis=0))

    def _eval_fn(tr: Any) -> float:
        return float(_eval_impl(tr.graphdef, tr.graphstate).item())

    return _eval_fn


def _bin_mean(values: Sequence[float], points: int) -> list[float]:
    """Bin-average *values* into *points* buckets."""
    n = len(values)
    if n == 0:
        return [0.0] * points
    out: list[float] = []
    for p in range(points):
        lo = (p * n) // points
        hi = max(lo + 1, ((p + 1) * n) // points)
        hi = min(hi, n)
        seg = values[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


def _snap_minibatch_size(
    requested: int, total_agents: int, horizon: int
) -> int:
    """Snap *requested* minibatch size to a valid divisor of ``total_agents * horizon``."""
    batch_size = total_agents * horizon
    req_rows = max(1, min(total_agents, int(round(requested / horizon))))
    divisors = [d for d in range(1, total_agents + 1) if total_agents % d == 0]
    best_rows = min(
        divisors, key=lambda d: (abs(d - req_rows), abs(d * horizon - requested))
    )
    return min(batch_size, best_rows * horizon)


def _write_puffer_ini_log(
    log_dir: Path,
    env_name: str,
    run_tag: str,
    config: Mapping[str, Any],
    metric_series: Mapping[str, Sequence[float]],
) -> Path:
    """Write a PufferLib v1 ``.ini`` sweep log."""
    env_dir = log_dir / env_name
    env_dir.mkdir(parents=True, exist_ok=True)
    path = env_dir / f"{run_tag}.ini"

    sections: dict[str, dict[str, Any]] = {}
    for full_key, val in config.items():
        if "." in full_key:
            sec, _, k = full_key.rpartition(".")
        else:
            sec, k = "base", full_key
        sections.setdefault(sec, {})[k] = val

    lines = ["# PufferLib log v1"]
    for sec, kv in sections.items():
        lines.append(f"\n[{sec}]")
        for k, v in kv.items():
            lines.append(f"{k} = {v}")

    lines.append("\n[metrics]")
    for k, seq in metric_series.items():
        if seq:
            lines.append(f"{k} = " + ",".join(f"{float(v):.17g}" for v in seq))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class ProteinSweep:
    """Pareto-frontier hyperparameter sweeper for JaxDEM RL environments.

    Accepts any JaxDEM :class:`Environment` (or factory accepting swept ``env.*``
    parameters) and runs ``pufferlib.sweep.Protein`` to discover the
    score-vs-wall-clock-cost Pareto frontier.

    Supports three execution backends:
    - ``backend="puffer"`` (default): :class:`PufferTrainer` (``pufferlib.pufferl.PuffeRL``).
    - ``backend="ppo"``: JaxDEM's full-JIT :class:`PPOTrainer`.
    - ``evaluate_fn``: custom callable ``(params, should_stop) -> dict`` for
      arbitrary training loops.
    """

    def __init__(
        self,
        env: Environment | Callable[[Mapping[str, Any]], Environment],
        params: Sequence[SweepParam] | None = None,
        *,
        backend: Literal["puffer", "ppo"] = "puffer",
        metric: str = "env/episode_return",
        fallback_metric: str | None = "env/active_return",
        goal: Literal["maximize", "minimize"] = "maximize",
        cost_param: str | None = "train.total_timesteps",
        max_runs: int = 30,
        num_random_samples: int = 8,
        downsample: int = 5,
        early_stop: bool = True,
        early_stop_quantile: float = 0.3,
        prune_pareto: bool = True,
        max_suggestion_cost: float = 3600.0,
        max_trial_seconds: float | None = None,
        # Base trainer / env defaults (overridden by swept params when present)
        num_envs: int = 64,
        num_steps_epoch: int = 32,
        skip_frames: int = 0,
        num_epochs: int = 50,
        num_minibatches: int = 4,
        hidden_size: int = 64,
        num_layers: int = 2,
        precision: str = "float32",
        action_space: ActionSpace | None = None,
        clip_actions: bool = False,
        clip_range: tuple[float, float] = (-1.0, 1.0),
        reward_scale: float = 1.0,
        model_type: str = "MinGRUActorCritic",
        model_factory: (
            Callable[[Environment, Mapping[str, float | int], nnx.Rngs], Model] | None
        ) = None,
        ppo_kwargs: Mapping[str, Any] | None = None,
        puffer_kwargs: Mapping[str, Any] | None = None,
        strict_constraints: bool = False,
        report_every: int = 5,
        eval_fn: Callable[[Any], float] | None = None,
        eval_metric_fn: Callable[[Environment, jax.Array], jax.Array] | None = None,
        eval_env: Environment | None = None,
        eval_n: int = 32,
        eval_max_steps: int | None = None,
        eval_seed: int = 101,
        eval_on_report: bool = False,
        evaluate_fn: (
            Callable[
                [Mapping[str, float | int], Callable[[float, float], bool]],
                Mapping[str, Any],
            ]
            | None
        ) = None,
        seed: int = 73,
        use_gpu: bool = True,
        log_dir: Path | str | None = None,
        env_name: str | None = None,
        protein_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if backend not in ("puffer", "ppo"):
            raise ValueError(f"backend must be 'puffer' or 'ppo', got {backend!r}")
        if max_runs < 1:
            raise ValueError(f"max_runs must be >= 1, got {max_runs}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")

        self._env_or_factory = env
        self.params = (
            tuple(params) if params is not None else tuple(default_sweep_params())
        )
        self.backend: Literal["puffer", "ppo"] = backend
        self.metric = metric
        self.fallback_metric = fallback_metric
        self.goal: Literal["maximize", "minimize"] = goal
        self.cost_param = cost_param
        self.max_runs = int(max_runs)
        self.num_random_samples = int(num_random_samples)
        self.downsample = int(downsample)
        self.early_stop = bool(early_stop)
        self.early_stop_quantile = float(early_stop_quantile)
        self.prune_pareto = bool(prune_pareto)
        self.max_suggestion_cost = float(max_suggestion_cost)
        self.max_trial_seconds = (
            float(max_trial_seconds) if max_trial_seconds is not None else None
        )

        self.base_num_envs = int(num_envs)
        self.base_num_steps_epoch = int(num_steps_epoch)
        self.base_skip_frames = int(skip_frames)
        self.base_num_epochs = int(num_epochs)
        self.base_num_minibatches = int(num_minibatches)
        self.base_hidden_size = int(hidden_size)
        self.base_num_layers = int(num_layers)
        self.precision = precision
        self.action_space = action_space
        self.clip_actions = clip_actions
        self.clip_range = clip_range
        self.base_reward_scale = float(reward_scale)
        if eval_fn is not None and eval_metric_fn is not None:
            raise ValueError("Pass either eval_fn or eval_metric_fn, not both.")
        if backend == "puffer" and model_factory is not None:
            raise ValueError(
                "model_factory is only supported with backend='ppo'; "
                "pass a custom PyTorch policy via puffer_kwargs={'policy': ...} for backend='puffer'."
            )
        if backend == "puffer" and eval_metric_fn is not None:
            raise ValueError(
                "eval_metric_fn is only supported with backend='ppo'; "
                "pass eval_fn for custom PufferTrainer evaluation."
            )

        self.model_type = model_type
        self.model_factory = model_factory
        self.ppo_kwargs = dict(ppo_kwargs) if ppo_kwargs is not None else {}
        self.puffer_kwargs = dict(puffer_kwargs) if puffer_kwargs is not None else {}
        self.strict_constraints = strict_constraints
        self.report_every = int(report_every)
        self.eval_fn = eval_fn
        self.eval_metric_fn = eval_metric_fn
        self.eval_env = eval_env
        self.eval_n = int(eval_n)
        self.eval_max_steps = eval_max_steps
        self.eval_seed = int(eval_seed)
        self.eval_on_report = bool(eval_on_report or eval_metric_fn is not None)
        base_eval_env: Environment | None
        if eval_env is not None:
            base_eval_env = eval_env
        elif isinstance(env, Environment):
            base_eval_env = env
        elif eval_metric_fn is not None:
            default_raw = {p.name: p.cast_value(p.default) for p in self.params}
            base_eval_env = self._build_env(default_raw)
            warnings.warn(
                "ProteinSweep received an env factory with eval_metric_fn but no "
                "explicit eval_env; freezing eval_env at the default env.* parameters "
                "so all trials are scored on a consistent horizon without recompiling.",
                UserWarning,
                stacklevel=2,
            )
        else:
            base_eval_env = None
        self._base_eval_env: Environment | None = base_eval_env
        self._eval_fn_cache: dict[int, Callable[[Any], float]] = {}
        if eval_metric_fn is not None and base_eval_env is not None:
            self._eval_fn_cache[self.base_skip_frames] = make_ppo_eval_fn(
                base_eval_env,
                metric_fn=eval_metric_fn,
                eval_n=eval_n,
                skip_frames=self.base_skip_frames,
                max_steps=eval_max_steps,
                seed=eval_seed,
            )
        self.evaluate_fn = evaluate_fn
        self.seed = int(seed)
        self.use_gpu = bool(use_gpu)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.env_name = env_name or (
            type(env).__name__ if isinstance(env, Environment) else "jaxdem_env"
        )
        self.protein_kwargs = dict(protein_kwargs) if protein_kwargs is not None else {}

    def _build_env(self, raw_params: Mapping[str, float | int]) -> Environment:
        if isinstance(self._env_or_factory, Environment):
            return self._env_or_factory
        env_kwargs = {
            k[len("env.") :]: v
            for k, v in raw_params.items()
            if k.startswith("env.") and k != "env.reward_scale"
        }
        return self._env_or_factory(env_kwargs)

    def _extract_score(self, metrics: Mapping[str, float]) -> float:
        val = metrics.get(self.metric, float("nan"))
        if math.isfinite(val):
            return float(val)
        if self.fallback_metric is not None:
            fb = metrics.get(self.fallback_metric, float("nan"))
            if math.isfinite(fb):
                return float(fb)
        return float("nan")

    def _run_puffer_trial(
        self,
        run_id: int,
        raw_params: Mapping[str, float | int],
        should_stop_fn: Callable[[float, float], bool],
    ) -> tuple[
        float,
        float,
        int,
        bool,
        list[float],
        list[float],
        list[float],
        dict[str, Any],
    ]:
        env = self._build_env(raw_params)
        agents_per_env = env.max_num_agents
        if "vec.num_envs" in raw_params:
            num_envs = max(1, int(round(float(raw_params["vec.num_envs"]))))
        elif "vec.total_agents" in raw_params:
            num_envs = max(
                1, int(round(float(raw_params["vec.total_agents"]) / agents_per_env))
            )
        else:
            num_envs = self.base_num_envs
        total_agents = num_envs * agents_per_env

        horizon = max(
            1,
            int(
                round(
                    float(
                        raw_params.get(
                            "train.bptt_horizon",
                            raw_params.get("train.horizon", self.base_num_steps_epoch),
                        )
                    )
                )
            ),
        )
        skip_frames = int(
            round(float(raw_params.get("train.skip_frames", self.base_skip_frames)))
        )
        batch_size = total_agents * horizon

        minibatch_size: int | None
        num_minibatches = self.base_num_minibatches
        if "train.minibatch_size" in raw_params:
            req_mb = int(round(float(raw_params["train.minibatch_size"])))
            if self.strict_constraints:
                minibatch_size = req_mb
            else:
                minibatch_size = _snap_minibatch_size(req_mb, total_agents, horizon)
        elif "train.num_minibatches" in raw_params:
            req_nmb = max(1, int(round(float(raw_params["train.num_minibatches"]))))
            if self.strict_constraints:
                num_minibatches = req_nmb
                minibatch_size = None
            else:
                minibatch_size = _snap_minibatch_size(
                    max(horizon, batch_size // req_nmb), total_agents, horizon
                )
        else:
            minibatch_size = _snap_minibatch_size(
                max(horizon, batch_size // num_minibatches), total_agents, horizon
            )

        if "train.total_timesteps" in raw_params:
            target_steps = int(round(float(raw_params["train.total_timesteps"])))
            num_epochs = max(2, int(round(target_steps / batch_size)))
        elif "train.num_epochs" in raw_params:
            num_epochs = max(2, int(round(float(raw_params["train.num_epochs"]))))
        else:
            num_epochs = max(2, self.base_num_epochs)

        hidden_size = int(
            round(float(raw_params.get("policy.hidden_size", self.base_hidden_size)))
        )
        num_layers = int(
            round(float(raw_params.get("policy.num_layers", self.base_num_layers)))
        )
        reward_scale = float(
            raw_params.get("env.reward_scale", self.base_reward_scale)
        )

        trainer_kwargs: dict[str, Any] = {
            "num_envs": num_envs,
            "num_steps_epoch": horizon,
            "skip_frames": skip_frames,
            "num_epochs": num_epochs,
            "minibatch_size": minibatch_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": float(raw_params.get("train.learning_rate", 0.015)),
            "gamma": float(raw_params.get("train.gamma", 0.99)),
            "gae_lambda": float(raw_params.get("train.gae_lambda", 0.95)),
            "clip_coef": float(raw_params.get("train.clip_coef", 0.2)),
            "vf_coef": float(raw_params.get("train.vf_coef", 2.0)),
            "vf_clip_coef": float(raw_params.get("train.vf_clip_coef", 0.2)),
            "ent_coef": float(raw_params.get("train.ent_coef", 0.001)),
            "max_grad_norm": float(raw_params.get("train.max_grad_norm", 1.5)),
            "momentum": float(raw_params.get("train.momentum", 0.95)),
            "replay_ratio": float(
                raw_params.get(
                    "train.update_epochs", raw_params.get("train.replay_ratio", 1.0)
                )
            ),
            "action_space": self.action_space,
            "clip_actions": self.clip_actions,
            "clip_range": self.clip_range,
            "reward_scale": reward_scale,
            "seed": self.seed + run_id,
            **self.puffer_kwargs,
        }

        hist_scores: list[float] = []
        hist_costs: list[float] = []
        hist_steps: list[float] = []
        stopped_early = False

        def _steady_cost(m: Mapping[str, float], fallback_s: float) -> float:
            if "perf/elapsed_s" in m and "perf/avg_epoch_ms" in m:
                return max(
                    float(m["perf/elapsed_s"]) + float(m["perf/avg_epoch_ms"]) * 1e-3,
                    1e-4,
                )
            if "perf/epoch_ms" in m:
                return max(float(m["perf/epoch_ms"]) * 1e-3, 1e-4)
            return max(fallback_s, 1e-4)

        with PufferTrainer.create(env, **trainer_kwargs) as trainer:
            t_train_start = time.perf_counter()
            report_every = max(
                1, min(self.report_every, max(1, num_epochs // max(self.downsample, 1)))
            )

            def _on_report(ep: int, m: Mapping[str, float]) -> None:
                sc = self._extract_score(m)
                if math.isfinite(sc):
                    cost_now = _steady_cost(m, time.perf_counter() - t_train_start)
                    hist_scores.append(sc)
                    hist_costs.append(cost_now)
                    hist_steps.append(float((ep + 1) * batch_size))

            def _should_stop(ep: int, m: Mapping[str, float]) -> bool:
                nonlocal stopped_early
                cost_now = _steady_cost(m, time.perf_counter() - t_train_start)
                if (
                    self.max_trial_seconds is not None
                    and cost_now >= self.max_trial_seconds
                ):
                    return True
                if not self.early_stop:
                    return False
                sc = self._extract_score(m)
                if should_stop_fn(sc, cost_now):
                    stopped_early = True
                    return True
                return False

            last_m = trainer.train(
                num_epochs=num_epochs,
                verbose=False,
                log=False,
                report_every=report_every,
                on_report=_on_report,
                should_stop=_should_stop,
            )
            total_cost = _steady_cost(last_m, time.perf_counter() - t_train_start)
            total_steps = trainer.epochs_completed * batch_size

            if self.eval_fn is not None:
                final_score = float(self.eval_fn(trainer))
            else:
                final_score = self._extract_score(last_m)

        if not math.isfinite(final_score) and hist_scores:
            final_score = hist_scores[-1]
        if not math.isfinite(final_score):
            raise RuntimeError("Trial produced non-finite objective score.")

        if hist_costs:
            total_cost = max(total_cost, hist_costs[-1])
        cfg_log = {
            "vec.total_agents": total_agents,
            "policy.hidden_size": hidden_size,
            "train.total_timesteps": total_steps,
            "train.horizon": horizon,
            "train.minibatch_size": minibatch_size,
            "train.learning_rate": trainer_kwargs["learning_rate"],
            "train.gamma": trainer_kwargs["gamma"],
            "train.gae_lambda": trainer_kwargs["gae_lambda"],
            "train.clip_coef": trainer_kwargs["clip_coef"],
            "train.vf_coef": trainer_kwargs["vf_coef"],
            "train.vf_clip_coef": trainer_kwargs["vf_clip_coef"],
            "train.ent_coef": trainer_kwargs["ent_coef"],
            "train.max_grad_norm": trainer_kwargs["max_grad_norm"],
            "train.momentum": trainer_kwargs["momentum"],
            "train.replay_ratio": trainer_kwargs["replay_ratio"],
        }
        return (
            final_score,
            total_cost,
            total_steps,
            stopped_early,
            hist_scores,
            hist_costs,
            hist_steps,
            cfg_log,
        )

    def _run_ppo_trial(
        self,
        run_id: int,
        raw_params: Mapping[str, float | int],
        should_stop_fn: Callable[[float, float], bool],
    ) -> tuple[
        float,
        float,
        int,
        bool,
        list[float],
        list[float],
        list[float],
        dict[str, Any],
    ]:
        env = self._build_env(raw_params)
        agents_per_env = env.max_num_agents
        if "vec.num_envs" in raw_params:
            num_envs = max(1, int(round(float(raw_params["vec.num_envs"]))))
        elif "vec.total_agents" in raw_params:
            num_envs = max(
                1, int(round(float(raw_params["vec.total_agents"]) / agents_per_env))
            )
        else:
            num_envs = self.base_num_envs
        total_agents = num_envs * agents_per_env

        horizon = max(
            2,
            int(round(float(raw_params.get("train.horizon", self.base_num_steps_epoch)))),
        )
        skip_frames = int(
            round(float(raw_params.get("train.skip_frames", self.base_skip_frames)))
        )
        batch_size = total_agents * horizon

        if "train.minibatch_size" in raw_params:
            mb_size = _snap_minibatch_size(
                int(round(float(raw_params["train.minibatch_size"]))),
                total_agents,
                horizon,
            )
        else:
            req_nmb = max(
                1,
                int(
                    round(
                        float(
                            raw_params.get(
                                "train.num_minibatches", self.base_num_minibatches
                            )
                        )
                    )
                ),
            )
            mb_size = _snap_minibatch_size(
                max(horizon, batch_size // req_nmb), total_agents, horizon
            )

        if "train.total_timesteps" in raw_params:
            target_steps = int(round(float(raw_params["train.total_timesteps"])))
            num_epochs = max(2, int(round(target_steps / batch_size)))
        elif "train.num_epochs" in raw_params:
            num_epochs = max(2, int(round(float(raw_params["train.num_epochs"]))))
        else:
            num_epochs = max(2, self.base_num_epochs)

        hidden_size = int(
            round(
                float(
                    raw_params.get(
                        "policy.hidden_features",
                        raw_params.get("policy.hidden_size", self.base_hidden_size),
                    )
                )
            )
        )
        num_layers = int(
            round(float(raw_params.get("policy.num_layers", self.base_num_layers)))
        )
        lr = float(raw_params.get("train.learning_rate", 1e-2))
        gamma = float(raw_params.get("train.gamma", 0.99))
        gae_lambda = float(raw_params.get("train.gae_lambda", 0.95))
        clip_eps = float(raw_params.get("train.clip_coef", 0.2))
        vf_coef = float(raw_params.get("train.vf_coef", 2.0))
        ent_coef = float(raw_params.get("train.ent_coef", 0.001))
        max_grad_norm = float(raw_params.get("train.max_grad_norm", 1.5))
        replay_ratio = float(raw_params.get("train.replay_ratio", 1.0))
        num_minibatches = max(1, int(round(replay_ratio * batch_size / mb_size)))

        key = jax.random.key(self.seed + run_id)
        model_key, tr_key = jax.random.split(key)

        if self.model_factory is not None:
            model = self.model_factory(env, raw_params, nnx.Rngs(model_key))
        else:
            model_kwargs: dict[str, Any] = {
                "key": nnx.Rngs(model_key),
                "observation_space_size": env.observation_space_size,
                "action_space_size": env.action_space_size,
                "action_space": self.action_space,
            }
            if "policy.actor_sigma_head" in raw_params:
                model_kwargs["actor_sigma_head"] = bool(
                    round(float(raw_params["policy.actor_sigma_head"]))
                )
            if self.model_type == "MinGRUActorCritic":
                model_kwargs["hidden_features"] = hidden_size
                model_kwargs["gru_features"] = int(
                    round(float(raw_params.get("policy.gru_features", hidden_size)))
                )
                model_kwargs["num_layers"] = num_layers
            elif self.model_type == "LSTMActorCritic":
                model_kwargs["hidden_features"] = hidden_size
                model_kwargs["lstm_features"] = int(
                    round(float(raw_params.get("policy.lstm_features", hidden_size)))
                )
            elif self.model_type == "SharedActorCritic":
                model_kwargs["architecture"] = [hidden_size] * num_layers
                for k in ("in_scale", "actor_scale", "critic_scale"):
                    if f"policy.{k}" in raw_params:
                        model_kwargs[k] = float(raw_params[f"policy.{k}"])
            elif self.model_type == "ActorCritic":
                arch = [hidden_size] * num_layers
                model_kwargs["actor_architecture"] = arch
                model_kwargs["critic_architecture"] = arch
                for k in ("in_scale", "actor_scale", "critic_scale"):
                    if f"policy.{k}" in raw_params:
                        model_kwargs[k] = float(raw_params[f"policy.{k}"])
            model = Model.create(self.model_type, **model_kwargs)

        extra_ppo: dict[str, Any] = dict(self.ppo_kwargs)
        for p_key, arg_name in (
            ("train.drip_decay", "drip_decay"),
            ("train.caps_temporal_coeff", "caps_temporal_coeff"),
            ("train.caps_spatial_coeff", "caps_spatial_coeff"),
            ("train.advantage_rho_clip", "advantage_rho_clip"),
            ("train.advantage_c_clip", "advantage_c_clip"),
        ):
            if p_key in raw_params:
                extra_ppo[arg_name] = float(raw_params[p_key])
        if "train.vtrace" in raw_params:
            extra_ppo["vtrace"] = bool(round(float(raw_params["train.vtrace"])))
        if "train.accumulate_n_gradients" in raw_params:
            extra_ppo["accumulate_n_gradients"] = max(
                1, int(round(float(raw_params["train.accumulate_n_gradients"])))
            )
        if "train.anneal_learning_rate" in raw_params:
            extra_ppo["anneal_learning_rate"] = bool(
                round(float(raw_params["train.anneal_learning_rate"]))
            )

        tr = Trainer.create(
            "PPO",
            env=env,
            model=model,
            key=tr_key,
            num_envs=num_envs,
            num_steps_epoch=horizon,
            skip_frames=skip_frames,
            num_epochs=num_epochs,
            num_minibatches=num_minibatches,
            minibatch_size=mb_size,
            learning_rate=lr,
            advantage_gamma=gamma,
            advantage_lambda=gae_lambda,
            ppo_clip_eps=clip_eps,
            ppo_value_coeff=vf_coef,
            ppo_entropy_coeff=ent_coef,
            max_grad_norm=max_grad_norm,
            clip_actions=self.clip_actions,
            clip_range=self.clip_range,
            **extra_ppo,
        )

        active_eval_fn: Callable[[Any], float] | None = self.eval_fn
        if active_eval_fn is None and self.eval_metric_fn is not None:
            cached = self._eval_fn_cache.get(skip_frames)
            if cached is None:
                eval_target = self._base_eval_env if self._base_eval_env is not None else env
                cached = make_ppo_eval_fn(
                    eval_target,
                    metric_fn=self.eval_metric_fn,
                    eval_n=self.eval_n,
                    skip_frames=skip_frames,
                    max_steps=self.eval_max_steps,
                    seed=self.eval_seed,
                )
                self._eval_fn_cache[skip_frames] = cached
            active_eval_fn = cached

        hist_scores: list[float] = []
        hist_costs: list[float] = []
        hist_steps: list[float] = []
        stopped_early = False
        t_start_all = time.perf_counter()
        t_steady_start: float | None = None
        eval_overhead_s = 0.0

        ep = 0
        last_score = float("nan")
        last_report_cost = 0.0
        report_every = max(
            1, min(self.report_every, max(1, num_epochs // max(self.downsample, 1)))
        )
        time_report_interval = (
            self.max_trial_seconds / max(self.downsample, 1)
            if self.max_trial_seconds is not None
            else None
        )
        while ep < num_epochs:
            tr, metrics = PPOTrainer._epoch_no_trajectory(tr)
            for val in metrics.values():
                val.block_until_ready()
            ep += 1
            if ep == 1:
                t_steady_start = time.perf_counter()
                if num_epochs > 1:
                    continue

            if t_steady_start is not None and ep > 1:
                steady_elapsed = max(
                    time.perf_counter() - t_steady_start - eval_overhead_s, 1e-4
                )
                cost_now = steady_elapsed * (ep / (ep - 1))
            else:
                cost_now = max(
                    time.perf_counter() - t_start_all - eval_overhead_s, 1e-4
                )

            hit_time_cap = (
                self.max_trial_seconds is not None
                and cost_now >= self.max_trial_seconds
            )
            hit_time_report = (
                time_report_interval is not None
                and cost_now - last_report_cost >= time_report_interval
            )
            if (
                ep % report_every == 0
                or ep == num_epochs
                or hit_time_cap
                or hit_time_report
            ):
                last_report_cost = cost_now
                if self.eval_on_report and active_eval_fn is not None:
                    t_ev0 = time.perf_counter()
                    sc = float(active_eval_fn(tr))
                    eval_overhead_s += time.perf_counter() - t_ev0
                else:
                    metrics_host = {
                        k: float(jnp.asarray(v).item()) for k, v in metrics.items()
                    }
                    sc = float("nan")
                    for candidate in (
                        self.metric,
                        self.metric.removeprefix("env/"),
                        "score",
                        "returns",
                    ):
                        if candidate in metrics_host and math.isfinite(
                            metrics_host[candidate]
                        ):
                            sc = metrics_host[candidate]
                            break
                if math.isfinite(sc):
                    last_score = sc
                    hist_scores.append(sc)
                    hist_costs.append(cost_now)
                    hist_steps.append(float(ep * batch_size))
                    if (
                        self.early_stop
                        and not hit_time_cap
                        and ep < num_epochs
                        and should_stop_fn(sc, cost_now)
                    ):
                        stopped_early = True
                        break
            if hit_time_cap:
                break

        if t_steady_start is not None and ep > 1:
            total_cost = max(
                time.perf_counter() - t_steady_start - eval_overhead_s, 1e-4
            ) * (ep / (ep - 1))
        else:
            total_cost = max(time.perf_counter() - t_start_all - eval_overhead_s, 1e-4)
        if hist_costs:
            total_cost = max(total_cost, hist_costs[-1])
        total_steps = ep * batch_size
        if active_eval_fn is not None:
            if self.eval_on_report and math.isfinite(last_score):
                final_score = last_score
            else:
                final_score = float(active_eval_fn(tr))
        else:
            final_score = last_score
        if not math.isfinite(final_score):
            raise RuntimeError("PPO trial produced non-finite objective score.")

        cfg_log = {
            "vec.total_agents": total_agents,
            "policy.hidden_size": hidden_size,
            "policy.num_layers": num_layers,
            "train.total_timesteps": total_steps,
            "train.horizon": horizon,
            "train.minibatch_size": mb_size,
            "train.learning_rate": lr,
            "train.gamma": gamma,
            "train.gae_lambda": gae_lambda,
            "train.clip_coef": clip_eps,
            "train.vf_coef": vf_coef,
            "train.ent_coef": ent_coef,
            "train.max_grad_norm": max_grad_norm,
            "train.replay_ratio": replay_ratio,
        }
        return (
            final_score,
            total_cost,
            total_steps,
            stopped_early,
            hist_scores,
            hist_costs,
            hist_steps,
            cfg_log,
        )

    def run(
        self,
        *,
        verbose: bool = True,
        on_trial: Callable[[SweepTrial, list[ParetoPoint]], None] | None = None,
    ) -> SweepResult:
        """Execute the full Protein hyperparameter sweep and return the Pareto frontier."""
        t_sweep0 = time.perf_counter()
        trials: list[SweepTrial] = []

        opt = ProteinOptimizer(
            self.params,
            cost_param=self.cost_param,
            goal=self.goal,
            num_random_samples=self.num_random_samples,
            use_success_prob=self.downsample == 1,
            prune_pareto=self.prune_pareto,
            max_suggestion_cost=self.max_suggestion_cost,
            early_stop_quantile=self.early_stop_quantile,
            use_gpu=self.use_gpu,
            **self.protein_kwargs,
        )

        try:
            for run_id in range(self.max_runs):
                if run_id == 0:
                    norm_params, raw_params = opt.default_point()
                    sugg_info: SuggestionInfo | None = None
                else:
                    norm_params, raw_params, sugg_info = opt.suggest()

                try:
                    if self.evaluate_fn is not None:
                        t0 = time.perf_counter()
                        out = self.evaluate_fn(raw_params, opt.should_stop)
                        final_score = float(out["score"])
                        total_cost = float(
                            out.get("cost", max(time.perf_counter() - t0, 1e-4))
                        )
                        total_steps = int(
                            out.get(
                                "steps",
                                int(
                                    round(
                                        float(raw_params.get("train.total_timesteps", 1))
                                    )
                                ),
                            )
                        )
                        stopped_early = bool(out.get("early_stopped", False))
                        hist_scores = [
                            float(x) for x in out.get("curve_scores", [final_score])
                        ]
                        hist_costs = [
                            float(x) for x in out.get("curve_costs", [total_cost])
                        ]
                        hist_steps = [
                            float(x) for x in out.get("curve_steps", [total_steps])
                        ]
                        cfg_log = dict(raw_params)
                    elif self.backend == "puffer":
                        (
                            final_score,
                            total_cost,
                            total_steps,
                            stopped_early,
                            hist_scores,
                            hist_costs,
                            hist_steps,
                            cfg_log,
                        ) = self._run_puffer_trial(run_id, raw_params, opt.should_stop)
                    else:
                        (
                            final_score,
                            total_cost,
                            total_steps,
                            stopped_early,
                            hist_scores,
                            hist_costs,
                            hist_steps,
                            cfg_log,
                        ) = self._run_ppo_trial(run_id, raw_params, opt.should_stop)

                    if len(hist_scores) >= 2 and self.downsample > 1:
                        ds_scores = _bin_mean(hist_scores, self.downsample)
                        ds_costs = _bin_mean(hist_costs, self.downsample)
                        ds_steps = _bin_mean(hist_steps, self.downsample)
                        ds_scores[-1] = final_score
                        ds_costs[-1] = total_cost
                        ds_steps[-1] = float(total_steps)
                    else:
                        ds_scores = [final_score]
                        ds_costs = [total_cost]
                        ds_steps = [float(total_steps)]

                    if opt.cost_idx >= 0 and len(ds_scores) > 1:
                        for sc_i, co_i, st_i in zip(
                            ds_scores, ds_costs, ds_steps, strict=True
                        ):
                            pt = list(norm_params)
                            pt[opt.cost_idx] = opt.normalize_value(opt.cost_idx, st_i)
                            opt.observe(pt, sc_i, co_i, is_failure=False)
                        final_pt = list(norm_params)
                        final_pt[opt.cost_idx] = max(
                            -0.999,
                            opt.normalize_value(opt.cost_idx, float(total_steps)),
                        )
                        opt.observe(final_pt, final_score, total_cost, is_failure=False)
                    else:
                        opt.observe(
                            norm_params, final_score, total_cost, is_failure=False
                        )

                    trial = SweepTrial(
                        run_id=run_id,
                        params=dict(raw_params),
                        norm_params=norm_params,
                        score=final_score,
                        cost=total_cost,
                        steps=total_steps,
                        early_stopped=stopped_early,
                        suggestion=sugg_info,
                        curve_scores=ds_scores,
                        curve_costs=ds_costs,
                        curve_steps=ds_steps,
                    )

                    if self.log_dir is not None:
                        _write_puffer_ini_log(
                            self.log_dir,
                            self.env_name,
                            f"sweep_{int(t_sweep0 * 1000)}_{run_id:04d}",
                            cfg_log,
                            {
                                "uptime": ds_costs,
                                "agent_steps": ds_steps,
                                "env/score": ds_scores,
                                self.metric: ds_scores,
                            },
                        )

                except Exception as exc:
                    opt.observe(
                        norm_params,
                        float("nan"),
                        self.max_suggestion_cost,
                        is_failure=True,
                    )
                    trial = SweepTrial(
                        run_id=run_id,
                        params=dict(raw_params),
                        norm_params=norm_params,
                        score=float("nan"),
                        cost=0.0,
                        steps=0,
                        is_failure=True,
                        error=f"{type(exc).__name__}: {exc}",
                        suggestion=sugg_info,
                    )

                trials.append(trial)
                frontier = opt.pareto_frontier(pruned=self.prune_pareto)

                if verbose:
                    tag = (
                        "FAIL"
                        if trial.is_failure
                        else ("STOP" if trial.early_stopped else "OK")
                    )
                    src = (
                        "default"
                        if run_id == 0
                        else ("sobol" if sugg_info and sugg_info.is_random else "gp")
                    )
                    print(
                        f"sweep run={run_id:03d}/{self.max_runs} [{tag}/{src}] "
                        f"score={trial.score:.4f} cost={trial.cost:.2f}s "
                        f"steps={trial.steps} pareto={len(frontier)}"
                        + (f" err={trial.error}" if trial.error else ""),
                        flush=True,
                    )

                if on_trial is not None:
                    on_trial(trial, frontier)

            result = SweepResult(
                goal=self.goal,
                metric=self.metric,
                trials=trials,
                pareto_frontier=opt.pareto_frontier(pruned=False),
                pruned_pareto_frontier=opt.pareto_frontier(pruned=True),
                elapsed_s=time.perf_counter() - t_sweep0,
            )
            if self.log_dir is not None:
                result.save_json(self.log_dir / self.env_name / "sweep_results.json")
            return result
        finally:
            opt.close()


__all__ = [
    "ParetoPoint",
    "ProteinOptimizer",
    "ProteinSweep",
    "SpaceDistribution",
    "SuggestionInfo",
    "SweepParam",
    "SweepResult",
    "SweepTrial",
    "default_sweep_params",
    "make_ppo_eval_fn",
]
