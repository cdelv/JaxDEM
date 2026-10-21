"""Matched N=64 follow-up with every initial distance doubled.

The original seeded 20-unit objective box and 10-unit padding ring are sampled
first, then their coordinates and the reflective domain are scaled by two.
This preserves the exact layout and jitter, unlike resampling a larger box.
Particle radii, sensing range, reward lengths, forces, drag and dt stay fixed.

Run this module in the frozen original experiment runtime to retain the action
mapping used in the earlier comparison::

    python -m benchmarks.swarm_mean_shift.spacing_followup \
        --seeds 0 1 2 3 4 --eval-envs 64 --checkpoint-at-120 \
        --output /tmp/swarm_double_spacing

Default episodes and training both have twice the original budget: 4000 policy
actions per episode and 240 training epochs. The planned learning-rate schedule
also doubles to 4320 epochs, preserving its progress per completed episode.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from math import isfinite
from pathlib import Path
import pickle
import sys
from typing import Any, Iterator

import jax
import jax.numpy as jnp

from jaxdem.rl.environments import SwarmNavigator

from benchmarks.swarm_mean_shift import variants


FOLLOWUP_NAMES = ("spacing_baseline", "spacing_rational_delta")
CLI_DEFAULTS = {
    "--variants": FOLLOWUP_NAMES,
    "--epochs": ("240",),
    "--schedule-epochs": ("4320",),
    "--max-physics-steps": ("200000",),
    "--eval-at": ("40", "120", "240"),
}


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SpacingFollowupNavigator(SwarmNavigator):
    """Scale reset geometry once and retain the baseline observation/dynamics."""

    reward_mode: str = jax.tree.static(default="baseline")
    spacing_scale: float = jax.tree.static(default=2.0)
    base_box_size: float = jax.tree.static(default=20.0)
    base_box_padding: float = jax.tree.static(default=10.0)

    @classmethod
    def Create(
        cls,
        *,
        reward_mode: str = "baseline",
        spacing_scale: float = 2.0,
        box_size: float = 20.0,
        box_padding: float = 10.0,
        **kwargs: Any,
    ) -> SpacingFollowupNavigator:
        if reward_mode not in {"baseline", "rational_delta"}:
            raise ValueError(f"Unknown reward_mode: {reward_mode!r}")
        if not isfinite(spacing_scale) or spacing_scale <= 0:
            raise ValueError("spacing_scale must be finite and positive")
        env = SwarmNavigator.Create.__func__(
            cls, box_size=box_size, box_padding=box_padding, **kwargs
        )
        env.reward_mode = reward_mode
        env.spacing_scale = float(spacing_scale)
        env.base_box_size = float(box_size)
        env.base_box_padding = float(box_padding)
        return env

    @staticmethod
    @jax.jit
    def reset(env: SpacingFollowupNavigator, key: jax.Array):
        # Read static original dimensions, never already-scaled env_params.
        # Thus identical reset keys always reproduce identical scaled geometry.
        env.env_params["box_size"] = jnp.asarray(env.base_box_size, dtype=float)
        env.env_params["box_padding"] = jnp.asarray(env.base_box_padding, dtype=float)
        env = SwarmNavigator.reset(env, key)
        scale = env.spacing_scale
        env.state.pos_c = env.state.pos_c * scale
        env.env_params["objective"] = env.env_params["objective"] * scale
        env.env_params["box_size"] = env.env_params["box_size"] * scale
        env.env_params["box_padding"] = env.env_params["box_padding"] * scale
        env.system.domain.box_size = env.system.domain.box_size * scale
        env.system.domain.inv_box_size = 1 / env.system.domain.box_size
        env.system.domain.anchor = env.system.domain.anchor * scale
        # Base reset checkpoint was evaluated before scaling, so replace it.
        return SpacingFollowupNavigator.checkpoint(
            env, jnp.zeros((env.max_num_agents, env.action_space_size))
        )

    @staticmethod
    @jax.jit(inline=True)
    def _score(env: SpacingFollowupNavigator) -> jax.Array:
        if env.reward_mode == "baseline":
            return SwarmNavigator._potential(env)
        if env.num_objectives == 0:
            return jnp.zeros_like(env.state.rad)
        peers, objectives, ids = SwarmNavigator._sense(env)
        _, scores, _ = variants.scan_goal_scores(objectives, peers, ids, env)
        return scores.max(axis=-1)

    @staticmethod
    @jax.jit(inline=True)
    def checkpoint(env: SpacingFollowupNavigator, action: jax.Array):
        env.env_params["prev_potential"] = SpacingFollowupNavigator._score(env)
        return env

    @staticmethod
    @jax.jit(inline=True)
    def reward(env: SpacingFollowupNavigator) -> jax.Array:
        return SpacingFollowupNavigator._score(env) - env.env_params["prev_potential"]


def make_followup_environment(
    variant: str,
    *,
    N: int = 64,
    num_objectives: int = 64,
    max_steps: int = 200000,
    seed: int | None = None,
    **kwargs: Any,
) -> SpacingFollowupNavigator:
    if variant not in FOLLOWUP_NAMES:
        raise ValueError(f"Unknown follow-up {variant!r}; expected {FOLLOWUP_NAMES}")
    env = SpacingFollowupNavigator.Create(
        reward_mode="baseline" if variant == "spacing_baseline" else "rational_delta",
        N=N, num_objectives=num_objectives, max_steps=max_steps, **kwargs,
    )
    if seed is not None:
        env = env.reset(env, jax.random.key(seed))
    return env


# Keep the conventional factory name available to checkpoint evaluation tools.
make_environment = make_followup_environment


@contextmanager
def matched_followup_runner(*, checkpoint_at_120: bool = False) -> Iterator[Any]:
    """Adapt the existing runner only within this process and hash dependencies."""
    from benchmarks.swarm_mean_shift import run

    original_factory = run.make_environment
    original_names = run.VARIANT_NAMES
    original_hashes = run.source_hashes
    original_evaluate = run.evaluate

    def hashes_with_followup(source_experiment: Path) -> dict[str, str | None]:
        hashes = original_hashes(source_experiment)
        for source in (
            Path(__file__).resolve(),
            Path(variants.__file__).resolve(),
            Path(__file__).with_name("spacing_evaluate.py").resolve(),
        ):
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        return hashes

    def evaluate_with_checkpoint(tr: Any, base_env: Any, args: Any, directory: Path, epoch: int):
        from benchmarks.swarm_mean_shift.spacing_evaluate import add_spacing_metrics

        result = original_evaluate(tr, base_env, args, directory, epoch)
        result = add_spacing_metrics(result, directory, epoch)
        if checkpoint_at_120 and epoch == 120:
            for name, value in (
                ("graphstate_epoch_0120.pkl", tr.graphstate),
                ("trainer_leaves_epoch_0120.pkl", jax.tree.leaves(tr)),
            ):
                with (directory / name).open("wb") as stream:
                    pickle.dump(jax.device_get(value), stream)
        return result

    run.make_environment = make_followup_environment
    run.VARIANT_NAMES = FOLLOWUP_NAMES
    run.source_hashes = hashes_with_followup
    run.evaluate = evaluate_with_checkpoint
    try:
        yield run
    finally:
        run.make_environment = original_factory
        run.VARIANT_NAMES = original_names
        run.source_hashes = original_hashes
        run.evaluate = original_evaluate


def main() -> None:
    original_argv = sys.argv
    checkpoint_at_120 = "--checkpoint-at-120" in original_argv[1:]
    arguments = [arg for arg in original_argv[1:] if arg != "--checkpoint-at-120"]
    defaults: list[str] = []
    for flag, values in CLI_DEFAULTS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            defaults.extend((flag, *values))
    sys.argv = [original_argv[0], *defaults, *arguments]
    try:
        with matched_followup_runner(checkpoint_at_120=checkpoint_at_120) as runner:
            runner.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
