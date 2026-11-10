# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Static checks of public result contracts; checked by mypy, not executed."""

from typing import assert_type
from collections.abc import Mapping

import jax

from jaxdem import State, System
from jaxdem.colliders.naive import NaiveSimulator
from jaxdem.rl.trainers import PPOTrainer, TrajectoryData
from jaxdem.utils.jamming import (
    JamResult,
    JammingInfo,
    bisection_jam,
    pe_band_jam,
    pressure_bisection_jam,
    rollback_pe_band_jam,
)


def jamming_results(state: State, system: System, return_info: bool) -> None:
    assert_type(bisection_jam(state, system), JamResult)
    assert_type(bisection_jam(state, system, return_info=False), JamResult)
    assert_type(
        bisection_jam(state, system, return_info=True),
        tuple[JamResult, JammingInfo],
    )
    assert_type(
        bisection_jam(state, system, return_info=return_info),
        JamResult | tuple[JamResult, JammingInfo],
    )
    assert_type(pressure_bisection_jam(state, system), JamResult)
    assert_type(
        pressure_bisection_jam(state, system, return_info=True),
        tuple[JamResult, JammingInfo],
    )
    assert_type(pe_band_jam(state, system), JamResult)
    assert_type(
        pe_band_jam(state, system, return_info=True),
        tuple[JamResult, JammingInfo],
    )
    assert_type(rollback_pe_band_jam(state, system), JamResult)
    assert_type(
        rollback_pe_band_jam(state, system, return_info=True),
        tuple[JamResult, JammingInfo],
    )


class CustomNaive(NaiveSimulator):
    """An application may extend the collider without losing its return type."""


def refreshed_subtype(collider: CustomNaive, state: State) -> None:
    assert_type(collider.refresh(state), CustomNaive)


def training_metrics(
    trainer: PPOTrainer, trajectory: TrajectoryData
) -> Mapping[str, jax.Array]:
    return {
        "entropy_coeff": trainer.ppo_entropy_coeff,
        "rollout_reward": trajectory.reward.mean(),
    }


def report_training(
    trainer: PPOTrainer, epoch: int, metrics: Mapping[str, float]
) -> PPOTrainer | None:
    return trainer if epoch >= 0 and metrics else None


def training_reporting(trainer: PPOTrainer) -> None:
    assert_type(
        PPOTrainer.train(
            trainer,
            metrics_fn=training_metrics,
            progress_metrics={"entropy_coeff": ".3f", "score": ".3f"},
            on_report=report_training,
        ),
        PPOTrainer,
    )
