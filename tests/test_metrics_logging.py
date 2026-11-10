# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Logged PPO values and hyperparameters survive the TensorBoard file format."""

import jax

import json
from dataclasses import replace

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from tests.test_ppo_math import BanditModel, TimeLimitCounterEnv


def test_training_writes_tensorboard_scalar_and_text(tmp_path) -> None:
    trainer = PPOTrainer.Create(
        env=TimeLimitCounterEnv.Create(),
        model=BanditModel(),
        key=jax.random.PRNGKey(0),
        num_epochs=2,
        num_envs=2,
        num_minibatches=1,
        num_steps_epoch=2,
    )
    trainer.train(trainer, verbose=False, log=True, directory=tmp_path)

    log_directory = next(tmp_path.iterdir())
    accumulator = EventAccumulator(str(log_directory))
    accumulator.Reload()
    assert [event.step for event in accumulator.Scalars("score")] == [0, 1]
    np.testing.assert_allclose(
        [event.value for event in accumulator.Scalars("score")], [1.0, 1.0]
    )
    assert accumulator.Scalars("steps_per_sec")[0].step == 1
    text = accumulator.Tensors("hparams/json/text_summary")
    assert len(text) == 1
    assert text[0].step == 0
    metadata = json.loads(text[0].tensor_proto.string_val[0])
    for key, expected in {
        "num_epochs": 2,
        "num_envs": 2,
        "num_steps_epoch": 2,
        "minibatch_size": 4,
    }.items():
        assert metadata[key] == expected


def test_custom_metrics_use_vectorized_environment_and_rollout(tmp_path) -> None:
    trainer = PPOTrainer.Create(
        env=TimeLimitCounterEnv.Create(),
        model=BanditModel(),
        key=jax.random.key(4),
        num_epochs=2,
        num_envs=2,
        num_minibatches=1,
        num_steps_epoch=2,
    )
    reports = []

    def metrics(current, trajectory):
        return {
            "accuracy": jax.vmap(lambda env: (env.state.rad <= 1).mean())(
                current.env
            ).mean(),
            "rollout_reward": trajectory.reward.mean(),
        }

    def callback(current, epoch, values):
        reports.append((epoch, values))
        return replace(current, ppo_entropy_coeff=current.ppo_entropy_coeff * 0.5)

    result = trainer.train(
        trainer,
        verbose=False,
        directory=tmp_path,
        metrics_fn=metrics,
        on_report=callback,
    )
    assert [epoch for epoch, _ in reports] == [0, 1]
    np.testing.assert_allclose(
        result.ppo_entropy_coeff, trainer.ppo_entropy_coeff * 0.25
    )
    accumulator = EventAccumulator(str(next(tmp_path.iterdir())))
    accumulator.Reload()
    assert [event.value for event in accumulator.Scalars("accuracy")] == [1.0, 1.0]
    assert [event.value for event in accumulator.Scalars("rollout_reward")] == [
        1.0,
        1.0,
    ]
