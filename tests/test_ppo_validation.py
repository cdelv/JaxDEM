"""Public PPO argument and resume-boundary validation."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from jaxdem.rl.trainers.ppo_trainer import PPOTrainer


class _UnusedEnvironment:
    max_num_agents = 1

    def agent_mask(self, _env):
        raise AssertionError("invalid numeric arguments must fail before env work")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_envs": 0}, "num_envs"),
        ({"num_envs": True}, "num_envs"),
        ({"num_envs": 1.5}, "num_envs"),
        ({"num_steps_epoch": 0}, "num_steps_epoch"),
        ({"num_minibatches": 0}, "num_minibatches"),
        ({"accumulate_n_gradients": 0}, "accumulate_n_gradients"),
        ({"accumulate_n_gradients": 1.5}, "accumulate_n_gradients"),
        ({"num_minibatches": 3, "accumulate_n_gradients": 2}, "divisible"),
        ({"minibatch_size": True}, "minibatch_size"),
        ({"minibatch_size": 1.5}, "minibatch_size"),
        ({"num_epochs": 0}, "num_epochs"),
        ({"total_timesteps": 1.5}, "total_timesteps"),
        ({"stop_at_epoch": 0}, "stop_at_epoch"),
        ({"skip_frames": -1}, "skip_frames"),
    ],
)
def test_create_rejects_invalid_numeric_arguments_before_environment_work(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        PPOTrainer.Create(_UnusedEnvironment(), object(), **kwargs)


class _FakeTrainer:
    def __init__(self, stop_at_epoch=3):
        self.stop_at_epoch = stop_at_epoch
        self.num_steps_epoch = 1
        self.skip_frames = 0
        self.calls = []
        collider = SimpleNamespace(overflow=jnp.asarray(False))
        system = SimpleNamespace(collider=collider)
        self.env = SimpleNamespace(max_num_agents=1, num_envs=1, system=system)

    def epoch(self, _trainer, epoch):
        self.calls.append(int(epoch))
        return self, None, {"score": jnp.asarray(0.0)}


def test_completed_resume_is_noop_before_writer_or_epoch(tmp_path):
    trainer = _FakeTrainer(stop_at_epoch=3)
    directory = tmp_path / "must-not-be-created"

    result = PPOTrainer.train(
        trainer,
        verbose=False,
        log=True,
        directory=directory,
        start_epoch=3,
    )

    assert result is trainer
    assert trainer.calls == []
    assert not directory.exists()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"save_every": 0}, "save_every"),
        ({"save_every": True}, "save_every"),
        ({"save_every": 1.5}, "save_every"),
        ({"start_epoch": -1}, "start_epoch"),
        ({"start_epoch": True}, "start_epoch"),
        ({"start_epoch": 1.5}, "start_epoch"),
        ({"start_epoch": 4}, "must not exceed"),
    ],
)
def test_train_rejects_invalid_bounds_before_writer_or_epoch(tmp_path, kwargs, message):
    trainer = _FakeTrainer(stop_at_epoch=3)
    directory = tmp_path / "must-not-be-created"

    with pytest.raises(ValueError, match=message):
        PPOTrainer.train(
            trainer,
            verbose=False,
            log=True,
            directory=directory,
            **kwargs,
        )

    assert trainer.calls == []
    assert not directory.exists()
