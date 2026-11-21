# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check report scheduling, custom metrics, TensorBoard logging, and cleanup."""

import json
from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from jaxdem.rl.trainers import ppo_trainer
from jaxdem.rl.trainers.ppo_trainer import PPOTrainer
from tests.test_ppo_math import BanditModel, TimeLimitCounterEnv


class _Collider(NamedTuple):
    overflow: jax.Array


class _System(NamedTuple):
    collider: _Collider


class _Environment(NamedTuple):
    value: jax.Array
    system: _System

    @property
    def max_num_agents(self):
        return 1

    @property
    def num_envs(self):
        return 1


@jax.tree_util.register_dataclass
@dataclass
class _ReportingTrainer:
    env: _Environment
    increment: jax.Array
    stop_at_epoch: int = jax.tree.static(default=6)
    num_steps_epoch: int = jax.tree.static(default=1)
    skip_frames: int = jax.tree.static(default=0)

    train = staticmethod(ppo_trainer.PPOTrainer.train)

    @staticmethod
    @jax.jit
    def epoch(tr, epoch):
        tr = replace(tr, env=tr.env._replace(value=tr.env.value + tr.increment))
        return tr, epoch, {"score": tr.env.value}


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(
        ppo_trainer, "_log_hparams_fallback", lambda *args, **kwargs: None
    )
    return _ReportingTrainer(
        env=_Environment(
            jnp.asarray(0.0, dtype=jnp.float32),
            _System(_Collider(jnp.asarray(False))),
        ),
        increment=jnp.asarray(1.0, dtype=jnp.float32),
    )


class _Progress:
    def __init__(self, *args):
        self.epochs = range(*args)
        self.reports = []
        self.closed = False

    def __iter__(self):
        return iter(self.epochs)

    def set_postfix(self, values):
        self.reports.append(dict(values))

    def close(self):
        self.closed = True


def _custom_metrics(tr, trajectory):
    return {"accuracy": tr.env.value / 20, "rollout_epoch": trajectory}


def test_metrics_reach_tensorboard_progress_and_callback(
    trainer, tmp_path, monkeypatch, capsys
):
    bars = []
    reports = []
    transfers = []
    metric_traces = []
    device_get = jax.device_get

    def make_progress(*args):
        bar = _Progress(*args)
        bars.append(bar)
        return bar

    def fetch(values):
        transfers.append(set(values))
        return device_get(values)

    def metrics(current, trajectory):
        metric_traces.append(True)
        return _custom_metrics(current, trajectory)

    def on_report(current, epoch, values):
        assert all(isinstance(value, float) for value in values.values())
        reports.append((epoch, dict(values)))
        values["score"] = -1000.0
        if epoch == 0:
            return replace(current, increment=jnp.asarray(2.0, dtype=jnp.float32))
        return None

    monkeypatch.setattr(ppo_trainer, "trange", make_progress)
    monkeypatch.setattr(jax, "device_get", fetch)
    result = trainer.train(
        trainer,
        directory=tmp_path,
        metrics_fn=metrics,
        progress_metrics={"accuracy": ".1%", "score": ".2f"},
        on_report=on_report,
    )

    assert float(result.env.value) == 11.0
    assert [epoch for epoch, _ in reports] == [0, 2, 4, 5]
    assert len(transfers) == 4
    assert len(metric_traces) == 1
    assert all(keys == {"score", "accuracy", "rollout_epoch"} for keys in transfers)
    assert bars[0].closed
    assert bars[0].reports == [
        {"accuracy": "25.0%", "score": "5.00"},
        {"accuracy": "45.0%", "score": "9.00"},
        {"accuracy": "55.0%", "score": "11.00"},
    ]
    assert "final accuracy: 55.0%, score: 11.00" in capsys.readouterr().out

    monkeypatch.setattr(jax, "device_get", device_get)
    accumulator = EventAccumulator(str(next(tmp_path.iterdir())))
    accumulator.Reload()
    assert [event.step for event in accumulator.Scalars("accuracy")] == [0, 2, 4, 5]
    np.testing.assert_allclose(
        [event.value for event in accumulator.Scalars("accuracy")],
        [0.05, 0.25, 0.45, 0.55],
    )
    assert [event.value for event in accumulator.Scalars("score")] == [1, 5, 9, 11]


@pytest.mark.parametrize(
    "stop,start,expected",
    [(1, 0, [0]), (6, 1, [1, 2, 4, 5]), (6, 6, [])],
)
def test_report_boundaries_without_standard_outputs(
    trainer, tmp_path, stop, start, expected
):
    reports = []
    trainer = replace(trainer, stop_at_epoch=stop)

    def callback(current, epoch, values):
        reports.append((epoch, dict(values)))

    trainer.train(
        trainer,
        verbose=False,
        log=False,
        directory=tmp_path / "unused",
        start_epoch=start,
        metrics_fn=_custom_metrics,
        on_report=callback,
    )
    assert [epoch for epoch, _ in reports] == expected
    assert not (tmp_path / "unused").exists()
    if stop == 1:
        assert reports[0][1]["elapsed"] == 0.0
        assert reports[0][1]["steps_per_sec"] == 0.0


@pytest.mark.parametrize("failure", ["exception", "shape", "structure"])
def test_callback_failures_close_writer(trainer, tmp_path, monkeypatch, failure):
    closed = []
    original_close = ppo_trainer.SummaryWriter.close

    def close(writer):
        closed.append(True)
        original_close(writer)

    def callback(current, epoch, values):
        if failure == "exception":
            if epoch == 0:
                return None
            raise RuntimeError("callback failed")
        if failure == "shape":
            return replace(current, increment=jnp.ones(2))
        return replace(current, stop_at_epoch=99)

    monkeypatch.setattr(ppo_trainer.SummaryWriter, "close", close)
    error = RuntimeError if failure == "exception" else ValueError
    with pytest.raises(error):
        trainer.train(trainer, verbose=False, directory=tmp_path, on_report=callback)
    assert closed


def test_training_writes_tensorboard_scalar_and_text(tmp_path) -> None:
    tr = PPOTrainer.Create(
        env=TimeLimitCounterEnv.Create(),
        model=BanditModel(),
        key=jax.random.PRNGKey(0),
        num_epochs=2,
        num_envs=2,
        num_minibatches=1,
        num_steps_epoch=2,
    )
    tr.train(tr, verbose=False, log=True, directory=tmp_path)

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
    tr = PPOTrainer.Create(
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

    result = tr.train(
        tr,
        verbose=False,
        directory=tmp_path,
        metrics_fn=metrics,
        on_report=callback,
    )
    assert [epoch for epoch, _ in reports] == [0, 1]
    np.testing.assert_allclose(result.ppo_entropy_coeff, tr.ppo_entropy_coeff * 0.25)
    accumulator = EventAccumulator(str(next(tmp_path.iterdir())))
    accumulator.Reload()
    assert [event.value for event in accumulator.Scalars("accuracy")] == [1.0, 1.0]
    assert [event.value for event in accumulator.Scalars("rollout_reward")] == [1.0, 1.0]
