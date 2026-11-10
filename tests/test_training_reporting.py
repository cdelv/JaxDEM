# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check report scheduling, custom metrics, state updates, and cleanup."""

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from jaxdem.rl.trainers import ppo_trainer


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


def test_default_display_and_empty_selection(trainer, tmp_path, monkeypatch, capsys):
    bars = []

    def make_progress(*args):
        bar = _Progress(*args)
        bars.append(bar)
        return bar

    monkeypatch.setattr(ppo_trainer, "trange", make_progress)
    trainer.train(trainer, log=False)
    assert all(set(values) == {"steps/s", "avg_score"} for values in bars[0].reports)
    assert "final avg_score: 6.00" in capsys.readouterr().out

    trainer.train(trainer, directory=tmp_path, progress_metrics={})
    assert all(values == {} for values in bars[1].reports)
    assert capsys.readouterr().out == ""
    accumulator = EventAccumulator(str(next(tmp_path.iterdir())))
    accumulator.Reload()
    assert accumulator.Scalars("score")[-1].value == 6.0


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        ({"metrics_fnn": None}, TypeError, "unexpected keyword"),
        ({"metrics_fn": 1}, TypeError, "metrics_fn must be callable"),
        ({"on_report": 1}, TypeError, "on_report must be callable"),
        ({"progress_metrics": ["score"]}, TypeError, "must map metric names"),
        ({"progress_metrics": {"": ".2f"}}, ValueError, "nonempty strings"),
        ({"progress_metrics": {"score": 2}}, TypeError, "must be a string"),
        ({"progress_metrics": {"score": "bad"}}, ValueError, "Invalid format"),
    ],
)
def test_invalid_options_fail_before_training(
    trainer, tmp_path, monkeypatch, kwargs, error, match
):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid options must fail before training")

    monkeypatch.setattr(_ReportingTrainer, "epoch", staticmethod(unexpected))
    with pytest.raises(error, match=match):
        trainer.train(trainer, directory=tmp_path / "unused", **kwargs)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "result,error,match",
    [
        ([1.0], TypeError, "mapping"),
        ({"accuracy": jnp.ones(2)}, ValueError, "accuracy.*shape"),
        ({"": 1.0}, ValueError, "nonempty strings"),
        ({1: 1.0}, ValueError, "nonempty strings"),
        ({"accuracy": 1j}, TypeError, "accuracy.*real numeric scalar"),
        ({"accuracy": "text"}, TypeError, "accuracy.*real numeric scalar"),
        ({"score": 1.0}, ValueError, "reserved metrics.*score"),
        ({"elapsed": 1.0}, ValueError, "reserved metrics.*elapsed"),
        ({"steps_per_sec": 1.0}, ValueError, "reserved metrics.*steps_per_sec"),
    ],
)
def test_invalid_metrics_close_writer(
    trainer, tmp_path, monkeypatch, result, error, match
):
    closed = []
    original_close = ppo_trainer.SummaryWriter.close

    def close(writer):
        closed.append(True)
        original_close(writer)

    monkeypatch.setattr(ppo_trainer.SummaryWriter, "close", close)
    with pytest.raises(error, match=match):
        trainer.train(
            trainer,
            verbose=False,
            directory=tmp_path,
            metrics_fn=lambda tr, td: result,
        )
    assert closed


def test_unknown_progress_metric_lists_available_names(trainer, tmp_path):
    with pytest.raises(
        ValueError,
        match="Unknown progress metrics.*missing.*Available metrics.*accuracy.*score",
    ):
        trainer.train(
            trainer,
            verbose=False,
            directory=tmp_path,
            metrics_fn=_custom_metrics,
            progress_metrics={"missing": ".2f"},
        )


@pytest.mark.parametrize("failure", ["exception", "return", "shape", "structure"])
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
        if failure == "return":
            return 42
        if failure == "shape":
            return replace(current, increment=jnp.ones(2))
        return replace(current, stop_at_epoch=99)

    monkeypatch.setattr(ppo_trainer.SummaryWriter, "close", close)
    error = (
        RuntimeError
        if failure == "exception"
        else TypeError if failure == "return" else ValueError
    )
    with pytest.raises(error):
        trainer.train(trainer, verbose=False, directory=tmp_path, on_report=callback)
    assert closed


def test_custom_metric_names_stay_fixed_across_compilations(trainer, tmp_path):
    def metrics(current, trajectory):
        name = (
            "floating"
            if jnp.issubdtype(current.env.value.dtype, jnp.floating)
            else "integer"
        )
        return {name: current.env.value}

    def callback(current, epoch, values):
        if epoch == 0:
            return replace(
                current,
                env=current.env._replace(value=current.env.value.astype(jnp.int32)),
                increment=current.increment.astype(jnp.int32),
            )

    with pytest.raises(ValueError, match="same metric names"):
        trainer.train(
            trainer,
            verbose=False,
            directory=tmp_path,
            metrics_fn=metrics,
            on_report=callback,
        )
