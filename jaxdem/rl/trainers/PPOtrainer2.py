from __future__ import annotations


from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import TYPE_CHECKING, Tuple, Optional, Sequence, cast

try:
    # Python 3.11+
    from typing import Self
except ImportError:
    from typing_extensions import Self

from dataclasses import dataclass, field
from functools import partial
import time
import datetime
from pathlib import Path

from flax import nnx
from flax.metrics import tensorboard
import optax
from tqdm.auto import trange

from . import Trainer, TrajectoryData
from ..envWrappers import clip_action_env, vectorise_env

if TYPE_CHECKING:
    from ..environments import Environment
    from ..models import Model


@Trainer.register("PPO2")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class PPOTrainer2(Trainer):
    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="PPOTrainer.one_epoch")
    def one_epoch(tr: "PPOTrainer", epoch):
        tr, td = tr.epoch(tr, epoch)
        model, optimizer, metrics, *rest = nnx.merge(tr.graphdef, tr.graphstate)
        data = metrics.compute()
        metrics.reset()
        tr.graphstate = nnx.state((model, optimizer, metrics, *rest))
        return tr, td, data

    @staticmethod
    @partial(jax.named_call, name="PPOTrainer.train")
    def train(
        tr: "PPOTrainer",
        verbose: bool = True,
        log: bool = True,
        directory: Path | str = "runs",
        save_every: int = 1,
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = {}

        log_folder = directory / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tensorboard.SummaryWriter(log_folder)

        tr, td, data = tr.one_epoch(tr, jnp.asarray(0, dtype=int))
        for k, v in data.items():
            writer.scalar(k, v, step=0)
        writer.flush()

        it = trange(1, tr.stop_at_epoch) if verbose else range(1, tr.stop_at_epoch)
        start_time = time.perf_counter()
        for epoch in it:
            tr, td, data = tr.one_epoch(tr, jnp.asarray(epoch, dtype=int))

            data["elapsed"] = time.perf_counter() - start_time
            steps_done = (epoch + 1) * tr.num_envs * tr.num_steps_epoch
            data["steps_per_sec"] = steps_done / data["elapsed"]

            if verbose:
                _sp = getattr(it, "set_postfix", None)
                if _sp is not None:
                    _sp(
                        {
                            "steps/s": f"{data['steps_per_sec']:.2e}",
                            "avg_score": f"{data['score']:.2f}",
                        }
                    )

            if log and epoch % save_every == 0:
                for k, v in data.items():
                    writer.scalar(k, v, step=int(epoch))
                writer.flush()

        print(
            f"steps/s: {data['steps_per_sec']:.2e}, final avg_score: {data['score']:.2f}"
        )
        return tr
