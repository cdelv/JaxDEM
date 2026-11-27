# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""PufferLib Training and Protein Sweeps
========================================

Use JaxDEM's optional PufferLib integration to:

1. Expose any JaxDEM :py:class:`~jaxdem.rl.environments.Environment` to
   external RL loops with :py:class:`~jaxdem.rl.env_wrappers.puffer.PufferEnvWrapper`
   (pure JAX, no PufferLib install required).
2. Train with PufferLib's PyTorch ``PuffeRL`` trainer via
   :py:class:`~jaxdem.rl.trainers.puffer_trainer.PufferTrainer`.
3. Discover score-vs-cost Pareto frontiers over trainer, model, action-space,
   and environment hyperparameters with
   :py:class:`~jaxdem.rl.trainers.protein_sweep.ProteinSweep` (using either
   JaxDEM's :py:class:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer` or
   :py:class:`~jaxdem.rl.trainers.puffer_trainer.PufferTrainer`).

``pufferlib`` (>=4.0) is an optional dependency required only by
``PufferTrainer`` and ``ProteinSweep``. Install it from GitHub with:

.. code-block:: bash

   pip install "git+https://github.com/PufferAI/PufferLib.git@4.0"

See :doc:`intro_to_rl` for core JaxDEM RL concepts.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

import jaxdem.rl as rl
from jaxdem.utils.linalg import norm

HAS_PUFFERLIB = importlib.util.find_spec("pufferlib") is not None

# %%
# 1. Wrap a JaxDEM Environment with ``PufferEnvWrapper``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.rl.env_wrappers.puffer.PufferEnvWrapper` keeps all
# physics stepping, action transforms, reward tracking, and auto-resets inside
# a single fused ``@jax.jit`` call while returning flat
# ``(total_agents, obs_dim)`` arrays for external trainers.

env = rl.Environment.create(
    "singleNavigator",
    min_box_size=4.0,
    max_box_size=4.0,
    max_steps=40,
)
wrapper = rl.PufferEnvWrapper(
    env,
    num_envs=4,
    skip_frames=4,
    action_space=rl.ActionSpace.create("maxNorm", max_norm=3.0),
    seed=0,
)

obs, rew, term = wrapper.reset()
print("Reset shapes:", obs.shape, rew.shape, term.shape)

# Advance a few decisions (each step runs 1 + skip_frames = 5 physics frames).
actions = jax.random.normal(
    jax.random.key(0),
    (wrapper.total_agents, wrapper.action_size),
    dtype=jnp.float32,
)
for _ in range(8):
    obs, rew, term = wrapper.step(actions)

metrics = wrapper.consume_metrics()
print("Wrapper metrics:", {k: round(v, 4) for k, v in metrics.items()})
print("Collider overflow:", wrapper.check_overflow())

# %%
# 2. Train with ``PufferTrainer`` (PufferLib 4.x ``Policy`` + ``Muon``)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.rl.trainers.puffer_trainer.PufferTrainer` wraps
# :py:class:`~jaxdem.rl.env_wrappers.puffer.PufferEnvWrapper` and drives
# PufferLib 4.x's PyTorch ``Policy`` (``DefaultEncoder`` + ``MinGRU``/``MLP`` +
# ``DefaultDecoder``) with ``pufferlib.muon.Muon`` (or Adam). You can also
# supply any custom PyTorch ``nn.Module`` via ``policy=...``.

if HAS_PUFFERLIB:
    with rl.PufferTrainer.create(
        env,
        num_envs=8,
        num_steps_epoch=8,
        skip_frames=4,
        num_epochs=2,
        num_minibatches=2,
        hidden_size=32,
        action_space=rl.ActionSpace.create("maxNorm", max_norm=3.0),
        seed=1,
    ) as trainer:
        summary = trainer.train(verbose=False, report_every=1)
        print(
            "PufferTrainer:",
            f"params={trainer.num_parameters},",
            f"phys_sps={summary['perf/phys_sps']:.1e},",
            f"step_reward={summary['env/step_reward']:.4f}",
        )
else:
    print("Skipping live PufferTrainer run (install `pufferlib` to execute).")

# %%
# 3. Sweep Hyperparameters with ``ProteinSweep``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.rl.trainers.protein_sweep.ProteinSweep` uses
# ``pufferlib.sweep.Protein`` to search the score-vs-wall-clock-cost Pareto
# frontier. With ``backend="ppo"``, it runs JaxDEM's native
# :py:class:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer`, so you can jointly
# sweep trainer, JaxDEM model, action-space, and environment parameters—and
# score every trial with any custom pure-JAX ``eval_metric_fn``.
#
# When sweeping environment parameters such as ``env.max_steps``, pass a fixed
# ``eval_env`` so every trial is scored on the exact same held-out horizon:

eval_env = rl.Environment.create(
    "singleNavigator", min_box_size=4.0, max_box_size=4.0, max_steps=40
)


def env_factory(env_params: Mapping[str, Any]) -> rl.Environment:
    """Build a training environment from swept ``env.*`` parameters."""
    return rl.Environment.create(
        "singleNavigator",
        min_box_size=4.0,
        max_box_size=4.0,
        max_steps=int(env_params["max_steps"]),
    )


def model_factory(
    cur_env: rl.Environment,
    params: Mapping[str, float | int],
    rngs: nnx.Rngs,
) -> rl.Model:
    """Build a JaxDEM model from swept ``policy.*`` and ``action.*`` params."""
    width = int(params["policy.width"])
    depth = int(params["policy.depth"])
    return rl.Model.create(
        "SharedActorCritic",
        key=rngs,
        observation_space_size=cur_env.observation_space_size,
        action_space_size=cur_env.action_space_size,
        architecture=[width] * depth,
        action_space=rl.ActionSpace.create(
            "maxNorm", max_norm=float(params["action.max_norm"])
        ),
    )


def target_fraction(final_env: rl.Environment, _rewards: jax.Array) -> jax.Array:
    """Fraction of agents within one radius of their target at episode end."""
    dist = norm(final_env.state.pos_c - final_env.env_params["objective"])
    return jnp.mean(dist <= final_env.state.rad)


sweep_params = [
    rl.SweepParam("train.learning_rate", "log_normal", 1e-3, 3e-2, 1e-2),
    rl.SweepParam("policy.width", "uniform_pow2", 16.0, 64.0, 32.0),
    rl.SweepParam("policy.depth", "int_uniform", 1.0, 2.0, 1.0),
    rl.SweepParam("action.max_norm", "uniform", 1.5, 4.5, 3.0),
    rl.SweepParam("env.max_steps", "int_uniform", 20.0, 60.0, 40.0),
]

print("Configured sweep parameters:", [p.name for p in sweep_params])

# %%
# Run a tiny 2-trial sweep (increase ``max_runs``, ``num_envs``, and
# ``num_epochs`` for real hyperparameter studies):

if HAS_PUFFERLIB:
    sweep = rl.ProteinSweep(
        env_factory,
        sweep_params,
        backend="ppo",
        cost_param=None,
        metric="eval/target_fraction",
        model_factory=model_factory,
        eval_metric_fn=target_fraction,
        eval_env=eval_env,
        eval_n=4,
        max_runs=2,
        num_random_samples=1,
        downsample=1,
        num_envs=4,
        num_steps_epoch=8,
        num_epochs=2,
        skip_frames=4,
    )
    result = sweep.run(verbose=False)
    print(result.summary_table())
else:
    # Even without pufferlib installed, make_ppo_eval_fn can score any PPOTrainer.
    demo_tr = rl.Trainer.create(
        "PPO",
        env=eval_env,
        model=model_factory(
            eval_env,
            {"policy.width": 16, "policy.depth": 1, "action.max_norm": 3.0},
            nnx.Rngs(0),
        ),
        num_envs=4,
        num_steps_epoch=8,
        num_epochs=1,
        num_minibatches=2,
        skip_frames=4,
    )
    eval_fn = rl.make_ppo_eval_fn(
        eval_env, metric_fn=target_fraction, eval_n=4, skip_frames=4
    )
    print(f"Held-out target_fraction (untrained): {eval_fn(demo_tr):.3f}")
    print("Skipping live ProteinSweep run (install `pufferlib` to execute).")
