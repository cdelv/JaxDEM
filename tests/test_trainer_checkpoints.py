# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Trainer checkpoints restore the complete trainer and continue training."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses
from functools import partial

import numpy as np
import optax
import pytest
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl

NUM_ENVS = 2
# Not importable by path, so checkpoint metadata cannot describe it.
UNIMPORTABLE_OPTIMIZER = partial(optax.adam, b1=0.8)


def _trainer(seed: int = 0, **kw):
    env = rl.Environment.create(
        "MultiNavigator", N=2, min_box_size=8.0, max_box_size=8.0
    )
    model = rl.Model.create(
        "MinGRUActorCritic",
        key=nnx.Rngs(seed),
        observation_space_size=env.observation_space_size,
        action_space_size=env.action_space_size,
        action_space=rl.ActionSpace.create("maxNorm", max_norm=2.0),
        hidden_features=8,
        gru_features=8,
        num_layers=1,
    )
    return rl.Trainer.create(
        "PPO",
        env=env,
        model=model,
        key=jax.random.key(seed),
        num_envs=NUM_ENVS,
        num_steps_epoch=4,
        num_minibatches=2,
        num_epochs=3,
        skip_frames=1,
        **kw,
    )


def _arrays(tree):
    return [
        np.asarray(
            jax.random.key_data(x)
            if jnp.issubdtype(jnp.asarray(x).dtype, jax.dtypes.prng_key)
            else x
        )
        for x in jax.tree.leaves(tree)
    ]


def _assert_same(actual, expected, *, exact: bool = True) -> None:
    """Compare the pytree structure and every leaf of two trainers.

    Separately built optimizers never compare equal, so the graph
    definitions are excluded; everything else in the structure must match.
    """
    assert jax.tree.structure(
        dataclasses.replace(actual, graphdef=expected.graphdef)
    ) == jax.tree.structure(expected)
    for a, b in zip(_arrays(actual), _arrays(expected), strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        if exact:
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train three epochs with a checkpoint every second epoch."""
    directory = tmp_path_factory.mktemp("trainer")
    trainer = _trainer()
    with jdem.CheckpointTrainerWriter(directory, save_every=2) as writer:
        final = trainer.train(
            trainer, verbose=False, log=False, checkpoint_writer=writer
        )
    return directory, final


def test_training_saves_each_interval_and_the_final_epoch(trained) -> None:
    directory, final = trained
    with jdem.CheckpointTrainerLoader(directory) as loader:
        assert loader.checkpointer.all_steps() == [2, 3]
        assert loader.latest_step() == 3
        _assert_same(loader.load(), final)


def test_forced_save_does_not_replace_an_existing_step(trained) -> None:
    directory, final = trained
    with jdem.CheckpointTrainerWriter(directory) as writer:
        with pytest.warns(UserWarning, match="already exists"):
            writer.save(final, 3, force=True)


def test_loaded_trainer_continues_the_run(trained) -> None:
    directory, final = trained
    with jdem.CheckpointTrainerLoader(directory) as loader:
        resumed = loader.load(step=2)
    resumed = resumed.train(resumed, verbose=False, log=False, start_epoch=2)
    # The restored state is exact. A recompiled epoch may round differently.
    _assert_same(resumed, final, exact=False)


def test_model_and_simulation_loaders_read_trainer_checkpoints(trained) -> None:
    directory, final = trained
    with jdem.CheckpointModelLoader(directory) as loader:
        model = loader.load()
    assert type(model) is type(final.model)
    for a, b in zip(
        _arrays(nnx.state(model)), _arrays(final.graphstate[0]), strict=True
    ):
        np.testing.assert_array_equal(a, b)

    with jdem.CheckpointLoader(directory) as loader:
        state, system = loader.load()
    restored = dataclasses.replace(final.env, state=state, system=system)
    assert jax.tree.structure(restored) == jax.tree.structure(final.env)
    for a, b in zip(_arrays(restored), _arrays(final.env), strict=True):
        np.testing.assert_array_equal(a, b)


def test_stacked_trainers_round_trip(tmp_path) -> None:
    trainer = _trainer()

    def lane(key):
        env = trainer.env.reset(trainer.env, jax.random.split(key, NUM_ENVS))
        return dataclasses.replace(trainer, env=env, key=key)

    stacked = jax.vmap(lane)(jax.random.split(jax.random.key(3), 3))
    assert stacked.env.state.pos.shape[:2] == (3, NUM_ENVS)
    with jdem.CheckpointTrainerWriter(tmp_path) as writer:
        writer.save(stacked, 0)
    with jdem.CheckpointTrainerLoader(tmp_path) as loader:
        _assert_same(loader.load(), stacked)
        _assert_same(loader.load(target=stacked), stacked)


def test_target_restores_a_trainer_that_metadata_cannot_describe(tmp_path) -> None:
    trainer = _trainer(optimizer=UNIMPORTABLE_OPTIMIZER)
    with jdem.CheckpointTrainerWriter(tmp_path) as writer:
        writer.save(trainer, 0)
    with jdem.CheckpointTrainerLoader(tmp_path) as loader:
        with pytest.raises(RuntimeError, match="optimizer_config.*target"):
            loader.load()
        target = _trainer(seed=1, optimizer=UNIMPORTABLE_OPTIMIZER)
        restored = loader.load(target=target)
    assert restored.graphdef == target.graphdef
    _assert_same(restored, trainer)
