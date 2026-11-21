# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Checkpoint options, custom-force serialization, and raw H5 bonded reattachment."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

import jaxdem as jdem

# ---------------------------------------------------------------------------
# Helpers (also imported by other serialization test modules)
# ---------------------------------------------------------------------------


def dummy_target_fn(state, system):
    return jnp.sum(state.pos**2)


def _assert_leaves_equal(
    original,
    restored,
    label: str = "",
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    """Check semantic/static configuration as well as exact stored array payloads."""
    from jaxdem.minimizers.optimizers import CustomGradientTransformation

    def static_config(left, right, normalize_sequence=False):
        if hasattr(left, "dtype"):
            right_array = jnp.asarray(right)
            assert left.shape == right_array.shape, label
            assert left.dtype == right_array.dtype, label
        elif isinstance(left, CustomGradientTransformation):
            assert left.metadata == right.metadata, label
            assert left.requires_value == right.requires_value, label
        elif is_dataclass(left):
            assert type(left) is type(right), label
            for field in fields(left):
                static_config(
                    getattr(left, field.name),
                    getattr(right, field.name),
                    (
                        isinstance(left, jdem.ForceManager)
                        and field.name
                        in {"force_functions", "energy_functions", "is_com_force"}
                    )
                    or (
                        isinstance(left, jdem.ForceModel)
                        and field.name in {"laws", "table"}
                    ),
                )
        elif isinstance(left, Mapping):
            assert left.keys() == right.keys(), label
            for key in left:
                static_config(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            # JSON metadata normalizes tuples to lists; both encode the same sequence.
            assert isinstance(right, (tuple, list)) and len(left) == len(right), label
            assert normalize_sequence or type(left) is type(right), label
            for a, b in zip(left, right, strict=True):
                static_config(a, b, normalize_sequence)
        elif callable(left):
            assert left is right, label
        else:
            assert type(left) is type(right), label
            assert left == right, label

    static_config(original, restored)
    leaves_orig = jax.tree.leaves(original)
    leaves_rest = jax.tree.leaves(restored)
    assert len(leaves_orig) == len(
        leaves_rest
    ), f"{label}: leaf count mismatch {len(leaves_orig)} vs {len(leaves_rest)}"
    for i, (lo, lr) in enumerate(zip(leaves_orig, leaves_rest, strict=False)):
        lo, lr = jnp.asarray(lo), jnp.asarray(lr)
        assert lo.shape == lr.shape and lo.dtype == lr.dtype, label
        if rtol or atol:
            assert jnp.allclose(
                lo, lr, rtol=rtol, atol=atol
            ), f"{label} leaf {i}: {lo} != {lr}"
        else:
            assert jnp.array_equal(lo, lr), f"{label} leaf {i}: {lo} != {lr}"


def _round_trip(
    tmp_path: Path, state: jdem.State, system: jdem.System
) -> tuple[jdem.State, jdem.System]:
    """Save then load a checkpoint and return the restored pair."""
    ckpt = tmp_path / "ckpt"
    with jdem.CheckpointWriter(ckpt) as w:
        w.save(state, system)
    with jdem.CheckpointLoader(ckpt) as loader:
        return loader.load()


class TestH5Load:
    """Regression tests for raw HDF5 save/load helpers."""

    def test_h5_load_reattaches_bonded_force(self, tmp_path):
        vertices = jnp.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
        )
        elements = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        adjacency = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

        dp = jdem.BondedForceModel.create(
            "deformableparticlemodel",
            vertices=vertices,
            elements=elements,
            edges=elements,
            element_adjacency=adjacency,
            ec=1.0,
            eb=0.5,
            el=0.3,
        )
        state = jdem.State.create(pos=vertices)
        system = jdem.System.create(
            state.shape, bonded_force_model=dp, interact_same_bond_id=False
        )

        expected_pe = jdem.utils.thermal.compute_potential_energy(state, system)

        state_path = str(tmp_path / "state.h5")
        system_path = str(tmp_path / "system.h5")
        jdem.utils.h5.save(state, state_path)
        jdem.utils.h5.save(system, system_path)

        state_r = jdem.utils.h5.load(state_path)
        system_r = jdem.utils.h5.load(system_path)

        assert system_r.bonded_force_model is not None
        assert len(system_r.force_manager.force_functions) == 1
        assert system_r.force_manager.force_functions[0].__name__ == "compute_forces"

        restored_pe = jdem.utils.thermal.compute_potential_energy(state_r, system_r)
        assert jnp.allclose(expected_pe, restored_pe)


class TestCustomForceFunctions:
    """Checkpoint round-trips for user-supplied force/energy functions and ragged bond_id."""

    def test_multiple_force_functions_and_continuation(self, tmp_path):
        """Mixed energy/COM flags and bond_id survive round-trip and continue identically."""
        from tests.custom_forces import (
            constant_push,
            harmonic_trap,
            harmonic_trap_energy,
        )

        state = jdem.State.create(
            pos=jnp.array([[2.0, 0.0], [2.5, 0.0]]),
            rad=jnp.array([0.5, 0.5]),
            bond_id=jnp.array([[1, -1], [0, -1]]),
        )
        system = jdem.System.create(
            state.shape,
            interact_same_bond_id=False,
            force_manager_kw={
                "force_functions": [
                    (harmonic_trap, harmonic_trap_energy, False),
                    (constant_push, None, True),
                ],
            },
            dt=0.05,
        )
        state, system = system.step(state, system)
        state_r, system_r = _round_trip(tmp_path, state, system)

        _assert_leaves_equal(state, state_r, "state")
        _assert_leaves_equal(system, system_r, "system")
        fm = system_r.force_manager
        assert len(fm.force_functions) == 2
        assert fm.force_functions[0].__name__ == "harmonic_trap"
        assert fm.force_functions[1].__name__ == "constant_push"
        assert fm.energy_functions[0].__name__ == "harmonic_trap_energy"
        assert fm.is_com_force == (False, True)

        state_a, _ = system.step(state, system)
        state_b, _ = system_r.step(state_r, system_r)
        _assert_leaves_equal(state_a, state_b, "post-restore step")

    def test_custom_force_with_bonded_model(self, tmp_path):
        """Custom force + bonded model restores both callbacks without double-counting."""
        from tests.custom_forces import harmonic_trap, harmonic_trap_energy

        vertices = jnp.array(
            [[0.0, 0.0], [1.2, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float
        )
        elements = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
        adjacency = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)

        dp = jdem.BondedForceModel.create(
            "deformableparticlemodel",
            vertices=vertices,
            elements=elements,
            edges=elements,
            element_adjacency=adjacency,
            em=1.0,
            eb=0.5,
            el=0.3,
            gamma=0.1,
        )

        state = jdem.State.create(pos=vertices)
        system = jdem.System.create(
            state.shape,
            bonded_force_model=dp,
            interact_same_bond_id=False,
            force_manager_kw={
                "force_functions": [(harmonic_trap, harmonic_trap_energy)],
            },
        )
        assert len(system.force_manager.force_functions) == 2

        state_r, system_r = _round_trip(tmp_path, state, system)
        _assert_leaves_equal(state, state_r, "state")

        fm = system_r.force_manager
        assert len(fm.force_functions) == 2
        assert fm.force_functions[0].__name__ == "harmonic_trap"
        np.testing.assert_allclose(
            jdem.utils.thermal.compute_potential_energy(state_r, system_r),
            jdem.utils.thermal.compute_potential_energy(state, system),
        )
        state_a, _ = system.step(state, system)
        state_b, _ = system_r.step(state_r, system_r)
        _assert_leaves_equal(state_a, state_b, "bonded+custom step")

    def test_nonimportable_function_rejected_on_save(self, tmp_path):
        """Saving a non-importable function fails before checkpoint I/O."""

        def _dummy_force(pos, state, system):
            return pos * 0, jnp.zeros_like(state.torque)

        _dummy_force.__module__ = "__main__"

        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw={"force_functions": [(_dummy_force,)]},
            dt=0.01,
        )
        state, system = system.step(state, system)

        with pytest.raises(TypeError, match="not importable|module-level"):
            with jdem.CheckpointWriter(tmp_path / "ckpt") as w:
                w.save(state, system)

    def test_unresolvable_force_or_energy_handling_on_load(self, tmp_path):
        """Unresolvable force/energy functions raise under strict=True and skip under strict=False."""
        from tests.custom_forces import harmonic_trap, harmonic_trap_energy

        state = jdem.State.create(pos=jnp.array([[2.0, 0.0]]))
        system = jdem.System.create(
            state.shape,
            force_manager_kw={
                "force_functions": [(harmonic_trap, harmonic_trap_energy)],
            },
            dt=0.01,
        )
        state, system = system.step(state, system)

        ckpt = tmp_path / "ckpt"
        with jdem.CheckpointWriter(ckpt) as w:
            w.save(state, system)

        meta_file = next(iter(ckpt.rglob("system_metadata/metadata")))
        original_meta = json.loads(meta_file.read_bytes())

        # Missing energy function fails loudly under strict=True.
        bad_energy = json.loads(json.dumps(original_meta))
        bad_energy["force_function_metadata"][0][
            "energy"
        ] = "tests.custom_forces.missing_energy"
        meta_file.write_text(json.dumps(bad_energy))
        with jdem.CheckpointLoader(ckpt) as loader:
            with pytest.raises(RuntimeError, match="Could not restore energy function"):
                loader.load()

        # Missing force function raises under strict=True, skips with warning under strict=False.
        bad_force = json.loads(json.dumps(original_meta))
        bad_force["force_function_metadata"][0][
            "force"
        ] = "nonexistent_module.fake_force"
        meta_file.write_text(json.dumps(bad_force))
        with jdem.CheckpointLoader(ckpt) as loader:
            with pytest.raises(RuntimeError, match="Could not restore"):
                loader.load()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _state_r, system_r = loader.load(strict=False)

        assert any("Could not restore" in str(w.message) for w in caught)
        assert len(system_r.force_manager.force_functions) == 0


class TestWriterOptions:
    """Tests for CheckpointWriter configuration options."""

    def test_multiple_saves_load_latest_and_specific_step(self, tmp_path):
        state = jdem.State.create(
            pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
            rad=jnp.array([1.0, 1.0]),
        )
        system = jdem.System.create(state.shape)

        ckpt = tmp_path / "ckpt"
        step_counts = []
        with jdem.CheckpointWriter(ckpt) as w:
            for _ in range(3):
                state, system = system.step(state, system)
                step_counts.append(int(system.step_count))
                w.save(state, system)

        with jdem.CheckpointLoader(ckpt) as loader:
            state_latest, system_latest = loader.load()
            _assert_leaves_equal(state, state_latest, "latest state")
            assert int(system_latest.step_count) == step_counts[-1]

            _state_first, system_first = loader.load(step=step_counts[0])
            assert int(system_first.step_count) == step_counts[0]

    def test_load_missing_step_raises(self, tmp_path):
        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([1.0]))
        system = jdem.System.create(state.shape)
        state, system = system.step(state, system)

        ckpt = tmp_path / "ckpt"
        with jdem.CheckpointWriter(ckpt) as w:
            w.save(state, system)

        with jdem.CheckpointLoader(ckpt) as loader:
            with pytest.raises(FileNotFoundError):
                loader.load(step=9999)

    def test_load_empty_dir_raises(self, tmp_path):
        with jdem.CheckpointLoader(tmp_path) as loader:
            with pytest.raises(FileNotFoundError):
                loader.load()

    def test_save_every(self, tmp_path):
        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([1.0]))
        system = jdem.System.create(state.shape)
        ckpt = tmp_path / "ckpt"
        with jdem.CheckpointWriter(ckpt, save_every=2) as w:
            for step in [2, 3, 4]:
                system.step_count = jnp.array(step, dtype=int)
                w.save(state, system)
                w.block_until_ready()

        with jdem.CheckpointLoader(ckpt) as loader:
            assert loader.checkpointer.all_steps() == [2, 4]

    def test_max_to_keep(self, tmp_path):
        state = jdem.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([1.0]))
        system = jdem.System.create(state.shape)
        ckpt = tmp_path / "ckpt"
        with jdem.CheckpointWriter(ckpt, max_to_keep=2) as w:
            for step in [1, 2, 3]:
                system.step_count = jnp.array(step, dtype=int)
                w.save(state, system)
                w.block_until_ready()

        with jdem.CheckpointLoader(ckpt) as loader:
            assert loader.checkpointer.all_steps() == [2, 3]


class TestModelCheckpoints:
    """Tests for CheckpointModelWriter and CheckpointModelLoader."""

    def test_model_load_specific_step(self, tmp_path):
        from flax import nnx
        import jaxdem.rl as rl
        from jaxdem.writers.checkpoints import (
            CheckpointModelLoader,
            CheckpointModelWriter,
        )

        model1 = rl.Model.create(
            "SharedActorCritic",
            observation_space_size=8,
            action_space_size=2,
            key=nnx.Rngs(42),
        )
        model2 = rl.Model.create(
            "SharedActorCritic",
            observation_space_size=8,
            action_space_size=2,
            key=nnx.Rngs(100),
        )
        model1.actor_mu.bias[...] = jnp.array([1.25, -0.75])
        model2.actor_mu.bias[...] = jnp.array([-2.0, 3.5])

        ckpt = tmp_path / "ckpt"
        with CheckpointModelWriter(ckpt, clean=False) as w:
            w.save(model1, step=10)
            w.save(model2, step=20)

        with CheckpointModelLoader(ckpt) as loader:
            model_latest = loader.load()
            _, state_orig2 = nnx.split(model2)
            _, state_loaded_latest = nnx.split(model_latest)
            for expected, actual in zip(
                jax.tree.leaves(state_orig2),
                jax.tree.leaves(state_loaded_latest),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)

            model_10 = loader.load(step=10)
            _, state_orig1 = nnx.split(model1)
            _, state_loaded_10 = nnx.split(model_10)
            for expected, actual in zip(
                jax.tree.leaves(state_orig1),
                jax.tree.leaves(state_loaded_10),
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)

    def test_model_load_missing_step_and_empty_dir_raise(self, tmp_path):
        from flax import nnx
        import jaxdem.rl as rl
        from jaxdem.writers.checkpoints import (
            CheckpointModelLoader,
            CheckpointModelWriter,
        )

        empty = tmp_path / "empty"
        empty.mkdir()
        with CheckpointModelLoader(empty) as loader:
            with pytest.raises(FileNotFoundError):
                loader.load()

        model = rl.Model.create(
            "SharedActorCritic",
            observation_space_size=8,
            action_space_size=2,
            key=nnx.Rngs(42),
        )
        ckpt = tmp_path / "ckpt"
        with CheckpointModelWriter(ckpt) as w:
            w.save(model, step=10)

        with CheckpointModelLoader(ckpt) as loader:
            with pytest.raises(FileNotFoundError):
                loader.load(step=9999)

    def test_model_clean_option(self, tmp_path):
        from flax import nnx
        import jaxdem.rl as rl
        from jaxdem.writers.checkpoints import (
            CheckpointModelLoader,
            CheckpointModelWriter,
        )

        model = rl.Model.create(
            "SharedActorCritic",
            observation_space_size=8,
            action_space_size=2,
            key=nnx.Rngs(42),
        )
        ckpt = tmp_path / "ckpt"

        with CheckpointModelWriter(ckpt, clean=True) as w:
            w.save(model, step=10)
            w.block_until_ready()
        with CheckpointModelLoader(ckpt) as loader:
            assert loader.checkpointer.all_steps() == [10]

        with CheckpointModelWriter(ckpt, clean=False) as w:
            w.save(model, step=20)
            w.block_until_ready()
        with CheckpointModelLoader(ckpt) as loader:
            assert loader.checkpointer.all_steps() == [10, 20]

        with CheckpointModelWriter(ckpt, clean=True) as w:
            w.save(model, step=30)
            w.block_until_ready()
        with CheckpointModelLoader(ckpt) as loader:
            assert loader.checkpointer.all_steps() == [30]
