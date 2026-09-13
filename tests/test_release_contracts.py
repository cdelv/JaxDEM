"""Public compatibility and core-only installation smoke checks."""

import subprocess
import sys

import pytest

import jaxdem as jd


def test_core_rollout_does_not_import_optional_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'h5py', 'vtk', 'vtkmodules', 'orbax', 'flax', 'distrax'}:
            raise ModuleNotFoundError(f'blocked optional dependency {fullname}', name=fullname)

sys.meta_path.insert(0, BlockOptional())
import jax.numpy as jnp
import jaxdem as jd
state = jd.State.create(pos=jnp.array([[0., 0.], [.8, 0.]]), rad=jnp.full(2, .5))
system = jd.System.create(state=state)
state, system = jd.System.initialize(state, system)
state, system = jd.System.step(state, system, n=2)
assert bool(jnp.all(jnp.isfinite(state.pos)))
for get_feature in (lambda: jd.CheckpointWriter, lambda: jd.utils.save,
                    lambda: jd.VTKWriter(directory="unused-optional-output")):
    try:
        get_feature()
    except ImportError as exc:
        assert 'JaxDEM[io]' in str(exc)
    else:
        raise AssertionError('missing optional dependency was not reported')
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_core_wildcard_exports_do_not_resolve_optional_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'h5py', 'vtk', 'vtkmodules', 'orbax', 'flax', 'distrax'}:
            raise ModuleNotFoundError(f'blocked optional dependency {fullname}', name=fullname)

sys.meta_path.insert(0, BlockOptional())
from jaxdem import *
from jaxdem.writers import *
assert 'orbax' not in sys.modules
assert 'vtk' not in sys.modules
assert 'vtkmodules' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_checkpoint_classes_remain_available_by_explicit_lazy_access():
    pytest.importorskip("orbax.checkpoint")
    import jaxdem.writers as writers

    from jaxdem.writers.checkpoints import (
        CheckpointLoader,
        CheckpointModelLoader,
        CheckpointModelWriter,
        CheckpointWriter,
    )

    assert jd.CheckpointLoader is CheckpointLoader
    assert jd.CheckpointModelLoader is CheckpointModelLoader
    assert jd.CheckpointModelWriter is CheckpointModelWriter
    assert jd.CheckpointWriter is CheckpointWriter
    assert writers.CheckpointLoader is CheckpointLoader
    assert writers.CheckpointModelLoader is CheckpointModelLoader
    assert writers.CheckpointModelWriter is CheckpointModelWriter
    assert writers.CheckpointWriter is CheckpointWriter


def test_cundall_strack_public_name_resolves_to_history_model():
    law = jd.ForceModel.create("cundallstrack")
    assert isinstance(law, jd.forces.CundallStrackForce)
    assert law.history_shape(2) != (0,)
    assert not law.supports_analytical_energy_gradient
