# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Public compatibility and core-only installation smoke checks."""

import subprocess
import sys

import pytest

import jaxdem as jd


def test_core_rollout_and_wildcard_exports_do_not_import_optional_dependencies():
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
for mod in ('orbax', 'vtk', 'vtkmodules', 'h5py'):
    assert mod not in sys.modules

import jax.numpy as jnp
import jaxdem as jd
state = jd.State.create(pos=jnp.array([[0., 0.], [.8, 0.]]), rad=jnp.full(2, .5))
system = jd.System.create(state=state)
state, system = jd.System.initialize(state, system)
state, system = jd.System.step(state, system, n=2)
assert bool(jnp.all(jnp.isfinite(state.pos)))
for get_feature in (lambda: jd.CheckpointWriter, lambda: jd.utils.save,
                    lambda: jd.VTKWriter(directory="unused-optional-output"),
                    lambda: jd.H5Writer(directory="unused-optional-output")):
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


def test_package_exposes_version_and_py_typed():
    from pathlib import Path

    assert isinstance(jd.__version__, str) and jd.__version__
    assert (Path(jd.__file__).resolve().parent / "py.typed").is_file()


def test_importing_core_and_analysis_does_not_initialize_jax_backend():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import jax._src.xla_bridge as xb
import jaxdem
import jaxdem.analysis

assert not xb.backends_are_initialized(), "Importing jaxdem/jaxdem.analysis initialized JAX backend"
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
