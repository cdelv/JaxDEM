"""Public compatibility and core-only installation smoke checks."""

import subprocess
import sys

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


def test_cundall_strack_public_name_resolves_to_history_model():
    law = jd.ForceModel.create("cundallstrack")
    assert isinstance(law, jd.forces.CundallStrackForce)
    assert law.history_shape(2) != (0,)
    assert not law.supports_analytical_energy_gradient
