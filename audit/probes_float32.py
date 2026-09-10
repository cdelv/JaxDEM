import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jaxdem as jd
from jaxdem.colliders._partition import _grid_params
p = _grid_params(jnp.array([100000., 100000.]), jnp.array(1.), False)
print("grid dtypes", p[0].dtype, p[1].dtype, "grid dims", p[0], "overflow", p[3], "iinfo(int)", jnp.iinfo(int).max)
s = jd.State.create(pos=jnp.array([[0., 0.], [1., 0.]]))
y = jd.System.create(state=s)
try:
    s, y = jd.utils.scale_to_packing_fraction(s, y, .5)
    print("float32 packing", s.pos)
except Exception as e:
    print(type(e).__name__, str(e))
