# `jaxdem.State`

Short description of what a **state** represents in JaxDEM.

## Creating a state
```python
import jax
import jax.numpy as jnp
import jaxdem as jd

state = jd.State(
    x=jnp.zeros((N, 3)),
    v=jnp.zeros((N, 3)),
    radii=jnp.full((N,), 0.5),
)