# Mutation and compiled state

`State`, `System`, `Domain`, `Quaternion`, and `ForceManager` are mutable Python
objects. Mutation is useful while assembling a simulation on the host. Inside
JAX transformations, treat them as pytrees of values: call a JaxDEM operation,
use its return value, and rebind it.

```python
state, system = system.step(state, system)
state = state.refresh_rotation_cache()
```

Python assignment to an object captured by an already compiled function does
not update that compiled computation. Pass changed objects as arguments, or
compile again when static configuration changes. Array-valued data fields are
dynamic pytree leaves. Class identity, callable configuration, tuple structure,
and fields declared static are compilation metadata; changing them produces a
different specialization.

Some value objects, including `MaterialTable`, optimizer results, analysis
specifications, and `BodyTopology`, are frozen. Construct a replacement rather
than assigning their attributes. This distinction is a Python API property; it
does not change the return-and-rebind rule under `jax.jit`.

## Derived caches

Several arrays are derived from other fields and have an explicit refresh
contract:

- Mutating nested `state.q.w` or `state.q.xyz` does not refresh rotated clump
  offsets. Call `state.refresh_rotation_cache()`. Assigning a complete `q` or
  `pos_p` field refreshes the cache automatically.
- Changing positions, domain geometry, or force-law search radii can invalidate
  a collider cache. Use `invalidate_collider` to rebuild it while preserving
  compatible pair history. Use `refresh_collider` when particle count or cache
  capacity changes; pass `reset_history=True` only when pair identity or the
  history layout has changed and old history cannot be mapped safely.
- Neighbor-list contact history belongs to cached directed pairs. Read-only
  force evaluation preserves it, while a physical step advances it once.
- A force law's `search_radii` bounds determine spatial-search reach. After
  changing law parameters that affect reach, refresh or recreate the collider.

`State.body_topology()` is a derived logical-body view. It keeps `N` padded body
slots so shapes remain static; `valid` selects occupied slots. The simulation
continues to store replicated particle-shaped clump fields for contact-kernel
performance and checkpoint compatibility. Call `State.validate()` first when
using `body_mass_properties()`, or pass ``validate=True`` to that method, when
using it as a trusted mass-property record. It reads replicated mass, volume,
and inertia from each body's representative.

Validate host-side mutations with `state.validate()` or `system.validate(state)`
before entering a long compiled run.
