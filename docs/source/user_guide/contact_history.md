# Cundall–Strack contact history

Select `force_model_type="cundallstrack"` for a normal spring/dashpot,
a tangential displacement spring with viscous damping and a Coulomb limit,
and optional rolling resistance. Tangential displacement persists when sliding
motion stops, providing static friction.

Use a `NeighborList` collider to retain pair history:

```python
import jax.numpy as jnp
import jaxdem as jd

state = jd.State.create(
    pos=jnp.array([[0.0, 0.0], [0.9, 0.0]]),
    rad=jnp.full(2, 0.5),
    vel=jnp.array([[0.0, 0.1], [0.0, 0.0]]),
)
material = jd.Material.create(
    "elasticfrict", density=1.0, young=100.0,
    poisson=0.3, e=0.8, mu=0.5, mu_r=0.0,
)
system = jd.System.create(
    state=state,
    dt=1e-3,
    mat_table=jd.MaterialTable.from_materials([material]),
    force_model_type="cundallstrack",
    collider_type="NeighborList",
    collider_kw={"cutoff": 1.0, "skin": 0.1, "max_neighbors": 1},
)
state, system = jd.System.initialize(state, system)
state, system = jd.System.step(state, system, n=10)
system.check_overflow()
```

The history layout per directed pair is `(2 * dim,)`: the tangential spring
displacement followed by the previous contact normal. The law transports the
spring into the current contact frame before adding the timestep's tangential
relative motion. In 3D it also follows common spin about the contact normal.
Sliding adjusts the stored spring to the Coulomb surface so displacement does
not accumulate beyond the force limit. Contact separation clears history.

Only an advancing force evaluation updates history. `System.initialize` and
force calls with `advance_history=False` preserve it, including when inspecting
a separated contact. Neighbor rebuilds remap surviving pairs, and advancing
calls reset excluded/padding slots. Checkpoints retain the history required to
continue the same trajectory.

Contact normals that reverse by exactly 180 degrees have no unique shortest
rotation in 3D. The implementation uses a deterministic perpendicular axis in
that case. Ordinary timestep refinement should keep contact-frame changes small.

`energy` reports **normal elastic energy only**. The current energy API does not
receive pair history, so it excludes tangential stored energy as well as
irreversible friction/damping losses. Accordingly this law sets
`supports_analytical_energy_gradient=False`: minimization requires an explicit
objective, and normal-energy diagnostics must not be interpreted as the full
mechanical energy of a frictional trajectory.

Changing an existing system from the former stateless approximation changes its
physics. Construct a new history-capable system; a checkpoint from that
approximation cannot reconstruct tangential displacement that was never stored.
