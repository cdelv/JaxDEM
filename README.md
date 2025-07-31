# JaxDEM

JaxDEM ( **Jax** Discrete‐Element Method ) is a lightweight, fully
JAX-compatible toolkit for particle simulations in 2-D or 3-D.
Everything—geometry, neighbour search, material laws—is written with
pure JAX arrays so that you can

* `jit`–compile the entire solver,
* run thousands of realisations in parallel with `vmap`,
* ship the computation seamlessly to CPU / GPU / TPU, and
* keep the code base short, hackable and fun.


## Getting started

### 1  Import `jax` before `jaxdem`

`jax.config.update` is evaluated **at import time**.  Therefore select a
device *first*, then import JaxDEM:

```python
import jax, jax.numpy as jnp

jax.config.update("jax_platform_name", "gpu")   # or "cpu" / "tpu"

import jaxdem as jdem
```

---

## 1. State  – the particle container

`State` is a dataclass with a “structure-of-arrays’’ layout:

```
State
 ├─ pos          (..., N, dim)  positions
 ├─ vel          (..., N, dim)  velocities
 ├─ accel        (..., N, dim)  accelerations
 ├─ rad          (..., N)       radii
 ├─ mass         (..., N)       masses
 ├─ ID           (..., N)       global identifiers
 ├─ mat_id       (..., N)       material index (0‥M-1)
 └─ species_id   (..., N)       species  index (0‥S-1)
```

* `N` number of particles  
* `dim` 2 or 3  
* the optional **leading axes** hold batches and/or trajectories

### 1.1 Create a state

```python
N, dim = 3, 2
coords = jnp.stack([2 * x * jnp.ones(dim) for x in range(N)])  # (N, 2)

state = jdem.State.create(coords)
print(state)
```

```
State(
  pos=[[0. 0.]
       [2. 2.]
       [4. 4.]],
  vel=[[0. 0.]
       [0. 0.]
       [0. 0.]],
  accel=[[0. 0.]
         [0. 0.]
         [0. 0.]],
  rad=[1. 1. 1.],
  mass=[1. 1. 1.],
  ID=[0 1 2],
  mat_id=[0 0 0],
  species_id=[0 0 0])
```

`pos` is the **only** mandatory argument; every other field is populated
with a shape-compatible default.  Override what you need:

```python
vel = jnp.stack([x * jnp.ones(dim) for x in range(N)])   # (N, 2)
state = jdem.State.create(coords, vel=vel, rad=jnp.array([.5, .6, .7]))
```

Shape checks are automatic—wrong input raises a clear error.

---

### 1.2 Batch simulations with `vmap`

```python
batch_size = 5
state_batch = jax.vmap(lambda _: jdem.State.create(coords))(jnp.arange(batch_size))

print(state_batch.pos.shape)   # (B, N, dim) = (5, 3, 2)
```

The first leading axis is now a **batch**; all kernels in JaxDEM honour that convention.

---

### 1.3 Add particles

```python
state = jdem.State.add(
    state,
    pos=jnp.array([[10., 10.]]),
    rad=jnp.array([1.2]),
    mat_id=jnp.array([1]),         # optional: give the new sphere a new material
)
print(state.N)   # 4
```

`add` re-uses `State.create`’s keyword interface; shapes must broadcast
with the existing batch layout.

---

### 1.4 Merge two snapshots

```python
state_big = jdem.State.merge(state, state)
print(state_big.N)  # 8   (particle axis doubled)
```

IDs of the second argument are shifted automatically so they stay
unique.

---

### 1.5 Build a trajectory

`State.stack` concatenates an iterable along the *time* axis:

```python
traj = jdem.State.stack([state, state, state])   # length-3 trajectory
print(traj.pos.shape)   # (3, N, dim)
```

The function validates dimension, batch Size and particle count.

---

### 1.6 Batch + trajectory

```python
def init(_):
    # one *snapshot*  ─ shape (N, dim)
    return jdem.State.create(coords)

# -----------------------------------------------------------
# 1. build one *batched* snapshot  – shape (B, N, dim)
# -----------------------------------------------------------
batch_snap = jax.vmap(init)(jnp.arange(4))        # B = 4
print(batch_snap.pos.shape)                       # (4, 3, 2)

# -----------------------------------------------------------
# 2. stack 10 such snapshots along the *time* axis
#    result shape  (T, B, N, dim)
# -----------------------------------------------------------
batch_traj = jdem.State.stack([batch_snap] * 10)  # T = 10
print(batch_traj.pos.shape)                       # (10, 4, 3, 2)
```

### 1.7 Axis semantics in JaxDEM

1. **First leading axis**  
   By convention the *outermost* axis (axis 0) is  
   • **time/trajectory** when you call `State.stack`, `System.step_rollout`,
     or when you pass `trajectory=True` to `VTKWriter.save`.
   • **batch** when the data come from a plain `jax.vmap` 

2. **Second, third, … leading axes**  
   Every additional axis is always treated as **trajectory**.  
   This rule makes mixed layouts possible without ambiguity:

   | shape                      | interpretation                            |
   |----------------------------|-------------------------------------------|
   | `(N, dim)`                 | single snapshot                           |
   | `(T, N, dim)`              | trajectory                                |
   | `(B, N, dim)`              | batch (all snapshots taken at the same t) |
   | `(T, B, N, dim)`           | trajectory of **batched** snapshots       |
   | `(T, T, N, dim)`           | trajectory of trajectories. (whatever that means) |

   In the code above we first created a **batch snapshot**
   `(B, N, dim)` and then stacked it → `(T, B, N, dim)`, so axis 0 is
   time and axis 1 is batch.

All JaxDEM kernels—colliders, integrators, material routers and VTK
writers—follow those rules automatically, so you can freely combine
`vmap`, `stack`, `jit`, and `pmap` without reshaping arrays by hand.


---

## 2. System – simulation configuration

`System` collects every *static* ingredient of a DEM run:

```python
System
 ├─ integrator          time-stepping scheme   (Euler, RK4, …)
 ├─ collider            contact–search + reduction strategy
 ├─ domain              boundary conditions
 ├─ force_model         force-law or router (may combine many laws)
 ├─ mat_table           per-material property table (may be a pair matrix)
 └─ dt                  time step
```

### 2.1 Quick start

```python
import jax.numpy as jnp
import jaxdem as jdem

dim = state.dim                          # must match state.dim
system = jdem.System.create(dim)
```

Only `dim` is mandatory:

| component     | default | factory key | kwargs dict                     |
|---------------|---------|-------------|---------------------------------|
| integrator    | Euler   | `"euler"`   | `integrator_kw={...}`           |
| collider      | naive   | `"naive"`   | `collider_kw={...}`             |
| domain        | free    | `"free"`    | `domain_kw={anchor, box_size}`  |
| force model   | spring  | `"spring"`  | `force_model_kw={...}`          |


The factory raises a helpful error if a key is unknown:

```
KeyError: Unknown Domain 'perodic'. Available: ['free', 'periodic', 'reflect']
```

---

### 2.2 Customising a component

Pass a different *factory key* plus an optional kwargs dict:

```python
system = jdem.System.create(
    dim          = 2,
    domain_type  = "periodic",
    domain_kw    = {
        "anchor":   jnp.ones(2),
        "box_size": 10.0 * jnp.ones(2)
    },
    integrator_type = "euler",
    dt = 1e-3,
)
```

---

### 2.3 Material table

If you do not supply one, a single default material is created (`ElasticMat`):

```python
from jaxdem.Materials import ElasticMat, MaterialTable
from jaxdem.MaterialMatchmaker import MaterialMatchmaker

steel = ElasticMat(young=2.1e11, poisson=0.29)
rubber= ElasticMat(young=1.0e7,  poisson=0.49)

matcher   = MaterialMatchmaker.create("harmonic")
mat_table = MaterialTable.from_materials([steel, rubber], matcher=matcher)

system = jdem.System.create(2, mat_table=mat_table)
```

`System.create` validates that the table contains every property needed by the chosen `force_model`. The MaterialTable can have materials of different types. This is important is one wants to use different force models.

---

### 2.4 `System` is **jit-friendly**

Because every field is a dataclass/PyTree leaf of plain arrays, you can
create or replicate a `System` **inside** `vmap` or `pmap`:

```python
batch_size = 100

@jax.jit
def build_system(_):
    return jdem.System.create(2)

sys_batch = jax.vmap(build_system)(jnp.arange(batch_size))
```

This is convenient when each simulation instance needs a different domain size or time step.

---

### 2.5 Driving a simulation

*One* step:

```python
state, system = jdem.System.step(state, system)
```

*Several* steps:

```python
state, system = jdem.System.step(state, system, n=100)
```

### 2.6 Trajectory rollout

`trajectory_rollout` runs `stride` integrator steps, stores the snapshot,
repeats `n` times, and returns

```python
final_state, final_system, trajectory
```

```python
# This is equivalent as running 200*5 steps saving data every 5 steps
final_st, final_sys, traj = jdem.System.trajectory_rollout(state, system, n=200,  stride = 5)
traj_state, traj_system = traj

print(traj_state.pos.shape)     # (200, N, dim) or (200, B, N, dim) if batched
```

Because the function is `@jax.jit`-ed you can nest it inside `vmap`
to generate hundreds of trajectories at GPU speed:

```python
T = 500        # frames
B = 64         # batch
traj = jax.vmap(
           lambda st: jdem.System.trajectory_rollout(st, system, n=T)[2]
       )(state_batch)            # shape (B, T, N, dim)
```

---

## 3. VTKWriter – visualising a run in ParaView

`VTKWriter` turns a `State / System` pair (or a whole batch / trajectory) into VTK PolyData files that you can open directly in
ParaView. Everything happens in one call; the function figures out by itself how many snapshots it has to write and where they belong on disk.

```python
writer = jdem.VTKWriter(                  # ← default: spheres + domain
            directory = "frames",         # output folder
            binary    = True,             # binary *.vtp*, smaller + faster
            clean     = True)             # purge folder on first save
```

TO DO: We will support different shapes when available in the code

### 3.1 A single snapshot

```python
state  = jdem.State.create(coords)
system = jdem.System.create(dim)

writer.save(state, system)                # → frames/spheres_00000000.vtp
                                          #   frames/domain_00000000.vtp
```

The VTKWriter has an internal counter. If you call save again it will make sure not to override the prev frame.

### 3.2 A trajectory

If the first leading axis of `state.pos` is time set `trajectory=True`:

```python
traj_state, traj_sys = jdem.System.trajectory_rollout(state, system, n=200, stride=1)[2]

writer.save(traj_state, traj_sys, trajectory=True)
```

Output:

```
frames/
  spheres_00000000.vtp
  domain_00000000.vtp
  …
  spheres_00000199.vtp
  domain_00000199.vtp
```

### 3.3 Batch of independent simulations

```python
state_batch = jax.vmap(lambda _: jdem.State.create(coords))(jnp.arange(4))
writer.save(state_batch, system)          # trajectory=False is default
```

Files are grouped in numbered sub-folders:

```
frames/
  batch_00000000/
        spheres_00000000.vtp
        domain_00000000.vtp
  batch_00000001/
  …
```

### 3.4 Batch + trajectory

Shape `(T, B, N, dim)` → first axis = time, second = batch.

```python
writer.save(batch_traj, batch_system, trajectory=True)
```

Layout:

```
frames/
  batch_00000000/
        spheres_00000000.vtp
        domain_00000000.vtp
        …
        spheres_00000009.vtp
  batch_00000001/
  …
```

### 3.5 What gets written?

* **spheres** – each particle centre; every scalar / vector field in
  `State` is exported as Point-Data (`vel`, `rad`, `mat_id`, …).
* **domain** – a rectangle (2-D) or cuboid (3-D) that encloses the
  current simulation box.

### 3.6 Adding your own writer

Register a subclass of `VTKBaseWriter`:

```python
@VTKBaseWriter.register("energy")
class EnergyWriter(VTKBaseWriter):
    @classmethod
    def write(cls, state, system, counter, directory, binary):
        filename = pathlib.Path(directory) / f"energy_{counter:08d}.txt"
        e = system.collider.compute_potential_energy(state, system).sum()
        filename.write_text(f"{e}\n")
        return counter + 1
```

Then tell `VTKWriter` to use it:

```python
writer = VTKWriter(writers=["spheres", "domain", "energy"])
```

By default it will use all the registered writers.

### 3.7 Performance—write once, stream fast

Because `VTKWriter.save` understands *both* batch and trajectory axes you do not have to interleave I/O with computation in a Python loop. 

Compare the two approaches:

```python
# classical – Python loop + I/O every n_every steps
for step in range(steps):
    if step % n_every == 0:
        writer.save(state, system)      # blocks until files are on disk

    state, system = jdem.System.step(state, system)
```

vs.

```python
# JAX-native – build the trajectory inside jit, write once
traj_state, traj_sys = jdem.System.trajectory_rollout(
        state, system,
        n      = steps // n_every,   # number of frames
        stride = n_every             # steps between frames
)[2]                                 # [2] → (trajectory_state, trajectory_system)

writer.save(traj_state, traj_sys, trajectory=True)
```

Advantages of the second pattern

| feature              | inside-loop I/O            | rollout + one save |
|----------------------|----------------------------|--------------------|
| JIT barrier          | every call to `save`       | **none**           |
| Python ↔ device sync | every `step`               | only once          |
| disk latency         | serial, blocks simulation  | overlaps via thread-pool |
| memory footprint     | minimal (single snapshot)  | \(≈\) `steps/n_every` snapshots in RAM |

Why it is fast

1. `trajectory_rollout` is implemented with `jax.lax.scan`, the most
   efficient way to accumulate data inside a `jit`ted section; no Python
   overhead is incurred per step.

2. The generated trajectory is still a pure PyTree of arrays, so
   `writer.save` can dispatch all frames to the thread-pool at once; the
   simulation does **not** wait for the operating system to write each
   file.

3. The only extra cost is RAM: one full snapshot per frame. For large
   scenes you can trade memory for speed by increasing `n_every`
   (fewer frames kept in memory) or by writing batches of, say, 100
   frames at a time.