<div class="github-only">
    <a href="https://cdelv.github.io/JaxDEM">
      <img src="https://img.shields.io/badge/Documentation-blue?style=flat&logo=readthedocs&logoColor=white" alt="Documentation Sticker">
    </a>
    <!-- Convert the Markdown badge to HTML <img> and <a> tags -->
    <a href="https://github.com/cdelv/JaxDEM/actions/workflows/docs.yml">
      <img src="https://img.shields.io/github/actions/workflow/status/cdelv/JaxDEM/docs.yml?branch=main&label=Docs%20Status&logo=github&style=flat-square" alt="Docs Build Status">
    </a>
</div>

----

JaxDEM is a lightweight, fully [JAX](https://docs.jax.dev/en/latest/)-compatible Python library that empowers researchers and engineers to easily perform high-performance Discrete Element Method (DEM) simulations in 2D and 3D. Every simulation component is written with pure [JAX](https://docs.jax.dev/en/latest/) arrays so that you can:

*   JIT-compile the entire solver.
*   Run thousands of simulations in parallel with `vmap`.
*   Collect trajectories with `jax.lax.scan` to avoid interrupting the simulation for I/O operations.
*   The provided VTK writer understands when you pass it a batched simulation state or a trajectory, and saves every VTK file concurrently.
*   Ship the computation seamlessly to CPU/GPU/TPU.
*   Interface easily with ML workloads.
*   Keep the codebase short, hackable, and fun.

Whether exploring granular materials, designing new manufacturing processes, working on molecular dynamics, or robotics, JaxDEM provides a robust and easily extendable framework to bring your simulations to life.

## Example

A minimal simulation with I/O output might look like this:

```python
import jaxdem as jdem
from jaxdem.utils import grid_state
import jax.numpy as jnp

state = grid_state(
    n_per_axis=(10, 10, 10), spacing=0.5, radius=0.1
)  # Initialize particles arranged in a grid
system = jdem.System.create(
    state.shape,
    domain_type="reflect",
    domain_kw=dict(box_size=20.0 * jnp.ones(state.dim)),
)
steps = 1000
n_every = 10
writer = jdem.VTKWriter(save_every=n_every)

for step in range(steps):
    writer.save(state, system)  # does not block until files are on disk
    state, system = jdem.System.step(state, system)

writer.block_until_ready()
```

However, there is an even simpler way! You can accumulate the trajectory with `jax.lax.scan`. As `VTKWriter` understands batch and trajectory axes, you do not have to interleave I/O with computation in a Python loop.

No need to complicate yourself with `scan`; we already did it for you:

```python
state, system, (traj_state, traj_sys) = system.trajectory_rollout(
    state,
    system,
    n=steps // n_every,  # number of frames
    stride=n_every,  # steps between frames
)

writer.n_every = 1
writer.save(
    traj_state, traj_sys, trajectory=True
)  # does not block until files are on disk
```

### Advantages of the second pattern

| Feature              | Inside-loop I/O            | Rollout + One Save       |
| :------------------  | :------------------------- | :----------------------- |
| I/O Barrier          | **None**                   | **None**                 |
| Python ↔ Device Sync | Every `save`               | Only once                |
| Memory Footprint     | Single snapshot            | `n` snapshots in RAM     |

### Why is it fast?

1.  `trajectory_rollout` is implemented with `jax.lax.scan`, the most efficient way to accumulate data inside a JIT-compiled section; no Python overhead is incurred per step.

2.  The generated trajectory is still a pure PyTree of arrays, so
    `writer.save` can simultaneously dispatch all frames to the thread-pool.

3.  The only extra cost is RAM. For large scenes, you can trade memory for speed by increasing `n_every`
    (fewer frames kept in memory) or by writing batches of, say, 100
    frames at a time.

<div class="github-only">
    For more details, visit the <a href="https://cdelv.github.io/JaxDEM">Documentation</a>.
</div>
