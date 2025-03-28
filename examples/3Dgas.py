import jax
import jax.numpy as jnp
from numpy import empty

import jaxdem as jdem
import time


def report(state, steps, end, start):
    fps = steps / (end - start)
    print(f"N: {state.N}")
    print(f"steps: {steps}")
    print(f"time: {(end - start):.2f} s")
    print(f"fps: {fps:.1f}")
    print(f"performance: {state.N * fps:.2e}")
    print()

def initial_conditions(N: int, r: float = 1.0, polydispersity: float = 1.0, seed: int = 0, shuffle: bool = False):
    side = int(jnp.ceil(N ** (1/3)))
    z = jax.lax.iota(int, N) // (side * side)
    y = (jax.lax.iota(int, N) // side) % side
    x = jax.lax.iota(int, N) % side

    spacing = 2.0 * polydispersity * r
    x = x * spacing + polydispersity * r
    y = y * spacing + polydispersity * r
    z = z * spacing + polydispersity * r

    pos = jnp.column_stack((x, y, z))
    key = jax.random.PRNGKey(seed)
    vel = -jax.random.uniform(key, shape=(N, 3), dtype=float, minval=0.0, maxval=2.0)
    key, subkey = jax.random.split(key)
    rad = jax.random.uniform(key, shape=(N,), dtype=float, minval=r, maxval=r * polydispersity)

    if shuffle:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        pos = pos[perm]
        rad = rad[perm]
    
    return pos, rad, vel

# --- Setup ---
N = 600
pos, rad, vel = initial_conditions(N, polydispersity=1.5)
state = jdem.State.create(dim=3, pos=pos, vel=vel, rad=rad)

L = 4 * N**(1/3)  # domain length in each direction
system = jdem.System(
    dt=0.005,
    domain=jdem.Domain.create('reflect', box_size= L * jnp.ones(3), anchor=-jnp.ones(3)),
    simulator=jdem.Simulator.create('Igrid'),
    integrator=jdem.Integrator.create('euler'),
    force_model=jdem.ForceModel.create('spring'),
    grid=jdem.Grid.create('Igrid')
)

cell_capacity, n_neighbors = system.grid.estimate_ocupancy(state, system)
system.grid = system.grid.build(state, system, cell_capacity=cell_capacity.item(), n_neighbors=n_neighbors.item())
print(system.grid)

writer = jdem.VTKWriter(empty=True)
writer.save(state, system)

# --- Run Simulation ---
start = time.time()
for j in range(500):
    state, system = jdem.System.step(state, system, steps=50)
    writer.save(state, system)

state.pos.block_until_ready()
end = time.time()
report(state, 500 * 50, end, start)
