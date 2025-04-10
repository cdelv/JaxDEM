import jax
device = 'gpu'
jax.config.update('jax_platform_name', device)

import time 
import jax.numpy as jnp

import jaxdem as jdem

def report(state, steps, end, start, batches = 1):
    fps = steps/(end - start)
    print(f"device: {device}")
    print(f"batches: {batches}")
    print(f"N: {state.N} x {batches} = {state.N*batches}")
    print(f"steps: {steps} x {batches} = {steps*batches}")
    print(f"fps: {fps:.1f} x {batches} = {batches*fps:.1f}")
    print(f"performance: {batches*state.N*fps:.2e}")
    print(f"time: {(end - start):.2f} s")
    print()


def initial_conditions(N: int, r: float = 1.0, polydispersity: float = 1.0, seed: int = 0, shuffle: bool = False):
    cols = jnp.ceil(jnp.sqrt(N)).astype(int)
    row = jax.lax.iota(int, N)//cols 
    col = jnp.mod(jax.lax.iota(int, N), cols)
    x = col * 2.0 * polydispersity * r + polydispersity * r
    y = row * 2.0 * polydispersity * r + polydispersity * r
    pos = jnp.column_stack((x, y))
    key = jax.random.PRNGKey(seed)
    vel = -jax.random.uniform(key, shape=(N, 2), dtype=float, minval=0.0, maxval=2.0)
    key, subkey = jax.random.split(key)
    rad = jax.random.uniform(key, shape=(N,), dtype=float, minval=r, maxval=r*polydispersity)
    if shuffle:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)
        pos = pos[perm]
        rad = rad[perm]
    return pos, rad, vel

N = 6000
pos, rad, vel = initial_conditions(N, polydispersity = 1.5)
state = jdem.State.create(dim = 2, pos = pos, vel = vel, rad = rad)

system = jdem.System(
    dt = 0.005, 
    domain = jdem.Domain.create('reflect', dim = state.dim, box_size = 3*jnp.sqrt(N)*jnp.ones(2), anchor=-jnp.ones(2)),
    simulator = jdem.Simulator.create('Igrid'), 
    integrator = jdem.Integrator.create('euler'),
    force_model = jdem.ForceModel.create('spring'),
    grid = jdem.Grid.create('Igrid')
)
cell_capacity, n_neighbors = system.grid.estimate_occupancy(state, system)
system.grid = system.grid.build(state, system, cell_capacity = cell_capacity.item(), n_neighbors = n_neighbors.item())

writer = jdem.VTKWriter(empty = True)
writer.save(state, system)

start = time.time()
for j in range(500):
    state, system = jdem.System.step(state, system, steps = 50)
    writer.save(state, system)

state.pos.block_until_ready()
end = time.time()
report(state, 500*50, end, start)