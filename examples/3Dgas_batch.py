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
    
    return jdem.State.create(dim = 3, pos = pos, vel = vel, rad = rad)

def System(state, seed):
    L = 4 * state.N**(1/3)
    system = jdem.System(
        dt = 0.005, 
        domain = jdem.Domain.create('reflect', box_size = seed + L*jnp.sqrt(N)*jnp.ones(3), anchor=-jnp.ones(3)),
        simulator = jdem.Simulator.create('Igrid'), 
        integrator = jdem.Integrator.create('euler'),
        force_model = jdem.ForceModel.create('spring'),
        grid = jdem.Grid.create('Igrid')
    )
    # this is no allowed in a vmaped function
    #cell_capacity, n_neighbors = system.grid.estimate_occupancy(state, system)
    #system.grid = system.grid.build(state, system, cell_capacity = cell_capacity.item(), n_neighbors = n_neighbors.item())
    system.grid = system.grid.build(state, system, cell_capacity = 7, n_neighbors = 1)
    return system

N = 6000
batches = 4
seed = jax.lax.iota(int, batches)
state = jax.vmap(initial_conditions, (None, 0))(N, seed)
system = jax.vmap(System)(state, seed)

writer = jdem.VTKWriter(empty = True)
writer.save(state, system)


start = time.time()
for j in range(500):
    state, system = jax.vmap(jdem.System.step, (0, 0, None))(state, system, 50)
    writer.save(state, system)

state.pos.block_until_ready()
end = time.time()
report(state, 500*50, end, start, batches)
