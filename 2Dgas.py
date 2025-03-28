import jax
import jax.numpy as jnp
from numpy import empty

import jaxdem as jdem

import time 

N = 6000

def report(state, steps, end, start):
    fps = steps/(end - start)
    print(f"N: {state.N}")
    print(f"steps: {steps}")
    print(f"time: {(end - start):.2f} s")
    print(f"fps: {fps:.1f}")
    print(f"performance: {state.N*fps:.2e}")
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

pos, rad, vel = initial_conditions(N, polydispersity = 1.5)

state = jdem.State.create(dim = 2, pos = pos, vel = vel, rad = rad)

domain = jdem.Domain.create('reflect', box_size = 3 * jnp.sqrt(N) * jnp.ones(2), anchor = -jnp.ones(2))

system = jdem.System(
    dt = 0.005, 
    domain = domain,
    simulator = jdem.Simulator.create('Igrid'), 
    integrator = jdem.Integrator.create('euler'),
    force_model = jdem.ForceModel.create('spring'),
    grid = jdem.Grid.create('Igrid')
)
cell_capacity, n_neighbors = system.grid.estimate_ocupancy(state, system)
print(cell_capacity, n_neighbors)

system.grid = system.grid.build(state, system, cell_capacity = 3, n_neighbors = 1)

writer = jdem.VTKWriter(empty = True)
writer.save(state, system)

start = time.time()
state, system = jdem.System.step(state, system, steps = 500)
state.pos.block_until_ready()
end = time.time()
report(state, 500, end, start)

exit()
start = time.time()

with jax.disable_jit():
    with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
        for j in range(10):
            state, system = jdem.System.step(state, system, steps = 1)
            state.pos.block_until_ready()
            #writer.save(state, system)

end = time.time()
report(state, 500*50, end, start)