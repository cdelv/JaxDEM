import jax
import jax.numpy as jnp

import jaxdem as jdem

N = 6000

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

domain = jdem.Domain.create('reflect', state = state, box_size = 3 * jnp.sqrt(N) * jnp.ones(2), anchor = -jnp.ones(2))

system = jdem.System(
    dt = 0.005, 
    domain = domain,
    simulator = jdem.Simulator.create('fgrid'), 
    integrator = jdem.Integrator.create('euler'),
    force_model = jdem.ForceModel.create('spring'),
    grid = jdem.Grid.create('Igrid', state = state, domain = domain, cell_size = 4)
)

writer = jdem.VTKWriter()
writer.save(state, system)

for j in range(500):
    state, system = jdem.System.step(state, system, steps = 50)
    writer.save(state, system)