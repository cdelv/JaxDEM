import jax, jax.numpy as jnp
import jaxdem as jdem
from jaxdem.utils import grid_state, random_state

import time

# 1. 10 × 10 grid in 2-D
state = grid_state(n_per_axis=(10, 10, 10), spacing=0.5, radius=0.1)
system = jdem.System.create(state.dim, domain_type="reflect", domain_kw={"box_size": 20.0 * jnp.ones(state.dim)})
writer = jdem.VTKWriter()

frames = 100
stride = 100
writer.save(state, system)
start = time.time()
state, system, (traj_state, traj_sys) = jdem.System.trajectory_rollout(state, system, frames, stride) 
state.pos.block_until_ready()
end = time.time()                                
writer.save(traj_state, traj_sys, trajectory=True)

print(f"Performance: {state.N * (frames * stride)/(end - start):1e}")