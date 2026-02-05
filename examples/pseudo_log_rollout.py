r"""
Pseudo-log trajectory rollout
----------------------------------------

This example demonstrates how to roll out dynamics while saving only a subset of
quantities at a pseudo-logarithmic set of step indices.


"""

# %%

import time
import jax.numpy as jnp
import jaxdem as jdem
from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration


def build_microstate(
    *,
    n_particles: int,
    packing_fraction: float,
    space_dim: int,
    config_seed: int,
):
    mass = 1.0
    e_int = 1.0
    dt = 1e-2
    if space_dim == 2:
        cr = [0.5, 0.5]
        sr = [1.0, 1.4]
    else:
        cr = [1.0]
        sr = [1.0]
    particle_radii = jdem.utils.dispersity.get_polydisperse_radii(n_particles, cr, sr)
    pos, box_size = random_sphere_configuration(
        particle_radii, packing_fraction, space_dim, config_seed
    )
    microstate = jdem.State.create(
        pos=pos, rad=particle_radii, mass=jnp.ones(n_particles) * mass
    )
    mats = [jdem.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jdem.MaterialMatchmaker.create("harmonic")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)
    microsystem = jdem.System.create(
        state_shape=microstate.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(box_size=box_size),
    )
    return microstate, microsystem


def save_positions(state: jdem.State, _system: jdem.System):
    return state.pos_c


# %%
# Setup

N = 100
phi = 0.7
dim = 2
seed = 0

num_steps = 100_000
reset_save_decade = 10_000
min_save_decade = 100
block_size = 5

save_steps = jdem.utils.make_save_steps_pseudolog(
    num_steps=num_steps,
    reset_save_decade=reset_save_decade,
    min_save_decade=min_save_decade,
    decade=10,
    include_step0=True,
)

# %%
# Rollout

st, sys = build_microstate(
    n_particles=N, packing_fraction=phi, space_dim=dim, config_seed=seed
)
save_steps_jax = jnp.asarray(save_steps)
st, sys, pos_traj = jdem.System.trajectory_rollout_at_steps(
    st, sys, save_steps=save_steps_jax, save_fn=save_positions, block=block_size
)

print("num saved frames:", pos_traj.shape[0])
print("saved positions shape:", pos_traj.shape)
print("final step:", int(sys.step_count))


# %%
# Speed comparison

min_stride = int(jnp.min(save_steps_jax[1:] - save_steps_jax[:-1]))
n_steps = int(save_steps[-1])

state_w, system_w = build_microstate(
    n_particles=N, packing_fraction=phi, space_dim=dim, config_seed=seed
)
_, _, out_w = jdem.System.trajectory_rollout_at_steps(
    state_w,
    system_w,
    save_steps=save_steps_jax,
    # save_fn=save_positions,
    block=block_size,
)
# out_w.block_until_ready()
out_w[0].pos.block_until_ready()

state_w, system_w = build_microstate(
    n_particles=N, packing_fraction=phi, space_dim=dim, config_seed=seed
)
_, _, (traj_w, _) = jdem.System.trajectory_rollout(
    state_w, system_w, n=n_steps // min_stride, stride=min_stride
)
traj_w.pos_c.block_until_ready()

state_t, system_t = build_microstate(
    n_particles=N, packing_fraction=phi, space_dim=dim, config_seed=seed
)
t0 = time.perf_counter()
_, _, out = jdem.System.trajectory_rollout_at_steps(
    state_t,
    system_t,
    save_steps=save_steps_jax,
    # save_fn=save_positions,
    block=block_size,
)
# out.block_until_ready()
out[0].pos.block_until_ready()
t_blocks = time.perf_counter() - t0

state_t, system_t = build_microstate(
    n_particles=N, packing_fraction=phi, space_dim=dim, config_seed=seed
)
t0 = time.perf_counter()
_, _, (traj, _) = jdem.System.trajectory_rollout(
    state_t, system_t, n=n_steps // min_stride, stride=min_stride
)
traj.pos_c.block_until_ready()
t_dense = time.perf_counter() - t0

print("min stride:", min_stride)
print("dense frames:", n_steps // min_stride)
print("pseudo-log frames:", int(pos_traj.shape[0]))
print("time pseudo-log (blocks):", t_blocks)
print("time dense trajectory_rollout:", t_dense)

