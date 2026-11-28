# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Save selected quantities on a pseudo-logarithmic schedule
============================================================

Combine :func:`~jaxdem.utils.rollout_schedules.make_save_steps_pseudolog` with
a custom ``save_fn`` in :meth:`~jaxdem.system.System.trajectory_rollout` to
record only lightweight observables at dense early times and sparse late times,
without storing full ``State``/``System`` snapshots in memory.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jdem
from jaxdem.utils.thermal import compute_translational_kinetic_energy

# %%
# 1) Create a Periodic 2D Grid with Random Initial Velocities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

state0 = jdem.utils.grid_state(n_per_axis=(4, 4), spacing=1.2, radius=0.5)
state0.vel = 0.2 * jax.random.normal(jax.random.key(0), state0.vel.shape)
system0 = jdem.System.create(
    state=state0,
    dt=0.01,
    rotation_integrator_type=None,
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(2, 4.8)},
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, poisson=0.3, density=1.0)]
    ),
)
state0, system0 = jdem.System.initialize(state0, system0)

# %%
# 2) Build Pseudolog Strides and a Custom ``save_fn``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setting ``include_step0=True`` with ``np.diff(save_steps, prepend=0)`` puts a
# leading ``0`` stride first so step 0 is recorded before stepping.
# ``save_fn(state, system)`` can return any JAX pytree (a single array or a
# dict of observables), keeping peak memory small:

num_steps = 2_000
save_steps = jdem.utils.make_save_steps_pseudolog(
    num_steps=num_steps,
    reset_save_decade=1_000,
    min_save_decade=10,
    decade=10,
    include_step0=True,
)
deltas = jnp.asarray(np.diff(save_steps, prepend=0))


def save_observables(st: jdem.State, sys: jdem.System) -> dict[str, jax.Array]:
    return {
        "time": sys.time,
        "pos": st.pos_c,
        "ke": compute_translational_kinetic_energy(st),
    }


state, system, saved = jdem.System.trajectory_rollout(
    state0, system0, strides=deltas, save_fn=save_observables
)
system.check_overflow()
print("Saved step indices (first 12):", save_steps[:12])
print("Saved pos shape (T, N, dim):", saved["pos"].shape)
print("Saved KE shape:", saved["ke"].shape, "| final step:", int(system.step_count))

# %%
# 3) Compare Storage and Runtime Against Constant-Stride Saving
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Warm up each compiled shape once, then call ``block_until_ready()`` so JAX's
# asynchronous dispatch finishes before stopping the timer:

min_stride = int(np.min(np.diff(save_steps)))
n_dense = int(save_steps[-1]) // min_stride


def run_sparse():
    return jdem.System.trajectory_rollout(
        state0, system0, strides=deltas, save_fn=lambda st, _: st.pos_c
    )[2]


def run_dense():
    return jdem.System.trajectory_rollout(
        state0, system0, n=n_dense, stride=min_stride, save_fn=lambda st, _: st.pos_c
    )[2]


for run in (run_sparse, run_dense):
    run().block_until_ready()

for label, run in (("pseudo-log", run_sparse), ("constant stride", run_dense)):
    t0 = time.perf_counter()
    pos = run()
    pos.block_until_ready()
    print(
        f"{label:15s}: {pos.shape[0]:3d} frames, {pos.nbytes:6d} bytes, "
        f"{time.perf_counter() - t0:.4f} s"
    )
