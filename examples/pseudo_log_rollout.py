"""Save selected quantities on a pseudo-logarithmic schedule
============================================================

Use ``save_fn`` to retain only positions while sampling densely near the
beginning of each time window. The full State and System remain available as
the final return values, without being stored at every saved frame.

This small 16-disk, 2,000-step run demonstrates scheduling and memory use.
The timing comparison warms compilation and synchronizes results;
its timings depend on the device and should not be read as a general benchmark.
"""

# %%
import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jdem

# %%
# A regular grid avoids a packing minimization before this rollout lesson.
# Unit masses, diameter 1 and stiffness 1 define reduced units. Random initial
# velocities make the trajectory nontrivial; the periodic box has side 4.8.
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
# Saving a trajectory has two independent choices: **when** to keep a frame
# and **what** that frame contains. The schedule changes storage cadence; the
# solver still executes every intervening integration step at the same dt.
# For ordered absolute step indices :math:`s_0,\ldots,s_{T-1}`, the rollout
# expects intervals
#
# .. math::
#
#     \Delta s_0=s_0,\qquad \Delta s_j=s_j-s_{j-1},\qquad
#     \sum_j\Delta s_j=s_{T-1}.
#
# A zero first interval records the initialized state. With a nonzero first
# interval, the first saved frame follows that many physical steps instead.
# Within each reset window, ``min_save_decade`` sets the smallest save spacing;
# the spacing grows by ``decade`` in successively later regions. Resetting the
# window restores dense sampling for a repeated transient. Inspect the returned
# integer indices when choosing the final duration: a requested endpoint that
# is not on the schedule need not be among the saved indices.
#
# Absolute save steps become per-frame strides through ``diff``. The leading
# zero records the initial state without advancing it. Periodic resets restore
# dense sampling at the beginning of each 1,000-step window.
num_steps = 2_000
save_steps = jdem.utils.make_save_steps_pseudolog(
    num_steps=num_steps,
    reset_save_decade=1_000,
    min_save_decade=10,
    decade=10,
    include_step0=True,
)
deltas = jnp.asarray(np.diff(save_steps, prepend=0))


# %%
# ``trajectory_rollout`` carries the complete evolving State/System internally.
# Its ``save_fn`` selects the per-frame payload, so returning ``state.pos_c``
# avoids storing forces, orientations, and search caches at every save point.
# The default callback instead records both State and System PyTrees. Selecting
# positions only *after* a full rollout would not provide this memory saving.
#
# Here positions have shape ``(T, N, dim)``. For a batched input, a leading batch
# axis is retained after the trajectory axis, giving ``(T, B, N, dim)``.
# Position storage is approximately
#
# .. math::
#
#     M_{\rm pos}=T\,N\,d\,b
#
# bytes, with ``b`` bytes per scalar (and another factor ``B`` for a batch).
# This measures the saved array, not peak compiler or integration memory.


def save_positions(state, system):
    return state.pos_c


state, system, pos_traj = jdem.System.trajectory_rollout(
    state0, system0, strides=deltas, save_fn=save_positions
)
system.check_overflow()
np.testing.assert_allclose(pos_traj[0], state0.pos_c)
assert int(system.step_count) == int(save_steps[-1])
print("Saved step indices (first 12):", save_steps[:12])
print("Saved position shape (frames, particles, dim):", pos_traj.shape)
print("Final step:", int(system.step_count))
print("Position storage (bytes):", pos_traj.nbytes)

# %%
# Compare with a constant-stride recording of the same integration interval.
# Both methods save only positions. The first calls compile the two different
# scan shapes; timing the second calls separates execution from compilation.
# Reusing the initialized inputs starts each run from the same microstate.
min_stride = int(np.min(np.diff(save_steps)))
n_dense = int(save_steps[-1]) // min_stride


def run_sparse():
    return jdem.System.trajectory_rollout(
        state0, system0, strides=deltas, save_fn=save_positions
    )[2]


def run_dense():
    return jdem.System.trajectory_rollout(
        state0, system0, n=n_dense, stride=min_stride, save_fn=save_positions
    )[2]


# %%
# JAX dispatch can return before the device has completed the calculation.
# ``block_until_ready`` places the timing endpoint after actual completion.
# Each scan shape is warmed independently, so the timed comparison omits its
# first compilation. Sparse output reduces recorded frames and transfer/storage
# costs, but it does not reduce the number of force evaluations needed to reach
# the final physical step. Any speedup therefore depends on the workload.
for run in (run_sparse, run_dense):
    run().block_until_ready()

for label, run in (("pseudo-log", run_sparse), ("constant stride", run_dense)):
    start = time.perf_counter()
    positions = run()
    positions.block_until_ready()
    print(
        f"{label}: {positions.shape[0]} frames, {positions.nbytes} bytes, "
        f"{time.perf_counter() - start:.4f} seconds after compilation"
    )
