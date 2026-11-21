# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Rigid, elastic and viscoplastic grains under the same load
==============================================================

Compare JaxDEM's three single-grain representations under the exact same
initial 12-node outline, total mass, and force-controlled loading--unloading
cycle:

1. **Rigid clump** (``clump_id=0``) — carries the load with zero deformation.
2. **Elastic deformable particle** (``"deformable_particle_model"``) —
   deforms under load and recovers its reference shape after unloading.
3. **Viscoplastic deformable particle** (``"PlasticDeformableParticleModel"``)
   — relaxes its reference edge lengths toward the current lengths with
   timescale :math:`\tau_s` via
   :math:`L_{e,0}^{n+1} = L_e^{n+1} + (L_{e,0}^n - L_e^{n+1}) e^{-\Delta t / \tau_s}`,
   leaving a permanent residual deformation after unloading.

Requires Matplotlib.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem

# %%
# 1) Common 12-Node Geometry and Balanced Force-Controlled Load
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Perimeter segments plus diametral struts resist flattening; linear drag
# damps oscillations (note that viscous drag also adds slight rate-dependent
# hysteresis even in the purely elastic grain).

N = 12
DT = 0.002
FRAME_DT = 0.02
N_FRAMES = 400
PEAK_LOAD = 3.0
DRAG = 2.0
TAU_S = 0.8

angles = jnp.arange(N) * (2.0 * jnp.pi / N)
vertices = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
indices = jnp.arange(N)
elements = jnp.stack((indices, jnp.roll(indices, -1)), axis=-1)
struts = jnp.stack((jnp.arange(N // 2), jnp.arange(N // 2) + N // 2), axis=-1)
edges = jnp.concatenate((elements, struts))
reference_lengths = jnp.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=-1)

load_pattern = jnp.stack(
    (-jnp.cos(angles) / jnp.sum(jnp.maximum(jnp.cos(angles), 0.0)), jnp.zeros(N)),
    axis=-1,
)


def applied_load(time: jax.Array) -> jax.Array:
    """Smooth ramp up over [0, 1], hold on [1, 3], smooth ramp down over [3, 4]."""
    ramp_up = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(time, 0.0, 1.0)))
    ramp_down = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(time - 3.0, 0.0, 1.0)))
    return PEAK_LOAD * ramp_up * ramp_down


def loading_and_drag(pos: jax.Array, state: jdem.State, system: jdem.System):
    del pos
    return applied_load(system.time) * load_pattern - DRAG * state.vel / N, jnp.zeros_like(
        state.torque
    )


# %%
# 2) Build the Three Constitutive Representations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Notice the key state-layout difference between rigid clumps and deformable
# particles:
#
# * **Rigid clump:** every member shares the COM ``pos=0``, stores its body
#   offset in ``pos_p=vertices``, shares ``clump_id=0``, and repeats the
#   **total** body mass ``1.0`` on every slot.
# * **Deformable particle:** each node is an independent vertex at
#   ``pos=vertices`` with **nodal** mass ``1 / N`` and ``bond_id`` connectivity.


def make_experiment(kind: str) -> tuple[jdem.State, jdem.System]:
    if kind == "rigid":
        state = jdem.State.create(
            pos=jnp.zeros_like(vertices),
            pos_p=vertices,
            clump_id=jnp.zeros(N, dtype=int),
            mass=jnp.ones(N),
            inertia=jnp.ones((N, 1)),
            rad=jnp.full(N, 0.08),
        )
        model = None
    else:
        state = jdem.State.create(
            pos=vertices,
            mass=jnp.full(N, 1.0 / N),
            rad=jnp.full(N, 0.08),
            bond_id=jnp.zeros(N, dtype=int),
        )
        model = jdem.BondedForceModel.create(
            "deformable_particle_model"
            if kind == "elastic"
            else "PlasticDeformableParticleModel",
            vertices=vertices,
            elements=elements,
            edges=edges,
            element_adjacency=elements,
            el=10.0,
            ec=10.0,
            eb=0.1,
            **({"tau_s": TAU_S} if kind == "viscoplastic" else {}),
        )
    system = jdem.System.create(
        state=state,
        dt=DT,
        bonded_force_model=model,
        collider_type=None,
        rotation_integrator_type=None,
        force_manager_kw={"force_functions": (loading_and_drag,)},
    )
    return jdem.System.initialize(state, system)


# %%
# 3) Roll Out All Three Models and Measure Compression & Internal Memory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Record 400 frames via :meth:`~jaxdem.system.System.trajectory_rollout`
# (4,000 physics steps = 8 time units: 4 loaded, 4 unloaded recovery):

kinds = ("rigid", "elastic", "viscoplastic")
histories = {}
scene_states, scene_systems = {}, {}
substeps = round(FRAME_DT / DT)

print("Model          peak compression   final compression   RMS ref-length change")
for kind in kinds:
    st0, sys0 = make_experiment(kind)
    _, _, (traj_s, traj_sys) = jdem.System.trajectory_rollout(
        st0, sys0, n=N_FRAMES, stride=substeps
    )
    # Keep a 40-frame subsample for interactive 3D rendering.
    scene_states[kind] = jax.tree.map(lambda x: x[::10], traj_s)
    scene_systems[kind] = jax.tree.map(lambda x: x[::10], traj_sys)

    pos_np = np.asarray(traj_s.pos)
    times = np.asarray(traj_sys.time)
    strain = 1.0 - (pos_np[:, :, 0].max(axis=1) - pos_np[:, :, 0].min(axis=1)) / 2.0
    if traj_sys.bonded_force_model is not None:
        ref_len = np.asarray(traj_sys.bonded_force_model.initial_edge_lengths)
        ref_change = np.sqrt(np.mean((ref_len - np.asarray(reference_lengths)) ** 2, axis=-1))
    else:
        ref_change = np.zeros_like(times)

    histories[kind] = (times, strain, ref_change)
    print(f"{kind:13s}  {strain.max():16.5f}  {strain[-1]:17.5f}  {ref_change[-1]:21.5f}")

# %%
# Compare Shape Compression, Load--Response Curves, and Reference Flow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

colors = {"rigid": "#52657c", "elastic": "#258685", "viscoplastic": "#d07a28"}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
for kind in kinds:
    times, strain, ref_change = histories[kind]
    axes[0].plot(times, 100 * strain, color=colors[kind], label=kind)
    axes[1].plot(100 * strain, applied_load(times), color=colors[kind], label=kind)
    axes[2].plot(times, ref_change, color=colors[kind], label=kind)

axes[0].axvspan(0, 4, color="0.5", alpha=0.08, label="load applied")
axes[0].set(xlabel="Time", ylabel="Horizontal compression (%)", title="Creep and recovery")
axes[0].legend(fontsize=8)
axes[1].set(
    xlabel="Horizontal compression (%)", ylabel="Load P", title="Force-controlled response"
)
axes[2].set(xlabel="Time", ylabel="RMS reference-length change", title="Viscoplastic memory")
for ax in axes:
    ax.grid(alpha=0.2)
fig.suptitle("Same outline, mass, and applied load; different internal mechanics", fontsize=13)
plt.show()
