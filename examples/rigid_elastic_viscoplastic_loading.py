r"""Rigid, elastic and viscoplastic grains under the same load
==========================================================

How much of a grain's deformation disappears when its load is removed?
Apply the same pair of opposing, distributed forces to three initially
identical twelve-node outlines: a rigid clump, an elastic deformable
particle, and a viscoplastic deformable particle. Plot their compression,
load--deformation curves and evolving reference lengths.

This is a **force-controlled** experiment. Prescribing the same nonzero
shape strain would be incompatible with a perfectly rigid grain. Here the
rigid grain carries a nonzero, self-balanced load without changing shape;
the deformable grains respond through their internal bonded forces. No
wall contact or overlap law is needed for this single-grain comparison.
The nodes, initial outline, total mass and applied forces are matched.

The small planar framework is a constitutive demonstration, not a
calibrated continuum material or a test of fracture. It uses reduced units
and runs for eight time units, including a four-unit unloaded recovery.

This example additionally requires Matplotlib for plotting:
``python -m pip install matplotlib``.
"""

# %%
# Elastic energy and reference-state flow
# ---------------------------------------
# The elastic and viscoplastic grains use exactly the same edge, enclosed-area
# and bending coefficients. With current edge lengths :math:`L_e`, enclosed
# area :math:`A`, and signed bending angles :math:`\theta_a`, the energy is
#
# .. math::
#
#    E = \frac{el}{2}\sum_e(L_e-L_{e,0})^2
#      + \frac{ec}{2A_0}(A-A_0)^2
#      + \frac{eb}{2}\sum_a w_{b,a}
#          \operatorname{wrap}(\theta_a-\theta_{a,0})^2.
#
# JaxDEM obtains the nodal forces from :math:`-\partial E/\partial\mathbf x_i`.
# The constructor computes :math:`w_b` from the reference polygon. The
# ``PlasticDeformableParticleModel`` changes only the reference edge lengths:
#
# .. math::
#
#    \dot L_{e,0}=\frac{L_e-L_{e,0}}{\tau_s},\qquad
#    L_{e,0}^{n+1}=L_e^{n+1}
#       +(L_{e,0}^{n}-L_e^{n+1})e^{-\Delta t/\tau_s}.
#
# The discrete update is exact while the current geometry is held fixed and
# occurs once per physical step. Area and bending references remain fixed.
# Finite :math:`\tau_s` permits reference flow for any edge extension: this
# model has no yield threshold. Setting :math:`\tau_s=\infty` recovers the
# elastic reference history. A finite-time residual shape demonstrates
# memory; its value also depends on the subsequent recovery time.

from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem

jax.config.update("jax_enable_x64", True)

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
# Perimeter edges plus diametral struts resist flattening and give the
# reference-length relaxation a clear internal deformation to remember.
struts = jnp.stack((jnp.arange(N // 2), jnp.arange(N // 2) + N // 2), axis=-1)
edges = jnp.concatenate((elements, struts))
adjacency = jnp.stack((indices, jnp.roll(indices, -1)), axis=-1)
reference_lengths = jnp.linalg.norm(
    vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=-1
)


# %%
# The same loading protocol
# -------------------------
# The total inward horizontal force on each half of the outline is
# :math:`P(t)`. Distribute it in proportion to the initial horizontal normal:
#
# .. math::
#
#    \mathbf f_i(t)=-\frac{P(t)\cos\alpha_i}{
#       \sum_{\cos\alpha_j>0}\cos\alpha_j}\,\mathbf e_x
#       -\frac{c}{N}\mathbf v_i.
#
# Reflection symmetry of the complete outline and loading pattern balances
# both resultant force and torque. These are dead loads attached to
# corresponding nodes, rather than pressure that follows
# the changing surface normal. A smooth ramp loads the grain over one time
# unit, a two-unit hold allows creep, and another smooth ramp unloads it.
# Linear drag removes oscillations in the deformable runs. It also produces
# finite-rate hysteresis in the elastic run, so loop area alone does not
# isolate reference-state dissipation. More precisely, the plotted load--width
# compression curve is a response curve: width compression is not generally
# work-conjugate to this distributed load under nonaffine deformation, and
# its loop area is not directly dissipated work. The work supplied by the
# prescribed nodal loads is
#
# .. math::
#
#    W_{\rm load}=\sum_i\int\mathbf f_i^{\rm load}(t)
#       \cdot d\mathbf x_i(t),\qquad
#    \mathbf f_i^{\rm load}=P(t)\,\mathbf p_i.
#
# Here :math:`\mathbf p_i` is the dimensionless loading pattern defined above.
# This expression excludes the drag contribution; an energy balance would
# account separately for drag dissipation, kinetic and elastic energy changes,
# and relaxation of the reference state.

load_pattern = jnp.stack(
    (
        -jnp.cos(angles) / jnp.sum(jnp.maximum(jnp.cos(angles), 0.0)),
        jnp.zeros(N),
    ),
    axis=-1,
)


def applied_load(time):
    ramp_up = 0.5 * (1.0 - jnp.cos(jnp.pi * jnp.clip(time, 0.0, 1.0)))
    ramp_down = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip(time - 3.0, 0.0, 1.0)))
    return PEAK_LOAD * ramp_up * ramp_down


def loading_and_drag(pos, state, system):
    del pos
    force = applied_load(system.time) * load_pattern - DRAG * state.vel / N
    return force, jnp.zeros_like(state.torque)


def make_experiment(kind, dt=DT):
    if kind == "rigid":
        # Rigid members share a COM, total mass and planar moment of inertia.
        # Unit-radius point nodes of total mass one have I_zz = 1.
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
            (
                "deformable_particle_model"
                if kind == "elastic"
                else "PlasticDeformableParticleModel"
            ),
            vertices=vertices,
            elements=elements,
            edges=edges,
            element_adjacency=adjacency,
            el=10.0,
            ec=10.0,
            eb=0.1,
            **({"tau_s": TAU_S} if kind == "viscoplastic" else {}),
        )
    system = jdem.System.create(
        state=state,
        dt=dt,
        bonded_force_model=model,
        collider_type="",  # This experiment applies loads directly to nodes.
        rotation_integrator_type="",  # The symmetric loading has zero torque.
        force_manager_kw={"force_functions": (loading_and_drag,)},
    )
    return jdem.System.initialize(state, system)


# %%
# Integrate once, sample compact diagnostics
# ------------------------------------------
# Translational velocity Verlet advances the actual rigid-body or nodal
# equations. The compiled scan stores only 400 frames from 4,000 steps.
# Compression is :math:`\epsilon_x=1-W(t)/W(0)`, where :math:`W` is the
# horizontal span of the node centers, initially two. The reference diagnostic
# :math:`\sqrt{\langle(L_{e,0}(t)-L_{e,0}(0))^2\rangle_e}` distinguishes
# reference flow from a temporarily deformed elastic shape.


@partial(jax.jit, static_argnames=("substeps",))
def run_experiment(state, system, *, substeps):
    def frame(carry, _):
        state, system = jdem.System.step(*carry, n=substeps)
        width = jnp.max(state.pos[:, 0]) - jnp.min(state.pos[:, 0])
        reference_change = jnp.asarray(0.0)
        if system.bonded_force_model is not None:
            reference_change = jnp.sqrt(
                jnp.mean(
                    (system.bonded_force_model.initial_edge_lengths - reference_lengths)
                    ** 2
                )
            )
        diagnostics = (
            state.pos,
            system.time,
            1.0 - width / 2.0,
            reference_change,
            jnp.max(jnp.abs(state.vel)),
        )
        return (state, system), diagnostics

    return jax.lax.scan(frame, (state, system), None, length=N_FRAMES)


kinds = ("rigid", "elastic", "viscoplastic")
histories = {}
for kind in kinds:
    state, system = make_experiment(kind)
    _, history = run_experiment(state, system, substeps=round(FRAME_DT / DT))
    histories[kind] = jax.device_get(history)


# %%
# Checks and timestep refinement
# ------------------------------
# A rigid outline must not change under balanced loading; the elastic
# reference lengths must not evolve. After release the elastic grain nearly
# recovers, while the viscoplastic grain retains a visible shape and reference
# change. Halving dt repeats the *same physical protocol* and checks its final
# compression. Thus the result is more than a plausible-looking trajectory.

np.testing.assert_allclose(jnp.sum(load_pattern, axis=0), 0.0, atol=1e-12)
np.testing.assert_allclose(
    jnp.sum(vertices[:, 0] * load_pattern[:, 1] - vertices[:, 1] * load_pattern[:, 0]),
    0.0,
    atol=1e-12,
)
np.testing.assert_allclose(histories["rigid"][2], 0.0, atol=1e-12)
np.testing.assert_allclose(histories["elastic"][3], 0.0, atol=1e-12)
for kind, (positions, times, strain, reference_change, speed) in histories.items():
    assert np.all(np.isfinite(positions)), f"Nonfinite {kind} trajectory"
    np.testing.assert_allclose(positions.mean(axis=1), 0.0, atol=1e-10)
    assert speed[-1] < 0.01, f"{kind} has not relaxed after unloading"
assert abs(histories["elastic"][2][-1]) < 5e-4
assert histories["viscoplastic"][2][-1] > 0.02
assert histories["viscoplastic"][3][-1] > 0.02

state, system = make_experiment("viscoplastic", dt=DT / 2)
_, refined = run_experiment(state, system, substeps=round(FRAME_DT / (DT / 2)))
refined_strain = float(refined[2][-1])
np.testing.assert_allclose(
    refined_strain, histories["viscoplastic"][2][-1], rtol=0.0, atol=2e-4
)
print("Model          peak compression   final compression   reference change")
for kind in kinds:
    _, _, strain, reference_change, _ = histories[kind]
    print(
        f"{kind:13s}  {strain.max():16.5f}  {strain[-1]:17.5f}"
        f"  {reference_change[-1]:16.5f}"
    )
print(f"Final viscoplastic compression at dt/2: {refined_strain:.5f}")


# %%
# Shape, creep and recovery
# -------------------------
# The solid outlines below are sampled at the end of the load hold
# (:math:`t=3`); dashed colored outlines are sampled after unloaded recovery
# (:math:`t=8`). The rigid load--compression curve is vertical because its
# compliance is zero. The viscoplastic reference continues to relax during
# recovery; the reported residual is at the stated observation time.
# Increase the recovery time, change :math:`\tau_s`, or vary the loading rate
# to distinguish delayed response from an ideal rate-independent plastic law.

colors = {"rigid": "#52657c", "elastic": "#258685", "viscoplastic": "#d07a28"}
fig, axes = plt.subplots(2, 3, figsize=(12, 7.3), layout="constrained")
closed_initial = np.vstack((vertices, vertices[0]))
hold_index = round(3.0 / FRAME_DT) - 1
for ax, kind in zip(axes[0], kinds):
    positions, times, strain, reference_change, speed = histories[kind]
    loaded = np.vstack((positions[hold_index], positions[hold_index, 0]))
    recovered = np.vstack((positions[-1], positions[-1, 0]))
    ax.plot(*closed_initial.T, ":", color="0.65", lw=2, label="initial")
    ax.fill(*loaded.T, color=colors[kind], alpha=0.13)
    ax.plot(*loaded.T, "o-", color=colors[kind], ms=3, label="loaded, t=3")
    ax.plot(*recovered.T, "--", color=colors[kind], lw=2, label="recovered, t=8")
    ax.set(
        title=kind.capitalize(),
        aspect="equal",
        xlim=(-1.25, 1.25),
        ylim=(-1.3, 1.3),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.18)
axes[0, 0].legend(fontsize=8, loc="upper right")

for kind in kinds:
    _, times, strain, reference_change, _ = histories[kind]
    axes[1, 0].plot(times, 100 * strain, color=colors[kind], label=kind)
    axes[1, 1].plot(100 * strain, applied_load(times), color=colors[kind], label=kind)
    axes[1, 2].plot(times, reference_change, color=colors[kind], label=kind)
axes[1, 0].axvspan(0, 4, color="0.5", alpha=0.08, label="load applied")
axes[1, 0].set(
    xlabel="time",
    ylabel="horizontal compression (%)",
    title="Creep and unloaded recovery",
)
axes[1, 0].legend(fontsize=8)
axes[1, 1].set(
    xlabel="horizontal compression (%)",
    ylabel="load on each half, P",
    title="Force-controlled response",
)
axes[1, 2].set(
    xlabel="time",
    ylabel="RMS reference-length change",
    title="Internal state remembers loading",
)
for ax in axes[1]:
    ax.grid(alpha=0.2)
fig.suptitle(
    "Same outline, mass and applied load; different internal mechanics", fontsize=14
)
plt.show()
