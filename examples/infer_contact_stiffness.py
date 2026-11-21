# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Infer contact stiffness from observed trajectories
====================================================

Recover an unknown normal contact stiffness ``k`` from the positions of two
colliding disks by differentiating straight through
:meth:`~jaxdem.system.System.trajectory_rollout` with ``jax.value_and_grad``.

Fit a dimensionless log-stiffness :math:`\theta = \log(k / k_{\mathrm{ref}})`
(guaranteeing :math:`k > 0`) with SciPy's ``L-BFGS-B`` optimizer, check the AD
derivative against finite differences, and validate on a held-out impact speed.
Requires Matplotlib and SciPy.
"""

from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import jaxdem as jdem

# %%
# 1) Build a Differentiable Collision Rollout
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Two key JaxDEM details for reverse-mode automatic differentiation:
#
# * **Update both ``props["young"]`` and ``pair["young_eff"]``** when varying
#   stiffness inside ``jit``: ``SpringForce`` reads the precomputed pair table
#   ``young_eff``, so updating ``props["young"]`` alone leaves forces unchanged.
# * **Use a reverse-AD-compatible collider** (``"naive"``, ``"neighbor_list"``,
#   or static cell lists with ``static=True``). For much longer multi-collision
#   rollouts, wrap chunks with ``jax.checkpoint`` (rematerialization) to cap
#   reverse-mode memory.

radius = 0.5
mass = 1.0
ref_stiffness = 1.0
true_stiffness = 40.0
init_stiffness = 10.0
dt = 0.002
n_frames = 80
save_stride = 5
train_speed = 0.5
val_speed = 0.35
times = dt * save_stride * np.arange(1, n_frames + 1)

initial_state = jdem.State.create(
    pos=jnp.array([[-0.54, 0.0], [0.54, 0.0]]),
    rad=jnp.full(2, radius),
    mass=jnp.full(2, mass),
)
material_table = jdem.MaterialTable.from_materials(
    [
        jdem.Material.create(
            "elastic",
            young=ref_stiffness,
            poisson=0.3,
            density=mass / (np.pi * radius**2),
        )
    ]
)
initial_system = jdem.System.create(
    state=initial_state,
    dt=dt,
    collider_type="naive",
    force_model_type="spring",
    rotation_integrator_type=None,
    mat_table=material_table,
)


@jax.jit
def simulate(log_stiffness: jax.Array, speed: jax.Array) -> jax.Array:
    k = ref_stiffness * jnp.exp(log_stiffness)
    table = replace(
        material_table,
        props={**material_table.props, "young": jnp.full((1,), k)},
        pair={**material_table.pair, "young_eff": jnp.full((1, 1), k)},
    )
    state = replace(initial_state, vel=jnp.array([[speed, 0.0], [-speed, 0.0]]))
    system = replace(initial_system, mat_table=table)
    state, system = jdem.System.initialize(state, system)
    _, _, positions = jdem.System.trajectory_rollout(
        state, system, n=n_frames, stride=save_stride, save_fn=lambda st, _: st.pos
    )
    return positions


true_log_k = jnp.log(true_stiffness / ref_stiffness)
init_log_k = jnp.log(init_stiffness / ref_stiffness)
observed_positions = simulate(true_log_k, jnp.asarray(train_speed))


@jax.jit
def loss(log_stiffness: jax.Array) -> jax.Array:
    pos = simulate(log_stiffness, jnp.asarray(train_speed))
    return jnp.mean(((pos[:, :, 0] - observed_positions[:, :, 0]) / radius) ** 2)


loss_and_grad = jax.jit(jax.value_and_grad(loss))

# %%
# 2) Verify AD Gradient Against Finite Differences
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

check_theta = jnp.log(18.0 / ref_stiffness)
eps = 1e-4
_, ad_grad = loss_and_grad(check_theta)
fd_grad = (loss(check_theta + eps) - loss(check_theta - eps)) / (2.0 * eps)
print(f"Gradient at k=18: AD={float(ad_grad):.8f}, FD={float(fd_grad):.8f}")

# %%
# 3) Fit Stiffness with ``L-BFGS-B`` and Validate on a Held-Out Speed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

iter_loss = [float(loss(init_log_k))]


def scipy_obj(params: np.ndarray):
    val, grad = loss_and_grad(jnp.asarray(params[0]))
    return float(val), np.array([float(grad)])


fit = minimize(
    scipy_obj,
    np.array([float(init_log_k)]),
    jac=True,
    method="L-BFGS-B",
    bounds=[(np.log(5.0 / ref_stiffness), np.log(100.0 / ref_stiffness))],
    callback=lambda p: iter_loss.append(float(loss(jnp.asarray(p[0])))),
    options={"maxiter": 30, "gtol": 1e-10, "ftol": 1e-14},
)
if not fit.success:
    raise RuntimeError(f"Stiffness fit failed: {fit.message}")

fitted_log_k = jnp.asarray(fit.x[0])
fitted_k = ref_stiffness * float(np.exp(fit.x[0]))
print(f"Recovered k = {fitted_k:.6f} (true = {true_stiffness:.1f}) in {fit.nit} iterations")

init_pos = simulate(init_log_k, jnp.asarray(train_speed))
fitted_pos = simulate(fitted_log_k, jnp.asarray(train_speed))
val_target = simulate(true_log_k, jnp.asarray(val_speed))
val_pred = simulate(fitted_log_k, jnp.asarray(val_speed))
val_rmse = float(jnp.sqrt(jnp.mean((val_pred - val_target) ** 2)))
print(f"Held-out impact speed v={val_speed}: position RMSE = {val_rmse:.3e}")

# %%
# Plot Trajectory Fit, Loss Convergence, and Held-Out Prediction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
axes[0].plot(times, init_pos[:, 1, 0], "--", label=f"Initial k={init_stiffness:g}")
axes[0].plot(times, fitted_pos[:, 1, 0], label=f"Fitted k={fitted_k:.2f}")
axes[0].plot(times[::4], observed_positions[::4, 1, 0], "o", ms=3, label="Observed")
axes[0].set(xlabel="Time", ylabel="Right disk x", title="Calibration collision")
axes[0].legend(fontsize=8)

axes[1].semilogy(np.maximum(iter_loss, 1e-18), "o-")
axes[1].set(xlabel="L-BFGS-B iteration", ylabel="Position MSE", title="Fit convergence")

axes[2].plot(times, val_pred[:, 1, 0], label="Prediction")
axes[2].plot(times[::4], val_target[::4, 1, 0], "o", ms=3, label="Held-out target")
axes[2].set(xlabel="Time", ylabel="Right disk x", title=f"Held-out speed v={val_speed}")
axes[2].legend(fontsize=8)

for ax in axes:
    ax.grid(alpha=0.2)
plt.show()
