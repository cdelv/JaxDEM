# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Run an ensemble of granular compaction experiments
======================================================

Relax twelve independent 2D bidisperse disk packings (three packing fractions
across four random seeds) in a single compiled ``jax.vmap`` call, and measure
how contact virial pressure
:math:`P = \frac{1}{2 L^2} \sum_{i<j} \mathbf{r}_{ij} \cdot \mathbf{F}_{ij}`
varies across preparations.

This illustrates JaxDEM's core ensemble workflow: ``State.stack`` +
``System.stack`` + ``jax.jit(jax.vmap(...))`` over heterogeneous box sizes.
Requires Matplotlib.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxdem as jdem
from jaxdem.utils.contacts import compute_contact_pressure

# %%
# 1) Stack Matching Replicas Across Density and Seed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# All replicas share particle count, force law, and minimizer configuration,
# while ``box_size`` and initial positions vary across the batch. For much
# larger ensembles (hundreds or thousands of replicas), generate states inside
# ``jax.vmap`` directly rather than in a Python loop.

N = 16
packing_fractions = np.array([0.74, 0.82, 0.90])
seeds = range(4)
force_tol = 1e-7
radii = jnp.array([0.5, 0.7] * (N // 2))
areas = jnp.pi * radii**2
materials = jdem.MaterialTable.from_materials(
    [jdem.Material.create("elastic", young=1.0, poisson=0.3, density=1.0)]
)

states, systems = [], []
for phi in packing_fractions:
    box_len = jnp.sqrt(areas.sum() / phi)
    for seed in seeds:
        st = jdem.State.create(
            pos=box_len * jax.random.uniform(jax.random.key(seed), (N, 2)),
            rad=radii,
            mass=jnp.ones(N),
            volume=areas,
        )
        sys = jdem.System.create(
            state=st,
            domain_type="periodic",
            domain_kw={"box_size": jnp.full(2, box_len)},
            collider_type="naive",
            rotation_integrator_type=None,
            mat_table=materials,
            minimizer=jdem.minimizers.fire,
            minimizer_kw={"dt": 0.05},
        )
        states.append(st)
        systems.append(sys)

batch_state = jdem.State.stack(states)
batch_system = jdem.System.stack(systems)
print("Batched position shape (B, N, dim):", batch_state.pos.shape)

# %%
# 2) Relax All Twelve Packings in One ``vmap`` Call
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Note that batched ``System.minimize`` loops until the slowest active replica
# converges. Afterward, slice any individual replica out of the batched PyTree
# with ``jax.tree.map(lambda x: x[idx], ...)``:


def relax_one(st: jdem.State, sys: jdem.System):
    return jdem.System.minimize(
        st, sys, max_steps=10_000, force_tol=force_tol, torque_tol=force_tol
    )


result = jax.jit(jax.vmap(relax_one))(batch_state, batch_system)
jax.block_until_ready(result)
if not np.all(np.asarray(result.converged)):
    raise RuntimeError(f"Some packings failed to relax: {result.info}")

pressures = np.asarray(
    jax.jit(jax.vmap(lambda st, sys: compute_contact_pressure(st, sys)[2]))(
        result.state, result.system
    )
).reshape(len(packing_fractions), len(seeds))

mean_p = pressures.mean(axis=1)
se_p = pressures.std(axis=1, ddof=1) / np.sqrt(len(seeds))
print("phi    mean P       SE(P)       individual pressures")
for phi, mean, err, row in zip(packing_fractions, mean_p, se_p, pressures, strict=True):
    print(f"{phi:.2f}   {mean:.4e}   {err:.2e}   {np.round(row, 4)}")

# Extract a single replica (e.g. phi=0.90, seed=0) for downstream inspection:
dense_state = jax.tree.map(lambda x: x[8], result.state)
dense_system = jax.tree.map(lambda x: x[8], result.system)
print("Extracted single replica pos shape:", dense_state.pos.shape, "| box:", dense_system.domain.box_size)

# %%
# 3) Plot Preparation Spread Across Packing Fractions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fig, ax = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
for col, seed in enumerate(seeds):
    ax.scatter(
        packing_fractions + (col - 1.5) * 0.0025,
        pressures[:, col],
        s=40,
        alpha=0.8,
        label=f"seed {seed}",
    )
ax.errorbar(
    packing_fractions,
    mean_p,
    yerr=se_p,
    color="black",
    fmt="D-",
    capsize=5,
    label="mean ± 1 SE",
)
ax.set(
    xlabel=r"Nominal area fraction $\phi$",
    ylabel="Contact pressure P",
    title="Preparation dependence across 12 vectorized minimizations",
)
ax.legend(fontsize=8, ncols=3)
ax.grid(alpha=0.2)
plt.show()
