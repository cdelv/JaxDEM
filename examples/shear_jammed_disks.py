"""Quasistatic shear of a small jammed disk packing
===================================================

An athermal quasistatic (AQS) step first applies an affine strain to particle
centers and periodic images, then minimizes the energy at fixed strain.
There is no physical loading rate or thermal noise in this protocol. The
minimizer's time step is a relaxation parameter, not a measured shear time.

We locate the jamming threshold of eight bidisperse disks and compress a
little above it before shearing. This gives a small finite-overlap solid
rather than a packing exactly at the vanishing-energy threshold. Its
jamming fraction depends strongly on size and seed. Twelve strain steps
illustrate the protocol; they are insufficient to infer a bulk yield stress.

Lengths, unit masses and unit spring stiffness use reduced units. Reported
energy is per disk; contact virial stress is normalized by box area, with
positive diagonal values for compression. Elastic branches and rearrangements
can be studied by extending the run and checking strain-step and force-tolerance
convergence. A rearrangement is not promised for this short deterministic run.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import jaxdem as jdem
from jaxdem.utils.contacts import compute_contact_stress_tensor
from jaxdem.utils.packing_utils import scale_to_packing_fraction

jax.config.update("jax_enable_x64", True)

# %%
# Affine loading followed by energy relaxation
# ---------------------------------------------
# Write positions relative to the domain anchor (zero in this run). A strain
# increment changes the reference box and centers according to
#
# .. math::
#
#     \begin{aligned}
#     \gamma_{n+1} &= \gamma_n+\Delta\gamma, \\
#     \mathbf r_i^{\rm aff} &=
#       \begin{pmatrix}1&\Delta\gamma\\0&1\end{pmatrix}\mathbf r_i, \\
#     \mathbf r_{n+1} &= \operatorname{local\,min}_{\mathbf r}
#                         U(\mathbf r;\gamma_{n+1}).
#     \end{aligned}
#
# The affine configuration starts the next minimization. The local minimum
# reached depends on the preceding state; AQS follows a branch of equilibria
# rather than finding the globally lowest energy at every strain. The library's
# ``domain.shear`` also updates the periodic geometry and any pair history.
#
# For overlapping disks the spring energy is
#
# .. math::
#
#     U=\frac12\sum_{i<j}k_{ij}
#            \left[\max(R_i+R_j-r_{ij},0)\right]^2.
#
# Relaxation stops when force (and any active torque) residuals meet their
# tolerances. A small energy alone does not establish equilibrium. Starting
# above the finite-system threshold gives contacts a finite overlap before
# loading; the energy printed by the minimizer is ``U/N``.
#
# With the compression-positive contact virial convention, a smooth relaxed
# branch of this conservative central-force system satisfies
#
# .. math::
#
#     \sigma^c_{xy}=-\frac{1}{A}\frac{dU_{\min}}{d\gamma}.
#
# On such a branch the energy and stress vary smoothly. A loss of stability
# can trigger nonaffine rearrangement and an energy drop; its location depends
# on the strain increment and relaxation accuracy. That interpretation requires
# inspecting the configurations as well as the scalar output.

# %%
# A bounded jamming search establishes a reproducible starting configuration.
N = 8
phi = 0.65
rad = jnp.concatenate([jnp.full(N // 2, 0.5), jnp.full(N // 2, 0.7)])
volume = jnp.pi * rad**2
L = jnp.sqrt(jnp.sum(volume) / phi)
state = jdem.State.create(
    pos=jax.random.uniform(jax.random.key(0), (N, 2), minval=0.0, maxval=L),
    rad=rad,
    mass=jnp.ones(N),
    volume=volume,
)
system = jdem.System.create(
    state=state,
    dt=0.01,
    rotation_integrator_type=None,
    minimizer=jdem.minimizers.fire,
    minimizer_kw={"dt": 0.05},
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(2, L)},
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", young=1.0, poisson=0.3, density=1.0)]
    ),
)
jam = jdem.utils.jamming.bisection_jam(
    state,
    system,
    pe_tol=1e-8,
    pe_band_factor=4.0,
    packing_fraction_increment=0.04,
    packing_fraction_tolerance=1e-4,
    n_jamming_steps=80,
    verbose=False,
    minimizer_kwargs={"max_steps": 5_000, "force_tol": 1e-7, "torque_tol": 1e-7},
)
if not bool(jam.converged):
    raise RuntimeError(f"Jamming did not converge: {jam.info}")
state, system = jam.jammed_state, jam.jammed_system
solid_phi = float(jam.packing_fraction) + 0.02
state, system = scale_to_packing_fraction(state, system, solid_phi)
print(
    f"Finite-system phi_J = {float(jam.packing_fraction):.6f}; "
    f"solid phi = {solid_phi:.6f}"
)

system = replace(
    system,
    domain=jdem.Domain.create(
        "leesedwards",
        dim=2,
        box_size=system.domain.box_size,
        anchor=system.domain.anchor,
    ),
)

# %%
# Relax the compressed state before the first strain increment, then repeat
# affine shear followed by relaxation. Every accepted point satisfies the
# same force/torque criterion; reaching the iteration cap is a failure.
print("gamma   E/N        contact_sigma_xy  relaxation_steps")
for step in range(13):
    if step:
        state, system = system.domain.shear(state, system, 0.002)
    result = system.minimize(
        state, system, max_steps=5_000, force_tol=1e-7, torque_tol=1e-7
    )
    if not bool(result.converged):
        raise RuntimeError(
            f"Relaxation failed at gamma={float(system.domain.gamma):.4f}"
        )
    state, system, steps, pe = result
    state, system, stress = compute_contact_stress_tensor(state, system)
    print(
        f"{float(system.domain.gamma):.4f}  {float(pe):.3e}  "
        f"{float(stress[0, 1]):.3e}  {int(steps)}"
    )
