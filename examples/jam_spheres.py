r"""Jamming of bidisperse spheres (or disks).
---------------------------------------------

Find an energy threshold for two tiny bidisperse disk packings in a periodic
box. The harmonic contact energy is zero without overlaps and increases
quadratically with overlap. Length, mass, and stiffness are in reduced units.

For radii :math:`R_i` and pair separations :math:`r_{ij}`, write

.. math::

    \delta_{ij}=\max(0,R_i+R_j-r_{ij}),\qquad
    U=\frac12\sum_{i<j}k_{ij}\delta_{ij}^2,\qquad e=U/N.

The spring force has magnitude :math:`k_{ij}\delta_{ij}`. Below jamming,
relaxation can remove every overlap; a compressed force-balanced network can
retain positive energy. We use this change to bracket a numerical energy
threshold. Each trial rescales the box, relaxes forces, and measures energy
per constituent sphere. Equilibrated states below ``pe_tol`` form a lower
bound; states above it form an upper bound. Bisection narrows these bounds and
accepts an upper state inside the specified energy band. Mechanical stability
still requires contact and spectral analysis; energy alone does not prove it.

Eight particles, two fixed seeds, moderate tolerances, and bounded iteration
counts keep the script quick. Such small packings have large finite-size
effects: their jamming fractions are not estimates of the bulk value.

:py:func:`~jaxdem.utils.jamming.bisection_jam` implements the bisection search.
"""

# %%
# Imports
# ~~~~~~~~~~~~~~~~~~~~~
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import jaxdem as jdem

# Double precision keeps the small energy threshold resolved.
jax.config.update("jax_enable_x64", True)

# %%
# Parameters
# ~~~~~~~~~~
# Half the disks have radius 0.5 and half 0.7. The 1.4 size ratio reduces
# the tendency toward a simple monodisperse crystal. Independent seeds give
# different initial positions at the same nominal packing fraction, allowing
# us to inspect finite-sample variation instead of treating one packing as a
# universal threshold. ``vmap`` adds the system axis; each member retains its
# own state, box, and minimization history. Batched relaxation proceeds until
# the slowest active member has finished.
N_systems = 2
N = 8
phi = 0.65
dim = 2
e_int = 1.0
dt = 5e-2


# %%
# For the one-call equivalent see
# :py:func:`~jaxdem.utils.particle_creation.build_sphere_system`
# (:doc:`sphere_construction example </auto_examples/sphere_construction>`).
def build_microstate(i):
    # assign bidisperse radii
    rad = jnp.ones(N)
    rad = rad.at[: N // 2].set(0.5)
    rad = rad.at[N // 2 :].set(0.7)

    # The d-dimensional particle volumes are V_i = c_d R_i**d, with
    # c_d = pi**(d/2) / Gamma(d/2+1). Set L = (sum(V_i)/phi)**(1/d),
    # so the periodic hypercube has exactly the requested nominal fraction.
    # This yields disk area in 2D and sphere volume in 3D; it counts nominal
    # particle volumes even when the initial random particles overlap.
    volume = (jnp.pi ** (dim / 2) / jax.scipy.special.gamma(dim / 2 + 1)) * rad**dim
    L = (jnp.sum(volume) / phi) ** (1 / dim)
    box_size = jnp.ones(dim) * L

    # create microstate
    key = jax.random.PRNGKey(i)
    pos = jax.random.uniform(key, (N, dim), minval=0.0, maxval=L)
    mass = jnp.ones(N)
    mats = [jdem.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jdem.MaterialMatchmaker.create("harmonic")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)

    # create system and state
    state = jdem.State.create(pos=pos, rad=rad, mass=mass, volume=volume)
    system = jdem.System.create(
        state_shape=state.shape,
        dt=dt,
        minimizer=jdem.minimizers.fire,
        minimizer_kw={"dt": dt},
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw={
            "box_size": box_size,
        },
    )
    return state, system


# %%
# A bounded search
# ~~~~~~~~~~~~~~~~
# The energy threshold classifies equilibrated states; ``force_tol`` controls
# the separate inner relaxation. First the search establishes a lower bound
# :math:`\phi_-` with :math:`e_-\leq e_{\rm tol}`. It increases packing
# fraction until an upper bound :math:`\phi_+` has :math:`e_+>e_{\rm tol}`.
# Subsequent trials use :math:`(\phi_-+\phi_+)/2` and update the appropriate
# bound only after force/torque relaxation has converged.
#
# A successful upper state satisfies both
#
# .. math::
#
#    e_{\rm tol}<e_+\leq b\,e_{\rm tol},\qquad
#    \frac{\phi_+}{\phi_-}-1\leq\varepsilon_\phi,
#
# where ``b = pe_band_factor`` and ``epsilon_phi`` is the relative bracket
# tolerance. Thus the returned upper energy is deliberately **above** the
# threshold; the last accepted lower state is available separately.
#
# ``n_jamming_steps`` limits outer trials and ``max_steps`` limits each inner
# relaxation. Exhausting either is diagnostic information, not a successful
# jam. Inspect convergence before using the returned state or scalar metrics.
pe_tol = 1e-8
search_kw = dict(
    pe_tol=pe_tol,
    pe_band_factor=4.0,
    n_jamming_steps=80,
    packing_fraction_increment=0.04,
    packing_fraction_tolerance=1e-4,
    minimizer_kwargs=dict(max_steps=5_000, force_tol=1e-7, torque_tol=1e-7),
    verbose=False,
)

# %%
# Run the Jamming Algorithm for Multiple Systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ``jax.vmap(build_microstate)`` creates two State/System pairs with leading
# batch axes. The second vmap applies the same search to each pair while
# retaining separate bounds and convergence records. Parameters and shapes
# must agree across the batch, but numerical arrays such as box sizes may vary.
state, system = jax.vmap(build_microstate)(jnp.arange(N_systems))
result = jax.vmap(
    lambda st, sys: jdem.utils.jamming.bisection_jam(st, sys, **search_kw)
)(state, system)
if not bool(jnp.all(result.converged)):
    raise RuntimeError(
        f"Jamming failed: reason={result.reason}, "
        f"minimizer_reason={result.minimizer_reason}"
    )
state, system = result.jammed_state, result.jammed_system
print("Jammed energy per constituent sphere:", result.potential_energy)
print("Finite-system jamming fractions:", result.packing_fraction)
print("Trial minimizations:", result.steps)
assert bool(jnp.all(result.potential_energy > pe_tol))
assert bool(jnp.all(result.potential_energy <= 4.0 * pe_tol))

# %%
# Inspect the accepted bounds and search cost
# -------------------------------------------
# Each marker represents one independent preparation. The lower error-bar
# endpoint is the stored unjammed fraction; its small span is the numerical
# bracket width. The shaded energy band shows the acceptance criterion.
# Different fractions across these two seeds reflect finite-system
# preparation dependence, not an uncertainty estimate for bulk jamming.
seeds = np.arange(N_systems)
phi_hi = np.asarray(result.packing_fraction)
phi_lo = np.asarray(result.unjammed_state.volume).sum(axis=-1) / np.prod(
    np.asarray(result.unjammed_system.domain.box_size), axis=-1
)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
axes[0].errorbar(
    seeds,
    phi_hi,
    yerr=np.stack((phi_hi - phi_lo, np.zeros_like(phi_hi))),
    fmt="o",
    capsize=5,
)
axes[0].set(ylabel=r"Packing fraction $\phi_J$", title="Accepted upper bounds")
axes[1].axhspan(1.0, 4.0, color="tab:green", alpha=0.12, label="Acceptance band")
axes[1].plot(
    seeds, np.asarray(result.potential_energy) / pe_tol, "o", color="tab:green"
)
axes[1].set(
    ylabel=r"Energy per sphere / $e_{\mathrm{tol}}$",
    title="Energy threshold is resolved",
    ylim=(0.0, 4.5),
)
axes[1].legend(fontsize="small")
axes[2].bar(seeds, np.asarray(result.steps), width=0.5, color="tab:orange")
axes[2].set(ylabel="Trial minimizations", title="Work to resolve each bracket")
for ax in axes:
    ax.set_xticks(seeds)
    ax.set_xlabel("Preparation seed")
    ax.grid(axis="y", alpha=0.2)
plt.show()

# %%
# Run the Jamming Algorithm for a Single System
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The same function also accepts one State/System pair without ``vmap``.
# This is useful when inspecting one failed minimization or studying the
# bracket sequence before scaling to an ensemble. The small example uses
# the same seed as the first batch member so both invocation patterns describe
# the same initial realization and numerical search criteria.
state, system = build_microstate(0)
result = jdem.utils.jamming.bisection_jam(state, system, **search_kw)
if not bool(result.converged):
    raise RuntimeError(f"Single-system jamming failed: reason={result.reason}")
state, system = result.jammed_state, result.jammed_system
print("Single-system energy per sphere:", float(result.potential_energy))
print("Single-system jamming fraction:", float(result.packing_fraction))
