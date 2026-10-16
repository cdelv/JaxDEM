"""Quasistatic shear of a jammed disk packing."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import jaxdem as jdem

jax.config.update("jax_enable_x64", True)

N = 32
phi = 0.4
dt = 1e-2
seed = 0
dgamma = 1e-3
n_shear_steps = 100

rad = jnp.concatenate([jnp.full(N // 2, 0.5), jnp.full(N // 2, 0.7)])
volume = jnp.pi * rad**2
L = jnp.sqrt(jnp.sum(volume) / phi)
pos = jax.random.uniform(jax.random.PRNGKey(seed), (N, 2), minval=0.0, maxval=L)

state = jdem.State.create(pos=pos, rad=rad, mass=jnp.ones(N), volume=volume)
mat_table = jdem.MaterialTable.from_materials(
    [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
)
system = jdem.System.create(
    state=state,
    dt=dt,
    minimizer=jdem.minimizers.fire,
    minimizer_kw={"dt": dt},
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(2, L)},
    force_model_type="spring",
    collider_type="naive",
    mat_table=mat_table,
)

jam = jdem.utils.jamming.bisection_jam(
    state, system, n_minimization_steps=100_000_000, verbose=False,
)
if not jam.converged:
    raise RuntimeError("Jamming did not converge.")
state, system = jam.jammed_state, jam.jammed_system
print(
    f"phi = {float(jam.packing_fraction):.8f}, E/N = {float(jam.potential_energy):.8e}"
)

system = replace(
    system,
    domain=jdem.Domain.create(
        "leesedwards",
        dim=state.dim,
        box_size=system.domain.box_size,
        anchor=system.domain.anchor,
    ),
)

print("gamma E/N steps")
for _ in range(n_shear_steps):
    state, system = system.domain.shear(state, system, dgamma)
    result = system.minimize(state, system, max_steps=100_000_000)
    if not result.converged:
        raise RuntimeError(
            f"Minimization failed at gamma = {float(system.domain.gamma)}."
        )
    state, system, steps, pe = result
    print(f"{float(system.domain.gamma):.6f} {float(pe):.8e} {int(steps)}")
