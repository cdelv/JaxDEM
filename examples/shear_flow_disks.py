"""Constant-rate shear of disks below jamming, without a thermostat."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import jaxdem as jdem
from jaxdem.utils.packing_utils import scale_to_packing_fraction
from jaxdem.utils.thermal import compute_potential_energy

jax.config.update("jax_enable_x64", True)

N = 32
phi = 0.4
delta_phi = 0.02  # phi_J - phi
dt = 1e-2
seed = 0
shear_rate = 1e-3
n_steps = 100_000
output_every = 10_000


def advance_shear(state, system):
    domain = replace(system.domain, gamma=system.domain.gamma_dot * system.time)
    return state, replace(system, domain=domain)


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
    linear_integrator_type="verlet",
    rotation_integrator_type=None,
    minimizer=jdem.minimizers.fire,
    minimizer_kw={"dt": dt},
    domain_type="periodic",
    domain_kw={"box_size": jnp.full(2, L)},
    force_model_type="spring",
    collider_type="naive",
    mat_table=mat_table,
)

jam = jdem.utils.jamming.bisection_jam(
    state, system, n_minimization_steps=100_000_000, verbose=False
)
if not jam.converged:
    raise RuntimeError("Jamming did not converge.")
state, system = jam.jammed_state, jam.jammed_system
flow_phi = float(jam.packing_fraction) - delta_phi
state, system = scale_to_packing_fraction(state, system, flow_phi)
print(f"phi_J = {float(jam.packing_fraction):.8f}, phi = {flow_phi:.8f}")

system = replace(
    system,
    domain=jdem.Domain.create(
        "leesedwards",
        dim=2,
        box_size=system.domain.box_size,
        anchor=system.domain.anchor,
        gamma_dot=shear_rate,
    ),
    user_pre_step_actions=advance_shear,
)
y = state.pos_c[:, 1]
state = replace(
    state, vel=jnp.zeros_like(state.vel).at[:, 0].set(shear_rate * (y - jnp.mean(y)))
)
state, system = system.initialize(state, system)

print("time gamma PE/N KE/N")
for step in range(0, n_steps, output_every):
    state, system = system.step(state, system, n=min(output_every, n_steps - step))
    pe = compute_potential_energy(state, system) / N
    ke = 0.5 * jnp.sum(state.mass[:, None] * state.vel**2) / N
    print(
        f"{float(system.time):.2f} {float(system.domain.gamma):.6f} "
        f"{float(pe):.8e} {float(ke):.8e}"
    )
