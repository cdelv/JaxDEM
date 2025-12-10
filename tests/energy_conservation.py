import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jd

def make_grid_state(n_per_axis, dim):
    radius = 0.5
    spacing = 3 * radius
    mass = 1.0

    spacing = jnp.array([spacing for _ in range(dim)])

    state = jd.utils.gridState.grid_state(
        n_per_axis=[n_per_axis for _ in range(dim)],
        spacing=spacing,
        radius=radius,
        mass=mass,
        jitter=0.0,
    )

    box_size = jnp.max(state.pos, axis=0) + spacing

    return state, box_size

def make_system_for_state(state, box_size, e_int):
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return system

def assign_velocities(state, rotational=True):
    seed = np.random.randint(0, 1000000)
    key = jax.random.PRNGKey(seed)
    key_vel, key_angVel = jax.random.split(key, 2)
    N_clumps = jnp.unique(state.ID).size
    clump_vel = jax.random.normal(key_vel, (N_clumps, state.dim))
    clump_vel -= jnp.mean(clump_vel, axis=0)
    clump_angVel = jax.random.normal(key_angVel, (N_clumps, state.angVel.shape[-1])) * (rotational)
    dof = state.dim * state.N + state.angVel.shape[-1] * state.N * (rotational)
    _, offsets = jnp.unique(state.ID, return_index=True)
    clump_mass = state.mass[offsets]
    clump_inertia = state.inertia[offsets]
    ke = jnp.sum((0.5 * clump_mass * jnp.sum(clump_vel ** 2, axis=-1)) + 0.5 * jnp.sum(clump_inertia * clump_angVel, axis=-1))
    temperature = 2 * ke / dof
    scale = jnp.sqrt(target_temperature / temperature)
    clump_vel *= scale
    clump_angVel *= scale
    state.vel = clump_vel[state.ID]
    state.angVel = clump_angVel[state.ID]
    return state

if __name__ == "__main__":
    dim = 2
    target_temperature = 1e-4
    e_int = 1.0
    n_per_axis = 4

    dts = np.linspace(1e-3, 1e-1, 5)
    fluctuation = np.zeros_like(dts)
    for j, dt in enumerate(dts):
        state, box_size = make_grid_state(n_per_axis, dim)
        system = make_system_for_state(state, box_size, e_int)
        state = assign_velocities(state, rotational=False)

        n_repeats = 100
        ke = np.zeros(n_repeats)
        pe = np.zeros(n_repeats)
        for i in range(n_repeats):
            n_steps = 1000
            state, system = system.step(state, system, n=n_steps)
            ke_t = 0.5 * (state.mass * jnp.sum(state.vel ** 2, axis=-1))
            ke[i] = jnp.sum(ke_t)
            pe[i] = jnp.sum(system.collider.compute_potential_energy(state, system))
        fluctuation[j] = np.std(ke + pe) / np.mean(ke + pe)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.plot(dts, fluctuation)
    plt.plot(dts, dts ** 2)
    plt.xscale('log')
    plt.yscale('log')
    # plt.gca().set_aspect('equal')
    # plt.scatter(*state.pos.T)
    # plt.xlim(0, system.domain.box_size[0])
    # plt.ylim(0, system.domain.box_size[1])
    # for pos, rad in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
    #     plt.gca().add_artist(Circle(pos, rad))
    plt.savefig('config.png')
    plt.close()