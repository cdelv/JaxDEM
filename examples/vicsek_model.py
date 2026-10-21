"""Vicsek Model
============

This example implements the Vicsek model in 2D.
The state has 100 rigid two-sphere clumps (200 spheres in total) in
periodic boundaries. Each sphere interacts through a pairwise
purely-repulsive harmonic potential.
Bodies move at a constant speed ```v0```. Two components set the
direction: a random component proportional to ```eta```, and a
component proportional to the average velocity of all neighboring
particles within a distance ```neighbor_radius```. Here the noise
randomly rotates the heading direction. This is the "intrinsic" noise
Vicsek model, as opposed to the "extrinsic" (vectorial) noise variant.
We use a ```trajectory_rollout``` to run the dynamics in a
jit-compiled loop. The rollout also returns the trajectory data.
We then compute the polarization order parameter (norm of the
average velocity) for each saved frame.

Both Vicsek integrators are defined and registered below as local
plugins using the existing linear-integrator interface.
"""

from dataclasses import dataclass
from functools import partial
from typing import cast

import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
import jax.numpy as jnp
import jaxdem as jdem
import numpy as np

from jaxdem.integrators import free_mask
from jaxdem.utils.linalg import cross, dot, unit
from jaxdem.utils.random_sphere_configuration import random_sphere_configuration

# %%
# Local Vicsek integrators
# ~~~~~~~~~~~~~~~~~~~~~~~~


def _vicsek_alignment(
    state: jdem.State,
    system: jdem.System,
    neighbor_radius: jax.Array,
    max_neighbors: int,
) -> tuple[jdem.State, jdem.System, jax.Array, jax.Array]:
    """Shared Vicsek pre-processing: neighbor query and clump-wise averages.

    Build the neighbor list and average the neighbor velocities (including
    self). Reduce the alignment vector and the accumulated force to one
    clump-wise value each. Broadcast these values to all clump members.
    Queries use the exact requested radius; incomplete queries propagate to
    the system overflow flag so checked stepping can reject the result.

    Returns
    -------
    Tuple[State, System, jax.Array, jax.Array]
        The state and system, the clump-broadcast force,
        and the clump-broadcast average neighbor velocity.
    """
    # Neighbor list query.
    state, system, nl, overflow = system.collider.create_neighbor_list(
        state, system, neighbor_radius, max_neighbors
    )

    system.search_overflow |= overflow
    vel = state.vel

    # Gather neighbor velocities (with padding-safe indexing).
    safe_nl = jnp.maximum(nl, 0)
    nbr_vel = vel[safe_nl]  # (N, K, dim)
    valid = nl != -1

    # Average neighbor velocity including self.
    sum_v = jnp.sum(nbr_vel * valid[..., None], axis=1) + vel
    count = jnp.sum(valid, axis=1) + 1
    avg_v = sum_v / count[..., None]

    # Clump-wise average of avg_v (so each clump has a single alignment vector)
    # and clump-wise force.
    counts = jnp.bincount(state.clump_id, length=state.N)
    counts_safe = jnp.where(counts > 0, counts, 1).astype(avg_v.dtype)
    clump_values = (
        jax.ops.segment_sum(
            jnp.concatenate((avg_v, state.force), axis=-1),
            state.clump_id,
            num_segments=state.N,
        )
        / counts_safe[..., None]
    )
    clump_values = clump_values[state.clump_id]
    avg_v = clump_values[..., : state.dim]
    force = clump_values[..., state.dim :]

    return state, system, force, avg_v


@jax.jit(inline=True)
def _apply_desired_velocity(
    state: jdem.State, system: jdem.System, v_des: jax.Array
) -> tuple[jdem.State, jdem.System]:
    """Shared Vicsek tail: set the velocities of the free particles and advance the positions."""
    state.vel = jnp.where(free_mask(state), v_des, state.vel)
    state.pos_c = state.pos_c + system.dt * state.vel
    return state, system


@jdem.LinearIntegrator.register("vicsek_extrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekExtrinsic(jdem.LinearIntegrator):
    """Vicsek-model integrator with **extrinsic** (vectorial) noise.

    Each step, the integrator sets the translational velocity magnitude to
    ``v0``. The direction comes from a vector that combines:

    - the current accumulated force vector (from colliders + force functions),
    - the average neighbor velocity direction (including self),
    - an additive random unit vector scaled by ``eta`` (extrinsic noise).

    Notes
    -----
    - The integrator draws the noise **per clump** (one sample per rigid body)
      and broadcasts it to all clump members, so clumps move coherently.
    - A collider may cache the neighbor list (e.g., NeighborList collider) or
      sort the state (e.g., some cell-list builders). This integrator uses the
      returned state from ``create_neighbor_list`` for consistency.

    """

    neighbor_radius: jax.Array
    eta: jax.Array
    v0: jax.Array
    max_neighbors: int = jax.tree.static()

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VicsekExtrinsic.step_after_force")
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System]:
        vicsek = cast(VicsekExtrinsic, system.linear_integrator)

        state, system, force, avg_v = _vicsek_alignment(
            state, system, vicsek.neighbor_radius, vicsek.max_neighbors
        )

        # Random unit vector per clump (extrinsic/vectorial noise).
        system.key, noise_key = jax.random.split(system.key)
        if state.dim == 2:
            angles = jax.random.uniform(
                noise_key, shape=(state.N,), minval=0.0, maxval=2.0 * jnp.pi
            )
            unit_clump = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        else:
            # Isotropic direction in R^3 by normalizing a Gaussian vector.
            raw = jax.random.normal(
                noise_key, shape=(state.N, 3), dtype=state.vel.dtype
            )
            unit_clump = unit(raw)
        noise_dir = unit_clump[state.clump_id]

        f = force + avg_v + vicsek.eta * noise_dir
        v_des = vicsek.v0 * unit(f)

        return _apply_desired_velocity(state, system, v_des)


@jdem.LinearIntegrator.register("vicsek_intrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekIntrinsic(jdem.LinearIntegrator):
    """Vicsek-model integrator with **intrinsic** noise.

    This variant perturbs the *direction* of the desired motion. It applies a
    random rotation to the normalized base direction. It does not add a random
    vector in force space as the extrinsic (vectorial-noise) variant does.

    The base direction comes from:
    - the current accumulated force vector (from colliders + force functions),
    - the average neighbor velocity direction (including self).

    The integrator then applies the noise per clump and broadcasts it to all
    clump members, so clumps move coherently.
    """

    neighbor_radius: jax.Array
    eta: jax.Array
    v0: jax.Array
    max_neighbors: int = jax.tree.static()

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VicsekIntrinsic.step_after_force")
    def step_after_force(
        state: jdem.State, system: jdem.System
    ) -> tuple[jdem.State, jdem.System]:
        vicsek = cast(VicsekIntrinsic, system.linear_integrator)

        state, system, force, avg_v = _vicsek_alignment(
            state, system, vicsek.neighbor_radius, vicsek.max_neighbors
        )

        base_dir = unit(force + avg_v)

        # Intrinsic noise: randomly rotate the base direction.
        # Angle perturbation in [-pi, pi] scaled by eta, generated per clump.
        system.key, noise_key = jax.random.split(system.key)
        dtheta_clump = (2.0 * jax.random.uniform(noise_key, shape=(state.N,)) - 1.0) * (
            jnp.pi * vicsek.eta
        )
        dtheta = dtheta_clump[state.clump_id]
        if state.dim == 2:
            c = jnp.cos(dtheta)
            s = jnp.sin(dtheta)
            # Rotate base_dir by dtheta: [x', y'] = [c -s; s c] [x, y]
            rot = jnp.stack([c, -s, s, c], axis=-1).reshape((-1, 2, 2))
            dir_clump = jnp.einsum("nij,nj->ni", rot, base_dir)
        else:
            # 3D: sample random rotation axis (uniform on sphere), then rotate by angle.
            system.key, axis_key = jax.random.split(system.key)
            raw_axis_clump = jax.random.normal(
                axis_key, shape=(state.N, 3), dtype=state.vel.dtype
            )
            axis = unit(raw_axis_clump[state.clump_id])

            # Rodrigues' rotation formula: v' = v c + (k x v) s + k (k·v) (1-c)
            c = jnp.cos(dtheta)[..., None]
            s = jnp.sin(dtheta)[..., None]
            k = axis
            v = base_dir
            k_cross_v = cross(k, v)
            k_dot_v = dot(k, v)[..., None]
            dir_clump = v * c + k_cross_v * s + k * k_dot_v * (1.0 - c)

        v_des = vicsek.v0 * dir_clump

        return _apply_desired_velocity(state, system, v_des)


# %%
# This function sets the initial data


# For one-call builders see
# :func:`~jaxdem.utils.particle_creation.build_sphere_system` and
# :func:`~jaxdem.utils.particle_creation.build_ga_system`
# (sphere_construction and general_particle_construction examples).
def build_microstate(
    N: int,
    phi: float,
    dim: int,
    dt: float,
    neighbor_radius: float,
    eta: float,
    v0: float,
    seed: int,
    integrator_type: str = "vicsek_extrinsic",
) -> tuple[jdem.State, jdem.System]:
    # neighbor-list capacity per particle. It must be large enough that the
    # collider does not overflow
    max_neighbors = 64

    # Mono-disperse spheres of radius 1.0. The ratio lists are relative and
    # normalize away. Only ``small_radius`` sets the absolute scale.
    particle_radii = jdem.utils.dispersity.get_polydisperse_radii(
        N, [1.0], [1.0], small_radius=1.0
    )

    assert N % 2 == 0
    N_clumps = N // 2
    # Bounding radius of each dumbbell: sphere radius 1.0 + center offset 0.5.
    clump_radii = jdem.utils.dispersity.get_polydisperse_radii(
        N_clumps, [1.0], [1.0], small_radius=1.5
    )
    pos_clumps, box_size = random_sphere_configuration(
        clump_radii.tolist(), phi, dim, seed
    )
    pos_c = jnp.repeat(pos_clumps, 2, axis=0)
    pos_p = jnp.tile(jnp.array([[-0.5, 0.0], [0.5, 0.0]]), (N_clumps, 1))
    clump_id = jnp.repeat(jnp.arange(N_clumps), 2)
    # No compute_clump_properties call here (cf. clump_guide Pitfall #1):
    # the Vicsek integrators prescribe velocities directly, so the dynamics
    # never use the clump mass and inertia.
    state = jdem.State.create(
        pos=pos_c,
        pos_p=pos_p,
        rad=particle_radii,
        mass=jnp.ones(N),
        clump_id=clump_id,
    )

    mats = [jdem.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = jdem.MaterialMatchmaker.create("harmonic")
    mat_table = jdem.MaterialTable.from_materials(mats, matcher=matcher)

    system = jdem.System.create(
        state_shape=state.shape,
        dt=dt,
        # Vicsek integrator:
        linear_integrator_type=integrator_type,
        linear_integrator_kw={
            "neighbor_radius": jnp.asarray(neighbor_radius, dtype=float),
            "eta": jnp.asarray(eta, dtype=float),
            "v0": jnp.asarray(v0, dtype=float),
            "max_neighbors": max_neighbors,
        },
        rotation_integrator_type=None,
        domain_type="periodic",
        domain_kw={"box_size": box_size},
        force_model_type="spring",
        mat_table=mat_table,
        # here, we use the naive (double for-loop) collider because the system is small.
        # For a larger system, use the CellList or the NeighborList instead.
        collider_type="naive",
        seed=seed,
    )
    return jdem.System.initialize(state, system)


# %%
# Build the initial data
state, system = build_microstate(
    N=200,  # 200 spheres -> 100 two-sphere clumps
    phi=0.65,  # just dense enough to order
    dim=2,  # 2D
    dt=1e-2,  # semi-arbitrary timestep
    neighbor_radius=2.0,  # particles will align within 2x their radii
    eta=0.2,  # small noise component
    v0=1.0,  # semi-arbitrary velocity
    seed=int(np.random.randint(0, int(1e9))),  # random seed
    integrator_type="vicsek_intrinsic",
)

# Run the dynamics for 5K steps, saving every 50th
n_steps = 5_000
save_stride = 50
n_frames = n_steps // save_stride

state_f, system_f, (traj_state, traj_system) = jdem.System.trajectory_rollout(
    state,
    system,
    n=n_frames,
    stride=save_stride,
)

polarization = jnp.linalg.norm(jnp.mean(traj_state.vel, axis=-2), axis=-1)
print("Polarization:")
print(polarization)
# %%
