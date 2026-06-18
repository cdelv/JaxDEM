from dataclasses import replace
import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxdem as jd
import numpy as np
from jaxdem.utils.dynamicsRoutines import run_packing_fraction_protocol
from jaxdem.utils.packingUtils import compute_packing_fraction
from jaxdem.utils.particleCreation import (
    create_dp_container,
    create_ga_state,
    create_sphere_state,
    distribute_bodies,
)


N_PARTICLES = 50
DIM = 3

N_VERTICES = 492
MESH_TYPE = "icosphere"

PARTICLE_RADIUS = 0.5
VERTEX_RADIUS = 0.05
INITIAL_BOUNDING_SPHERE_PHI = 0.5
TARGET_TRUE_DP_PHI = 0.55
COMPRESSION_STEP = 0.02
STEPS_PER_COMPRESSION = 100
DT = 1e-4
SEED = 0
DP_BOND_ID = 0
DRAG_STIFFNESS = 20.0
VERTEX_DAMPING = 5.0
DRAG_SPEED = 0.05
DRAG_STEPS = 100
DRAG_STRIDE = 10_000
DP_VERTEX_TRAJECTORY_PATH = "dp_vertex_trajectory.npy"


def use_dp_container_volumes(state, container):
    """Use the DP container's enclosed mesh volumes for packing-fraction control."""
    dp_mask = state.bond_id == DP_BOND_ID
    counts = jax.ops.segment_sum(
        dp_mask.astype(state.pos_c.dtype),
        state.bond_id,
        num_segments=state.N,
    )
    dp_node_volume = container.initial_body_contents[0] / counts[DP_BOND_ID]
    node_volumes = jnp.where(dp_mask, dp_node_volume, state.volume)
    return replace(state, volume=node_volumes)


def move_dp_to_box_center(state, box_size):
    dp_mask = state.bond_id == DP_BOND_ID
    dp_center = jnp.mean(state.pos_c[dp_mask], axis=0)
    offset = box_size / 2.0 - dp_center
    return replace(state, pos_c=jnp.where(dp_mask[:, None], state.pos_c + offset, state.pos_c))


def fix_spheres(state):
    sphere_mask = state.bond_id != DP_BOND_ID
    return replace(
        state,
        fixed=sphere_mask,
        vel=jnp.where(sphere_mask[:, None], 0.0, state.vel),
    )


def dp_vertex_positions(state):
    idx_map = jnp.zeros((state.N,), dtype=int).at[state.unique_id].set(jnp.arange(state.N))
    return state.pos[idx_map[jnp.arange(N_VERTICES)]]


def make_drag_force(start_com):
    def drag_force(pos, state, system):
        dp_mask = state.bond_id == DP_BOND_ID
        n_dp = jnp.sum(dp_mask.astype(pos.dtype))
        dp_com = jnp.sum(jnp.where(dp_mask[:, None], pos, 0.0), axis=0) / n_dp
        target = start_com + jnp.array([DRAG_SPEED * system.time, 0.0, 0.0])
        spring_force = DRAG_STIFFNESS * (target - dp_com)
        force = jnp.where(
            dp_mask[:, None],
            spring_force / n_dp - VERTEX_DAMPING * state.vel,
            0.0,
        )
        return force, jnp.zeros_like(state.torque)

    return drag_force


def make_system(
    state,
    box_size,
    container,
    mat_table,
    linear_integrator_type,
    dt,
    force_manager_kw=None,
):
    return jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type=linear_integrator_type,
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw={
            "state": state,
            "cutoff": float(2.0 * jnp.max(state.rad)),
            "skin": 0.2,
            "safety_factor": 5.0,
        },
        mat_table=mat_table,
        domain_kw={"box_size": box_size},
        bonded_force_model=container,
        force_manager_kw=force_manager_kw,
    )


dp_state = create_ga_state(
    N=1,
    nv=N_VERTICES,
    dim=DIM,
    particle_radius=PARTICLE_RADIUS,
    asperity_radius=VERTEX_RADIUS,
    particle_type="dp",
    core_type="hollow",
    n_samples=1,
    seed=SEED,
    mesh_type=MESH_TYPE,
)
sphere_state = create_sphere_state(
    radii=jnp.full((N_PARTICLES - 1,), PARTICLE_RADIUS),
    dim=DIM,
)
state = jd.State.merge(dp_state, sphere_state)

print('a')

state, box_size = distribute_bodies(
    state,
    phi=INITIAL_BOUNDING_SPHERE_PHI,
    domain_type="periodic",
    seed=SEED,
    randomize_orientation=True,
    group_by="bond",
)
state = move_dp_to_box_center(state, box_size)
dp_state = replace(dp_state, pos_c=state.pos_c[:N_VERTICES])

print('b')

container = create_dp_container(
    dp_state,
    em=1.0,
    ec=1.0,
    eb=0.01,
    el=0.3,
    gamma=None,
    # tau_s=1.0,
    # plasticity_type="bending",
)

print('c')

state = use_dp_container_volumes(state, container)

material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
mat_table = jd.MaterialTable.from_materials(
    [material],
    matcher=jd.MaterialMatchmaker.create("harmonic"),
)

system = make_system(
    state,
    box_size,
    container,
    mat_table,
    linear_integrator_type="verlet",
    dt=DT,
)

initial_true_phi = float(compute_packing_fraction(state, system))
remaining_phi = TARGET_TRUE_DP_PHI - initial_true_phi
n_compression_frames = math.ceil(abs(remaining_phi) / COMPRESSION_STEP)
if n_compression_frames > 0:
    phi_schedule = jnp.linspace(
        initial_true_phi + remaining_phi / n_compression_frames,
        TARGET_TRUE_DP_PHI,
        n_compression_frames,
    )
    strides = jnp.full((n_compression_frames,), STEPS_PER_COMPRESSION, dtype=int)
    state, system, _ = run_packing_fraction_protocol(
        state,
        system,
        strides=strides,
        phi_at_frames=phi_schedule,
    )
final_true_phi = float(compute_packing_fraction(state, system))
print(f"true DP phi: {initial_true_phi:.4f} -> {final_true_phi:.4f}")



print(dp_vertex_positions(state))


state = fix_spheres(state)
drag_start_com = jnp.mean(dp_vertex_positions(state), axis=0)
system = make_system(
    state,
    system.domain.box_size,
    container,
    mat_table,
    linear_integrator_type="verlet",
    dt=DT,
    force_manager_kw={"force_functions": [make_drag_force(drag_start_com)]},
)
state, system, dp_vertex_trajectory = jd.System.trajectory_rollout(
    state,
    system,
    n=DRAG_STEPS,
    stride=DRAG_STRIDE,
    save_fn=lambda st, sys: dp_vertex_positions(st),
)
np.save(DP_VERTEX_TRAJECTORY_PATH, np.asarray(dp_vertex_trajectory))
print(f"saved DP vertex trajectory with shape {dp_vertex_trajectory.shape}")

print(dp_vertex_positions(state))


jd.utils.h5.save(state, 'state.h5')
jd.utils.h5.save(system, 'system.h5')