from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxdem as jd
import numpy as np
from jaxdem.utils.packing_utils import compute_packing_fraction
from jaxdem.utils.particle_creation import (
    create_dp_container,
    create_ga_state,
    create_sphere_state,
)


N_PARTICLES = 50
DIM = 3

N_VERTICES = 492
MESH_TYPE = "icosphere"

PARTICLE_RADIUS = 0.5
VERTEX_RADIUS = 0.05
RING_GAP_RADIUS = 0.4
BOX_SIZE = jnp.array([8.0, 5.0, 5.0])
DT = 1e-4
SEED = 0
DP_BOND_ID = 0
DRAG_STIFFNESS = 20.0
VERTEX_DAMPING = 5.0
DRAG_SPEED = 0.05
DRAG_STEPS = 200
DRAG_STRIDE = 20_000
DP_START_OFFSET_X = -1.0
DP_VERTEX_TRAJECTORY_PATH = "dp_ring_extrusion_vertex_trajectory.npy"


def use_dp_container_volumes(current_state, dp_container):
    """Use the DP container's enclosed mesh volumes for packing-fraction control."""
    dp_mask = current_state.bond_id == DP_BOND_ID
    counts = jax.ops.segment_sum(
        dp_mask.astype(current_state.pos_c.dtype),
        current_state.bond_id,
        num_segments=current_state.N,
    )
    dp_node_volume = dp_container.initial_body_contents[0] / counts[DP_BOND_ID]
    node_volumes = jnp.where(dp_mask, dp_node_volume, current_state.volume)
    return replace(current_state, volume=node_volumes)


def place_dp_com(current_state, target_com):
    dp_mask = current_state.bond_id == DP_BOND_ID
    dp_center = jnp.mean(current_state.pos_c[dp_mask], axis=0)
    offset = target_com - dp_center
    return replace(
        current_state,
        pos_c=jnp.where(dp_mask[:, None], current_state.pos_c + offset, current_state.pos_c),
    )


def fix_spheres(current_state):
    sphere_mask = current_state.bond_id != DP_BOND_ID
    return replace(
        current_state,
        fixed=sphere_mask,
        vel=jnp.where(sphere_mask[:, None], 0.0, current_state.vel),
    )


def dp_vertex_positions(current_state):
    idx_map = jnp.zeros((current_state.N,), dtype=int).at[current_state.unique_id].set(
        jnp.arange(current_state.N)
    )
    return current_state.pos[idx_map[jnp.arange(N_VERTICES)]]


def ring_sphere_positions():
    n_spheres = N_PARTICLES - 1
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n_spheres, endpoint=False)
    center = BOX_SIZE / 2.0
    centerline_radius = RING_GAP_RADIUS + PARTICLE_RADIUS
    return jnp.stack(
        [
            jnp.full_like(theta, center[0]),
            center[1] + centerline_radius * jnp.cos(theta),
            center[2] + centerline_radius * jnp.sin(theta),
        ],
        axis=1,
    )


def make_drag_force(start_com):
    def drag_force(pos, current_state, current_system):
        dp_mask = current_state.bond_id == DP_BOND_ID
        n_dp = jnp.sum(dp_mask.astype(pos.dtype))
        dp_com = jnp.sum(jnp.where(dp_mask[:, None], pos, 0.0), axis=0) / n_dp
        target = start_com + jnp.array([DRAG_SPEED * current_system.time, 0.0, 0.0])
        spring_force = DRAG_STIFFNESS * (target - dp_com)
        force = jnp.where(
            dp_mask[:, None],
            spring_force / n_dp - VERTEX_DAMPING * current_state.vel,
            0.0,
        )
        return force, jnp.zeros_like(current_state.torque)

    return drag_force


def make_system(
    current_state,
    box_size,
    dp_container,
    material_table,
    linear_integrator_type,
    dt,
    force_manager_kw=None,
):
    return jd.System.create(
        state_shape=current_state.shape,
        dt=dt,
        linear_integrator_type=linear_integrator_type,
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw={
            "state": current_state,
            "cutoff": float(2.0 * jnp.max(current_state.rad)),
            "skin": 0.2,
            "safety_factor": 5.0,
        },
        mat_table=material_table,
        domain_kw={"box_size": box_size},
        bonded_force_model=dp_container,
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
    pos=ring_sphere_positions(),
)
state = jd.State.merge(dp_state, sphere_state)

print('a')

dp_start_com = BOX_SIZE / 2.0 + jnp.array([DP_START_OFFSET_X, 0.0, 0.0])
state = place_dp_com(state, dp_start_com)
dp_state = replace(dp_state, pos_c=state.pos_c[:N_VERTICES])
print("dp initial COM:", np.asarray(jnp.mean(dp_vertex_positions(state), axis=0)))

print('b')

container = create_dp_container(
    dp_state,
    em=1.0,
    ec=1.0,
    eb=0.05,
    el=0.3,
    gamma=None,
    tau_s=1.0,
    plasticity_type="bending",
)

print('c')

state = use_dp_container_volumes(state, container)
state = fix_spheres(state)

material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
mat_table = jd.MaterialTable.from_materials(
    [material],
    matcher=jd.MaterialMatchmaker.create("harmonic"),
)

system = make_system(
    state,
    BOX_SIZE,
    container,
    mat_table,
    linear_integrator_type="verlet",
    dt=DT,
    force_manager_kw={"force_functions": [make_drag_force(dp_start_com)]},
)
print(f"initial packing fraction: {float(compute_packing_fraction(state, system)):.4f}")

state, system, dp_vertex_trajectory = jd.System.trajectory_rollout(
    state,
    system,
    n=DRAG_STEPS,
    stride=DRAG_STRIDE,
    save_fn=lambda st, sys: dp_vertex_positions(st),
)
np.save(DP_VERTEX_TRAJECTORY_PATH, np.asarray(dp_vertex_trajectory))
print(f"saved DP vertex trajectory with shape {dp_vertex_trajectory.shape}")
print("dp final COM:", np.asarray(jnp.mean(dp_vertex_positions(state), axis=0)))

jd.utils.h5.save(state, 'ring_state.h5')
jd.utils.h5.save(system, 'ring_system.h5')
