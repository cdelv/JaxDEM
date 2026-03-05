"""
Pyramid Test – SwarmRoller3D
----------------------------

Five spheres must form a small pyramid: four at the base and one on
top.  The single elevated objective forces agents to cooperate using
the magnetic attractor to stack.

This script creates the environment, runs it with random actions for
visual verification, and saves VTK frames.
"""

# %%
# Imports
# ~~~~~~~
import jax
import jax.numpy as jnp

import jaxdem as jdem
import jaxdem.rl as rl

# %%
# Environment
# ~~~~~~~~~~~
# Five agents, one elevated objective at ~3 radii high (one sphere
# sitting on top of four requires z ≈ rad + 2*rad*√(2/3) ≈ 3.3*rad).
# The four base objectives sit on the floor in a tight square.

N = 5
rad = 0.05
# Height of the apex of a square-base pyramid of equal spheres
apex_z = rad + 2 * rad * jnp.sqrt(2.0 / 3.0)

env = rl.Environment.create(
    "swarmRoller3D",
    N=N,
    n_objectives=5,
    k_objectives=5,
    n_lidar_rays=8,
    min_box_size=0.6,
    max_box_size=0.6,
    box_padding=5.0,
    max_steps=10_000,
    friction=0.08,
    goal_weight=0.01,
    global_weight=0.005,
    goal_radius_factor=2.0,
    work_weight=0.001,
    seek_weight=0.3,
    lidar_range=0.5,
    magnet_strength=1.0,
    magnet_range=3.5 * rad,
    max_obj_height=float(apex_z),
)

# %%
# Custom reset – place objectives in a fixed pyramid pattern
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
key = jax.random.PRNGKey(42)
env = env.reset(env, key)

# Base objectives: tight square of side 2*rad centred in the box
cx, cy = 0.3, 0.3  # centre of pyramid
d = rad  # half-side
base = jnp.array(
    [
        [cx - d, cy - d, rad],
        [cx + d, cy - d, rad],
        [cx - d, cy + d, rad],
        [cx + d, cy + d, rad],
    ]
)
apex = jnp.array([[cx, cy, float(apex_z)]])
objectives = jnp.concatenate([base, apex], axis=0)
env.env_params["objective"] = objectives

print("Pyramid objectives:")
for i, o in enumerate(objectives):
    label = "apex" if i == 4 else "base"
    print(f"  [{label}] obj {i}: x={o[0]:.3f}  y={o[1]:.3f}  z={o[2]:.4f}")

# %%
# Run with random actions + magnets ON
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
writer = jdem.VTKWriter(directory="/tmp/pyramid_frames")


def save_frame(env):
    """Save agents + objective markers."""
    n_obj = env.env_params["objective"].shape[0]
    combined = env.state.add(
        env.state,
        pos=env.env_params["objective"],
        rad=rad / 4 * jnp.ones(n_obj),
    )
    writer.save(combined, env.system)


save_frame(env)

n_steps = 2000
print(f"\nRunning {n_steps} steps with random torques + magnets ON …")
for step in range(1, n_steps + 1):
    key, k_act = jax.random.split(key)
    # Random torques, all magnets activated
    torques = jax.random.normal(k_act, (N, 3)) * 0.3
    magnets = jnp.ones((N, 1))
    action = jnp.concatenate([torques, magnets], axis=-1).reshape(-1)
    env = env.step(env, action)

    if step % 50 == 0:
        save_frame(env)

    if step % 500 == 0:
        obs = env.observation(env)
        rew = env.reward(env)
        print(
            f"  step {step:5d} | "
            f"mean_z={float(jnp.mean(env.state.pos[:, 2])):.4f} | "
            f"max_z={float(jnp.max(env.state.pos[:, 2])):.4f} | "
            f"reward={float(jnp.mean(rew)):.6f} | "
            f"obs_ok={not bool(jnp.any(jnp.isnan(obs)))}"
        )

# %%
# Final diagnostics
# ~~~~~~~~~~~~~~~~~
print("\nFinal agent positions:")
for i in range(N):
    p = env.state.pos[i]
    print(f"  agent {i}: x={p[0]:.4f}  y={p[1]:.4f}  z={p[2]:.4f}")

print(f"\nObjective apex height: {float(apex_z):.4f}")
print(f"Highest agent z:      {float(jnp.max(env.state.pos[:, 2])):.4f}")

obs = env.observation(env)
rew = env.reward(env)
print(f"Observation shape: {obs.shape}  (expected ({N}, {env.observation_space_size}))")
print(f"Reward:  {rew}")
print(f"NaN check: obs={not bool(jnp.any(jnp.isnan(obs)))}, rew={not bool(jnp.any(jnp.isnan(rew)))}")
