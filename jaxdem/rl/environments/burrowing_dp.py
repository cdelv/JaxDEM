from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import jaxdem as jdem
import jaxdem.utils.thermal as thermal

from ...state import State
from ...system import System
from . import Environment
from typing import cast, Any
from jaxdem.bonded_forces.deformable_particle import DeformableParticleModel


@Environment.register("burrowing_dp")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class BurrowingDP(Environment):
    """2D environment where a deformable particle burrows into a granular bed.

    The agent changes the rest lengths of the particle's perimeter and
    cross springs to move down into the bed. The reward gives the depth
    gained per step minus a kinetic-energy penalty. The episode ends when
    ``step_count`` exceeds ``max_steps``.
    """

    @classmethod
    @partial(jax.named_call, name="BurrowingDP.Create")
    def Create(
        cls,
        box_size: float = 40.0,
        max_steps: int = 10000,
    ) -> BurrowingDP:
        L = 28.0
        H = 40.0
        max_rad = 1.0
        max_rad * 2.0 + 0.01

        N_dp = 64
        dp_R = 5.0
        rad_vertex = dp_R * jnp.sin(jnp.pi / N_dp)

        # Bottom 3 layers (2X bigger)
        rad_bottom_base = 2.0 * rad_vertex
        nx_bottom = int(L / (2.0 * rad_bottom_base))
        ny_bottom = 3
        N_bottom = nx_bottom * ny_bottom
        xs_bottom = jnp.linspace(rad_bottom_base, L - rad_bottom_base, nx_bottom)
        y_start_bottom = 1.0
        ys_bottom = jnp.linspace(
            y_start_bottom,
            y_start_bottom + (ny_bottom - 1) * 2.0 * rad_bottom_base,
            ny_bottom,
        )
        grid_x_b, grid_y_b = jnp.meshgrid(xs_bottom, ys_bottom)
        pos_bottom = jnp.stack([grid_x_b.flatten(), grid_y_b.flatten()], axis=1)

        # Rest of the grains
        N_rest = 414 * 3
        nx_rest = 50
        ny_rest = int(jnp.ceil(N_rest / nx_rest))
        xs_rest = jnp.linspace(0.5, L - 0.5, nx_rest)
        y_start_rest = ys_bottom[-1] + rad_bottom_base + rad_vertex + 0.1
        ys_rest = jnp.linspace(
            y_start_rest, y_start_rest + (ny_rest - 1) * 2.1 * rad_vertex, ny_rest
        )
        grid_x_r, grid_y_r = jnp.meshgrid(xs_rest, ys_rest)
        pos_rest = jnp.stack([grid_x_r.flatten(), grid_y_r.flatten()], axis=1)[:N_rest]

        N_spheres = N_bottom + N_rest

        key = jax.random.PRNGKey(42)
        key_pos, key_rad1, key_rad2 = jax.random.split(key, 3)

        pos_s = jnp.concatenate([pos_bottom, pos_rest], axis=0)
        pos_s = pos_s + jax.random.uniform(
            key_pos, (N_spheres, 2), minval=-0.05, maxval=0.05
        )

        rad_s_bottom = jax.random.uniform(
            key_rad1, (N_bottom,), minval=1.6 * rad_vertex, maxval=2.0 * rad_vertex
        )
        rad_s_rest = jax.random.uniform(
            key_rad2, (N_rest,), minval=0.8 * rad_vertex, maxval=1.0 * rad_vertex
        )
        rad_s = jnp.concatenate([rad_s_bottom, rad_s_rest], axis=0)

        total_dp_mass = 8.0
        # Substrate density smaller by 50% from its original absolute value of 0.1
        base_mass = 0.05
        mass_s = base_mass * (rad_s / rad_vertex) ** 2

        # Spawn the DP higher up relative to the top of the generated bed
        y_top_bed = ys_rest[-1] + 2.0 * rad_vertex
        dp_center = jnp.array([L / 2, y_top_bed + dp_R + 2.0])
        angles = jnp.linspace(0, 2 * jnp.pi, N_dp, endpoint=False)
        pos_dp = dp_center + dp_R * jnp.stack(
            [jnp.cos(angles), jnp.sin(angles)], axis=1
        )
        rad_dp = jnp.ones(N_dp) * rad_vertex
        mass_dp = jnp.ones(N_dp) * (total_dp_mass / N_dp)

        pos = jnp.concatenate([pos_s, pos_dp], axis=0)
        rad = jnp.concatenate([rad_s, rad_dp], axis=0)
        mass = jnp.concatenate([mass_s, mass_dp], axis=0)
        mat_id = jnp.zeros(N_spheres + N_dp, dtype=int)

        # bond_id: DP nodes connected to neighbors and opposite nodes
        bond_id: list[list[int]] = []
        for i in range(N_spheres):
            bond_id.append([])
        for i in range(N_dp):
            bond_id.append(
                [
                    N_spheres + (i - 1) % N_dp,
                    N_spheres + (i + 1) % N_dp,
                    N_spheres + (i + 32) % N_dp,
                ]
            )

        state = State.create(
            pos=pos,
            rad=rad,
            mass=mass,
            bond_id=bond_id,
            mat_id=mat_id,
            clump_id=jnp.arange(N_spheres + N_dp),
        )

        elements = jnp.stack([jnp.arange(N_dp), (jnp.arange(N_dp) + 1) % N_dp], axis=1)

        internal_edges = jnp.stack([jnp.arange(32), jnp.arange(32) + 32], axis=1)

        all_edges = jnp.concatenate([elements, internal_edges], axis=0)
        all_edges_global = all_edges + N_spheres
        elements_global = elements + N_spheres

        adjacency = elements

        dp = jdem.BondedForceModel.create(
            "deformable_particle_model",
            vertices=state.pos,
            elements=elements_global,
            edges=all_edges_global,
            element_adjacency=adjacency,
            em=200.0,
            el=jnp.concatenate([jnp.full(64, 0.2), jnp.full(32, 3 * 14.0)]),
            eb=0.01,
            ec=200.0,
            elements_id=jnp.zeros(N_dp, dtype=int),
        )

        def dynamic_dp_forces(pos: jax.Array, s: State, sys: System) -> Any:
            force_fn = sys.bonded_force_model.force_and_energy_fns[0]  # type: ignore
            return force_fn(pos, s, sys)

        def frictional_floor_force(pos: jax.Array, state: State, system: System) -> Any:
            k = 2e5
            mu_wall = 0.5
            restitution = 0.6
            n = jnp.array([0.0, 1.0])
            floor_y = 0.1

            dist = pos[:, 1] - floor_y - state.rad
            penetration = jnp.minimum(0.0, dist)
            force_n = (-k * penetration)[..., None] * n

            v_n_scalar = jnp.sum(state.vel * n, axis=-1, keepdims=True)
            in_contact = (penetration < 0)[..., None]
            c_n = (2.0 * (1.0 - restitution) * jnp.sqrt(k * state.mass))[..., None]
            c_n = jnp.minimum(c_n, (0.5 * state.mass / system.dt)[..., None])
            force_damping = -c_n * v_n_scalar * n * in_contact

            radius_vec = -state.rad[..., None] * n
            w = state.ang_vel
            cross_w_r = jnp.concatenate(
                [-w * radius_vec[:, 1:2], w * radius_vec[:, 0:1]], axis=-1
            )
            v_at_contact = state.vel + cross_w_r

            v_n = jnp.sum(v_at_contact * n, axis=-1, keepdims=True) * n
            v_t = v_at_contact - v_n

            v_t_norm = jnp.sqrt(jnp.sum(v_t**2, axis=-1, keepdims=True) + 1e-12)
            t_dir = jnp.where(v_t_norm > 1e-8, v_t / v_t_norm, jnp.zeros_like(v_t))

            f_t_mag = mu_wall * jnp.sum(force_n * n, axis=-1, keepdims=True)
            force_t = -f_t_mag * t_dir * in_contact

            total_force = force_n + force_damping + force_t
            torque = (
                radius_vec[:, 0:1] * force_t[:, 1:2]
                - radius_vec[:, 1:2] * force_t[:, 0:1]
            )

            return total_force, torque

        frictional = jdem.Material.create(
            "elasticfrict",
            density=1.0,
            young=1000.0,
            poisson=0.35,
            mu=0.1,
            e=0.8,
        )
        mat_table = jdem.MaterialTable.from_materials(
            [frictional], matcher=jdem.MaterialMatchmaker.create("harmonic")
        )

        system = System.create(
            state.shape,
            dt=1e-3,
            domain_type="periodic",
            domain_kw={"box_size": jnp.array([L, H])},
            force_model_type="cundallstrack",
            collider_type="MultiCellList",
            collider_kw={"state": state},
            mat_table=mat_table,
            force_manager_kw={
                "gravity": [0.0, -9.81],
                "force_functions": (dynamic_dp_forces, frictional_floor_force),
            },
        )
        import dataclasses

        system = dataclasses.replace(system, bonded_force_model=dp)

        dp_center_y = jnp.mean(state.pos[N_spheres:, 1])
        env_params = {
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "prev_depth": jnp.asarray(dp_center_y),
            "curr_depth": jnp.asarray(dp_center_y),
            "curr_ke": jnp.asarray(0.0),
            "base_internal_lengths": cast(
                jax.Array, cast(DeformableParticleModel, dp).initial_edge_lengths
            )[N_dp:],
            "base_outer_lengths": cast(
                jax.Array, cast(DeformableParticleModel, dp).initial_edge_lengths
            )[:N_dp],
            "initial_state": state,
            "initial_system": system,
        }

        return cls(state=state, system=system, env_params=env_params)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="BurrowingDP.reset")
    def reset(env: BurrowingDP, key: ArrayLike) -> Environment:
        env.state = env.env_params["initial_state"]
        env.system = env.env_params["initial_system"]
        N_spheres = env.state.pos.shape[0] - 64
        env.env_params["curr_depth"] = jnp.mean(env.state.pos[N_spheres:, 1])
        env.env_params["prev_depth"] = env.env_params["curr_depth"]
        env.env_params["curr_ke"] = jnp.asarray(0.0)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="BurrowingDP.step")
    def step(env: BurrowingDP, action: jax.Array) -> Environment:
        # action shape: (1, 96)
        actuation_outer = jnp.clip(action[0, :64], -1.0, 1.0)
        actuation_inner = jnp.clip(action[0, 64:], -1.0, 1.0)

        base_outer = env.env_params["base_outer_lengths"]
        base_inner = env.env_params["base_internal_lengths"]

        # Map actuation from [-1, 1] to [0.8, 1.2]
        new_outer_lengths = base_outer * (1.0 + 0.3 * actuation_outer)
        new_inner_lengths = base_inner * (1.0 + 0.2 * actuation_inner)

        current_dp = cast(DeformableParticleModel, env.system.bonded_force_model)
        current_initial_lengths = cast(jax.Array, current_dp.initial_edge_lengths)
        updated_lengths = (
            current_initial_lengths.at[:64]
            .set(new_outer_lengths)
            .at[64:]
            .set(new_inner_lengths)
        )

        import dataclasses

        new_bonded = dataclasses.replace(
            current_dp, initial_edge_lengths=updated_lengths
        )
        env.system = dataclasses.replace(env.system, bonded_force_model=new_bonded)

        env.env_params["prev_depth"] = env.env_params["curr_depth"]
        env.state, env.system = env.system.step(env.state, env.system)

        N_spheres = env.state.pos.shape[0] - 64
        dp_center_y = jnp.mean(env.state.pos[N_spheres:, 1])
        env.env_params["curr_depth"] = dp_center_y

        ke = thermal.compute_translational_kinetic_energy_per_particle(env.state)[
            N_spheres:
        ]
        env.env_params["curr_ke"] = jnp.sum(ke)

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="BurrowingDP.observation")
    def observation(env: BurrowingDP) -> jax.Array:
        N_spheres = env.state.pos.shape[0] - 64
        dp_pos = env.state.pos[N_spheres:]
        dp_vel = env.state.vel[N_spheres:]
        center = jnp.mean(dp_pos, axis=0)
        rel_pos = dp_pos - center

        phase = jnp.array([(env.system.step_count % 500) / 500.0])
        current_dp = cast(DeformableParticleModel, env.system.bonded_force_model)
        spring_targets = cast(jax.Array, current_dp.initial_edge_lengths)

        obs = jnp.concatenate(
            [
                rel_pos.flatten(),
                dp_vel.flatten(),
                jnp.array([center[1]]),
                phase,
                spring_targets,
            ]
        )
        return obs[None, :]

    @staticmethod
    @jax.jit(inline=True)
    def reward(env: BurrowingDP) -> jax.Array:
        depth_reward = env.env_params["prev_depth"] - env.env_params["curr_depth"]
        ke_penalty = 1e-4 * env.env_params["curr_ke"]
        rew = 10.0 * depth_reward - ke_penalty
        return jnp.array([rew])

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="BurrowingDP.done")
    def done(env: BurrowingDP) -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def max_num_agents(self) -> int:
        return 1

    @property
    def action_space_size(self) -> int:
        return 96

    @property
    def action_space_shape(self) -> tuple[int]:
        return (96,)

    @property
    def observation_space_size(self) -> int:
        return 64 * 2 + 64 * 2 + 1 + 1 + 96
