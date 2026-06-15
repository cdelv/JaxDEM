# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Multi-agent 2-D environment with three gears and a triangle objective."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from ...colliders import DynamicCellList
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils.linalg import norm, unit
from . import Environment
from .two_gears import (
    N,
    _rad,
    frictional_floor_force,
    inertia,
    pos,
    pos_p,
    q,
    rad,
    rr,
    volume,
    y_min,
)


@partial(jax.named_call, name="three_gears._gear_indices")
def _gear_indices(state: State) -> jax.Array:
    idx_0 = jnp.argmax(state.clump_id == 0)
    idx_1 = jnp.argmax(state.clump_id == 1)
    idx_2 = jnp.argmax(state.clump_id == 2)
    return jnp.array([idx_0, idx_1, idx_2], dtype=int)


@partial(jax.named_call, name="three_gears._total_objective_distance")
def _total_objective_distance(env: "ThreeGears") -> jax.Array:
    idx = _gear_indices(env.state)
    pos_c = env.state.pos_c[idx]
    delta = env.system.domain.displacement(
        pos_c, env.env_params["objective"], env.system
    )
    return jnp.sum(norm(delta))


@Environment.register("ThreeGears")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ThreeGears(Environment):
    r"""Multi-agent 2-D environment with three gears.

    The environment consists of three active gears composed of spheres. All gears
    can apply torque to themselves. The shared objective is to navigate the gears
    to form a triangular structure defined by a randomized target position.

    Note
    ----
    Similar to the TwoGears environment, if one wants some realistic parameters
    for training, ``skip_frames = 50`` will give a response rate of 200 Hz,
    meaning that ``num_steps_epoch = 100`` gives a horizon of 0.5 seconds.
    """

    @classmethod
    @partial(jax.named_call, name="ThreeGears.Create")
    def Create(
        cls,
        box_size: float = 10.0,
        max_steps: int = 10000 * 10,
        friction: float = 0.2,
        ke_weight: float = 0.001,
    ) -> "ThreeGears":
        r"""Create a three-gears 2-D environment.

        Parameters
        ----------
        box_size : float
            Size of the square bounding box.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
        ke_weight : float
            Weight for the differential kinetic energy penalty.

        Returns
        -------
        ThreeGears
            A freshly constructed environment (call :meth:`reset` before use).
        """
        dim = 2
        state = State.create(pos=jnp.zeros((3 * N, dim)))
        system = System.create(state.shape)

        env_params = {
            "box_size": jnp.asarray(box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ke_weight": jnp.asarray(ke_weight, dtype=float),
            "objective": jnp.zeros((3, dim), dtype=float),
            "prev_dist": jnp.asarray(0.0, dtype=float),
            "curr_dist": jnp.asarray(0.0, dtype=float),
            "action": jnp.zeros((3, 1), dtype=float),
            "curr_ke": jnp.zeros(3, dtype=float),
            "prev_ke": jnp.zeros(3, dtype=float),
        }
        return cls(state=state, system=system, env_params=env_params)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ThreeGears.reset")
    def reset(env: "ThreeGears", key: jax.Array) -> Environment:
        """Reset the environment to a random initial configuration.

        Parameters
        ----------
        env : Environment
            The environment instance to reset.
        key : jax.Array
            PRNG key used to sample the initial positions and objective triangle.

        Returns
        -------
        Environment
            The environment with a fresh episode state.
        """
        key, k_span, k_center, k_x0, k_x1, k_x2, k_y = jax.random.split(key, 7)

        box = jnp.array([env.env_params["box_size"], env.env_params["box_size"]])
        y_floor = 1.0

        # Objective: random triangle
        base_span = jax.random.uniform(k_span, minval=2.2 * rr, maxval=3.2 * rr)
        x_margin = 1.5 * rr
        center_x = jax.random.uniform(
            k_center,
            minval=x_margin + base_span / 2.0,
            maxval=box[0] - x_margin - base_span / 2.0,
        )
        top_height = jnp.sqrt(3.0) * base_span / 2.0

        base_y = y_floor + rr
        objective = jnp.array(
            [
                [center_x - base_span / 2.0, base_y],
                [center_x + base_span / 2.0, base_y],
                [center_x, base_y + top_height],
            ]
        )
        objective = objective.at[:, 1].add(-rr - y_min)

        # Initial positions: disjoint x regions to prevent overlap
        x0 = jax.random.uniform(k_x0, minval=rr, maxval=box[0] / 3 - rr)
        x1 = jax.random.uniform(
            k_x1, minval=box[0] / 3 + rr, maxval=2 * box[0] / 3 - rr
        )
        x2 = jax.random.uniform(k_x2, minval=2 * box[0] / 3 + rr, maxval=box[0] - rr)

        # Sample Y positions such that they are within the box
        # We can use a single key to sample 3 values
        y_vals = jax.random.uniform(k_y, (3,), minval=y_floor + rr, maxval=box[1] - rr)

        init_pos = jnp.array(
            [
                [x0, y_vals[0] - rr - y_min],
                [x1, y_vals[1] - rr - y_min],
                [x2, y_vals[2] - rr - y_min],
            ]
        )

        state = State.create()
        for i in range(3):
            state = State.add_clump(
                state,
                pos=pos + init_pos[i][None, :],
                rad=rad,
                pos_p=pos_p,
                volume=volume,
                inertia=inertia,
                q=q,
            )
        env.state = state

        # Match TwoGears materials and system params exactly
        mat = Material.create(
            "elasticfrict", density=1.0 / volume, young=2e5, poisson=0.3, mu=0.1, e=0.88
        )
        mat_table = MaterialTable.from_materials([mat])

        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflect",
            domain_kw={"box_size": box, "anchor": jnp.zeros(2)},
            force_manager_kw={
                "gravity": [0.0, -1.0],
                "force_functions": (frictional_floor_force,),
            },
            collider_type="",
            mat_table=mat_table,
            force_model_type="cundallstrack",
        )
        env.system.collider = DynamicCellList(
            neighbor_mask=jnp.array(
                [
                    [-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [0, -1],
                    [0, 0],
                    [0, 1],
                    [1, -1],
                    [1, 0],
                    [1, 1],
                ],
                dtype=int,
            ),
            cell_size=jnp.array(2 * _rad, dtype=float),
        )

        env.env_params["objective"] = objective
        dist = _total_objective_distance(env)
        env.env_params["prev_dist"] = dist
        env.env_params["curr_dist"] = dist
        env.env_params["action"] = jnp.zeros((3, 1), dtype=float)

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        ke_total = ke_t + ke_r
        # Three active clumps, sum over each clump_id
        ke_0 = jnp.sum(jnp.where(env.state.clump_id == 0, ke_total, 0.0))
        ke_1 = jnp.sum(jnp.where(env.state.clump_id == 1, ke_total, 0.0))
        ke_2 = jnp.sum(jnp.where(env.state.clump_id == 2, ke_total, 0.0))
        ke_arr = jnp.array([ke_0, ke_1, ke_2])
        env.env_params["curr_ke"] = ke_arr
        env.env_params["prev_ke"] = ke_arr

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ThreeGears.step")
    def step(env: "ThreeGears", action: jax.Array) -> Environment:
        r"""Advance the environment by one physics step.

        Applies torque to the active gears, computes inter-gear forces,
        and applies viscous drag.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            Actions for all gears.

        Returns
        -------
        Environment
            Updated environment after physics integration and sensor updates.
        """
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        env.env_params["action"] = reshaped_action
        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]

        # Action is shape (3, 1). Clumps are 0, 1, 2.
        # So we can just index it by clump_id.
        action_torque = reshaped_action[env.state.clump_id]

        env.system = env.system.force_manager.add_torque(
            env.state, env.system, action_torque
        )

        force_drag = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(
            env.state, env.system, force_drag, is_com=True
        )

        torque_drag = -env.env_params["friction"] * env.state.ang_vel
        env.system = env.system.force_manager.add_torque(
            env.state, env.system, torque_drag
        )

        env.state, env.system = env.system.step(env.state, env.system)
        env.env_params["curr_dist"] = _total_objective_distance(env)

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        ke_total = ke_t + ke_r
        ke_0 = jnp.sum(jnp.where(env.state.clump_id == 0, ke_total, 0.0))
        ke_1 = jnp.sum(jnp.where(env.state.clump_id == 1, ke_total, 0.0))
        ke_2 = jnp.sum(jnp.where(env.state.clump_id == 2, ke_total, 0.0))
        env.env_params["curr_ke"] = jnp.array([ke_0, ke_1, ke_2])

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ThreeGears.observation")
    def observation(env: "ThreeGears") -> jax.Array:
        r"""Build the observation vector.

        The observation vector contains 22 features:

        ====================================  ====================================
        Feature                               Size
        ====================================  ====================================
        Distance to floor                     ``1``
        Distance to left/right walls          ``2``
        Unit vector to target                 ``2``
        Clamped displacement to target        ``2``
        Unit vector to neighbor j             ``2``
        Clamped displacement to neighbor j    ``2``
        :math:`\sin(\Delta\theta_j)`          ``1``
        :math:`\cos(\Delta\theta_j)`          ``1``
        Unit vector to neighbor k             ``2``
        Clamped displacement to neighbor k    ``2``
        :math:`\sin(\Delta\theta_k)`          ``1``
        :math:`\cos(\Delta\theta_k)`          ``1``
        Velocity (x, y)                       ``2``
        Angular velocity                      ``1``
        ====================================  ====================================

        Returns
        -------
        jax.Array
            Observation vector of size ``22``.
        """
        idx = _gear_indices(env.state)  # [0, 1, 2]
        idx_j = jnp.array([1, 2, 0])
        idx_k = jnp.array([2, 0, 1])

        pos_c = env.state.pos_c[idx]  # (3, 2)
        pos_j = env.state.pos_c[idx_j]
        pos_k = env.state.pos_c[idx_k]

        q_z = env.state.q.xyz[idx, 2]
        q_w = env.state.q.w[idx, 0]
        theta = 2 * jnp.arctan2(q_z, q_w)[:, None]  # (3, 1)

        q_z_j = env.state.q.xyz[idx_j, 2]
        q_w_j = env.state.q.w[idx_j, 0]
        theta_j = 2 * jnp.arctan2(q_z_j, q_w_j)[:, None]

        q_z_k = env.state.q.xyz[idx_k, 2]
        q_w_k = env.state.q.w[idx_k, 0]
        theta_k = 2 * jnp.arctan2(q_z_k, q_w_k)[:, None]

        delta_theta_j = theta_j - theta
        delta_theta_k = theta_k - theta

        vel = env.state.vel[idx]  # (3, 2)
        w = env.state.ang_vel[idx].reshape(3, 1)  # (3, 1)

        obj = env.env_params["objective"]  # (3, 2)
        delta_obj = env.system.domain.displacement(pos_c, obj, env.system)

        delta_j = env.system.domain.displacement(pos_c, pos_j, env.system)
        delta_k = env.system.domain.displacement(pos_c, pos_k, env.system)

        dist_left = pos_c[:, 0:1]
        dist_right = env.env_params["box_size"] - pos_c[:, 0:1]
        dist_floor = pos_c[:, 1:2] + y_min - 1.0

        obs = jnp.concatenate(
            [
                dist_floor,  # 1
                dist_left,  # 1
                dist_right,  # 1
                unit(delta_obj),  # 2
                jnp.clip(delta_obj, -3.0, 3.0),  # 2
                unit(delta_j),  # 2
                jnp.clip(delta_j, -3.0, 3.0),  # 2
                jnp.sin(delta_theta_j),  # 1
                jnp.cos(delta_theta_j),  # 1
                unit(delta_k),  # 2
                jnp.clip(delta_k, -3.0, 3.0),  # 2
                jnp.sin(delta_theta_k),  # 1
                jnp.cos(delta_theta_k),  # 1
                vel,  # 2
                w,  # 1
            ],
            axis=-1,
        )  # shape (3, 22)
        return obs

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ThreeGears.reward")
    def reward(env: "ThreeGears") -> jax.Array:
        r"""Compute the cooperative reward.

        The shared reward is based on the differential distance to the objective
        minus a penalty for the change in kinetic energy:

        .. math::

            R_t = (d_{t-1} - d_t) - w_{\text{ke}} \sum_i (K_t^i - K_{t-1}^i)

        where :math:`d_t` is the total distance to the objective at step :math:`t`,
        :math:`K_t^i` is the kinetic energy of agent :math:`i`, and :math:`w_{\text{ke}}`
        is the kinetic energy weight.

        Returns
        -------
        jax.Array
            Reward value, identical for all agents.
        """
        shaping_reward = env.env_params["prev_dist"] - env.env_params["curr_dist"]

        ke_diff = env.env_params["curr_ke"] - env.env_params["prev_ke"]
        ke_penalty = env.env_params["ke_weight"] * ke_diff

        # Shared reward
        return shaping_reward - ke_penalty

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ThreeGears.done")
    def done(env: "ThreeGears") -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 1

    @property
    def action_space_shape(self) -> tuple[int]:
        return (1,)

    @property
    def observation_space_size(self) -> int:
        return 22

    @property
    def max_num_agents(self) -> int:
        return 3
