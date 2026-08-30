# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where N dynamic gears assemble a triangular stack."""

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
    _clump_first_indices,
    _measure,
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


def _triangle_layout(n: int) -> tuple[int, list[int]]:
    r"""Gears per row (bottom to top) for an ``n``-gear triangular stack.

    The bottom row is the smallest ``m`` with ``m(m+1)/2 >= n``. Each row
    above shrinks by one, and the top row takes the remainder. Examples:
    ``3 -> [2,1]``, ``4 -> [3,1]``, ``5 -> [3,2]``, ``6 -> [3,2,1]``,
    ``7 -> [4,3]``. Returns ``(m, rows)``.
    """
    m = 1
    while m * (m + 1) // 2 < n:
        m += 1
    rows: list[int] = []
    remaining, size = n, m
    while remaining > 0:
        c = min(size, remaining)
        rows.append(c)
        remaining -= c
        size -= 1
    return m, rows


@Environment.register("ThreeGears")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ThreeGears(Environment):
    r"""N dynamic gears that must assemble a triangular stack.

    The dynamics, pairwise attraction, nearest-neighbor observation, and
    per-gear reward match :class:`TwoGears`. Only the objective differs: the
    ``num_gears`` targets form a triangular stack. The rows shrink by one
    from bottom to top and the gears touch. ``num_gears=3`` gives the
    classic triangle ``[2,1]``, ``5 -> [3,2]``, and ``6 -> [3,2,1]``.
    Gear ``i`` is paired with objective ``i``.

    Note
    ----
    As with :class:`TwoGears`, ``skip_frames = 50`` gives a 200 Hz response
    rate, so ``num_steps_epoch = 100`` is a 0.5 s horizon. ``box_size`` must
    fit the stack: width ``2*m*rr`` and height ``(2 + (m-1)*sqrt(3))*rr``
    (``m`` = bottom row size). It must also be ``>= 2*rr*(num_gears+1)``
    wide for a non-overlapping spawn.
    """

    num_gears: int = jax.tree.static()
    """Number of gears (agents) forming the triangular stack."""

    @classmethod
    @partial(jax.named_call, name="ThreeGears.Create")
    def Create(
        cls,
        num_gears: int = 6,
        box_size: float = 30.0,
        max_steps: int = 10000 * 10,  # 10000 steps = 1 second
        friction: float = 0.2,
        ke_weight: float = 0.1,
        attraction_mag: float = 2.0,
    ) -> ThreeGears:
        r"""Create an N-gear triangular-stack environment.

        Parameters
        ----------
        num_gears : int
            Number of dynamic gears (agents) forming the stack.
        box_size : float
            Size of the square bounding box.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
        ke_weight : float
            Weight for the differential kinetic energy penalty.
        attraction_mag : float
            Magnitude of the pairwise attraction force between gears.

        Returns
        -------
        ThreeGears
            The constructed environment. Call :meth:`reset` before use.
        """
        dim = 2
        n = int(num_gears)
        state = State.create(pos=jnp.zeros((n * N, dim)))
        system = System.create(state.shape)

        env_params = {
            "box_size": jnp.asarray(box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ke_weight": jnp.asarray(ke_weight, dtype=float),
            "attraction_mag": jnp.asarray(attraction_mag, dtype=float),
            "action": jnp.zeros((n, 1)),
            "objective": jnp.zeros((n, 2)),
            "curr_dist": jnp.zeros((n,)),
            "prev_dist": jnp.zeros((n,)),
            "curr_ke": jnp.zeros((n,)),
            "prev_ke": jnp.zeros((n,)),
        }
        return cls(state=state, system=system, env_params=env_params, num_gears=n)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ThreeGears.reset")
    def reset(env: ThreeGears, key: jax.Array) -> Environment:
        """Reset with the gears on the floor and a random triangular-stack objective."""
        n = env.num_gears
        key, key_obj, key_x = jax.random.split(key, 3)
        box = jnp.array([env.env_params["box_size"], env.env_params["box_size"]])
        y_floor = 1.0

        # Triangular-stack objective: rows shrink by one from bottom to top
        # (3 -> [2,1], 5 -> [3,2], 6 -> [3,2,1]). Gears touch within a row
        # (2*rr apart, centred) and rows stack sqrt(3)*rr high (nestled).
        m, rows = _triangle_layout(n)
        sqrt3 = jnp.sqrt(3.0)
        x_obj = jax.random.uniform(key_obj, minval=m * rr, maxval=box[0] - m * rr)
        pts = []
        for r, c in enumerate(rows):
            yy = y_floor + r * rr * sqrt3 - y_min
            for i in range(c):
                xx = x_obj + (i - (c - 1) / 2.0) * 2.0 * rr
                pts.append([xx, yy])
        objective = jnp.array(pts)  # (n, 2)
        env.env_params["objective"] = objective

        # Spawn gears on the floor on a jittered 1-D grid, kept >= 2*rr apart.
        w = (box[0] - 2.0 * rr) / n
        centers_x = rr + (jnp.arange(n) + 0.5) * w
        jitter_amp = jnp.maximum(0.0, (w - 2.0 * rr) / 2.0)
        jitter = jax.random.uniform(key_x, (n,), minval=-jitter_amp, maxval=jitter_amp)
        xs = centers_x + jitter

        y_shifted = y_floor - y_min
        state = State.create()
        for i in range(n):
            pos_i_c = jnp.array([[xs[i], y_shifted]])
            state = State.add_clump(
                state,
                pos=pos + pos_i_c,
                rad=rad,
                pos_p=pos_p,
                volume=volume,
                inertia=inertia,
                q=q,
            )
        env.state = state

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

        env.env_params["action"] = jnp.zeros((n, 1))

        curr_dist, curr_ke = _measure(env.state, env.system, env.env_params)
        env.env_params["curr_dist"] = curr_dist
        env.env_params["prev_dist"] = curr_dist
        env.env_params["curr_ke"] = curr_ke
        env.env_params["prev_ke"] = curr_ke

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ThreeGears.step")
    def step(env: ThreeGears, action: jax.Array) -> Environment:
        r"""Advance one step: per-gear torque, pairwise attraction, viscous drag.

        Attraction on gear :math:`i` from gear :math:`j` is
        :math:`-(C/d_{ij}^3)\,\hat{n}_{ij}` when :math:`d_{ij} < 3r`, with
        :math:`\hat{n}_{ij}=\mathrm{unit}(\mathbf{r}_i-\mathbf{r}_j)` and
        :math:`C = m_{\text{attr}}(2r)^3`. Net force on :math:`i` is
        :math:`\sum_{j\ne i}`.
        """
        n = env.num_gears
        action = action.reshape(env.max_num_agents, *env.action_space_shape)
        env.env_params["action"] = action

        action_torque = action[env.state.clump_id]
        env.system = env.system.force_manager.add_torque(
            env.state,
            env.system,
            action_torque - env.env_params["friction"] * env.state.ang_vel,
        )
        env.system = env.system.force_manager.add_force(
            env.state,
            env.system,
            -env.env_params["friction"] * env.state.vel,
            is_com=True,
        )

        # Pairwise attraction between all gears.
        idx = _clump_first_indices(env.state, n)
        centers = env.state.pos_c[idx]  # (n, 2)
        pair = env.system.domain.displacement(
            centers[:, None, :], centers[None, :, :], env.system
        )  # (n, n, 2)
        dist = norm(pair)
        dist = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist)
        C = env.env_params["attraction_mag"] * (2.0 * rr) ** 3
        F_mag = (C / dist**3) * (dist < 3.0 * rr)
        F_per_gear = -jnp.sum(F_mag[..., None] * unit(pair), axis=1)
        F_attraction = F_per_gear[env.state.clump_id]
        env.system = env.system.force_manager.add_force(
            env.state, env.system, F_attraction, is_com=True
        )

        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]

        env.state, env.system = env.system.step(env.state, env.system)

        env.env_params["curr_dist"], env.env_params["curr_ke"] = _measure(
            env.state, env.system, env.env_params
        )
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ThreeGears.observation")
    def observation(env: ThreeGears) -> jax.Array:
        r"""Per-gear observation (16 features). The "other gear" slot holds the nearest neighbor.

        ====================================  ====================================
        Feature                               Size
        ====================================  ====================================
        Distance to floor                     ``1``
        Distance to left/right walls          ``2``
        Unit vector to target                 ``2``
        Clamped displacement to target        ``2``
        Unit vector to nearest gear           ``2``
        Clamped displacement to nearest gear  ``2``
        :math:`\sin(\Delta\theta)`            ``1``
        :math:`\cos(\Delta\theta)`            ``1``
        Velocity (x, y)                       ``2``
        Angular velocity                      ``1``
        ====================================  ====================================
        """
        n = env.num_gears
        idx = _clump_first_indices(env.state, n)
        pos_c = env.state.pos_c[idx]  # (n, 2)

        pair = env.system.domain.displacement(
            pos_c[:, None, :], pos_c[None, :, :], env.system
        )
        dists = norm(pair)
        dists = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dists)
        nearest = jnp.argmin(dists, axis=1)
        pos_c_other = pos_c[nearest]

        q_z = env.state.q.xyz[idx, 2]
        q_w = env.state.q.w[idx, 0]
        theta = 2 * jnp.arctan2(q_z, q_w)[:, None]

        q_z_other = env.state.q.xyz[nearest, 2]
        q_w_other = env.state.q.w[nearest, 0]
        theta_other = 2 * jnp.arctan2(q_z_other, q_w_other)[:, None]
        delta_theta = theta_other - theta

        vel = env.state.vel[idx]
        w = env.state.ang_vel[idx].reshape(n, 1)

        delta_obj = env.system.domain.displacement(
            pos_c, env.env_params["objective"], env.system
        )
        delta_other = env.system.domain.displacement(pos_c, pos_c_other, env.system)

        dist_left = pos_c[:, 0:1]
        dist_right = env.env_params["box_size"] - pos_c[:, 0:1]
        dist_floor = pos_c[:, 1:2] + y_min - 1.0

        return jnp.concatenate(
            [
                dist_floor,
                dist_left,
                dist_right,
                unit(delta_obj),
                jnp.clip(delta_obj, -3.0, 3.0),
                unit(delta_other),
                jnp.clip(delta_other, -3.0, 3.0),
                jnp.sin(delta_theta),
                jnp.cos(delta_theta),
                vel,
                w,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="ThreeGears.reward")
    def reward(env: ThreeGears) -> jax.Array:
        r"""Per-gear shaping reward.

        .. math::

            R_i = (d_{i,t-1} - d_{i,t}) - w_{\text{ke}} (K_{i,t} - K_{i,t-1})
        """
        shaping = env.env_params["prev_dist"] - env.env_params["curr_dist"]
        ke_diff = env.env_params["curr_ke"] - env.env_params["prev_ke"]
        return shaping - env.env_params["ke_weight"] * ke_diff

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="ThreeGears.done")
    def done(env: ThreeGears) -> jax.Array:
        return jnp.asarray(env.system.step_count > env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 1

    @property
    def action_space_shape(self) -> tuple[int]:
        return (1,)

    @property
    def observation_space_size(self) -> int:
        return 16

    @property
    def max_num_agents(self) -> int:
        return self.num_gears


__all__ = ["ThreeGears"]
