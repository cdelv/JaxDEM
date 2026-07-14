# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Two-dimensional environment with two gears for RL training."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from ...colliders import DynamicCellList
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils.linalg import (
    cross,
    cross_3X3D_1X2D,
    dot,
    norm,
    unit,
    unit_and_norm,
)
from . import Environment

pos_p = jnp.asarray(
    [
        [0.7802483, -0.05938501],
        [0.85563521, -0.06901893],
        [0.93102213, -0.07865286],
        [1.0039932, -0.06766816],
        [1.05620428, -0.04753258],
        [1.10222651, -0.02348541],
        [1.10222651, 0.02348541],
        [1.05251482, 0.04921093],
        [1.00011246, 0.06883811],
        [0.94521566, 0.07928084],
        [0.87145329, 0.07104037],
        [0.7802483, 0.05938501],
        [0.7695042, 0.13599226],
        [0.75230416, 0.21135057],
        [0.72874561, 0.285033],
        [0.80084688, 0.3090623],
        [0.87294815, 0.33309159],
        [0.93392673, 0.37464942],
        [0.97223079, 0.41544449],
        [1.00326171, 0.45707854],
        [0.98288183, 0.49939778],
        [0.92693126, 0.50100658],
        [0.87120245, 0.49595353],
        [0.81721121, 0.48154327],
        [0.75432902, 0.44211458],
        [0.67721323, 0.39204109],
        [0.6342945, 0.45640015],
        [0.58610105, 0.51683283],
        [0.53290593, 0.57299673],
        [0.58744101, 0.62592995],
        [0.64197609, 0.67886316],
        [0.67888462, 0.74276308],
        [0.69569506, 0.79613768],
        [0.70558862, 0.84711247],
        [0.66886536, 0.8763983],
        [0.6177576, 0.85357174],
        [0.56974012, 0.82483927],
        [0.52734807, 0.78843015],
        [0.48780064, 0.72562257],
        [0.44004779, 0.64704863],
        [0.373455, 0.6864124],
        [0.30381344, 0.71995],
        [0.23151768, 0.74747143],
        [0.25768523, 0.8188245],
        [0.28385278, 0.89017756],
        [0.28938108, 0.96376341],
        [0.2813684, 1.01914604],
        [0.26816505, 1.06936539],
        [0.22237189, 1.07981738],
        [0.18622947, 1.03707654],
        [0.15543376, 0.99035548],
        [0.13303717, 0.93915877],
        [0.12465736, 0.86541211],
        [0.11572548, 0.77390025],
        [0.03864815, 0.78047225],
        [-0.03864815, 0.78047225],
        [-0.11572548, 0.77390025],
        [-0.12310826, 0.84954081],
        [-0.13049105, 0.92518137],
        [-0.15743793, 0.99387857],
        [-0.18868673, 1.04030003],
        [-0.22237189, 1.07981738],
        [-0.26816505, 1.06936539],
        [-0.28218369, 1.01517561],
        [-0.28965816, 0.95971963],
        [-0.28762337, 0.90387548],
        [-0.26317585, 0.83379617],
        [-0.23151768, 0.74747143],
        [-0.30381344, 0.71995],
        [-0.373455, 0.6864124],
        [-0.44004779, 0.64704863],
        [-0.47951865, 0.71199515],
        [-0.51898952, 0.77694167],
        [-0.57307442, 0.8271439],
        [-0.62137013, 0.85540984],
        [-0.66886536, 0.8763983],
        [-0.70558862, 0.84711247],
        [-0.69470691, 0.79220671],
        [-0.67737973, 0.73899955],
        [-0.65131658, 0.68956856],
        [-0.59888384, 0.63703667],
        [-0.53290593, 0.57299673],
        [-0.58610105, 0.51683283],
        [-0.6342945, 0.45640015],
        [-0.67721323, 0.39204109],
        [-0.7409545, 0.43343012],
        [-0.80469576, 0.47481914],
        [-0.8752065, 0.49658323],
        [-0.93098356, 0.50109523],
        [-0.98288183, 0.49939778],
        [-1.00326171, 0.45707854],
        [-0.96963491, 0.41233155],
        [-0.93093793, 0.37191154],
        [-0.88600855, 0.33868414],
        [-0.81597555, 0.31410425],
        [-0.72874561, 0.285033],
        [-0.75230416, 0.21135057],
        [-0.7695042, 0.13599226],
        [-0.7802483, 0.05938501],
        [-0.85563521, 0.06901893],
        [-0.93102213, 0.07865286],
        [-1.0039932, 0.06766816],
        [-1.05620428, 0.04753258],
        [-1.10222651, 0.02348541],
        [-1.10222651, -0.02348541],
        [-1.05251482, -0.04921093],
        [-1.00011246, -0.06883811],
        [-0.94521566, -0.07928084],
        [-0.87145329, -0.07104037],
        [-0.7802483, -0.05938501],
        [-0.7695042, -0.13599226],
        [-0.75230416, -0.21135057],
        [-0.72874561, -0.285033],
        [-0.80084688, -0.3090623],
        [-0.87294815, -0.33309159],
        [-0.93392673, -0.37464942],
        [-0.97223079, -0.41544449],
        [-1.00326171, -0.45707854],
        [-0.98288183, -0.49939778],
        [-0.92693126, -0.50100658],
        [-0.87120245, -0.49595353],
        [-0.81721121, -0.48154327],
        [-0.75432902, -0.44211458],
        [-0.67721323, -0.39204109],
        [-0.6342945, -0.45640015],
        [-0.58610105, -0.51683283],
        [-0.53290593, -0.57299673],
        [-0.58744101, -0.62592995],
        [-0.64197609, -0.67886316],
        [-0.67888462, -0.74276308],
        [-0.69569506, -0.79613768],
        [-0.70558862, -0.84711247],
        [-0.66886536, -0.8763983],
        [-0.6177576, -0.85357174],
        [-0.56974012, -0.82483927],
        [-0.52734807, -0.78843015],
        [-0.48780064, -0.72562257],
        [-0.44004779, -0.64704863],
        [-0.373455, -0.6864124],
        [-0.30381344, -0.71995],
        [-0.23151768, -0.74747143],
        [-0.25768523, -0.8188245],
        [-0.28385278, -0.89017756],
        [-0.28938108, -0.96376341],
        [-0.2813684, -1.01914604],
        [-0.26816505, -1.06936539],
        [-0.22237189, -1.07981738],
        [-0.18622947, -1.03707654],
        [-0.15543376, -0.99035548],
        [-0.13303717, -0.93915877],
        [-0.12465736, -0.86541211],
        [-0.11572548, -0.77390025],
        [-0.03864815, -0.78047225],
        [0.03864815, -0.78047225],
        [0.11572548, -0.77390025],
        [0.12310826, -0.84954081],
        [0.13049105, -0.92518137],
        [0.15743793, -0.99387857],
        [0.18868673, -1.04030003],
        [0.22237189, -1.07981738],
        [0.26816505, -1.06936539],
        [0.28218369, -1.01517561],
        [0.28965816, -0.95971963],
        [0.28762337, -0.90387548],
        [0.26317585, -0.83379617],
        [0.23151768, -0.74747143],
        [0.30381344, -0.71995],
        [0.373455, -0.6864124],
        [0.44004779, -0.64704863],
        [0.47951865, -0.71199515],
        [0.51898952, -0.77694167],
        [0.57307442, -0.8271439],
        [0.62137013, -0.85540984],
        [0.66886536, -0.8763983],
        [0.70558862, -0.84711247],
        [0.69470691, -0.79220671],
        [0.67737973, -0.73899955],
        [0.65131658, -0.68956856],
        [0.59888384, -0.63703667],
        [0.53290593, -0.57299673],
        [0.58610105, -0.51683283],
        [0.6342945, -0.45640015],
        [0.67721323, -0.39204109],
        [0.7409545, -0.43343012],
        [0.80469576, -0.47481914],
        [0.8752065, -0.49658323],
        [0.93098356, -0.50109523],
        [0.98288183, -0.49939778],
        [1.00326171, -0.45707854],
        [0.96963491, -0.41233155],
        [0.93093793, -0.37191154],
        [0.88600855, -0.33868414],
        [0.81597555, -0.31410425],
        [0.72874561, -0.285033],
        [0.75230416, -0.21135057],
        [0.7695042, -0.13599226],
    ],
    dtype=float,
)

rr = 1.0  # gear radius
_rad = 0.04  # radius of spheres that compose the gears
N = pos_p.shape[0]  # number of spheres per gear
pos = jnp.broadcast_to(jnp.asarray([[0.0, 0.0]]), (N, 2))
rad = jnp.broadcast_to(jnp.asarray([_rad]), (N,))
volume = 3.024044
inertia = 1.56906198
q = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
y_min = jnp.min((pos + pos_p - rad[..., None])[:, 1])
one_second = 2.021757e1

# Units used in the simulation
# 1 m' = 24 mm
# 1 kg' = 98 g
# 1 m'/s'^2 = 9.81 m/s^2 (gravity)
#
# This is for training. 1 action every 1/200 seconds and 0.5 second horizon.
# num_steps_epoch = 100
# reset_every = 20
# skip_frames = 50
# max_torque = 28.6 (*mgr)
# attraction force = 5 (*mg)


@partial(jax.named_call, name="two_gears.frictional_floor_force")
def frictional_floor_force(
    pos: jax.Array, state: State, system: System
) -> Tuple[jax.Array, jax.Array]:
    # 1. Wall Definition
    n = jnp.array([0.0, 1.0])
    p = jnp.array([0.0, 1.0])
    pos_p = pos - state.pos_c - state.rad[..., None] * n

    # 2. Material Properties
    k_n = 2e5
    mu = 0.4
    restitution = 0.6
    gamma_n = (-2.0 * jnp.log(restitution) * jnp.sqrt(k_n * state.mass)) / jnp.sqrt(
        jnp.pi**2 + jnp.log(restitution) ** 2
    )

    # 3. Compute normal force
    vc = state.vel + cross_3X3D_1X2D(state.ang_vel, pos_p)
    vn_scalar = dot(vc, n)
    dist = dot(pos - p, n) - state.rad
    overlap = jnp.maximum(0.0, -dist)
    active = jnp.where(overlap > 0.0, 1.0, 0.0)
    Fn_scalar = k_n * overlap - gamma_n * vn_scalar
    Fn_scalar = jnp.maximum(0.0, Fn_scalar)
    Fn = Fn_scalar[..., None] * n

    # 4. Compute tangential force
    gamma_t = gamma_n
    vt = vc - vn_scalar[..., None] * n
    t, vt_scalar = unit_and_norm(vt)
    Ft_viscous = gamma_t * vt_scalar
    Ft_coulomb = mu * Fn_scalar
    Ft_scalar = jnp.minimum(Ft_viscous, Ft_coulomb)
    Ft = -Ft_scalar[..., None] * t

    # 5. Total Force & Torque
    F_total = Fn + Ft
    Torque = cross(pos_p, F_total)

    return F_total * active[..., None], Torque * active[..., None]


def _clump_first_indices(state: State, n: int) -> jax.Array:
    """First particle index of each clump ``0..n-1``. Shape ``(n,)``."""
    return jnp.stack([jnp.argmax(state.clump_id == i) for i in range(int(n))])


def _measure(
    state: State, system: System, env_params: dict
) -> Tuple[jax.Array, jax.Array]:
    """Per-gear distance to its objective and total kinetic energy.

    Gear ``i`` (``clump_id == i``) is paired with ``objective[i]``. Returns
    ``(curr_dist, curr_ke)`` each of shape ``(num_gears,)``.
    """
    n = env_params["objective"].shape[0]
    idx = _clump_first_indices(state, n)
    pos_c = state.pos_c[idx]
    delta = system.domain.displacement(pos_c, env_params["objective"], system)
    curr_dist = norm(delta)

    import jaxdem.utils.thermal as thermal

    ke_total = thermal.compute_translational_kinetic_energy_per_particle(
        state
    ) + thermal.compute_rotational_kinetic_energy_per_particle(state)
    cid = state.clump_id
    curr_ke = jnp.stack(
        [jnp.sum(jnp.where(cid == i, ke_total, 0.0)) for i in range(int(n))]
    )
    return curr_dist, curr_ke


@Environment.register("TwoGears")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class TwoGears(Environment):
    r"""Two-dimensional environment with N dynamic gears building a tower.

    All ``num_gears`` gears are dynamic agents that each apply torque to
    themselves. Each episode samples a random target x and stacks ``num_gears``
    objectives vertically into a tower (gear ``i`` must reach level ``i``,
    bottom to top). The gears spawn at random, non-overlapping floor positions —
    not necessarily under the tower — and must navigate to assemble the stack.
    Gears attract each other pairwise via a magnetic force, and each gear
    observes its nearest neighbour.

    Note
    ----
    After experimentation, one needs the max torque to be at least ``4.0 * mgr``
    for the gear to be able to climb correctly, and attraction at least ``1 * mg``.
    If one wants some realistic parameters for training, ``skip_frames = 50``
    will give a response rate of 200 Hz, meaning that ``num_steps_epoch = 100``
    gives a horizon of 0.5 seconds. ``box_size`` must fit ``num_gears`` gears of
    radius ``rr`` side by side on the floor (``box_size >= 2*rr*(num_gears+1)``)
    and fit the tower height ``2*rr*num_gears`` vertically.
    """

    num_gears: int = jax.tree.static()
    """Number of gears (agents) that must form the tower."""

    @classmethod
    @partial(jax.named_call, name="TwoGears.Create")
    def Create(
        cls,
        num_gears: int = 3,
        box_size: float = 20.0,
        max_steps: int = 10000 * 10,  # 10000 steps = 1 second
        friction: float = 0.2,
        ke_weight: float = 0.1,
        attraction_mag: float = 4.0,
    ) -> TwoGears:
        r"""Create an N-gear tower environment.

        Parameters
        ----------
        num_gears : int
            Number of dynamic gears (agents) that must form the tower.
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
        TwoGears
            A freshly constructed environment (call :meth:`reset` before use).
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
    @partial(jax.named_call, name="TwoGears.reset")
    def reset(env: TwoGears, key: jax.Array) -> Environment:
        """Reset the environment to a random initial configuration.

        Parameters
        ----------
        env : Environment
            The environment instance to reset.
        key : jax.Array
            PRNG key used to sample the initial positions and objective.

        Returns
        -------
        Environment
            The environment with a fresh episode state.
        """
        n = env.num_gears
        key, key_obj, key_x = jax.random.split(key, 3)
        box = jnp.array([env.env_params["box_size"], env.env_params["box_size"]])
        y_floor = 1.0

        # Random tower location: n objectives stacked vertically at the same x.
        x_obj = jax.random.uniform(key_obj, minval=rr, maxval=box[0] - rr)
        levels_y = y_floor + 2.0 * rr * jnp.arange(n) - y_min  # (n,)
        objective = jnp.stack(
            [jnp.broadcast_to(x_obj, (n,)), levels_y], axis=1
        )  # (n, 2)
        env.env_params["objective"] = objective

        # Spawn gears on the floor on a jittered 1-D grid, kept >= 2*rr apart
        # (cell width w; jitter is capped at (w - 2*rr)/2 so neighbours don't overlap).
        w = (box[0] - 2.0 * rr) / n
        centers_x = rr + (jnp.arange(n) + 0.5) * w  # (n,)
        jitter_amp = jnp.maximum(0.0, (w - 2.0 * rr) / 2.0)
        jitter = jax.random.uniform(key_x, (n,), minval=-jitter_amp, maxval=jitter_amp)
        xs = centers_x + jitter  # (n,)

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

        # In the paper, the microcontroller updates at 200 Hz (5 ms),
        # so we can make an action every 50 time steps
        env.system = System.create(
            env.state.shape,
            dt=2e-3,  # 1 / 10000 (0.1 ms) in real units
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

        env.env_params["action"] = jnp.zeros((env.num_gears, 1))

        curr_dist, curr_ke = _measure(env.state, env.system, env.env_params)
        env.env_params["curr_dist"] = curr_dist
        env.env_params["prev_dist"] = curr_dist
        env.env_params["curr_ke"] = curr_ke
        env.env_params["prev_ke"] = curr_ke

        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="TwoGears.step")
    def step(env: TwoGears, action: jax.Array) -> Environment:
        r"""Advance the environment by one step.

        Applies each gear's torque, computes the pairwise attraction force
        between all gears, and applies viscous drag.

        The attraction on gear :math:`i` from gear :math:`j` is:

        .. math::

            \mathbf{F}_{ij} = - \frac{C}{d_{ij}^3} \hat{n}_{ij},

        when :math:`d_{ij} < 3 r`, where :math:`d_{ij}` is the center-to-center
        distance, :math:`\hat{n}_{ij} = \mathrm{unit}(\mathbf{r}_i - \mathbf{r}_j)`
        (so the force points from :math:`i` toward :math:`j`), and
        :math:`C = m_{\text{attr}} (2r)^3` with :math:`r` the gear radius.
        The net force on gear :math:`i` is :math:`\sum_{j \ne i} \mathbf{F}_{ij}`.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            Torque action for each gear, shape ``(num_gears, 1)``.

        Returns
        -------
        Environment
            Updated environment after physics integration and sensor updates.
        """
        action = action.reshape(env.max_num_agents, *env.action_space_shape)
        env.env_params["action"] = action

        # Apply each gear's torque to its own clump (clump_id 0 and 1).
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

        # Pairwise attraction between all gears (gear i pulled toward every
        # other gear j within 3*rr): F_on_i = -sum_j (C/d_ij^3) * unit(r_i - r_j).
        n = env.num_gears
        idx = _clump_first_indices(env.state, n)
        centers = env.state.pos_c[idx]  # (n, 2)
        pair = env.system.domain.displacement(
            centers[:, None, :], centers[None, :, :], env.system
        )  # (n, n, 2): centers[i] - centers[j]
        dist = norm(pair)  # (n, n)
        dist = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dist)  # exclude self
        C = env.env_params["attraction_mag"] * (2.0 * rr) ** 3
        F_mag = (C / dist**3) * (dist < 3.0 * rr)  # self -> 0
        F_per_gear = -jnp.sum(F_mag[..., None] * unit(pair), axis=1)  # (n, 2)
        F_attraction = F_per_gear[env.state.clump_id]  # (num_particles, 2)

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
    @partial(jax.named_call, name="TwoGears.observation")
    def observation(env: TwoGears) -> jax.Array:
        r"""Build the per-gear observation vector.

        Each gear receives a 16-feature observation; the "other gear" slot is
        filled by its nearest neighbour:

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

        Returns
        -------
        jax.Array
            Observation of shape ``(num_gears, 16)`` — one row per gear.
        """
        n = env.num_gears
        idx = _clump_first_indices(env.state, n)
        pos_c = env.state.pos_c[idx]  # (n, 2)

        # Nearest other gear per gear (for n == 2 this is just the other gear).
        pair = env.system.domain.displacement(
            pos_c[:, None, :], pos_c[None, :, :], env.system
        )  # (n, n, 2)
        dists = norm(pair)
        dists = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, dists)
        nearest = jnp.argmin(dists, axis=1)  # (n,)
        pos_c_other = pos_c[nearest]  # (n, 2)

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
    @partial(jax.named_call, name="TwoGears.reward")
    def reward(env: TwoGears) -> jax.Array:
        r"""Compute the reward.

        The reward is based on the differential distance to the objective
        minus a penalty for the change in kinetic energy:

        .. math::

            R_t = (d_{t-1} - d_t) - w_{\text{ke}} (K_t - K_{t-1})

        where :math:`d_t` is the distance from gear :math:`i` to its objective at
        step :math:`t`, :math:`K_t` is that gear's kinetic energy at step
        :math:`t`, and :math:`w_{\text{ke}}` is the weight for the kinetic energy
        penalty.

        Returns
        -------
        jax.Array
            Per-gear reward of shape ``(num_gears,)``.
        """
        shaping_reward = env.env_params["prev_dist"] - env.env_params["curr_dist"]

        ke_diff = env.env_params["curr_ke"] - env.env_params["prev_ke"]
        ke_penalty = env.env_params["ke_weight"] * ke_diff

        return shaping_reward - ke_penalty

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="TwoGears.done")
    def done(env: TwoGears) -> jax.Array:
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
