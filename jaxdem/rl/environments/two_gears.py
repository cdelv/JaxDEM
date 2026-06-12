# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Two-dimensional environment with two gears for RL training."""

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
    vt_scalar = jnp.squeeze(vt_scalar)
    Ft_viscous = gamma_t * vt_scalar
    Ft_coulomb = mu * Fn_scalar
    Ft_scalar = jnp.minimum(Ft_viscous, Ft_coulomb)
    Ft = -Ft_scalar[..., None] * t

    # 5. Total Force & Torque
    F_total = Fn + Ft
    Torque = cross(pos_p, F_total)

    return F_total * active[..., None], Torque * active[..., None]


@Environment.register("TwoGears")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class TwoGears(Environment):
    r"""Two-dimensional environment with two gears.

    The environment consists of two gears composed of spheres. One gear is
    frozen on the floor, and the other is an active agent that can apply torque
    to itself. The objective is to navigate the active gear to a specified target
    position above the frozen gear. The active gear is attracted to the frozen gear by a magnetic force.

    Note
    ----
    After experimentation, one needs the max torque to be at least ``4.0 * mgr``
    for the gear to be able to climb correctly, and attraction at least ``1 * mg``.
    If one wants some realistic parameters for training, ``skip_frames = 50``
    will give a response rate of 200 Hz, meaning that ``num_steps_epoch = 100``
    gives a horizon of 0.5 seconds.
    """

    @classmethod
    @partial(jax.named_call, name="TwoGears.Create")
    def Create(
        cls,
        box_size: float = 10.0,
        max_steps: int = 10000 * 10,  # 10000 steps = 1 second
        friction: float = 0.2,
        ke_weight: float = 0.1,
        attraction_mag: float = 4.0,
    ) -> "TwoGears":
        r"""Create a two-gears 2-D environment.

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
        attraction_mag : float
            Magnitude of the attraction force between the two gears.

        Returns
        -------
        TwoGears
            A freshly constructed environment (call :meth:`reset` before use).
        """
        dim = 2
        state = State.create(pos=jnp.zeros((2 * N, dim)))
        system = System.create(state.shape)

        env_params = {
            "box_size": jnp.asarray(box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "ke_weight": jnp.asarray(ke_weight, dtype=float),
            "attraction_mag": jnp.asarray(attraction_mag, dtype=float),
            "action": jnp.zeros((1, 1)),
            "objective": jnp.zeros((1, 2)),
            "curr_dist": jnp.zeros((1,)),
            "prev_dist": jnp.zeros((1,)),
            "curr_ke": jnp.zeros((1,)),
            "prev_ke": jnp.zeros((1,)),
        }
        return cls(state=state, system=system, env_params=env_params)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="TwoGears.reset")
    def reset(env: "TwoGears", key: jax.Array) -> Environment:
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
        key, key_x1, key_y1, key_obj = jax.random.split(key, 4)
        box = jnp.array([env.env_params["box_size"], env.env_params["box_size"]])

        x_obj = jax.random.uniform(key_obj, minval=rr, maxval=box[0] - rr)
        y_floor = 1.0

        # Objective for the second gear (clump 1)
        # Frozen gear bottom is at y_floor, top is at y_floor + 2*rr.
        # Active gear bottom should be at y_floor + 2*rr, so its ideal center is at y_floor + 3*rr.
        objective = jnp.array([[x_obj, y_floor + 2 * rr - y_min]])
        env.env_params["objective"] = objective

        x0_f = x_obj

        # Sample x1_f ensuring it is at least 3*rr away from x0_f and 1*rr away from walls
        a = rr
        b = jnp.maximum(a, x0_f - 3.0 * rr)
        c = jnp.minimum(box[0] - rr, x0_f + 3.0 * rr)
        d = box[0] - rr

        len1 = b - a
        len2 = d - c

        valid1 = len1 >= 2.0 * rr
        valid2 = len2 >= 2.0 * rr

        len1 = jnp.where(valid1, len1, 0.0)
        len2 = jnp.where(valid2, len2, 0.0)

        x = jax.random.uniform(key_x1, minval=0.0, maxval=len1 + len2)
        x1_f = jnp.where(x < len1, a + x, c + (x - len1))

        y_max_val = y_floor + 2 * rr

        # Frozen gear on the floor (ideal center at y_floor + rr)
        y0_f = y_floor + rr
        # Active gear randomly placed
        y1_f = jax.random.uniform(key_y1, minval=y_floor + rr, maxval=y_max_val)

        # Shift the user's ideal center so the actual bottom of the gear is at y - rr
        y0_shifted = y0_f - rr - y_min
        y1_shifted = y1_f - rr - y_min

        pos0_c = jnp.array([[x0_f, y0_shifted]])
        pos1_c = jnp.array([[x1_f, y1_shifted]])

        state = State.create()
        state = State.add_clump(
            state,
            pos=pos + pos0_c,
            rad=rad,
            pos_p=pos_p,
            volume=volume,
            inertia=inertia,
            q=q,
            fixed=jnp.ones((1,), dtype=bool),
        )
        state = State.add_clump(
            state,
            pos=pos + pos1_c,
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

        env.env_params["action"] = jnp.zeros((1, 1))

        idx_1 = jnp.argmax(env.state.clump_id == 1)
        idx = jnp.array([idx_1])
        pos_c = env.state.pos_c[idx]
        delta = env.system.domain.displacement(
            pos_c, env.env_params["objective"], env.system
        )
        dist = norm(delta)
        env.env_params["curr_dist"] = dist
        env.env_params["prev_dist"] = dist

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        ke_total = ke_t + ke_r
        ke_agent = jnp.sum(jnp.where(env.state.clump_id == 1, ke_total, 0.0))
        ke_agent_arr = jnp.array([ke_agent])
        env.env_params["curr_ke"] = ke_agent_arr
        env.env_params["prev_ke"] = ke_agent_arr

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="TwoGears.step")
    def step(env: "TwoGears", action: jax.Array) -> Environment:
        r"""Advance the environment by one step.

        Applies torque to the active agent, computes the attraction force
        between the gears, and applies viscous drag.

        The attraction force is defined as:

        .. math::

            \mathbf{F}_{\text{attraction}} = - \frac{C}{d^3} \hat{n},

        when :math:`d < 3 r`, where :math:`d` is the distance between the centers, :math:`\hat{n}` is the unit
        vector from the frozen gear to the active gear, and :math:`C` is determined by
        ``attraction_mag`` as :math:`C = m_{\text{attr}} (2r)^3`. r is the gear radius.

        Parameters
        ----------
        env : Environment
            Current environment.
        action : jax.Array
            Actions for the active gear.

        Returns
        -------
        Environment
            Updated environment after physics integration and sensor updates.
        """
        action = action.reshape(env.max_num_agents, *env.action_space_shape)
        env.env_params["action"] = action

        # Apply torque only to active agent (clump_id 1)
        action_torque = jnp.where(
            env.state.clump_id[:, None] == 1,
            action[0, :],
            jnp.zeros_like(action[0, :]),
        )

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

        # Attraction force between the two gears
        idx_0 = jnp.argmax(env.state.clump_id == 0)
        idx_1 = jnp.argmax(env.state.clump_id == 1)
        pos_0 = env.state.pos_c[idx_0]
        pos_1 = env.state.pos_c[idx_1]
        delta = env.system.domain.displacement(pos_1, pos_0, env.system)
        direction, dist = unit_and_norm(delta)

        C = env.env_params["attraction_mag"] * (2.0 * rr) ** 3
        F_mag = C / dist**3

        # Use norm and unit utilities
        F_1 = -F_mag * direction * (dist < 3.0 * rr)
        F_0 = -F_1

        F_attraction = jnp.where(
            env.state.clump_id[:, None] == 1,
            F_1,
            jnp.where(
                env.state.clump_id[:, None] == 0, F_0, jnp.zeros_like(env.state.pos_c)
            ),
        )

        env.system = env.system.force_manager.add_force(
            env.state, env.system, F_attraction, is_com=True
        )

        env.env_params["prev_dist"] = env.env_params["curr_dist"]
        env.env_params["prev_ke"] = env.env_params["curr_ke"]

        env.state, env.system = env.system.step(env.state, env.system)

        # Compute observables and rewards
        idx_1 = jnp.argmax(env.state.clump_id == 1)
        idx = jnp.array([idx_1])
        pos_c = env.state.pos_c[idx]
        delta = env.system.domain.displacement(
            pos_c, env.env_params["objective"], env.system
        )
        env.env_params["curr_dist"] = norm(delta)

        import jaxdem.utils.thermal as thermal

        ke_t = thermal.compute_translational_kinetic_energy_per_particle(env.state)
        ke_r = thermal.compute_rotational_kinetic_energy_per_particle(env.state)
        ke_total = ke_t + ke_r
        ke_agent = jnp.sum(jnp.where(env.state.clump_id == 1, ke_total, 0.0))
        env.env_params["curr_ke"] = jnp.array([ke_agent])

        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="TwoGears.observation")
    def observation(env: "TwoGears") -> jax.Array:
        r"""Build the observation vector.

        The observation vector contains 16 features:

        ====================================  ====================================
        Feature                               Size
        ====================================  ====================================
        Distance to floor                     ``1``
        Distance to left/right walls          ``2``
        Unit vector to target                 ``2``
        Clamped displacement to target        ``2``
        Unit vector to frozen gear            ``2``
        Clamped displacement to frozen gear   ``2``
        :math:`\sin(\Delta\theta)`            ``1``
        :math:`\cos(\Delta\theta)`            ``1``
        Velocity (x, y)                       ``2``
        Angular velocity                      ``1``
        ====================================  ====================================

        Returns
        -------
        jax.Array
            Observation vector of size ``16``.
        """
        idx_0 = jnp.argmax(env.state.clump_id == 0)
        idx_1 = jnp.argmax(env.state.clump_id == 1)
        idx = jnp.array([idx_1])
        idx_other = jnp.array([idx_0])

        pos_c = env.state.pos_c[idx]
        pos_c_other = env.state.pos_c[idx_other]

        q_z = env.state.q.xyz[idx, 2]
        q_w = env.state.q.w[idx, 0]
        theta = 2 * jnp.arctan2(q_z, q_w)[:, None]

        q_z_other = env.state.q.xyz[idx_other, 2]
        q_w_other = env.state.q.w[idx_other, 0]
        theta_other = 2 * jnp.arctan2(q_z_other, q_w_other)[:, None]

        delta_theta = theta_other - theta

        vel = env.state.vel[idx]
        w = env.state.ang_vel[idx].reshape(env.max_num_agents, 1)

        delta_obj = env.system.domain.displacement(
            pos_c, env.env_params["objective"], env.system
        )
        delta_frozen = env.system.domain.displacement(pos_c, pos_c_other, env.system)

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
                unit(delta_frozen),
                jnp.clip(delta_frozen, -3.0, 3.0),
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
    def reward(env: "TwoGears") -> jax.Array:
        r"""Compute the reward.

        The reward is based on the differential distance to the objective
        minus a penalty for the change in kinetic energy:

        .. math::

            R_t = (d_{t-1} - d_t) - w_{\text{ke}} (K_t - K_{t-1})

        where :math:`d_t` is the distance to the objective at step :math:`t`,
        :math:`K_t` is the kinetic energy at step :math:`t`, and :math:`w_{\text{ke}}` is the
        weight for the kinetic energy penalty.

        Returns
        -------
        jax.Array
            Reward value for the active agent.
        """
        shaping_reward = env.env_params["prev_dist"] - env.env_params["curr_dist"]

        ke_diff = env.env_params["curr_ke"] - env.env_params["prev_ke"]
        ke_penalty = env.env_params["ke_weight"] * ke_diff

        return shaping_reward - ke_penalty

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="TwoGears.done")
    def done(env: "TwoGears") -> jax.Array:
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
        return 1
