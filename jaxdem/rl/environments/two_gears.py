# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Two-dimensional environment where N dynamic gears assemble a tower."""

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
        [0.98118480, 0.00000000],
        [0.97396423, 0.02258206],
        [0.95933126, 0.04159232],
        [0.94411538, 0.06015988],
        [0.93521554, 0.08215346],
        [0.94049978, 0.10529838],
        [0.95231984, 0.12619441],
        [0.96332838, 0.14750621],
        [0.96615682, 0.17103258],
        [0.95480521, 0.19186102],
        [0.93701956, 0.20796408],
        [0.91885969, 0.22366147],
        [0.90658297, 0.24394742],
        [0.90810372, 0.26765351],
        [0.91617450, 0.29026529],
        [0.92320153, 0.31319695],
        [0.92153437, 0.33682156],
        [0.90645946, 0.35515266],
        [0.88609178, 0.36783775],
        [0.86552420, 0.38020974],
        [0.85019285, 0.39827920],
        [0.84789814, 0.42193718],
        [0.85197573, 0.44559872],
        [0.85479097, 0.46941392],
        [0.84868665, 0.49228443],
        [0.83042772, 0.50746630],
        [0.80812883, 0.51633320],
        [0.78575791, 0.52502382],
        [0.76776895, 0.54042761],
        [0.76170500, 0.56342446],
        [0.76166302, 0.58743593],
        [0.76017127, 0.61136717],
        [0.74984914, 0.63265592],
        [0.72905662, 0.64415107],
        [0.70553642, 0.64892222],
        [0.68201643, 0.65368443],
        [0.66183043, 0.66604335],
        [0.65214219, 0.68777840],
        [0.64797634, 0.71142686],
        [0.64222132, 0.73470000],
        [0.62805460, 0.75363446],
        [0.60546619, 0.76103810],
        [0.58147081, 0.76156659],
        [0.55748647, 0.76226871],
        [0.53561572, 0.77128226],
        [0.52254406, 0.79118292],
        [0.51437296, 0.81376349],
        [0.50453695, 0.83562283],
        [0.48703963, 0.85151385],
        [0.46345369, 0.85456759],
        [0.43974171, 0.85084101],
        [0.41598801, 0.84746976],
        [0.39298323, 0.85292250],
        [0.37686001, 0.87045982],
        [0.36492249, 0.89129688],
        [0.35132116, 0.91102983],
        [0.33113015, 0.92329773],
        [0.30737680, 0.92189714],
        [0.28469534, 0.91403726],
        [0.26185727, 0.90669718],
        [0.23829371, 0.90846306],
        [0.21953418, 0.92316637],
        [0.20418274, 0.94163402],
        [0.18725587, 0.95859462],
        [0.16510800, 0.96679067],
        [0.14201987, 0.96098815],
        [0.12108111, 0.94924634],
        [0.09981393, 0.93815586],
        [0.07627722, 0.93620057],
        [0.05536992, 0.94767055],
        [0.03706096, 0.96321177],
        [0.01735879, 0.97684223],
        [-0.00593584, 0.98066580],
        [-0.02755286, 0.97066743],
        [-0.04609379, 0.95541604],
        [-0.06518329, 0.94090027],
        [-0.08811146, 0.93528082],
        [-0.11061670, 0.94320150],
        [-0.13133615, 0.95534536],
        [-0.15317009, 0.96519497],
        [-0.17675774, 0.96450537],
        [-0.19615247, 0.95066247],
        [-0.21171764, 0.93238243],
        [-0.22808816, 0.91486292],
        [-0.24984397, 0.90572590],
        [-0.27335013, 0.90987380],
        [-0.29585838, 0.91825019],
        [-0.31910786, 0.92399083],
        [-0.34212245, 0.91881278],
        [-0.35862410, 0.90160880],
        [-0.37073042, 0.88087406],
        [-0.38392119, 0.86085589],
        [-0.40397201, 0.84843268],
        [-0.42785046, 0.84868459],
        [-0.45146956, 0.85303600],
        [-0.47536870, 0.85447384],
        [-0.49696388, 0.84499735],
        [-0.51000461, 0.82502979],
        [-0.51827844, 0.80248896],
        [-0.52792164, 0.78054555],
        [-0.54577857, 0.76514677],
        [-0.56938575, 0.76148370],
        [-0.59340172, 0.76167371],
        [-0.61715999, 0.75875663],
        [-0.63654059, 0.74533034],
        [-0.64567432, 0.72328936],
        [-0.64986236, 0.69964505],
        [-0.65569287, 0.67640126],
        [-0.67092189, 0.65840827],
        [-0.69361686, 0.65091785],
        [-0.71730123, 0.64693601],
        [-0.74012949, 0.63975629],
        [-0.75658101, 0.62287481],
        [-0.76149791, 0.59951884],
        [-0.76147440, 0.57550562],
        [-0.76333646, 0.55161946],
        [-0.77556852, 0.53147543],
        [-0.79673043, 0.52035035],
        [-0.81936234, 0.51231307],
        [-0.84049914, 0.50110545],
        [-0.85341446, 0.48139143],
        [-0.85395035, 0.45752055],
        [-0.84972043, 0.43388200],
        [-0.84757084, 0.41002569],
        [-0.85651021, 0.38822452],
        [-0.87555587, 0.37375872],
        [-0.89644414, 0.36190658],
        [-0.91518136, 0.34704143],
        [-0.92408359, 0.32522258],
        [-0.92022438, 0.30165093],
        [-0.91192303, 0.27911665],
        [-0.90583142, 0.25595749],
        [-0.91126245, 0.23303218],
        [-0.92766281, 0.21561367],
        [-0.94616727, 0.20030469],
        [-0.96187424, 0.18227660],
        [-0.96643449, 0.15915882],
        [-0.95831558, 0.13668678],
        [-0.94620283, 0.11594996],
        [-0.93634864, 0.09413164],
        [-0.93814048, 0.07064162],
        [-0.95143600, 0.05074281],
        [-0.96698737, 0.03244207],
        [-0.97913270, 0.01185422],
        [-0.97918234, -0.01170831],
        [-0.96708284, -0.03232107],
        [-0.95153515, -0.05062481],
        [-0.93820145, -0.07050007],
        [-0.93631096, -0.09398219],
        [-0.94612534, -0.11581673],
        [-0.95824318, -0.13655072],
        [-0.96641331, -0.15900616],
        [-0.96195053, -0.18214268],
        [-0.94628273, -0.20020260],
        [-0.92778065, -0.21551434],
        [-0.91134510, -0.23290209],
        [-0.90581808, -0.25580394],
        [-0.91186950, -0.27897211],
        [-0.92017741, -0.30150413],
        [-0.92409172, -0.32506867],
        [-0.91528151, -0.34692427],
        [-0.89657591, -0.36182664],
        [-0.87568894, -0.37368096],
        [-0.85661241, -0.38810915],
        [-0.84758222, -0.40987199],
        [-0.84969249, -0.43373043],
        [-0.85393032, -0.45736773],
        [-0.85345166, -0.48124186],
        [-0.84061952, -0.50100922],
        [-0.81950619, -0.51225776],
        [-0.79687481, -0.52029642],
        [-0.77568764, -0.53137762],
        [-0.76337234, -0.55146958],
        [-0.76147291, -0.57535150],
        [-0.76150552, -0.59936491],
        [-0.75664596, -0.62273505],
        [-0.74026579, -0.63968435],
        [-0.71745262, -0.64690708],
        [-0.69376831, -0.65088927],
        [-0.67105491, -0.65833043],
        [-0.65575238, -0.67625909],
        [-0.64988737, -0.69949297],
        [-0.64570936, -0.72313927],
        [-0.63663096, -0.74520549],
        [-0.61730736, -0.75871150],
        [-0.59355584, -0.76167204],
        [-0.56953986, -0.76148128],
        [-0.54592216, -0.76509078],
        [-0.52800332, -0.78041485],
        [-0.51832927, -0.80234346],
        [-0.51006597, -0.82488840],
        [-0.49707641, -0.84489204],
        [-0.47552190, -0.85445703],
        [-0.45162154, -0.85306160],
        [-0.42800274, -0.84870837],
        [-0.40412260, -0.84839987],
        [-0.38402292, -0.86074011],
        [-0.37080556, -0.88073949],
        [-0.35870983, -0.90148072],
        [-0.34225303, -0.91873091],
        [-0.31926151, -0.92400287],
        [-0.29600345, -0.91830225],
        [-0.27349617, -0.90992305],
        [-0.24999784, -0.90571701],
        [-0.22820732, -0.91476516],
        [-0.21181486, -0.93226284],
        [-0.19625982, -0.95055189],
        [-0.17690157, -0.96444998],
        [-0.15331882, -0.96523540],
        [-0.13146976, -0.95542219],
        [-0.11075231, -0.94327474],
        [-0.08826484, -0.93529602],
        [-0.06531674, -0.94082318],
        [-0.04621021, -0.95531504],
        [-0.02767836, -0.97057796],
        [-0.00608762, -0.98063901],
        [0.01722017, -0.97690959],
        [0.03694299, -0.96331095],
        [0.05524864, -0.94776565],
        [0.07612808, -0.93623944],
        [0.09966969, -0.93810154],
        [0.12094898, -0.94916699],
        [0.14188031, -0.96092274],
        [0.16495390, -0.96679349],
        [0.18713213, -0.95868652],
        [0.20408407, -0.94175241],
        [0.21943068, -0.92328058],
        [0.23815240, -0.90852460],
        [0.26170611, -0.90666715],
        [0.28455147, -0.91398198],
        [0.30722773, -0.92185802],
        [0.33097946, -0.92333009],
        [0.35121657, -0.91114304],
        [0.36484617, -0.89143078],
        [0.37677724, -0.87058983],
        [0.39285317, -0.85300520],
        [0.41583396, -0.84746489],
        [0.43959044, -0.85081149],
        [0.46330000, -0.85455603],
        [0.48689797, -0.85157458],
        [0.50445508, -0.83575341],
        [0.51432137, -0.81390872],
        [0.52248431, -0.79132499],
        [0.53550001, -0.77138408],
        [0.55733371, -0.76228921],
        [0.58131671, -0.76156376],
        [0.60531294, -0.76105453],
        [0.62792728, -0.75372131],
        [0.64216493, -0.73484343],
        [0.64795103, -0.71157889],
        [0.65210713, -0.68792848],
        [0.66173185, -0.66616183],
        [0.68186913, -0.65372977],
        [0.70538417, -0.64894620],
        [0.72890888, -0.64419499],
        [0.74974092, -0.63276567],
        [0.76014224, -0.61151854],
        [0.76166477, -0.58759005],
        [0.76169559, -0.56357829],
        [0.76768983, -0.54055988],
        [0.78562010, -0.52509283],
        [0.80798307, -0.51638328],
        [0.83029042, -0.50753632],
        [0.84860161, -0.49241297],
        [0.85479027, -0.46956805],
        [0.85200443, -0.44575014],
        [0.84791461, -0.42209042],
        [0.85013508, -0.39842209],
        [0.86539969, -0.38030058],
        [0.88595695, -0.36791243],
        [0.90633719, -0.35524649],
        [0.92147571, -0.33696409],
        [0.92322911, -0.31334858],
        [0.91622926, -0.29040936],
        [0.90814555, -0.26780185],
        [0.90654790, -0.24409750],
        [0.91875193, -0.22377165],
        [0.93689981, -0.20806110],
        [0.95470206, -0.19197554],
        [0.96612675, -0.17118374],
        [0.96338326, -0.14765024],
        [0.95239893, -0.12632669],
        [0.94056575, -0.10543767],
        [0.93520398, -0.08230715],
        [0.94402737, -0.06028641],
        [0.95923026, -0.04170874],
        [0.97388369, -0.02271347],
    ],
    dtype=float,
)

rr = 1.0  # gear radius
_rad = 0.04  # radius of spheres that compose the gears
N = pos_p.shape[0]  # number of spheres per gear
pos = jnp.broadcast_to(jnp.asarray([[0.0, 0.0]]), (N, 2))
rad = jnp.broadcast_to(jnp.asarray([_rad]), (N,))
volume = 2.895698
inertia = 0.461264
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
    themselves. Each episode samples a random target x and stacks
    ``num_gears`` objectives vertically into a tower. Gear ``i`` must reach
    level ``i``, bottom to top. The gears spawn at random, non-overlapping
    floor positions, not necessarily under the tower, and must navigate to
    assemble the stack. A pairwise magnetic force attracts the gears to each
    other, and each gear observes its nearest neighbor.

    Note
    ----
    The maximum torque must be at least ``4.0 * mgr`` so the gear can climb
    correctly, and the attraction must be at least ``1 * mg``. For realistic
    training parameters, ``skip_frames = 50`` gives a response rate of
    200 Hz, so ``num_steps_epoch = 100`` gives a horizon of 0.5 seconds.
    ``box_size`` must fit ``num_gears`` gears of radius ``rr`` side by side
    on the floor (``box_size >= 2*rr*(num_gears+1)``) and fit the tower
    height ``2*rr*num_gears`` vertically.
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
    @partial(jax.named_call, name="TwoGears.reset")
    def reset(env: TwoGears, key: jax.Array) -> Environment:
        """Reset the environment to a random initial configuration.

        Parameters
        ----------
        env : Environment
            The current environment.
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

        The step applies each gear's torque, computes the pairwise attraction
        force between all gears, and applies viscous drag.

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

        Each gear receives a 16-feature observation. The "other gear" slot
        holds its nearest neighbor:

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
        r"""Compute the per-gear reward.

        The reward is the differential distance to the objective minus a
        penalty for the change in kinetic energy:

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
