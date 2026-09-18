# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple rolling agents cooperatively cover a set of objectives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import isfinite

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...material_matchmakers import MaterialMatchmaker
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils import cross_lidar_2d, lidar_2d
from . import Environment
from .single_roller import frictional_wall_force


@jax.jit(static_argnames=("N",))
@partial(jax.named_call, name="swarm_roller._sample_objectives_3d")
def _sample_objectives_3d(
    key: ArrayLike, N: int, box: jax.Array, gap: float, rad: float
) -> jax.Array:
    r"""Sample *N* positions on a jittered X-Y grid at floor level (``z = rad``).

    Parameters
    ----------
    key : ArrayLike
        JAX random number generator key.
    N : int
        Number of points, static under JIT. Zero returns an empty array.
    box : jax.Array
        Horizontal side lengths, shape ``(2,)``.
    gap : float
        Requested minimum center separation in the horizontal plane.
    rad : float
        Particle radius and height assigned to every point.

    Returns
    -------
    jax.Array
        Positions of shape ``(N, 3)`` relative to the horizontal box origin.

    Notes
    -----
    Jitter leaves a margin of ``gap / 2`` inside sufficiently large cells.
    When a cell dimension is smaller than ``gap``, jitter is zero in that
    direction and the requested separation cannot be guaranteed.
    """
    if N == 0:
        return jnp.zeros((0, 3))
    i = jax.lax.iota(int, N)
    Lx, Ly = box[0], box[1]
    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)
    ix, iy = jnp.mod(i, nx), i // nx
    dx, dy = Lx / nx, Ly / ny
    base = jnp.stack([(ix + 0.5) * dx, (iy + 0.5) * dy, jnp.full((N,), rad)], axis=1)
    # Jitter capped so each centre stays >= gap/2 inside its X-Y cell.
    noise = jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0) * jnp.asarray(
        [jnp.maximum(0.0, dx / 2 - gap / 2), jnp.maximum(0.0, dy / 2 - gap / 2), 0.0]
    )
    return base + noise


def _sample_padding_ring_3d(
    key: ArrayLike, N: int, box: float, pad: float, gap: float, rad: float
) -> jax.Array:
    r"""Sample *N* points on jittered grids filling the padding ring around ``box``.

    Parameters
    ----------
    key : ArrayLike
        JAX random number generator key, split for the four strips.
    N : int
        Number of points. Zero returns an empty array.
    box : float
        Side length of the inner square containing objectives.
    pad : float
        Total additional horizontal side length. Each strip extends
        ``pad / 2`` beyond the inner square.
    gap : float
        Requested center separation passed to :func:`_sample_objectives_3d`.
    rad : float
        Particle radius and height assigned to each point.

    Returns
    -------
    jax.Array
        Positions of shape ``(N, 3)``, outside the inner square and inside
        its padded boundary, all with ``z = rad``.

    Notes
    -----
    Top and bottom strips include the corners; left and right strips cover
    only the inner square's height. Each strip receives ``N // 4`` points,
    with the remainder assigned to the right strip. Cell size limits the
    achievable separation as described in :func:`_sample_objectives_3d`.
    """
    if N == 0:
        return jnp.zeros((0, 3))
    t = pad / 2.0
    L = box + pad
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = N // 4
    n4 = N - 3 * n
    bottom = _sample_objectives_3d(k1, n, jnp.asarray([L, t]), gap, rad) + jnp.asarray(
        [-t, -t, 0.0]
    )
    top = _sample_objectives_3d(k2, n, jnp.asarray([L, t]), gap, rad) + jnp.asarray(
        [-t, box, 0.0]
    )
    left = _sample_objectives_3d(k3, n, jnp.asarray([t, box]), gap, rad) + jnp.asarray(
        [-t, 0.0, 0.0]
    )
    right = _sample_objectives_3d(
        k4, n4, jnp.asarray([t, box]), gap, rad
    ) + jnp.asarray([box, 0.0, 0.0])
    return jnp.concatenate([bottom, top, left, right], axis=0)


@Environment.register("swarmRoller")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmRoller(Environment):
    r"""Cooperative objective coverage with torque-controlled rolling spheres.

    Objectives are unassigned points on a jittered grid at floor height;
    agents spawn in the surrounding padding ring. Reward and objective/peer
    LiDAR semantics match :class:`SwarmNavigator`. Rolling dynamics match
    :class:`MultiRoller`: gravity, frictional contact with the :math:`z=0`
    floor, normal spring particle contacts, and the naive all-pairs collider.
    No tangential particle-contact history is stored.

    Each physics step smooths the requested torque:

    .. math::

       \mathbf{u}_{i,t}=\alpha\mathbf{a}_{i,t}
           +(1-\alpha)\mathbf{u}_{i,t-1},\qquad
       \boldsymbol{\tau}_{i,t}=\mathbf{u}_{i,t}-\mu\boldsymbol{\omega}_{i,t},
       \qquad \mathbf{f}_{i,t}=-\mu\mathbf{v}_{i,t}.

    ``action_alpha`` sets :math:`\alpha` (default 0.22); ``friction`` sets
    :math:`\mu`. The previous applied torque belongs to the immediately
    preceding physics step, independent of checkpoints. Reset initializes it
    to zero. A constant request completes 99 percent of its transition in
    19 physics steps with the default smoothing coefficient.

    The shared coverage potential and per-action reward are

    .. math::

       \Phi_i(s)=\max_{k\in V_i}(w_{ik}B(d_{ik}))
           +\eta\sum_{k\in V_i}w_{ik}P(d_{ik}/a_i),\qquad
       r_{i,t}=\Phi_i(s_{t+1})-\Phi_i(s_{\mathrm{checkpoint}}).

    Here :math:`a_i` is the particle radius, :math:`B(d)=\exp[-(d/(1.5a_i))^4]`,
    and :math:`w_{ik}=1/(1+C_{ik})` shares objective value with locally
    estimated peer claims. :meth:`reward` documents the approach potential
    :math:`P`, claim estimate :math:`C`, parameters, and references. Empty
    scans give zero potential. There is no separate occupancy bonus or
    kinetic-energy term. Agent and objective counts may differ.

    By default, observations hide objectives below half the LiDAR range if
    another detected agent is estimated to cover them. Goals within the
    observer's own radius remain visible. Reward always uses the raw scans.
    ``prev_potential`` stores only the action-start baseline; current LiDAR
    readings are computed from live state without persistent sensor caches.

    Notes
    -----
    The observation vector per agent is:

    ============================  =================
    Feature                       Size
    ============================  =================
    Planar velocity               ``2``
    Angular velocity              ``3``
    Normalized objective LiDAR     ``n_lidar_rays``
    Normalized agent/wall LiDAR    ``n_lidar_rays``
    ============================  =================

    With the default 12 angular bins, each observation has 29 entries.
    Velocities are not normalized. As in :class:`MultiRoller`, the 2-D LiDAR
    utility bins by azimuth but uses full center distances in three dimensions.
    All centers initially lie at the same height, so these are planar ranges.
    If particles lift off the floor, the bin-center occupancy estimate ignores
    relative elevation and becomes less accurate.

    With ``dt=0.002`` and ``skip_frames=49``, a policy action spans 50 physics
    steps (0.1 seconds). :func:`jaxdem.utils.advance_action` checkpoints once
    before these steps; reward is read before the next checkpoint or reset.
    Reset and unchanged states give zero reward. Methods describe one
    environment; use ``jax.vmap`` for independent batches.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxdem.rl import Environment
    >>> from jaxdem.utils import advance_action
    >>> env = Environment.create("swarmRoller")
    >>> env = env.reset(env, jax.random.key(0))
    >>> action = jnp.zeros((env.max_num_agents, env.action_space_size))
    >>> env, terminated, truncated = advance_action(env, action, skip_frames=49)
    >>> reward = env.reward(env)
    >>> observation = env.observation(env)
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of angular bins for each LiDAR sensor."""

    num_objectives: int = jax.tree.static()
    """Number of objectives sampled per environment."""

    @classmethod
    @partial(jax.named_call, name="SwarmRoller.Create")
    def Create(
        cls,
        N: int = 64,
        num_objectives: int = 64,
        box_size: float = 20.0,
        box_padding: float = 10.0,
        max_steps: int = 10000,
        friction: float = 0.2,
        action_alpha: float = 0.22,
        lidar_range: float = 16.0,
        n_lidar_rays: int = 12,
        attraction_coeff: float = 0.1,
        attraction_constant: float = 0.01,
        attraction_quadratic: float = 0.02,
        attraction_decay: float = 1 / 3,
        sharing_range: float = 1.5,
        hide_occupied_objectives: bool = True,
    ) -> SwarmRoller:
        r"""Create a swarm roller with shared objective value and filtered LiDAR.

        Parameters
        ----------
        N : int
            Number of agents.
        num_objectives : int
            Number of objectives sampled per environment.
        box_size : float
            Side length of the square domain that holds the objectives.
        box_padding : float
            Total extra domain side length in particle radii. The spawn ring
            extends by half this amount on each side of the objective box.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Translational drag and angular damping coefficient, applied as
            ``-friction * vel`` and ``-friction * ang_vel``.
        action_alpha : float
            Fraction of the requested torque applied by the smoothing update
            each physics step, in ``[0, 1]``. Defaults to 0.22, as in
            :class:`MultiRoller`. One disables smoothing.
        lidar_range : float
            Maximum detection range :math:`L` for the LiDAR sensors.
        n_lidar_rays : int
            Number of angular LiDAR bins spanning :math:`[-\pi, \pi)`.
        attraction_coeff : float
            Nonnegative weight of the summed approach potential, default 0.1.
            Peer claims reduce both approach and settling values. Zero
            disables only the approach term.
        attraction_constant, attraction_quadratic : float
            Nonnegative coefficients :math:`c,b` of the attraction strength
            :math:`(c+b x^2)e^{-\lambda x}`, with :math:`x=d/\mathrm{rad}_i`.
        attraction_decay : float
            Positive exponential decay :math:`\lambda` in radius units.
        sharing_range : float
            Positive quartic claim scale in observing-agent radii. The default
            matches the settling scale, 1.5.
        hide_occupied_objectives : bool
            Enabled by default. Hide readings strictly below half the LiDAR
            range when a detected peer appears within one observing-agent
            radius of the objective. Preserve goals within the observer's radius.
            A hidden reading becomes zero; farther goals behind it are not
            revealed. Applies only to observations, never reward or checkpoints.
            Set to False to recover the unfiltered observation.

        Returns
        -------
        SwarmRoller
            The constructed environment. Call :meth:`reset` before use.

        Raises
        ------
        ValueError
            If an attraction coefficient is nonfinite or negative, or
            ``attraction_decay`` or ``sharing_range`` is nonfinite or nonpositive;
            or ``action_alpha`` is outside ``[0, 1]``.

        Notes
        -----
        Construction allocates placeholder agent and objective arrays. Reset
        samples the geometry and initializes the physical system and reward
        baseline. ``N``, ``num_objectives``, and ``n_lidar_rays`` determine
        array shapes. The remaining controls are stored in ``env_params``
        as JAX arrays.
        """
        if not 0.0 <= action_alpha <= 1.0:
            raise ValueError("action_alpha must be in [0, 1]")
        if not isfinite(attraction_coeff) or attraction_coeff < 0:
            raise ValueError("attraction_coeff must be finite and nonnegative")
        for name, value in (
            ("attraction_constant", attraction_constant),
            ("attraction_quadratic", attraction_quadratic),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("attraction_decay", attraction_decay),
            ("sharing_range", sharing_range),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        dim = 3
        n_obj = int(num_objectives)
        state = State.create(pos=jnp.zeros((int(N), dim)))
        env_params = {
            "objective": jnp.zeros((n_obj, dim)),
            "box_size": jnp.asarray(box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "action_alpha": jnp.asarray(action_alpha, dtype=float),
            "applied_action": jnp.zeros_like(state.torque),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "attraction_coeff": jnp.asarray(attraction_coeff, dtype=float),
            "attraction_constant": jnp.asarray(attraction_constant, dtype=float),
            "attraction_quadratic": jnp.asarray(attraction_quadratic, dtype=float),
            "attraction_decay": jnp.asarray(attraction_decay, dtype=float),
            "sharing_range": jnp.asarray(sharing_range, dtype=float),
            "hide_occupied_objectives": jnp.asarray(
                hide_occupied_objectives, dtype=bool
            ),
            "prev_potential": jnp.zeros(int(N)),
        }
        return cls(
            state=state,
            system=System.create(state.shape),
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            num_objectives=n_obj,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.reset")
    def reset(env: SwarmRoller, key: ArrayLike) -> Environment:
        """Sample floor-level agents and objectives and initialize rolling physics.

        Parameters
        ----------
        env : SwarmRoller
            Environment retaining its configured geometry, sensor, actuator,
            and shared-reward parameters.
        key : ArrayLike
            JAX random number generator key, split for agents and objectives.

        Returns
        -------
        Environment
            Initialized environment with zero velocities and applied torque,
            restarted step count, and the initial potential checkpointed.
            Reward is initially zero.

        Notes
        -----
        Agents have unit radius and mass. Objectives and agents start at
        ``z=1``. The jittered grid requests center spacing 2.05; undersized
        cells cannot guarantee that separation. The domain's horizontal side
        length is ``box_size + box_padding``; its lower vertical boundary is
        ``-1``, allowing floor contact before boundary reflection.
        Reset installs gravity ``(0, 0, -1)``, the same frictional floor force
        as MultiRoller, elastic spring particle contacts, and a naive collider.
        The physics time step is 0.002. Objective coordinates of shape
        ``(num_objectives, 3)`` are stored in ``env_params["objective"]``.
        """
        key_pos, key_obj = jax.random.split(key)
        N, rad = env.max_num_agents, 1.0
        gap = 2.05 * rad
        box_s = env.env_params["box_size"]
        box = box_s * jnp.ones(2)
        box3 = box_s * jnp.ones(env.state.dim)
        padding = env.env_params["box_padding"] * rad

        env.env_params["objective"] = _sample_objectives_3d(
            key_obj, env.num_objectives, box, gap, rad
        )
        pos = _sample_padding_ring_3d(key_pos, int(N), box_s, padding, gap, rad)
        env.state = State.create(pos=pos, rad=rad * jnp.ones(N), mass=jnp.ones(N))

        matcher = MaterialMatchmaker.create("harmonic")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elastic",
                    density=1.0 / (4.0 / 3.0 * jnp.pi),
                    young=2e5,
                    poisson=0.3,
                )
            ],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflectsphere",
            domain_kw={
                "box_size": box3 + padding,
                "anchor": jnp.array([-padding / 2, -padding / 2, -rad]),
            },
            force_manager_kw={
                "gravity": [0.0, 0.0, -1.0],
                "force_functions": (frictional_wall_force,),
            },
            mat_table=mat_table,
            force_model_type="spring",
            collider_type="naive",
        )
        env.state, env.system = System.initialize(env.state, env.system)

        env.env_params["applied_action"] = jnp.zeros_like(env.state.torque)
        return SwarmRoller.checkpoint(
            env, jnp.zeros((N, env.action_space_size), dtype=env.state.pos.dtype)
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller._sense")
    def _sense(env: SwarmRoller) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Read raw agent/wall and objective LiDAR from the live state.

        Parameters
        ----------
        env : SwarmRoller
            Environment supplying positions, objectives, domain, and sensor
            configuration. No sensor cache or checkpoint is modified.

        Returns
        -------
        lidar : jax.Array
            Agent/wall proximity, shape ``(N, n_lidar_rays)``. Each bin keeps
            its nearest return; walls can hide peers in the same bin.
        lidar_obj : jax.Array
            Objective proximity, shape ``(N, n_lidar_rays)``. These readings
            are unfiltered, including objectives covered by other agents.
        ids : jax.Array
            Agent/wall return IDs, shape ``(N, n_lidar_rays)``. Valid peer
            returns use indices into the current particle array. Walls use
            ``-1``; empty bins use the observer's own index. Occupancy tests
            exclude self, negative IDs, and zero-proximity readings.

        Notes
        -----
        Proximity is :math:`\ell=\max(0,L-d)`, where :math:`L` is the sensor
        range and :math:`d` is center distance for agents and objectives.
        Empty bins return zero. Recover distance as :math:`d=L-\ell` only
        for a valid detection; a zero reading does not prove a target exists
        at distance :math:`L`. Observation normalization is applied later.
        """
        objective = env.env_params["objective"]
        lr = env.env_params["lidar_range"]

        _, _, lidar, ids, _ = lidar_2d(
            env.state, env.system, lr, env.n_lidar_rays, sense_edges=True
        )
        lidar_obj, _, _ = cross_lidar_2d(
            env.state.pos, objective, env.system, lr, env.n_lidar_rays
        )
        return lidar, lidar_obj, ids

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller._potential")
    def _potential(env: SwarmRoller) -> jax.Array:
        """Evaluate the shared per-agent potential using raw live scans.

        Parameters
        ----------
        env : SwarmRoller
            Current environment with reward coefficients in ``env_params``.

        Returns
        -------
        jax.Array
            Potential values of shape ``(N,)``. No objectives, or no valid
            objective returns for an agent, gives zero for that agent.

        Notes
        -----
        Peer claims are estimated by comparing all pairs of objective and
        peer bins, including across angular boundaries. The observation
        mask does not affect these scans. This method neither subtracts nor
        updates ``prev_potential``; see :meth:`reward` for all formulas.
        """
        if env.num_objectives == 0:
            return jnp.zeros_like(env.state.rad)
        peers, objectives, peer_ids = SwarmRoller._sense(env)
        lr = env.env_params["lidar_range"]
        distance = lr - objectives
        valid = objectives > 0
        # All bin pairs matter: a peer and its nearby goal may straddle a bin edge.
        angle = jnp.arange(env.n_lidar_rays) * (2 * jnp.pi / env.n_lidar_rays)
        cosine = jnp.cos(angle[:, None] - angle[None, :])
        peer_distance = lr - peers
        separation_sq = jnp.maximum(
            distance[:, :, None] ** 2
            + peer_distance[:, None, :] ** 2
            - 2 * distance[:, :, None] * peer_distance[:, None, :] * cosine,
            0.0,
        )
        peer_valid = (
            (peers > 0)
            & (peer_ids >= 0)
            & (peer_ids != jnp.arange(env.max_num_agents)[:, None])
        )
        scale = env.env_params["sharing_range"] * env.state.rad[:, None, None]
        claims = jnp.where(
            peer_valid[:, None, :],
            jnp.exp(-((separation_sq / scale**2) ** 2)),
            0.0,
        ).sum(axis=-1)
        # Include one prospective claim for self, even during approach.
        weight = 1 / (1 + claims)
        settling = jnp.exp(-((distance / (1.5 * env.state.rad[:, None])) ** 4))
        # Settling at several nearby goals must not add their values.
        local = jnp.where(valid, weight * settling, 0.0).max(axis=-1)
        x = distance / env.state.rad[:, None]
        decay = env.env_params["attraction_decay"]
        # Integral from x to infinity of (c + b*x**2)*exp(-decay*x).
        attraction = jnp.exp(-decay * x) * (
            env.env_params["attraction_constant"] / decay
            + env.env_params["attraction_quadratic"]
            * (x**2 / decay + 2 * x / decay**2 + 2 / decay**3)
        )
        return local + env.env_params["attraction_coeff"] * jnp.where(
            valid, weight * attraction, 0.0
        ).sum(axis=-1)

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller.checkpoint")
    def checkpoint(env: SwarmRoller, action: jax.Array) -> Environment:
        """Save the LiDAR-derived potential at the start of a policy action.

        Physics steps preserve this potential through all skipped frames.
        The baseline is independent of ``action``. Read reward before the
        next checkpoint or reset replaces it. Unchanged states give zero reward.

        Parameters
        ----------
        env : SwarmRoller
            Environment immediately before the next policy action.
        action : jax.Array
            Per-agent torque requests, normally shape ``(N, 3)``. Accepted
            for the environment checkpoint interface; its value is unused.

        Returns
        -------
        Environment
            Environment with ``env_params["prev_potential"]`` replaced by
            the current per-agent potential of shape ``(N,)``. Physical state
            and observations are not advanced. The immediately preceding
            applied torque is preserved independently.
        """
        env.env_params["prev_potential"] = SwarmRoller._potential(env)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller.step")
    def step(env: SwarmRoller, action: jax.Array) -> Environment:
        """Smooth the requested torque, then advance physics with damping.

        ``applied_action`` follows each accepted physics step. ``prev_potential``
        remains fixed until the next checkpoint or reset.

        Parameters
        ----------
        env : Environment
            The current environment.
        action : jax.Array
            The per-agent torque vectors.

        Returns
        -------
        Environment
            The updated environment state.

        """
        reshaped_action = action.reshape(env.max_num_agents, *env.action_space_shape)
        alpha = env.env_params["action_alpha"]
        applied_action = (
            alpha * reshaped_action + (1 - alpha) * env.env_params["applied_action"]
        )
        env.env_params["applied_action"] = applied_action
        torque = applied_action - env.env_params["friction"] * env.state.ang_vel
        force = -env.env_params["friction"] * env.state.vel
        env.system = env.system.force_manager.add_force(env.state, env.system, force)
        env.system = env.system.force_manager.add_torque(env.state, env.system, torque)

        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmRoller.observation")
    def observation(env: SwarmRoller) -> jax.Array:
        r"""Return planar velocity, angular velocity, and filtered local LiDAR.

        Parameters
        ----------
        env : SwarmRoller
            Environment whose current state is observed. Checkpointed
            potentials are not used to construct the observation.

        Returns
        -------
        jax.Array
            Shape ``(N, 5 + 2 * n_lidar_rays)``. Each row concatenates planar
            velocity (two components), angular velocity (three components),
            normalized objective bins, then normalized agent/wall bins.
            Velocities are in simulation units; LiDAR channels lie in ``[0, 1]``.
            This matches :class:`SwarmNavigator` sensor semantics and
            :class:`MultiRoller` velocity channels.

        Notes
        -----
        A valid raw LiDAR return is normalized as
        :math:`p_k=\max(0,1-d_k/L)`. Empty and masked bins are zero.
        ``hide_occupied_objectives=False`` returns unfiltered goal readings.
        Only LiDAR channels are normalized. With filtering enabled, hide
        objective bin :math:`k` when

        .. math::

           a_i < d_{ik} < L/2,\qquad
           \min_{q\in Q_i}\widehat\delta_{ikq}\le a_i,

        where :math:`a_i` is the observing radius, :math:`L` the LiDAR range,
        and :math:`Q_i` the valid detected peers, excluding walls and self.
        Bin-center directions give the approximate separation

        .. math::

           \widehat\delta_{ikq}^2=\max\!\left(
           d_{ik}^2+d_{iq}^2-2d_{ik}d_{iq}\cos(\theta_k-\theta_q),0\right).

        Every objective bin is compared with every peer bin. Own goals within
        one radius and readings at or beyond half range remain visible.
        Hidden readings become zero without revealing occluded goals. Reward
        retains the raw scans. Angular quantization and hidden peers can cause
        occupancy errors; this is not an exact global coverage test.

        The half-range mask and own-goal exception were developed for this
        environment's experiments. The congestion-game and reward-shaping
        references in :meth:`reward` motivate the reward, not this observation
        filter, and do not guarantee that discarding observations is harmless.
        """
        lidar, lidar_obj, peer_ids = SwarmRoller._sense(env)
        lr = env.env_params["lidar_range"]
        distance = lr - lidar_obj
        peer_distance = lr - lidar
        theta = jnp.arange(env.n_lidar_rays) * (2 * jnp.pi / env.n_lidar_rays)
        cosine = jnp.cos(theta[:, None] - theta[None, :])
        delta2 = jnp.maximum(
            distance[:, :, None] ** 2
            + peer_distance[:, None, :] ** 2
            - 2 * distance[:, :, None] * peer_distance[:, None, :] * cosine,
            0,
        )
        valid_peer = (
            (lidar > 0)
            & (peer_ids >= 0)
            & (peer_ids < env.max_num_agents)
            & (peer_ids != jnp.arange(env.max_num_agents)[:, None])
        )
        occupied = jnp.any(
            valid_peer[:, None, :] & (delta2 <= env.state.rad[:, None, None] ** 2),
            axis=-1,
        )
        hide = (
            env.env_params["hide_occupied_objectives"]
            & (lidar_obj > 0)
            & (distance < lr / 2)
            & (distance > env.state.rad[:, None])
            & occupied
        )
        lidar_obj = jnp.where(hide, 0, lidar_obj)
        return jnp.concatenate(
            [
                env.state.vel[..., :2],
                env.state.ang_vel,
                lidar_obj / lr,
                lidar / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller.reward")
    def reward(env: SwarmRoller) -> jax.Array:
        r"""Return the action-start difference in the shared coverage potential.

        Parameters
        ----------
        env : SwarmRoller
            Environment at the endpoint of the action interval, with
            ``prev_potential`` captured by :meth:`checkpoint` or :meth:`reset`.

        Returns
        -------
        jax.Array
            Per-agent rewards of shape ``(N,)``. Positive values indicate
            increased potential; negative values indicate decreased potential.
            Reading reward does not update the baseline or physical state.

        Notes
        -----
        Reward uses raw objective detections regardless of the observation
        mask. It is evaluated over the complete action interval, not summed
        once per skipped physics frame. There is no separate occupancy bonus.

        Normalized goal distance is :math:`x=d/a_i`,
        where :math:`a_i` is the observing agent's radius. Attraction strength
        and its integrated potential are

        .. math::

           A(x)=(c+bx^2)e^{-\lambda x},\qquad
           P(x)=e^{-\lambda x}\left[\frac{c}{\lambda}
             +b\left(\frac{x^2}{\lambda}+\frac{2x}{\lambda^2}
             +\frac{2}{\lambda^3}\right)\right],\qquad -P'(x)=A(x).

        ``attraction_constant``, ``attraction_quadratic``, and
        ``attraction_decay`` set :math:`c,b,\lambda`. Defaults 0.01, 0.02,
        and 1/3 place the strength peak near 5.92 radii; strength at two radii
        is about 47 percent of the peak. The potential itself is monotone.

        The potential estimates peer claims on each visible
        objective from all pairs of objective and peer LiDAR bins:

        .. math::

           C_{ik}=\sum_{q\in Q_i}
               \exp\!\left[-\left(\frac{\widehat\delta_{ikq}}{\rho a_i}\right)^4\right],
           \qquad w_{ik}=\frac{1}{1+C_{ik}},\qquad
           B_{ik}=\exp\!\left[-\left(\frac{d_{ik}}{1.5a_i}\right)^4\right],

           \Phi_i=\max_{k\in V_i}(w_{ik}B_{ik})
                    +\eta\sum_{k\in V_i}w_{ik}P(d_{ik}/a_i),\qquad
           r_{i,t}=\Phi_i(s_{t+1})-\Phi_i(s_{\mathrm{checkpoint}}).

        ``sharing_range`` sets :math:`\rho` (default 1.5) and
        ``attraction_coeff`` sets :math:`\eta` (default 0.1). The denominator
        includes one prospective claim for self, so arrival never suppresses
        the agent's own value. An objective with no detected claimants has
        full value; :math:`n-1` unit peer claims divide both terms by :math:`n`.
        Settling uses the best shared score, rather than an unweighted nearest
        goal or a sum of nearby settling bonuses. All weights are positive;
        there is no contact penalty or negative attraction.

        This approximates equal sharing in a congestion game (Rosenthal,
        1973; see References). Soft counts, angular quantization,
        occlusion, approach shaping, and discounted physical motion do not
        inherit the discrete game's full-coverage equilibrium guarantee.
        Empty scans have zero potential. No claim assignments are stored.

        Potential-based reward shaping is studied by Ng, Harada, and Russell
        (1999), with a stochastic-game extension by Lu, Schwartz, and Givigi
        (2011); see References. Their invariance results
        concern adding :math:`\gamma\Phi_i(s')-\Phi_i(s)` to an existing reward,
        with suitable terminal conditions. Here the undiscounted difference
        is itself the reward; for :math:`\gamma<1` it is not the discount-correct
        shaping term. These results do not guarantee coverage or convergence,
        nor do they apply to the observation filter.

        Separation for objective bin :math:`k` and valid peer bin :math:`q`
        uses bin-center angles:

        .. math::

           \widehat{\delta}_{ikq}^2=\max\!\left(
               d_{ik}^2+d_{iq}^2-2d_{ik}d_{iq}\cos(\theta_k-\theta_q),0\right).

        All bin pairs are compared, including across the angular wraparound.
        Walls, self IDs, and empty peer bins contribute zero. No peer positions
        or other agents' objective scans are used. A wall can hide a peer in
        its bin, and bin-center directions approximate the true bearing.
        :math:`V_i` contains nonempty raw objective bins; empty scans have zero
        potential, including goals exactly at sensor range. Setting
        ``attraction_coeff=0`` retains only the best shared settling value.

        Reset and unchanged states give zero reward to floating-point
        precision. Undiscounted action rewards telescope across contiguous
        intervals with matching endpoint checkpoints and fixed parameters;
        episode resets begin a new baseline.

        References
        ----------
        Rosenthal, R. W. (1973). "A class of games possessing pure-strategy
        Nash equilibria." International Journal of Game Theory, 2, 65-67.
        https://doi.org/10.1007/BF01737559

        Ng, A. Y., Harada, D., and Russell, S. (1999). "Policy invariance
        under reward transformations: Theory and application to reward
        shaping." Proceedings of ICML, 278-287.
        https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf

        Lu, X., Schwartz, H. M., and Givigi, S. N. (2011). "Policy invariance
        under reward transformations for general-sum stochastic games."
        Journal of Artificial Intelligence Research, 41, 397-406.
        https://doi.org/10.1613/jair.3384
        """
        return SwarmRoller._potential(env) - env.env_params["prev_potential"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmRoller.truncated")
    def truncated(env: SwarmRoller) -> jax.Array:
        """Return whether the environment has reached its physics-step limit.

        Parameters
        ----------
        env : SwarmRoller
            Current environment supplying ``system.step_count`` and
            ``env_params["max_steps"]``.

        Returns
        -------
        jax.Array
            Scalar boolean, true when ``step_count >= max_steps``. The limit
            counts physics steps, including skipped frames, not policy actions.
            This method does not test coverage or reset the environment.
        """
        return jnp.asarray(env.system.step_count >= env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        """Flattened action size per agent.

        Returns
        -------
        int
            Three torque components. :meth:`step` normally receives shape
            ``(max_num_agents, 3)``.
        """
        return 3

    @property
    def action_space_shape(self) -> tuple[int]:
        """Unflattened shape of one agent's torque request.

        Returns
        -------
        tuple[int]
            ``(3,)``, used to reshape per-agent torque vectors in :meth:`step`.
        """
        return (3,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent.

        Returns
        -------
        int
            ``5 + 2 * n_lidar_rays`` (29 by default). :meth:`observation`
            returns shape ``(max_num_agents, observation_space_size)``.
            Filtering does not change sensor sizes or channel order.
        """
        return 5 + 2 * self.n_lidar_rays


__all__ = ["SwarmRoller"]
