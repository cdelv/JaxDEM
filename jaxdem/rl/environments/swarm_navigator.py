# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Environment where multiple agents cooperatively cover a set of objectives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import isfinite
from numbers import Integral

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ...material_matchmakers import MaterialMatchmaker
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils import cross_lidar_2d, cross_lidar_3d, lidar_2d, lidar_3d
from . import Environment
from ._swarm import filter_objectives, shared_potential, vacant_objective_lidar


@jax.jit(static_argnames=("N",))
@partial(jax.named_call, name="swarm_navigator._sample_objectives")
def _sample_objectives(key: ArrayLike, N: int, box: jax.Array, gap: float) -> jax.Array:
    r"""Sample positions on a jittered grid inside a 2-D or 3-D box.

    Parameters
    ----------
    key : ArrayLike
        JAX random number generator key.
    N : int
        Number of positions. Static under JIT; zero returns an empty array.
    box : jax.Array
        Positive side lengths, shape ``(2,)`` or ``(3,)``.
    gap : float
        Requested minimum center separation, in simulation length units.

    Returns
    -------
    jax.Array
        Sampled positions of shape ``(N, dim)`` relative to the box origin.

    Notes
    -----
    Each grid cell contains at most one point. Jitter leaves a margin of
    ``gap / 2`` at cell edges when both cell dimensions are at least ``gap``.
    Smaller cells receive zero jitter in that direction; the requested
    separation cannot be guaranteed when the box is too small for the grid.
    """
    if N == 0:
        return jnp.zeros((0, box.shape[0]))
    i = jax.lax.iota(int, N)
    if box.shape[0] == 3:
        Lx, Ly, Lz = box.astype(float)
        nx = jnp.ceil(jnp.cbrt(N * Lx**2 / (Ly * Lz))).astype(int)
        ny = jnp.ceil(jnp.sqrt(N * Ly / (nx * Lz))).astype(int)
        nz = jnp.ceil(N / (nx * ny)).astype(int)
        cell = box / jnp.asarray([nx, ny, nz])
        ijk = jnp.stack([i % nx, (i // nx) % ny, i // (nx * ny)], axis=-1)
        base = (ijk + 0.5) * cell
        noise = jax.random.uniform(key, (N, 3), minval=-1.0, maxval=1.0)
        return base + noise * jnp.maximum(0.0, (cell - gap) / 2)
    Lx, Ly = box.astype(float)
    nx = jnp.ceil(jnp.sqrt(N * Lx / Ly)).astype(int)
    ny = jnp.ceil(N / nx).astype(int)
    ix, iy = jnp.mod(i, nx), i // nx
    dx, dy = Lx / nx, Ly / ny
    base = jnp.stack([(ix + 0.5) * dx, (iy + 0.5) * dy], axis=1)
    # Jitter is capped so each centre stays >= gap/2 inside its cell, hence
    # adjacent centres are >= gap apart (no overlap when gap >= 2*rad).
    noise = jax.random.uniform(key, (N, 2), minval=-1.0, maxval=1.0) * jnp.asarray(
        [jnp.maximum(0.0, dx / 2 - gap / 2), jnp.maximum(0.0, dy / 2 - gap / 2)]
    )
    return base + noise


def _sample_padding_ring(
    key: ArrayLike, N: int, box: float, pad: float, gap: float
) -> jax.Array:
    r"""Sample *N* points on jittered grids filling the padding ring around ``box``.

    Parameters
    ----------
    key : ArrayLike
        JAX random number generator key, split once per strip.
    N : int
        Number of positions. Zero returns an empty array.
    box : float
        Side length of the inner square containing the objectives.
    pad : float
        Total additional side length of the outer square. The ring extends
        ``pad / 2`` beyond each side of the inner square.
    gap : float
        Requested center separation passed to :func:`_sample_objectives`.

    Returns
    -------
    jax.Array
        Positions of shape ``(N, 2)`` outside the inner square, whose lower
        corner is at the origin, and inside the padded square.

    Notes
    -----
    Bottom and top strips include the corners; left and right strips span
    only the inner square's height. Each receives ``N // 4`` points, with
    the remainder assigned to the right strip. Separation depends on the
    available cell sizes as described in :func:`_sample_objectives`.
    """
    if N == 0:
        return jnp.zeros((0, 2))
    t = pad / 2.0
    L = box + pad
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = N // 4
    n4 = N - 3 * n
    bottom = _sample_objectives(k1, n, jnp.asarray([L, t]), gap) + jnp.asarray([-t, -t])
    top = _sample_objectives(k2, n, jnp.asarray([L, t]), gap) + jnp.asarray([-t, box])
    left = _sample_objectives(k3, n, jnp.asarray([t, box]), gap) + jnp.asarray(
        [-t, 0.0]
    )
    right = _sample_objectives(k4, n4, jnp.asarray([t, box]), gap) + jnp.asarray(
        [box, 0.0]
    )
    return jnp.concatenate([bottom, top, left, right], axis=0)


def _sample_padding_shell_3d(
    key: ArrayLike, N: int, box: float, pad: float, gap: float
) -> jax.Array:
    """Fill six nonoverlapping slabs around the inner objective cube."""
    t = pad / 2
    L = box + pad
    sizes = (
        (t, L, L),
        (t, L, L),
        (box, t, L),
        (box, t, L),
        (box, box, t),
        (box, box, t),
    )
    anchors = (
        (-t, -t, -t),
        (box, -t, -t),
        (0, -t, -t),
        (0, box, -t),
        (0, 0, -t),
        (0, 0, box),
    )
    keys = jax.random.split(key, 6)
    return jnp.concatenate(
        [
            _sample_objectives(keys[i], N // 6 + (i < N % 6), jnp.asarray(size), gap)
            + jnp.asarray(anchor)
            for i, (size, anchor) in enumerate(zip(sizes, anchors, strict=True))
        ],
        axis=0,
    )


@Environment.register("swarmNavigator")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class SwarmNavigator(Environment):
    r"""Multi-agent cooperative objective coverage with local sensing.

    Each agent controls a force vector that acts on a sphere in a reflective
    2-D or 3-D box. Each step adds viscous drag ``-friction * vel``.
    Objectives lie on a jittered grid inside the box; agents spawn in its
    padding ring or shell. Observations read live objective and agent/wall
    LiDAR. By default, observations hide nearby objectives estimated
    to be covered by another agent, preserving the observing agent's own goal.
    Reward still uses the unfiltered scans and shares value among nearby claimants,
    combining flat settling with a wider approach potential. Local objective and
    agent/wall LiDAR estimate peer-to-objective distances across every pair of
    angular bins.
    ``prev_potential`` holds the action-start potential, captured once by
    ``checkpoint`` and preserved through all physics steps.
    Reset initializes this baseline. Only the historical potential is stored;
    current sensor values are computed when observations or rewards are read.

    The shared potential combines the best settling value with
    the summed approach value of visible objectives:

    .. math::

       \Phi_i(s)=\max_{k\in V_i}\bigl(w_{ik}B(d_{ik})\bigr)
           +\eta\sum_{k\in V_i}w_{ik}P(d_{ik}/a_i),\qquad
       r_{i,t}=\Phi_i(s_{t+1})-\Phi_i(s_{\mathrm{checkpoint}}).

    Here :math:`a_i` is the particle radius, :math:`V_i` contains raw
    objective detections, :math:`B(d)=\exp[-(d/(1.5a_i))^4]`, and
    :math:`w_{ik}=1/(1+C_{ik})` shares value with estimated peer claims
    :math:`C_{ik}`. See :meth:`reward` for the approach potential and references.
    Objectives have no assigned owners; "own goal" in the observation filter
    means a detected goal within the observing agent's radius. Agent and
    objective counts may differ.

    Notes
    -----
    The observation vector per agent is:

    ============================  =================
    Feature                       Size
    ============================  =================
    Velocity                      ``dim``
    Normalized objective LiDAR    ``lidar_bins``
    Normalized agent/wall LiDAR   ``lidar_bins``
    ============================  =================

    Velocity is not normalized. LiDAR stores proximity, with zero for empty
    or masked bins and one for coincident centers after normalization.
    With two spatial dimensions and the default 12 bins, each observation
    has 26 entries. In 3-D, ``lidar_bins = n_lidar_rays * n_lidar_elevation``;
    default settings give 291 entries. Methods describe one environment;
    use ``jax.vmap`` for independent environment batches.

    With ``dt = 0.002`` and ``skip_frames = 49``, one policy action spans
    50 physics steps (0.1 seconds). :func:`jaxdem.utils.advance_action <jaxdem.utils.environment.advance_action>`
    checkpoints once before these steps. Read reward at their endpoint,
    before another checkpoint or reset replaces the baseline. Forces are
    applied directly with viscous drag; this environment has no action
    smoothing state.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxdem.rl import Environment
    >>> from jaxdem.utils import advance_action
    >>> env = Environment.create("swarmNavigator")
    >>> env = env.reset(env, jax.random.key(0))
    >>> action = jnp.zeros((env.max_num_agents, env.action_space_size))
    >>> env, terminated, truncated = advance_action(env, action, skip_frames=49)
    >>> reward = env.reward(env)
    >>> observation = env.observation(env)
    """

    n_lidar_rays: int = jax.tree.static()
    """Number of angular bins for each LiDAR sensor."""

    n_lidar_elevation: int = jax.tree.static()
    """Number of elevation bins in 3-D; unused in 2-D."""

    num_objectives: int = jax.tree.static()
    """Number of objectives sampled per environment."""

    vacancy_aware_lidar: bool = jax.tree.static(default=False)
    """Opt in to removing occupied goals before objective LiDAR bin reduction."""

    approach_kernel: str = jax.tree.static(default="legacy")
    """Select the original attraction or an opt-in raised-cosine approach."""

    @classmethod
    @partial(jax.named_call, name="SwarmNavigator.Create")
    def Create(
        cls,
        N: int = 64,
        num_objectives: int = 64,
        box_size: float = 20.0,
        box_padding: float = 10.0,
        max_steps: int = 10000,
        friction: float = 0.2,
        lidar_range: float = 16.0,
        n_lidar_rays: int = 12,
        attraction_coeff: float = 0.1,
        attraction_constant: float = 0.01,
        attraction_quadratic: float = 0.02,
        attraction_decay: float = 1 / 3,
        sharing_range: float = 1.5,
        hide_occupied_objectives: bool = True,
        dim: int = 2,
        n_lidar_elevation: int = 12,
        *,
        vacancy_aware_lidar: bool = False,
        approach_kernel: str = "legacy",
        occupancy_radius_scale: float = 1.2,
    ) -> SwarmNavigator:
        r"""Create a swarm navigator environment.

        Parameters
        ----------
        N : int
            Number of agents.
        num_objectives : int
            Number of objectives sampled per environment.
        box_size : float
            Side length of the square or cube that holds the objectives.
        box_padding : float
            Total extra domain side length in particle radii. The spawn ring or shell
            extends by half this amount on each side of the objective box.
        max_steps : int
            Episode length in physics steps.
        friction : float
            Viscous drag coefficient applied as ``-friction * vel``.
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
            :math:`(c+b x^2)e^{-\lambda x}`, with :math:`x=d/\mathrm{rad}_i`,
            used by the legacy approach kernel.
        attraction_decay : float
            Positive exponential decay :math:`\lambda` in radius units.
            Used by the legacy approach kernel.
        sharing_range : float
            Positive quartic claim scale in observing-agent radii. The default
            matches the settling scale, 1.5.
        hide_occupied_objectives : bool
            Enabled by default. Hide readings strictly below half the LiDAR
            range when a detected peer appears within ``occupancy_radius_scale``
            observing-agent radii of the objective. Preserve goals within the
            observer's radius. A hidden reading becomes zero; farther goals are not
            revealed. Applies only to observations, never reward or checkpoints.
            Set to False to recover the unfiltered observation.
        vacancy_aware_lidar : bool
            Disabled by default. When enabled, remove goals occupied by visible
            peers before nearest-per-bin reduction, revealing farther vacant
            goals in the same bin. Preserve goals covered by the observer.
            Requires all goal and peer detections within the observer's LiDAR
            range before binning; rewards still use the original raw scans.
            ``hide_occupied_objectives=False`` disables either occupancy mask.
        dim : int
            Spatial dimension, either 2 (default) or 3. Three-dimensional
            agents and objectives occupy a volume, with agents in its
            padding shell.
        n_lidar_elevation : int
            Positive number of elevation bins in 3-D; unused in 2-D.
        approach_kernel : {"legacy", "cosine"}
            The default retains the original approach potential. ``"cosine"``
            replaces only that term with ``(1 + cos(pi*d/lidar_range)) / 2``
            for visible goals, preserving settling, sharing, observations,
            and potential-difference timing. This option uses the existing
            attraction weight and sharing range; the constant, quadratic,
            and decay attraction controls have no effect on its score.
        occupancy_radius_scale : float
            Positive peer-to-objective masking tolerance in observing-agent
            radii, default 1.2. Set to 1.0 for the original cutoff. Applies to
            the default post-bin mask only, not ``vacancy_aware_lidar``.

        Returns
        -------
        SwarmNavigator
            The constructed environment. Call :meth:`reset` before use.

        Raises
        ------
        ValueError
            If an attraction coefficient is nonfinite or negative, or
            ``attraction_decay``, ``sharing_range``, or ``occupancy_radius_scale``
            is nonfinite or nonpositive, or ``approach_kernel`` is not
            ``"legacy"`` or ``"cosine"``.

        Notes
        -----
        Construction allocates placeholder agent and objective arrays. Reset
        samples the geometry and initializes the physical system and reward
        baseline. ``N``, ``num_objectives``, and ``n_lidar_rays`` determine
        array shapes. Kernel and sensing selections are static fields.
        Physical, reward, and observation coefficients are stored in
        ``env_params`` as JAX arrays.
        """
        if approach_kernel not in ("legacy", "cosine"):
            raise ValueError("approach_kernel must be 'legacy' or 'cosine'")
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
            ("occupancy_radius_scale", occupancy_radius_scale),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if (
            isinstance(n_lidar_elevation, bool)
            or not isinstance(n_lidar_elevation, Integral)
            or n_lidar_elevation < 1
        ):
            raise ValueError("n_lidar_elevation must be a positive integer")
        n_obj = int(num_objectives)
        state = State.create(pos=jnp.zeros((int(N), dim)))
        env_params = {
            "objective": jnp.zeros((n_obj, dim)),
            "box_size": jnp.asarray(box_size, dtype=float),
            "box_padding": jnp.asarray(box_padding, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "attraction_coeff": jnp.asarray(attraction_coeff, dtype=float),
            "attraction_constant": jnp.asarray(attraction_constant, dtype=float),
            "attraction_quadratic": jnp.asarray(attraction_quadratic, dtype=float),
            "attraction_decay": jnp.asarray(attraction_decay, dtype=float),
            "sharing_range": jnp.asarray(sharing_range, dtype=float),
            "occupancy_radius_scale": jnp.asarray(occupancy_radius_scale, dtype=float),
            "hide_occupied_objectives": jnp.asarray(
                hide_occupied_objectives, dtype=bool
            ),
            "prev_potential": jnp.zeros(int(N)),
        }
        return cls(
            state=state,
            system=System.create(state.shape, rotation_integrator_type=None),
            env_params=env_params,
            n_lidar_rays=int(n_lidar_rays),
            n_lidar_elevation=int(n_lidar_elevation),
            num_objectives=n_obj,
            vacancy_aware_lidar=bool(vacancy_aware_lidar),
            approach_kernel=approach_kernel,
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.reset")
    def reset(env: SwarmNavigator, key: ArrayLike) -> Environment:
        """Sample agents in the padding ring and objectives in the inner box.

        Parameters
        ----------
        env : SwarmNavigator
            The environment to initialize, retaining its configured sizes
            and reward/observation parameters.
        key : ArrayLike
            JAX random number generator key, split for agents and objectives.

        Returns
        -------
        Environment
            The initialized environment with new physical state, zero initial
            velocities, restarted step count, and an action-start potential
            equal to the initial potential. Reward is initially zero.

        Notes
        -----
        Agents have radius one. Sampling uses jittered cells with requested
        center separation 2.05; undersized cells cannot guarantee this spacing.
        The reflecting domain has side length ``box_size + box_padding``
        and lower corner ``-box_padding / 2`` in each coordinate. Reset
        constructs elastic material with unit-mass density (``1 / pi`` in
        2-D, ``3 / (4*pi)`` in 3-D), Young's modulus ``2e5``, and Poisson ratio
        0.3. The physics time step is 0.002 and
        rotation is disabled. Objective coordinates have shape
        ``(num_objectives, dim)`` and are stored in ``env_params["objective"]``.
        """
        key_pos, key_obj = jax.random.split(key)
        N, rad = env.max_num_agents, 1.0
        gap = 2.05 * rad
        box_s = env.env_params["box_size"]
        box = box_s * jnp.ones(env.state.dim)
        padding = env.env_params["box_padding"] * rad

        env.env_params["objective"] = _sample_objectives(
            key_obj, env.num_objectives, box, gap
        )
        if env.state.dim == 3:
            pos = _sample_padding_shell_3d(key_pos, int(N), box_s, padding, gap)
        else:
            pos = _sample_padding_ring(key_pos, int(N), box_s, padding, gap)
        env.state = State.create(pos=pos, rad=rad * jnp.ones(N))

        matcher = MaterialMatchmaker.create("linear")
        mat_table = MaterialTable.from_materials(
            [
                Material.create(
                    "elastic",
                    density=(1 / jnp.pi if env.state.dim == 2 else 3 / (4 * jnp.pi)),
                    young=2e5,
                    poisson=0.3,
                )
            ],
            matcher=matcher,
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            rotation_integrator_type=None,
            domain_type="reflectsphere",
            domain_kw={
                "box_size": box + padding,
                "anchor": -padding / 2 * jnp.ones(env.state.dim),
            },
            mat_table=mat_table,
        )

        return SwarmNavigator.checkpoint(
            env, jnp.zeros((N, env.action_space_size), dtype=env.state.pos.dtype)
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator._sense")
    def _sense(env: SwarmNavigator) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Read raw agent/wall and objective LiDAR from the live state.

        Parameters
        ----------
        env : SwarmNavigator
            Environment supplying positions, objectives, domain, and sensor
            configuration. No sensor cache or checkpoint is modified.

        Returns
        -------
        lidar : jax.Array
            Agent/wall proximity, shape ``(N, lidar_bins)``. Each bin keeps
            its nearest return; walls can hide peers in the same bin.
        lidar_obj : jax.Array
            Objective proximity, shape ``(N, lidar_bins)``. These readings
            are unfiltered, including objectives covered by other agents.
        ids : jax.Array
            Agent/wall return IDs, shape ``(N, lidar_bins)``. Valid peer
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
        Three-dimensional bins flatten azimuth then elevation, with
        ``lidar_bins = n_lidar_rays * n_lidar_elevation``.
        """
        objective = env.env_params["objective"]
        lr = env.env_params["lidar_range"]

        if env.state.dim == 3:
            _, _, lidar, ids, _ = lidar_3d(
                env.state,
                env.system,
                lr,
                env.n_lidar_rays,
                env.n_lidar_elevation,
                sense_edges=True,
            )
            lidar_obj, _, _ = cross_lidar_3d(
                env.state.pos,
                objective,
                env.system,
                lr,
                env.n_lidar_rays,
                env.n_lidar_elevation,
            )
        else:
            _, _, lidar, ids, _ = lidar_2d(
                env.state, env.system, lr, env.n_lidar_rays, sense_edges=True
            )
            lidar_obj, _, _ = cross_lidar_2d(
                env.state.pos, objective, env.system, lr, env.n_lidar_rays
            )
        return lidar, lidar_obj, ids

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator._potential")
    def _potential(env: SwarmNavigator) -> jax.Array:
        """Evaluate the shared per-agent potential using raw live scans.

        Parameters
        ----------
        env : SwarmNavigator
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
        peers, objectives, peer_ids = SwarmNavigator._sense(env)
        return shared_potential(
            objectives,
            peers,
            peer_ids,
            env,
            n_elevation=env.n_lidar_elevation if env.state.dim == 3 else None,
            approach_kernel=env.approach_kernel,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.checkpoint")
    def checkpoint(env: SwarmNavigator, action: jax.Array) -> Environment:
        """Save the LiDAR-derived potential at the start of a policy action.

        Physics steps preserve this potential through all skipped frames.
        The baseline is independent of ``action``. Read reward before the
        next checkpoint or reset replaces it. Unchanged states give zero reward.

        Parameters
        ----------
        env : SwarmNavigator
            Environment immediately before the next policy action.
        action : jax.Array
            Per-agent force requests, normally shape ``(N, 2)``. Accepted
            for the environment checkpoint interface; its value is unused.

        Returns
        -------
        Environment
            Environment with ``env_params["prev_potential"]`` replaced by
            the current per-agent potential of shape ``(N,)``. Physical state
            and observations are not advanced.
        """
        env.env_params["prev_potential"] = SwarmNavigator._potential(env)
        return env

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.step")
    def step(env: SwarmNavigator, action: jax.Array) -> Environment:
        r"""Advance one physics step with the requested force and viscous drag.

        The added force on agent :math:`i` is

        .. math::

           \mathbf{f}_i=\mathbf{a}_i-\mu\mathbf{v}_i,

        where :math:`\mu` is ``friction``. The physics system adds contact
        forces and advances the state. Requests are not smoothed or clipped
        by this method.

        Parameters
        ----------
        env : SwarmNavigator
            Current environment with an initialized physical system.
        action : jax.Array
            Force requests, normally shape ``(N, 2)``, reshaped internally
            to ``(N, *action_space_shape)``.

        Returns
        -------
        Environment
            Environment after one physics step. The action-start potential
            is preserved and no sensor readings are cached.

        Notes
        -----
        Use :func:`jaxdem.utils.advance_action <jaxdem.utils.environment.advance_action>` to checkpoint once before
        repeating steps for a policy action. Calling this method directly
        neither creates a checkpoint nor handles episode reset.
        """
        N = env.max_num_agents
        force = (
            action.reshape(N, *env.action_space_shape)
            - env.env_params["friction"] * env.state.vel
        )
        env.system = env.system.force_manager.add_force(env.state, env.system, force)

        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="SwarmNavigator.observation")
    def observation(env: SwarmNavigator) -> jax.Array:
        r"""Return velocity, filtered objective LiDAR, and agent/wall LiDAR.

        Parameters
        ----------
        env : SwarmNavigator
            Environment whose current state is observed. Checkpointed
            potentials are not used to construct the observation.

        Returns
        -------
        jax.Array
            Shape ``(N, observation_space_size)``. Each row concatenates the
            velocity components, normalized objective proximity bins,
            then normalized agent/wall proximity bins. Velocity is in
            simulation units; LiDAR channels lie in ``[0, 1]``.

        Notes
        -----
        A valid raw LiDAR return is normalized as
        :math:`p_k=\max(0,1-d_k/L)`. Empty and masked bins are zero.
        ``hide_occupied_objectives=False`` returns unfiltered goal readings.
        Only LiDAR channels are normalized. With filtering enabled, hide
        objective bin :math:`k` when

        .. math::

           a_i < d_{ik} < L/2,\qquad
           \min_{q\in Q_i}\widehat\delta_{ikq}\le \sigma a_i,

        where :math:`a_i` is the observing radius, :math:`L` the LiDAR range,
        and :math:`Q_i` the valid detected peers, excluding walls and self.
        ``occupancy_radius_scale`` sets :math:`\sigma` (default 1.2).
        Bin-center directions give the approximate separation

        .. math::

           \widehat\delta_{ikq}^2=\max\!\left(
           d_{ik}^2+d_{iq}^2-2d_{ik}d_{iq}\cos(\theta_k-\theta_q),0\right).

        Every objective bin is compared with every peer bin. Own goals within
        one radius and readings at or beyond half range remain visible.
        Hidden readings become zero without revealing occluded goals. Reward
        retains the raw scans. Angular quantization and hidden peers can cause
        occupancy errors; this is not an exact global coverage test.
        In 3-D, the cosine is the dot product of spherical bin-center unit
        directions, including elevation.

        The half-range mask and own-goal exception were developed for this
        environment's experiments. The congestion-game and reward-shaping
        references in :meth:`reward` motivate the reward, not this observation
        filter, and do not guarantee that discarding observations is harmless.

        With ``vacancy_aware_lidar=True``, occupied objectives are removed
        before nearest-per-bin reduction over all local detections instead.
        Visible peers within range can mask goals at any distance below range;
        occupancy uses each peer's own radius and preserves the observer's
        covered goals. This can reveal farther vacant goals behind occupied
        ones, while retaining the same observation layout. Setting
        ``hide_occupied_objectives=False`` disables occupancy filtering.
        """
        lidar, lidar_obj, peer_ids = SwarmNavigator._sense(env)
        lr = env.env_params["lidar_range"]
        if env.vacancy_aware_lidar:
            lidar_obj = jnp.where(
                env.env_params["hide_occupied_objectives"],
                vacant_objective_lidar(env),
                lidar_obj,
            )
        else:
            lidar_obj = filter_objectives(
                lidar_obj,
                lidar,
                peer_ids,
                env,
                n_elevation=env.n_lidar_elevation if env.state.dim == 3 else None,
            )
        return jnp.concatenate(
            [
                env.state.vel,
                lidar_obj / lr,
                lidar / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.reward")
    def reward(env: SwarmNavigator) -> jax.Array:
        r"""Return the action-start difference in the shared coverage potential.

        Parameters
        ----------
        env : SwarmNavigator
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

        With ``approach_kernel="cosine"``, replace this approach potential
        by :math:`H(d)=(1+\cos(\pi d/L))/2` for visible goals, where :math:`L`
        is the sensor range. Its value and slope vanish at :math:`L`, and
        the three attraction-shape controls have no effect on it. Settling,
        soft claim sharing, attraction weight, and reward timing are unchanged.
        This scalar potential adapts the exploration kernel of Sun et al.
        (2023), rather than implementing their mean-shift velocity controller.

        The potential estimates peer claims on each visible
        objective from all pairs of objective and peer LiDAR bins:

        .. math::

           \begin{aligned}
           C_{ik} &= \sum_{q\in Q_i}
               \exp\!\left[-\left(\frac{\widehat\delta_{ikq}}{\rho a_i}\right)^4\right],
               \\
           w_{ik} &= \frac{1}{1+C_{ik}}, \\
           B_{ik} &= \exp\!\left[-\left(\frac{d_{ik}}{1.5a_i}\right)^4\right], \\
           \Phi_i &= \max_{k\in V_i}(w_{ik}B_{ik})
                    +\eta\sum_{k\in V_i}w_{ik}P(d_{ik}/a_i), \\
           r_{i,t} &= \Phi_i(s_{t+1})-\Phi_i(s_{\mathrm{checkpoint}}).
           \end{aligned}

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
        In 3-D, spherical bin-center direction dot products replace the
        planar cosine and account for both azimuth and elevation.
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
        Sun, G. et al. (2023). "Mean-shift exploration in shape assembly of
        robot swarms." Nature Communications, 14, 3476.
        https://doi.org/10.1038/s41467-023-39251-5

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
        return SwarmNavigator._potential(env) - env.env_params["prev_potential"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="SwarmNavigator.truncated")
    def truncated(env: SwarmNavigator) -> jax.Array:
        """Return whether the environment has reached its physics-step limit.

        Parameters
        ----------
        env : SwarmNavigator
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
            Number of force components, equal to ``state.dim`` (two or three).
            Actions passed to :meth:`step` normally have shape
            ``(max_num_agents, action_space_size)``.
        """
        return self.state.dim

    @property
    def action_space_shape(self) -> tuple[int]:
        """Unflattened shape of one agent's force request.

        Returns
        -------
        tuple[int]
            ``(state.dim,)``, either ``(2,)`` or ``(3,)``. Used when
            reshaping the batch of per-agent actions inside :meth:`step`.
        """
        return (self.state.dim,)

    @property
    def observation_space_size(self) -> int:
        """Flattened observation size per agent.

        Returns
        -------
        int
            ``state.dim + 2 * lidar_bins`` (26 with 2-D default settings).
            :meth:`observation` returns an array of shape
            ``(max_num_agents, observation_space_size)``. Filtering changes
            objective readings but not the observation size or channel order.
        """
        lidar_bins = self.n_lidar_rays * (
            self.n_lidar_elevation if self.state.dim == 3 else 1
        )
        return self.state.dim + 2 * lidar_bins


__all__ = ["SwarmNavigator"]
