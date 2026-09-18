Swarm coverage environments
===========================

``SwarmNavigator``, ``SwarmRoller``, and ``SwarmRoller3D`` cooperatively cover
objectives rather than assigning a fixed target to each agent. Navigator
actions are forces; roller actions are torques. ``SwarmRoller3D`` retains
its pyramid objectives, 3-D LiDAR, and pairwise magnetic attraction.

Action-start checkpoints
------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jaxdem.rl import Environment
   from jaxdem.utils import advance_action

   env = Environment.create("swarmRoller", N=64, num_objectives=64)
   env = env.reset(env, jax.random.key(0))
   action = jnp.zeros((env.max_num_agents, env.action_space_size))
   env, terminated, truncated = advance_action(env, action, skip_frames=49)
   reward = env.reward(env)
   observation = env.observation(env)

``advance_action`` calls ``checkpoint`` once before executing the physics
steps. Navigator and SwarmRoller store one scalar per agent in
``prev_potential``. SwarmRoller3D stores objective and peer-only LiDAR readings
in ``lidar_obj_prev`` and ``lidar_agt_prev``. These baselines remain fixed
throughout the action, including skipped frames. With ``dt=0.002`` and
``skip_frames=49``, each action spans 50 physics steps, or 0.1 seconds.

Read reward before the next checkpoint or reset replaces the historical
readings. Each batched environment stops at its own time limit, and reward
uses that environment's actual final state. Reset initializes the history from
the new episode. SwarmRoller separately updates ``applied_action`` every
physics step; checkpoints preserve that actuator state.

Navigator and SwarmRoller: shared objective value
-------------------------------------------------

Both environments use the same shared reward, which divides an objective's
settling and approach value among the observing agent and estimated nearby
claimants.
It uses the existing local objective and agent/wall LiDAR, comparing every
pair of angular bins to estimate peer-to-objective separation
:math:`\widehat\delta_{ikq}`. Define

.. math::

   C_{ik}=\sum_{q\in Q_i}
      \exp\!\left[-\left(\frac{\widehat\delta_{ikq}}{\rho a_i}\right)^4\right],
   \qquad w_{ik}=\frac{1}{1+C_{ik}},\qquad
   B_{ik}=\exp\!\left[-\left(\frac{d_{ik}}{1.5a_i}\right)^4\right].

Here :math:`a_i` is the observing agent's radius, :math:`Q_i` contains valid
detected peers, and ``sharing_range`` sets :math:`\rho` (default 1.5 radii).
Walls, self IDs, and empty bins are excluded. The constant one in the
denominator represents the observing agent's prospective claim, including
while it approaches. The potential is

.. math::

   \Phi_i=\max_{k\in V_i}(w_{ik}B_{ik})
       +\eta\sum_{k\in V_i}w_{ik}P(d_{ik}/a_i),\qquad
   r_{i,t}=\Phi_i(s_{t+1})-\Phi_i(s_{\mathrm{checkpoint}}).

``attraction_coeff`` sets :math:`\eta` (default 0.1). The wider monotone
approach potential :math:`P` is defined below. Empty scans give zero potential.
Settling uses the best shared goal value: nearby settling bonuses do not add,
and a crowded nearest objective need not dominate an available alternative.
With no peers, an objective retains full value. With :math:`n-1` unit peer
claims, its value is divided by :math:`n`. Arrival does not cancel the agent's
own claim. Weights remain positive and there is no penalty for making contact.

This is a smooth local approximation to equal sharing in a congestion game
[Rosenthal1973]_. In the ideal discrete game with equal
numbers of agents and goals and unrestricted choices, a shared goal and an
empty goal cannot coexist at a pure Nash equilibrium. The LiDAR approximation
and discounted physical travel do **not** inherit that guarantee.

Potential differences are related to potential-based reward shaping
[Ng1999]_ and its extension to general-sum stochastic games [Lu2011]_. Those
invariance results concern adding
:math:`F_i(s,s')=\gamma\Phi_i(s')-\Phi_i(s)` to an existing reward, with the
appropriate discount and terminal conditions. These environments instead use
the undiscounted difference :math:`\Phi_i(s')-\Phi_i(s)` as its reward.
For training with :math:`\gamma<1`, this is not the discount-correct shaping
term. The cited results therefore do not guarantee preservation of a separate
coverage objective, convergence, or full coverage here.

Reward always uses the raw scans, independently of the observation filter
below. Neither environment adds an endpoint occupancy bonus or kinetic-energy
term. Their reward parameter names and defaults match.

Shared approach potential and sensing
-------------------------------------

The attraction strength at distance
:math:`x=d/\mathrm{rad}_i` and its integrated potential are

.. math::

   A(x)=(c+bx^2)e^{-\lambda x},\qquad
   P(x)=e^{-\lambda x}\left[\frac{c}{\lambda}
     +b\left(\frac{x^2}{\lambda}+\frac{2x}{\lambda^2}+\frac{2}{\lambda^3}\right)\right].

Thus :math:`-P'(x)=A(x)`: the incentive to approach can peak away from zero,
while the potential always increases toward an objective. The defaults
``attraction_constant=0.01``, ``attraction_quadratic=0.02``, and
``attraction_decay=1/3`` place the strength peak at about 5.92 particle radii.
At two radii the strength is about 47 percent of that peak. Positive constant
strength near zero does not give a standing reward: reward is a potential difference.

Peer-to-objective separation compares every objective bin with every peer
bin using the law of cosines:

.. math::

   \widehat{\delta}_{ikq}^2=\max\!\left(
       d_{ik}^2+d_{iq}^2-2d_{ik}d_{iq}\cos(\theta_k-\theta_q),0\right).

This includes bins across the angular wraparound. Walls, self IDs, and empty
peer bins are excluded. Bin-center directions are approximate; unseen peers
cannot reduce an objective's value, and walls can hide peers in the same bin.
The computation uses an ``N x n_lidar_rays x n_lidar_rays`` array. Reward uses
only the observing agent's raw scans; no objective assignments or pooled
coverage scans are required.

Set ``attraction_coeff=0`` to retain only the best shared settling value.
The sharing weights still apply. For example:

.. code-block:: python

   env = Environment.create(
       "swarmNavigator", N=64, num_objectives=64, box_size=40,
       attraction_coeff=0.1, sharing_range=1.5,
   )

SwarmRoller3D: existing reward preserved
----------------------------------------

SwarmRoller3D retains its contention-shaped potential and endpoint occupancy bonus:

.. math::

   r_{i,t} = b\,\mathbf{1}[d_{i,\min}<\mathrm{rad}_i]
       + 10\left(\phi_{i,t}-\phi_{i,\mathrm{checkpoint}}\right).

Here :math:`b` is ``near_goal_bonus`` and :math:`d_{i,\min}` is the closest
objective distance inferred from the current LiDAR. The potential sums
exponentials of contention-adjusted objective-bin distances:

.. math::

   \phi_i = \sum_k \exp\left[-2\left(d_{i,k}+P_{i,k}\right)\right],
   \qquad
   P_{i,k} =
   \begin{cases}
       P_{\max}\exp(-D_{i,k}/\tau), & D_{i,k}<L/4,\\
       0, & \text{otherwise},
   \end{cases}
   \qquad \tau=1.

``contention_strength`` is :math:`P_{\max}`, ``lidar_range`` is :math:`L`,
and :math:`D_{i,k}` is the minimum peer-to-objective distance estimated from
the sensor bins using the existing law-of-cosines calculation.
``SwarmRoller3D`` continues to use azimuth alignment for its flattened
azimuth/elevation bins; its reward geometry has not been changed.

The only timing change is that the previous potential now refers to the
start of the complete action, rather than the immediately preceding physics
step. The occupancy bonus is still evaluated once at the endpoint. It is not
summed over skipped frames. At an unchanged state, or immediately after
reset/checkpoint, the shaping difference is zero but an agent within its goal
radius still receives the existing bonus. Consequently, only the shaping
component telescopes when an action is split into multiple intervals.

Live observations and physics
-----------------------------

Navigator observations contain planar velocity followed by normalized objective
and agent/wall LiDAR. SwarmRoller uses the same sensor channels and inserts
three angular-velocity components after planar velocity, matching MultiRoller.
Its observation size is now ``5 + 2 * n_lidar_rays`` (29 by default), instead
of the former 30: vertical velocity is no longer included. Existing policies
trained on the former layout must be rebuilt and retrained.

SwarmRoller3D keeps its full translational/angular velocity channels and
separate peer-only scan for the legacy contention reward.

Both ``SwarmNavigator`` and ``SwarmRoller`` default to
``hide_occupied_objectives=True``. For the policy
observation only, an objective reading becomes zero when its distance is
strictly below half the LiDAR range, strictly beyond the observer's own radius,
and a detected peer is estimated to be within one radius of that objective:

.. math::

   a_i < d_{ik} < L/2,\qquad
   \min_{q\in Q_i}\widehat\delta_{ikq}\le a_i.

The separation estimate compares all pairs of angular bins using the same
bin-center geometry as the reward. Walls, self IDs, and invalid detections
cannot occupy objectives. Own goals within one radius and readings at or beyond
half range remain visible. This is the simple removal filter: it does not
reveal a farther goal behind a hidden reading. Velocity and peer/wall channels
are unchanged. Angular quantization and unseen peers can misclassify occupancy.

Reward, action checkpoints, and coverage accuracy continue to use raw scans;
occupied goals are not excluded from the potential. No density switch is used.
Set ``hide_occupied_objectives=False`` to restore the previous observation
semantics for those sensor channels. This does not restore SwarmRoller's old
velocity layout. SwarmRoller3D observations remain unchanged.

The half-range visibility rule, own-goal exception, and LiDAR occupancy mask
were developed for these experiments; they are not algorithms taken from the
papers cited here. They modify the policy's information and may alias an
occupied goal with an empty sensor bin. Reward-shaping invariance results
[Ng1999]_ [Lu2011]_ do not justify this observation change; its benefit is
supported by the measured SwarmNavigator training comparisons. SwarmRoller
uses the same rule, but its rolling dynamics require separate training evaluation.

Current scans are computed from the live state when observations or rewards
are read. There are no persistent ``lidar``, ``lidar_obj``, or ``lidar_agt``
caches. Physics steps update only dynamics; they neither scan LiDAR nor advance
reward history. The internal sensing helper returns arrays without modifying
the environment.

SwarmRoller matches MultiRoller's action smoothing and rolling physics:

.. math::

   \mathbf{u}_{i,t}=\alpha\mathbf{a}_{i,t}
       +(1-\alpha)\mathbf{u}_{i,t-1}.

``action_alpha`` defaults to 0.22. The previous applied torque comes from the
immediately preceding physics step, not the checkpoint. Reset sets it to zero.
Translational and angular damping, gravity, and the frictional floor force
match MultiRoller. Particle contacts now use the spring model and naive
all-pairs collider; no tangential particle-contact history is stored.
The lower vertical domain boundary is one radius below the floor so floor
contact can occur before boundary reflection.

Like MultiRoller, SwarmRoller bins LiDAR returns by azimuth but uses full 3-D
center distances. These equal planar distances when centers share a height.
The bin-center occupancy estimate ignores relative elevation if particles lift
off the floor. SwarmRoller3D retains its Cundall-Strack contacts, neighbor-list
history, and magnetic forces.

LiDAR coverage accuracy
-----------------------

The experiment script counts distinct objectives detected within one particle
radius of any agent. Objective IDs returned by cross-LiDAR are deduplicated
across all agents and sensor bins separately in each environment.
``accuracy_cutoff_radii`` defaults to 1.0 and can be set to 2.0 for a looser
coverage criterion. For proximity :math:`\ell_{i,k}` and target ID
:math:`q_{i,k}`,

.. math::

   \mathrm{accuracy} = \frac{1}{M}\left|
       \left\{q_{i,k}:q_{i,k}\ge0,\;
           L-\ell_{i,k}\le c\,\mathrm{rad}_i\right\}\right|.

Here :math:`M` is the number of objectives and :math:`c` is the cutoff in
particle radii. Several agents near the same objective count once. Invalid
(empty) bins never count, even if the cutoff exceeds the sensor range. This
metric supports unequal agent and objective counts and divides by objective
count, not agent count. With 32 agents and 64 sufficiently separated goals,
maximum simultaneous coverage is 50 percent at the one-radius cutoff.

Only objectives returned by LiDAR can count: a farther objective hidden behind
a closer one in the same angular bin may be missed, particularly with a larger
cutoff. MultiNavigator and MultiRoller retain their assigned-goal accuracy.

References
----------

.. [Rosenthal1973] Robert W. Rosenthal (1973).
   `A class of games possessing pure-strategy Nash equilibria
   <https://doi.org/10.1007/BF01737559>`_.
   *International Journal of Game Theory*, 2, 65–67.

.. [Ng1999] Andrew Y. Ng, Daishi Harada, and Stuart Russell (1999).
   `Policy invariance under reward transformations: Theory and application
   to reward shaping <https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf>`_.
   *Proceedings of the 16th International Conference on Machine Learning*,
   278–287.

.. [Lu2011] Xiaosong Lu, Howard M. Schwartz, and Sidney N. Givigi (2011).
   `Policy invariance under reward transformations for general-sum stochastic
   games <https://doi.org/10.1613/jair.3384>`_.
   *Journal of Artificial Intelligence Research*, 41, 397–406.
