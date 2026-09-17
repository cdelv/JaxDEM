Multi-agent navigation
======================

``MultiNavigator`` assigns each particle a fixed objective in a reflective
2-D box. Agents apply force vectors with viscous drag and particle contacts.
Objectives are assigned by a random permutation at reset;
``env.env_params["objective"][i]`` is already agent ``i``'s assigned objective,
and ``env.env_params["permutation"][i]`` stores its original target index.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jaxdem.rl import Environment
   from jaxdem.utils import advance_action

   env = Environment.create("multi_navigator", N=64, action_alpha=0.22)
   env = env.reset(env, jax.random.key(0))
   observation = env.observation(env)
   action = jnp.zeros((64, 2))  # Replace with the policy's requested forces.
   env, terminated, truncated = advance_action(env, action, skip_frames=49)
   reward = env.reward(env)

Radius-scaled goal potential
----------------------------

Each agent receives the change in a quartic exponential potential:

.. math::

   R_i = 2\,\mathrm{rad}_i,
   \qquad
   \varphi_i(d) = \exp\!\left[-\left(\frac{d}{R_i}\right)^4\right],

.. math::

   r_{i,t} = \varphi_i(d_{i,t})
       - \varphi_i(d_{i,\mathrm{checkpoint}}).

Here :math:`d_i` is the distance from agent :math:`i`'s center to its own
objective. Each potential uses that particle's radius, so equally sized
displacements relative to particle size receive equal credit. Particle radii
remain fixed during an episode; reset creates unit-radius particles.

The peak is flat: for small distances,
:math:`\varphi_i(d) \approx 1 - (d/R_i)^4`. Small departures from an objective
therefore have a small cost. Far away, the potential rapidly approaches zero;
it has no linear tail. At distances of one, two, three, and four particle radii,
the potential is approximately 0.9394, 0.3679, 0.0063, and
:math:`1.13\times10^{-7}`, respectively.

By default, reward depends only on each agent's own goal progress. There is no neighbor
shaping, occupancy bonus, or collision penalty. Physical contacts still affect
motion. The previous experimental ``neighbor_reward_coeff`` argument has been
removed.

Optional kinetic-energy potential
---------------------------------

To encourage agents to settle at their objectives, enable the energy factor:

.. code-block:: python

   env = Environment.create(
       "multi_navigator",
       N=64,
       kinetic_energy_coeff=0.1,
       kinetic_energy_scale=1.0,
   )

The coefficient above is an example starting point, not a measured optimum.
The default coefficient is zero, preserving the distance-only reward.
The full potential is

.. math::

   K_i = \tfrac12 m_i\|\mathbf{v}_i\|^2,
   \qquad
   \Phi_i(d,K) = \varphi_i(d)
       \exp\!\left[-\beta\frac{K}{K_{\mathrm{ref}}}\right].

``kinetic_energy_coeff`` is the nonnegative strength :math:`\beta`;
``kinetic_energy_scale`` is the positive reference energy
:math:`K_{\mathrm{ref}}`, defaulting to one simulation energy unit. Rotation
is disabled in this environment, so only translational energy enters.
At fixed distance, slowing down increases the potential. The goal factor
makes this effect strong near the objective and negligible far away. At rest,
the existing flat goal potential is recovered.

Reward is the full potential difference, including the energy at both ends:

.. math::

   r_{i,t} = \Phi_i(d_{i,t},K_{i,t})
       - \Phi_i(d_{i,\mathrm{checkpoint}},K_{i,\mathrm{checkpoint}}).

Unchanged distance and energy give zero reward: this does not charge for
motion every step or introduce physical damping. Set reward parameters before
reset or take a new checkpoint after changing them.

Action checkpoints and smoothing
--------------------------------

``advance_action`` checkpoints the full starting potential in ``prev_potential``
and keeps the starting distance in ``prev_dist`` for inspection. It then
executes ``1 + skip_frames`` physics steps. Read reward before starting
the next action. Reset initializes the baseline, so its reward is zero;
unchanged distance and energy also give zero reward. With the default ``dt=0.002`` and
``skip_frames=49``, a policy action lasts 0.1 seconds.

Each physics step filters the requested force:

.. math::

   \mathbf{u}_{i,t} = \alpha\,\mathbf{a}_{i,t}
       + (1-\alpha)\,\mathbf{u}_{i,t-1}.

``action_alpha`` defaults to 0.22. ``applied_action`` stores the immediately
preceding filtered force, independently of the reward checkpoint. Reset clears
it; checkpoints preserve it. Drag is applied after filtering.

Observations
------------

The per-agent observation contains the unit direction to its objective, the
displacement clipped componentwise to ``[-3, 3]``, velocity, and normalized
agent/wall LiDAR proximity. Objective locations are not included in the LiDAR
scan. The observation has ``6 + n_lidar_rays`` features, or 22 with the default
16 bins. LiDAR is computed from the live state only when observations are read;
reward and checkpoint calculations do not perform LiDAR scans.
