Multi-agent rolling
===================

``MultiRoller`` assigns rolling spheres to fixed objectives on a floor.
It uses the same floor friction and torque control as ``SingleRoller`` and
a radius-scaled goal potential with optional energy weighting, as in
``MultiNavigator``. Particle contacts use normal spring forces with the direct
naive all-pairs collider. There is no neighbor cache or tangential
particle-contact friction. Floor friction still drives rolling.
Reflective walls account for particle radii; the frictional floor is at
:math:`z=0`.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jaxdem.rl import Environment
   from jaxdem.utils import advance_action

   env = Environment.create("multi_roller", N=64, action_alpha=0.22)
   env = env.reset(env, jax.random.key(0))
   action = jnp.zeros((64, 3))  # Requested torque vectors.
   env, terminated, truncated = advance_action(env, action, skip_frames=49)
   observation = env.observation(env)
   reward = env.reward(env)

Reward and checkpoints
----------------------

With the default zero energy coefficient, each agent receives a change in
the distance-only potential:

.. math::

   \varphi_i(d) = \exp\!\left[-\left(
       \frac{d}{1.5\,\mathrm{rad}_i}\right)^4\right],
   \qquad
   r_{i,t} = \varphi_i(d_{i,t})
       - \varphi_i(d_{i,\mathrm{checkpoint}}).

Distance is measured in 3-D from the particle center to its assigned objective,
as in ``SingleRoller``. Objectives lie at the resting center height. The
potential is flat near the target and rapidly decays far away, using the
scale of 1.5 particle radii. Radii remain fixed
during an episode; reset creates unit-radius spheres.

``advance_action`` saves the full starting potential in ``prev_potential``
and the distance in ``prev_dist`` once, then
executes ``1 + skip_frames`` physics steps. Reward reads the live endpoint;
read it before the next action or reset replaces the baseline. Reset and
unchanged distance and energy give zero reward. The previous ``ke_tau``, ``ke_gate``,
and ``near_goal_bonus`` parameters have been removed.

Optional kinetic-energy potential
---------------------------------

Enable energy weighting with the same parameters as ``MultiNavigator``:

.. code-block:: python

   env = Environment.create(
       "multi_roller",
       N=64,
       kinetic_energy_coeff=0.1,
       kinetic_energy_scale=1.0,
   )

The coefficient above is an example to test, not a measured optimum. The
default coefficient is zero and the default reference energy is one simulation
energy unit. The full potential includes both translation and rotation:

.. math::

   K_i = \tfrac12m_i\|\mathbf{v}_i\|^2
       + \tfrac12\boldsymbol{\omega}_{i,b}^{\top}
           I_{i,b}\boldsymbol{\omega}_{i,b},
   \qquad
   \Phi_i(d,K) = \varphi_i(d)
       \exp\!\left[-\beta\frac{K}{K_{\mathrm{ref}}}\right].

Here :math:`\beta` is ``kinetic_energy_coeff`` and :math:`K_{\mathrm{ref}}`
is ``kinetic_energy_scale``. Angular velocity is transformed into the body's
principal frame to match its stored inertia. At fixed distance, reducing
translation or spin earns credit near the target and has negligible effect
far away. Reward subtracts the complete checkpoint potential from the live
potential, including the energy at both ends of the action.

This is a potential difference, not a per-step charge for motion or added
physical damping. Set the parameters before reset or take a new checkpoint
after changing them. The calculations are inline in reset, checkpoint, and
reward; physics steps perform no reward calculation or LiDAR scan.

Torque smoothing
----------------

Every physics step filters each requested torque:

.. math::

   \mathbf{u}_{i,t} = \alpha\,\mathbf{a}_{i,t}
       + (1-\alpha)\,\mathbf{u}_{i,t-1}.

``action_alpha`` defaults to 0.22 and accepts values in ``[0, 1]``; 1 disables
smoothing. ``applied_action`` stores the immediately preceding filtered
torque. Checkpoints preserve it and reset clears it. Angular damping is applied
after filtering, and translational drag acts on velocity. With ``dt=0.002``
and ``skip_frames=49``, an action lasts 0.1 seconds.

Observations and assignments
----------------------------

Each observation contains the planar unit direction to the objective,
planar displacement clipped to ``[-3, 3]``, planar velocity, all three angular
velocity components, and normalized agent/wall LiDAR proximity. The size is
``9 + n_lidar_rays``, or 25 with the default 16 bins and detection range 6.
Observations and LiDAR read the live state; physics steps and rewards do not
compute LiDAR. Objective locations are not included in the scan.

Reset assigns objectives one-to-one using a random permutation.
``env.env_params["objective"][i]`` is already agent ``i``'s assigned objective;
``env.env_params["permutation"][i]`` stores its original target index.
