"""Intro to JaxDEM reinforcement learning
=========================================

This tutorial runs a tiny
:py:class:`Proximal Policy Optimization (PPO) <jaxdem.rl.trainers.ppo_trainer.PPOTrainer>`
training loop for a sphere navigating toward a target in a reflective box,
using :py:class:`~jaxdem.rl.environments.single_navigator.SingleNavigator`.
A :py:class:`~jaxdem.rl.models.mlp.SharedActorCritic` MLP
maps observations to a continuous force policy and a value
estimate. Install ``JaxDEM[rl,io]`` for the experimental RL tools and optional
VTK export used here.

Four environments, short rollouts and two update epochs keep the walkthrough
quick. This budget checks the training and evaluation pipeline; it does not
produce a reliably trained navigator. First-call JAX compilation can dominate
runtime. Increase the budget and evaluate many independent episodes before
making a claim about learning quality.
"""

# %%
# Environment and action intervals
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

import jaxdem as jdem
import jaxdem.rl as rl
from jaxdem import utils

num_steps_epoch = 16
skip_frames = 4
num_envs = 4
num_epochs = 2
# Each decision requests five physics steps. The evaluation below requests
# 60 decisions (300 frames); its episode horizon must allow all of them.
evaluation_steps = 60
env = rl.Environment.create(
    "single_navigator",
    min_box_size=8.0,
    max_box_size=8.0,
    max_steps=(evaluation_steps + 1) * (1 + skip_frames),
)

# %%
# A scalar environment has fixed agent slots. Observations have shape
# ``(max_num_agents, observation_space_size)``; continuous actions have shape
# ``(max_num_agents, action_space_size)``. Vectorization prepends an environment
# axis. Inactive slots are masked out of the learning objective.
#
# ``skip_frames=k`` repeats an action for at most ``1+k`` physics frames,
# stopping at an episode boundary. PPO distinguishes termination from external
# truncation: only the latter bootstraps the value at the boundary. Evaluation
# helpers freeze completed episodes until the caller explicitly resets them.

# %%
# Reward and the actor–critic decomposition
# -----------------------------------------
# :py:meth:`~jaxdem.rl.environments.single_navigator.SingleNavigator.reward`
# rewards progress through a distance potential:
#
# .. math::
#
#     \Phi(d)=e^{-2d},\qquad
#     r_t=\Phi(d_{\rm end})-\Phi(d_{\rm start}).
#
# The start is checkpointed once before repeating an action; the end is the
# live state after all accepted physics frames. Positive reward therefore
# means that the agent ended closer to the target. It does not require the
# distance to decrease on every intermediate frame. Observations and rewards
# read the live endpoint without advancing that checkpoint.
#
# The shared MLP has policy and value heads. The policy produces a distribution
# :math:`\pi_\theta(a\mid o)` for exploration; the value head estimates expected
# discounted return. An advantage compares a sampled return with this baseline,
# guiding the policy toward actions with better-than-expected outcomes.

# %%
# Model and constrained actions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.rl.action_spaces.max_norm_space.MaxNormSpace`
# constrains sampled force magnitudes to one while retaining a
# differentiable probability density. Constructor registry keys ignore case
# and underscores. The small architecture is sufficient for an API walkthrough.
# The action space is a bijector, so transformed action probabilities include its
# Jacobian correction. It preserves direction and maps a standard isotropic
# Gaussian to a uniform ball; learned means and variances still shape the
# resulting policy. This lets PPO use a bounded action distribution without
# applying an additional hard clipping operation that changes sampled actions.
# See :py:class:`~jaxdem.rl.action_spaces.max_norm_space.MaxNormSpace` for the radial transform
# and its analytic entropy expression.
model = rl.Model.create(
    "SharedActorCritic",
    key=nnx.Rngs(jax.random.key(1)),
    architecture=[16, 16],
    observation_space_size=env.observation_space_size,
    action_space_size=env.action_space_size,
    action_space=rl.ActionSpace.create("max_norm", max_norm=1.0),
)

# %%
# Configure PPO
# ~~~~~~~~~~~~~
# One epoch collects 4*16=64 single-agent transitions. Two minibatches each
# contain two complete 16-step environment segments. ``num_epochs`` counts
# rollout/update iterations, not replay passes.
# `optax.adam <https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html>`_
# keeps this introductory
# example small; optimizer choice and larger budgets require separate tuning.
# PPO compares the live policy with the behavior policy that collected the
# rollout. For stored actions, its clipped surrogate is
#
# .. math::
#
#     \begin{aligned}
#     \rho_t(\theta) &=
#       \frac{\pi_\theta(a_t\mid o_t)}{\pi_{\rm old}(a_t\mid o_t)}, \\
#     L_{\rm policy} &= -\mathbb E_t\left[
#       \min\left(\rho_t A_t,
#       \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right].
#     \end{aligned}
#
# The ratio clipping limits the objective's incentive for large policy changes;
# it is unrelated to constraining the physical force magnitude. The trainer
# also fits the value head and includes entropy regularization. Active-agent
# masks determine which transitions contribute to these averages.
#
# The default ``vtrace=False`` uses generalized advantage estimation with live
# minibatch value predictions and fixed collection-time bootstraps at rollout
# and truncation boundaries. Advantages are used without normalization, while
# behavior log-probabilities and the value-clipping reference stay fixed.
# A minibatch contains complete fixed-horizon segments; those segments may
# cross episode boundaries and need not be complete episodes.
#
# Increase ``num_envs`` for more independent data, ``num_steps_epoch`` for longer
# segments, and ``num_epochs`` for more collect/update iterations. These affect
# data and optimization differently. A longer run can use
# :py:meth:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer.train` and its
# logging/directory options, or retain the explicit
# :py:meth:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer.epoch` loop shown here
# to inspect diagnostics and implement a stopping criterion.
tr = rl.Trainer.create(
    "PPO",
    env=env,
    model=model,
    key=jax.random.key(6),
    optimizer=optax.adam,
    num_steps_epoch=num_steps_epoch,
    num_envs=num_envs,
    num_epochs=num_epochs,
    skip_frames=skip_frames,
    num_minibatches=2,
    learning_rate=1e-3,
)

# %%
# Run and inspect two update epochs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer.train` is the convenience
# loop (``train(tr, verbose=False, log=False)``). Calling
# :py:meth:`~jaxdem.rl.trainers.ppo_trainer.PPOTrainer.epoch` directly exposes
# the scalar diagnostics for this short tutorial.
# Finite losses establish a working update, not successful target reaching.
for epoch in range(num_epochs):
    tr, _transitions, metrics = tr.epoch(tr, jnp.asarray(epoch, dtype=int))
    loss = float(metrics["loss"])
    assert np.isfinite(loss)
    print(f"Epoch {epoch}: loss={loss:.4g}, mean score={float(metrics['score']):.4g}")
tr.env.system.check_overflow()

# %%
# Evaluate the live model
# ~~~~~~~~~~~~~~~~~~~~~~~
# The trainer has its own vectorized environments. Reset the serial environment
# separately with :py:meth:`~jaxdem.rl.environments.single_navigator.SingleNavigator.reset`
# and a reproducible key. Recurrent models need their memory reset
# too; the feedforward model used here ignores this operation.
tr.key, subkey = jax.random.split(tr.key)
env = env.reset(env, subkey)
base_model = tr.model
base_model.eval()
base_model.reset(shape=(env.max_num_agents, 1), mask=None)
graphdef, graphstate = nnx.split(base_model)


# %%
# `flax.nnx.split <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.split>`_
# separates static graph structure (``graphdef``) from mutable graph
# state (``graphstate``). The callback uses
# `flax.nnx.merge <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/graph.html#flax.nnx.merge>`_
# to combine them, samples an action with the
# supplied key, then returns the new state. Threading that state through the
# evaluator also supports recurrent models. Such models need memory resets at
# episode/agent boundaries;
# :py:meth:`~jaxdem.rl.models.Model.reset` is harmless for this feedforward MLP.
# Setting evaluation mode changes model behavior such as dropout; sampling the
# returned action distribution remains stochastic in this callback.
@jax.jit
def policy_model(obs, key, graphstate, graphdef):
    model = nnx.merge(graphdef, graphstate)
    distribution, _value = model(obs, sequence=False)
    action = distribution.sample(seed=key)
    _, graphstate = nnx.split(model)
    return action, graphstate


# %%
# Save twelve frames to an isolated persistent directory. A small marker sphere
# identifies the target. This short evaluation changes the objective three times;
# distance summaries are descriptive, with no expectation of improvement after
# only two updates. A context manager completes asynchronous output before exit.
# Each :py:func:`~jaxdem.utils.environment.env_step` call with
# ``n=5, skip_frames=4`` requests 25 physics frames and
# returns only the endpoint. Four calls make 20 decisions per target, and three
# targets make all 60 decisions. The horizon above exceeds the resulting 300
# physical frames; otherwise evaluation would freeze at the episode boundary.
# The target marker is appended with :py:meth:`~jaxdem.state.State.add` only
# to the state sent to :py:class:`~jaxdem.writers.vtk_writer.VTKWriter`. It does
# not become an interacting particle in the environment or affect the policy.
frames_dir = Path(tempfile.mkdtemp(prefix="jaxdem-rl-tutorial-"))
with jdem.VTKWriter(directory=frames_dir) as writer:
    for target_index in range(3):
        initial_distance = float(
            jnp.linalg.norm(env.state.pos_c - env.env_params["objective"])
        )
        for _ in range(4):
            env, tr.key, graphstate = utils.env_step(
                env,
                policy_model,
                tr.key,
                graphstate,
                graphdef=graphdef,
                n=5,
                skip_frames=skip_frames,
            )
            marker_state = env.state.add(
                env.state, pos=env.env_params["objective"], rad=env.state.rad / 5
            )
            writer.save(marker_state, env.system)
        final_distance = float(
            jnp.linalg.norm(env.state.pos_c - env.env_params["objective"])
        )
        print(
            f"Target {target_index}: distance "
            f"{initial_distance:.3f} -> {final_distance:.3f}"
        )
        if target_index < 2:
            tr.key, subkey = jax.random.split(tr.key)
            low = env.state.rad[0]
            env.env_params["objective"] = jax.random.uniform(
                subkey,
                env.env_params["objective"].shape,
                minval=low,
                maxval=env.system.domain.box_size - low,
            )

assert int(env.system.step_count) == evaluation_steps * (1 + skip_frames)
print(f"Evaluation advanced {int(env.system.step_count)} physics frames.")
print(f"Open the PVD collection in ParaView: {frames_dir}")
