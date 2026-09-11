# RL support and execution contract

The DEM core's 1.0 compatibility promise does not extend to `jaxdem.rl`.
Reinforcement learning remains an experimental optional subsystem installed with
`JaxDEM[rl]`. Its interfaces and numerical behavior are tested separately, but
model internals, training-state layouts, and scenario constructors may change
before an RL stable release. This does not relax correctness requirements for
the contracts below.

The regression matrix covers Python 3.12 and 3.14 on CPU in float32 and float64,
with an optional CUDA job. It exercises feed-forward, LSTM, and MinGRU policies,
PPO numerical behavior, action spaces, metrics, and selected environment resets.
The configured matrix is distinct from evidence that a remote job has passed.
The DEM minimum-JAX wheel test does not establish minimum versions for every RL
dependency combination.

## Environment interface

A scalar environment has a fixed maximum agent count `A`, including `A=1` for
a single agent. Observations have shape `(A, observation_space_size)`; continuous
actions have shape `(A, action_space_size)`; rewards and `agent_mask` have shape
`(A,)`. A vectorized environment prepends one environment axis `E`, including
`E=1`. Agent identities occupy fixed slots; padding must contain finite values.
Every environment must have at least one active agent at every observation,
including immediately after reset. Entirely inactive environments are unsupported;
PPO validates this condition during construction and the initial reset.

Environment subclasses define:

- `terminated(env)`: a scalar true terminal condition, such as reaching an
  absorbing task outcome. No value is bootstrapped beyond this boundary.
- `truncated(env)`: a scalar external boundary, such as a time limit. The final
  observation retains its value bootstrap.
- `agent_mask(env)`: a boolean mask identifying active agent slots. The default
  includes every slot. Inactive actions are zero and inactive samples do not
  contribute to the learning objective or replay priorities.

`done(env)` combines termination and truncation. Both stop the advantage trace
and reset episode memory. If both flags are true, termination takes precedence
for bootstrapping. Built-in time limits use `step_count >= max_steps` and report
truncation. Custom environments should override the two explicit boundary hooks,
rather than overriding `done`.

PPO records the transition before resetting finished environments. A truncation
bootstrap uses the final observation before reset, including the recurrent carry
after processing the preceding observation. The bootstrap probe does not advance
the policy's persistent carry. The next rollout transition sees the reset
environment and fresh memory. An agent disappearing from an active slot is a
terminal transition for that agent, even when the environment continues.
Inactive recurrent slots are cleared; a reactivated slot starts with fresh memory.

`skip_frames=k` repeats the sampled action for up to `1+k` physics frames, stopping
each environment at its first episode boundary. Rewards are sampled from the
last accepted frame; they are not summed across repeated frames. The trainer
avoids running environment resets on steps where no environment has finished.

## Models, objectives, and schedules

Calling a model returns its action distribution and value estimate. Use those
outputs and the PPO loss metrics for diagnostics. `Model.carry` exposes recurrent
state; `Model.reset` clears it. Policy internals such as individual dense layers
are not a stable diagnostic API. Training and replay must reset recurrent state
at the same recorded boundaries.

Learning-rate and importance-sampling schedules are expressed in outer PPO
epochs. Gradient accumulation averages raw gradients before clipping and
applying the optimizer. Changing the minibatch count does not redefine an epoch.
Checkpointed optimizer state preserves its update position; changing the
training layout during continuation is outside the compatibility contract.

`BoxSpace` constrains components independently. `MaxNormSpace` constrains the
vector radius, with an exact inverse/Jacobian and a smooth origin. Its entropy
uses tensor-product quadrature and supports at most six action dimensions;
larger requests fail before allocating the grid. Use componentwise Box or Free
actions for higher-dimensional policies. No custom bijector equality overrides
are provided.

## Names, examples, and persistence

Wrappers live only in `jaxdem.rl.env_wrappers`; update old `envWrappers` imports.
There is no duplicate compatibility package. `vectorise_env` accepts a positive
integer batch size and reuses wrapper classes so equivalent wrapping preserves
JIT compilation identity.

Navigation, roller, gear, and burrowing environments are research scenarios,
not part of the DEM core's stable API. Their geometry, rewards, and initial
conditions remain subject to change. New scenario-specific datasets and setup
belong in examples. The unused hard-coded bed arrays were moved from the runtime
package to `examples/data/settled_bed.json`; no runtime module imports that data.

Metrics use TensorBoardX and do not need TensorFlow GPU support. Exact RL
continuation requires compatible model code, dependencies, trainer layout,
optimizer state, recurrent carry, environment state, and PRNG state. A model-only
checkpoint does not preserve a complete training run. Old RL training-state
layouts are not silently migrated to the new termination/masking contract.
