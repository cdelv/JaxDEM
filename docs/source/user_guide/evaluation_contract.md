# Evaluation and physical evolution

JaxDEM returns the new state and system explicitly. Always retain both return
values inside and outside JAX transformations.

| Operation | Physical evolution | Returned cache changes |
| --- | --- | --- |
| `System.evaluate_forces` | None; evaluates instantaneous pair and managed forces, including queued loads without consuming them | Search geometry and neighbor caches may be rebuilt |
| `Collider.evaluate_force` | None; evaluates pair forces with frozen contact history | Search caches may be rebuilt |
| Energy, objectives, Hessians | No contact or plastic history advancement | Returned search caches may change |
| `System.initialize` | Applies queued loads and configured integrator initialization; no contact/plastic evolution or time advance | Initializes force/search data |
| `System.step` | Executes the physical stages below once per timestep | Advances search caches and pair history |

The timestep order is:

1. Advance the clock and run the pre-step protocol callback.
2. Apply the domain update and the linear/rotational pre-force stages.
3. Evolve bonded reference state, then evaluate contacts and advance their history.
4. Apply managed forces and consume queued loads; aggregate body force/torque.
5. Complete both integrator kicks, then run both `finalize_step` hooks.
6. Run the post-step protocol callback.

Velocity rescaling belongs to `finalize_step` so temperature includes both final
kicks. Plastic reference evolution belongs to `update_reference_state`, never to
an energy or force probe. A quasistatic plastic protocol must explicitly evolve
the reference between minimizations. Custom force callbacks must be pure;
externally prescribed evolution belongs in a timestep callback and its returned
state/system.

`System.initialize` remains the user's responsibility before a new trajectory.
It is not repeated when continuing an initialized checkpoint. An instantaneous
force probe does not perform an integrator's initial velocity staggering.

## Checked execution

`System.step` retains its unchecked device loop. For a run that must stop on
invalid values or insufficient search capacity, use:

```python
system.validate(state)  # optional host validation after construction
state, system = system.initialize(state, system)
result = system.step_checked(state, system, n=100)
result.check()  # host synchronization; raises on any failed snapshot
state, system = result.state, result.system
```

`StepResult.steps` counts accepted steps. `status` contains `SimulationStatus`
bits for search overflow and nonfinite particle/search/load/domain data. On
failure, the result contains the state and system before the rejected step,
including their contact history, PRNG, time, and pending loads. The status stays
in the result even though the returned system predates the failure. An invalid
input is returned unchanged with zero accepted steps. Batched execution reports
one count and status per snapshot.

Checked execution adds device reductions and uses dynamic stopping; it does not
support reverse-mode differentiation. It does not automatically initialize,
resize capacity, change the timestep, or retry. Correct the cause and explicitly
retry from the returned snapshot. Device rollback cannot undo external effects
from callbacks, so callbacks used in this path must return their evolution as
data rather than writing files or changing host objects.

Minimization exposes `MinimizationResult.reason`. Quasistatic compression exposes
`CompressionResult.reason` and `minimizer_reason`; it stops on failed relaxation
and preserves the last accepted packing. Its historical four-value unpacking
remains available. Reaching an outer iteration limit is distinct from reaching
the requested packing fraction. Inspect these results before accepting output.

Jamming drivers return `JamResult` with the same historical six-value iteration
order. Its `reason` distinguishes a target hit, bracket convergence, initial
over-compression, outer-step exhaustion, nonfinite data, search overflow, and
other minimizer failure. `minimizer_reason` retains the last inner
`TerminationReason`, while `steps` totals all inner minimizer iterations.
Failed searches preserve the last accepted packing; callers should inspect
`reason` before using `jammed_state` as a successful jammed configuration.
`potential_energy` is `NaN` on failure because it does not describe the rolled
back state; it is meaningful only for an accepted result.
