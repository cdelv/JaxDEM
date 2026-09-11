# Serialization and continuation

JaxDEM writes new simulation checkpoints with serialization schema 3. Orbax
checkpoints and HDF5 files use the same manifest: it identifies the storage
format and payload, marks the file as trusted input, and records the topology,
coordinate, unit, and dtype conventions required to interpret the arrays.

Schema 3 stores dynamic state as arrays. This includes particle and body state,
the PRNG key, collider buffers, contact history, plastic reference values,
topology, time, and step count. Static metadata records the component factories
and their configuration, material layout, callbacks, custom force and energy
functions, minimizer construction, and domain convention. The manifest also
records JAX's ``jax_enable_x64`` setting. Current-schema loads reject missing
static fields, changed conventions, or a different precision setting before
decoding instead of silently casting arrays or filling defaults.

Callables are stored by import path. Saving first proves that importing the path
returns the same module-level object. Local functions, closures, `__main__`
functions, and unsupported custom objects therefore fail at save time. Move
custom callbacks and force functions into an importable module before saving.

Loading is a trusted-input operation. It imports modules named by the saved
metadata, which can execute Python module code. Load only checkpoints and HDF5
files produced by code and users you trust.

Orbax metadata schemas 1 and 2 remain supported through their legacy migration
paths. Unversioned HDF5 files are treated as legacy schema 1 and retain the
previous best-effort field rename, default, and dtype behavior. A schema 3 payload must
contain its manifest; unknown versions, formats, payload kinds, or conventions
are rejected. The project does not promise migration for future schema versions
until an explicit migration is added and tested.

A restored simulation is already initialized. Continue it directly with
`System.step`; do not call `System.initialize` again. Exact continuation requires
the same JaxDEM code and compatible dependencies/backend semantics, importable
custom callables, and the complete saved state and system payload. Changing
precision, replacing callbacks or components, loading with permissive legacy
fallbacks, or editing checkpoint metadata can change subsequent evolution.

HDF5's generic `save` and `load` helpers can also store state, system, or an
object tree containing both. For exact simulation continuation, save both state
and system together. Saving only one object is useful for inspection or partial
workflows but cannot reproduce the omitted half of the simulation.
