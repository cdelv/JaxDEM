# Naming conventions

For consistency, there are some naming conventions everyone should follow.

## Submodules

When creating new files (submodules), use the snake_case convention.

```
colliders/
forces/router.py
```

## Classes

Class names should be descriptive and follow the PascalCase (CapWords) convention, meaning each word's first letter should be capitalized.

```python
class Collider:
    ...

class ForceRouter:
    ...
```

## Functions and methods

Functions and class methods use snake_case. The established factory hook
`Create` is the explicit exception: subclasses implement `Create`, while callers
use the registry root's `create(type_name, **kwargs)`. `Factory` is not generic.

For example:

```python
import jax

from dataclasses import dataclass

import jaxdem as jdem


@jdem.Collider.register("no_contact")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class NoContactCollider(jdem.Collider):
    """Disable pair forces while following the collider interface."""

    @staticmethod
    @jax.jit(inline=True, static_argnames=("advance_history", "quasistatic"))
    def compute_force(
        state: jdem.State,
        system: jdem.System,
        *,
        advance_history: bool = True,
        quasistatic: bool = False,
    ) -> tuple[jdem.State, jdem.System]:
        """Return zero pair forces; this collider has no contact history."""
        del advance_history, quasistatic
        state.force *= 0
        state.torque *= 0
        return state, system
```

Collider overrides accept both keyword arguments even when they do not use
contact history or quasistatic evaluation. Initialization and force inspection
call this interface with `advance_history=False`.

## Variables

Variables should also follow the snake_case convention. This includes identifiers which should use `_id` suffix (e.g., `clump_id`, `bond_id`, `mat_id`).

```python
pos_c: jax.Array
ang_vel: jax.Array
clump_id: jax.Array
bond_id: jax.Array
```

Keep Python's special method names, uppercase constants, and type-variable
names in their conventional forms. Established public names and scientific
symbols, such as `State.N` and `k_B`, retain their spelling for compatibility;
prefer descriptive snake_case names for new identifiers.
