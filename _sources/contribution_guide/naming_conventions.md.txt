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

Functions and class methods should follow the snake_case convention:

```python
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Collider(Factory["Collider"], ABC):
    """
    ...

    """

    @staticmethod
    @abstractmethod
    @jax.jit
    def compute_force(state: "State", system: "System") -> Tuple["State", "System"]:
```

## Variables

Variables should also follow the snake_case convention. This includes identifiers which should use `_id` suffix (e.g., `clump_id`, `bond_id`, `mat_id`).

```python
pos_c: jax.Array
ang_vel: jax.Array
clump_id: jax.Array
bond_id: jax.Array
```
