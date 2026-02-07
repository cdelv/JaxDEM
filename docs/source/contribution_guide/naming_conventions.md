# Naming conventions

For consistency, there are some naming conventions everyone should follow.

## Submodules

When creating new files (submodules), use the name of the base class in the submodule in lowercase. For names with multiple words, use uppercase for the first letter of the second word onwards. This is the camelCase or medial capitals convention.

```
colliders/
forces/router.py
```

## Classes

Class names should be descriptive and follow the Pascal Case convention, meaning each word's first letter should be capitalized.

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

Variables should also follow the snake_case convention.
