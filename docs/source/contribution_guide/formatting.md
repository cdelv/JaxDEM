# Formatting

We adopt an automatic formatting tool to maintain the look of the codebase and avoid fights about styling and formatting.

## Black

The formatting tool of choice is black. To install it:

```
pip install black
```

and to use it

```
# Format one or more files
black your_module.py another_file.py

# Format an entire directory
black .

# Check for changes without applying them
black --check .

# See diff of changes without applying them
black --diff .
```

Your code editor of choice may have a black extension.

## Import Order

When importing inside a submodule, we will follow this order: Jax-related imports, Python's standard library modules, other external libraries, internal library modules, and lastly, type-checking imports.

Python requires `from __future__` imports to come first. Keep deferred imports
and initialization-dependent imports at their required execution points;
apply the grouping above within each independent import block. Typing helpers
belong with the standard library, while `if TYPE_CHECKING:` blocks come last.

```python
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from .factory import Factory

if TYPE_CHECKING:
    from .state import State
    from .system import System
```
