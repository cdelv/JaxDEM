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

```python
import jax
import jax.numpy as jnp

from typing import ClassVar, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .factory import Factory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import State
    from .system import System
```
