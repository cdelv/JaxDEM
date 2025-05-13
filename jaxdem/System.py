# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax

from dataclasses import dataclass
from typing import Optional

from .State import State
from .Simulator import Simulator
from .Integrator import Integrator

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class System:
    dt: float = 0.1

    simulator: Optional['Simulator'] = None
    integrator: Optional['Integrator'] = None

    @classmethod
    def create(cls, state: "State", dt: float = 0.1, simulator_type: str = "naive", integrator_type: str = "euler") -> "System":
        return cls(
            simulator = Simulator.create(simulator_type),
            integrator = Integrator.create(integrator_type)
        )