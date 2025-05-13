# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, List, Tuple

from .State import State
from .Simulator import Simulator
from .Integrator import Integrator
from .Material import Material
from .MaterialMatchmaker import MaterialMatchmaker
from .Forces import ForceModel

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class System:
    dt: jax.Array = jnp.asarray(0.1, dtype=float)

    simulator: Optional['Simulator'] = None
    integrator: Optional['Integrator'] = None
    material_matchmaker: Optional['MaterialMatchmaker'] = None
    force_model: Optional['ForceModel'] = None

    material: Optional["Material"] = None

    @classmethod
    def create(cls, 
            state: "State", 
            dt: ArrayLike = 0.1, 
            simulator_type: str = "naive", 
            integrator_type: str = "euler",
            material: "Material" = Material.create("elastic"),
            material_matchmaker_type = "linear",
            force_model_type = "spring"
        ) -> "System":

        return cls(
            dt = jnp.asarray(dt, dtype=float),
            simulator = Simulator.create(simulator_type),
            integrator = Integrator.create(integrator_type),
            material = material,
            material_matchmaker = MaterialMatchmaker.create(material_matchmaker_type),
            force_model = ForceModel.create(force_model_type)
        )