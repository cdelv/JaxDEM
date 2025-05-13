# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, field
from typing import Optional, Dict

from .State import State
from .Simulator import Simulator
from .Integrator import Integrator
from .Material import Material
from .MaterialMatchmaker import MaterialMatchmaker

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class System:
    dt: jax.Array = jnp.asarray(0.1, dtype=float)

    simulator: Optional['Simulator'] = None
    integrator: Optional['Integrator'] = None

    materials: Dict[int, "Material"] = field(default_factory=dict)
    material_matchmaker: Optional['MaterialMatchmaker'] = None

    @classmethod
    def create(cls, 
            state: "State", 
            dt: ArrayLike = 0.1, 
            simulator_type: str = "naive", 
            integrator_type: str = "euler",
            materials: Dict[int, "Material"] = {0: Material.create("elastic")},
            material_matchmaker_type = "linear"
        ) -> "System":

        return cls(
            dt = jnp.asarray(dt, dtype=float),
            simulator = Simulator.create(simulator_type),
            integrator = Integrator.create(integrator_type),
            materials = materials,
            material_matchmaker = MaterialMatchmaker.create(material_matchmaker_type)
        )


    @classmethod
    def add_material(cls, mat) -> int:
        return 0