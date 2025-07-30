# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import final, Tuple

#from .Collider import Collider
from .Integrator import Integrator
#from .Forces import ForceModel
#from .Domain import Domain

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State

@final
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class System:
    integrator: "Integrator"
    dt: jax.Array

    @staticmethod
    def create(
            dim: int, *, 
            dt:float = 0.1, 
            integrator_type: str = "euler",
            **integrator_kw
        ) -> "System":

        return System(
                integrator = Integrator.create(integrator_type, **integrator_kw),
                dt = jnp.asarray(dt, dtype=float),
            )

    @staticmethod
    @partial(jax.jit, static_argnums=2)      
    def step_rollout(state: "State", system: "System", n: int) -> Tuple["State", "System", "State"]:
        """
        Roll the system forward *n* integrator steps and return:
          - final_state
          - final_system
          - trajectory  (state PyTree with leading axis = n)
        """
        def body(carry, _):
            st, sys = carry
            st, sys = sys.integrator.step(st, sys)
            return (st, sys), st                 

        (final_state, final_system), traj = jax.lax.scan(body, (state, system), xs=None, length=n)
        return final_state, final_system, traj

    @staticmethod
    @partial(jax.jit, static_argnums=2)
    def steps(state: "State", system: "System", n: int) -> Tuple["State", "System"]:
        """
        Advance the simulation exactly `n` steps using `system.integrator.step`
        """
        def body(carry, _):
            st, sys = carry                 
            st, sys = sys.integrator.step(st, sys)
            return (st, sys), None

        (final_state, final_system), _ = jax.lax.scan(body, (state, system), xs=None, length=n)
        return final_state, final_system

    @staticmethod
    @partial(jax.jit, static_argnums=2)
    def step(state: "State", system: "System", n: int=1) -> Tuple["State", "System"]:
        if n == 1:
            return system.integrator.step(state, system)
        return system.steps(state, system, n)