# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import final, Tuple, Optional, Dict, Any, Sequence

from .Integrator import Integrator
from .Collider import Collider
from .Domain import Domain
from .Forces import ForceModel
from .Materials import MaterialTable, Material
from .MaterialMatchmaker import MaterialMatchmaker

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .State import State


def _check_material_table(table, required: Sequence[str]):
    missing = [k for k in required if not hasattr(table, k)]
    if missing:
        raise KeyError(f"MaterialTable lacks fields {missing}, required by the selected force model.")

@final
@jax.tree_util.register_dataclass
@dataclass(slots=True, frozen=True)
class System:
    integrator: "Integrator"
    collider: "Collider"
    domain: "Domain"
    force_model: "ForceModel"
    mat_table: "MaterialTable"
    dt: jax.Array

    @staticmethod
    def create(
            dim: int, *, 
            dt:float = 0.01, 
            integrator_type:  str = "euler",
            collider_type:    str = "naive",
            domain_type:      str = "free",
            force_model_type: str = "spring",
            mat_table:      Optional["MaterialTable"] = None,
            integrator_kw:  Optional[Dict[str, Any]] = None,
            collider_kw:    Optional[Dict[str, Any]] = None,
            domain_kw:      Optional[Dict[str, Any]] = None,
            force_model_kw: Optional[Dict[str, Any]] = None,
        ) -> "System":

        integrator_kw  = {} if integrator_kw  is None else dict(integrator_kw)
        collider_kw    = {} if collider_kw    is None else dict(collider_kw)
        force_model_kw = {} if force_model_kw is None else dict(force_model_kw)

        if domain_kw is None:
            domain_kw = {
                "box_size": jnp.ones(dim, dtype=float),
                "anchor": jnp.zeros(dim, dtype=float)
            }
        else:
            domain_kw = dict(domain_kw)
            missing = [k for k in ("box_size", "anchor") if k not in domain_kw]
            for miss in missing:
                domain_kw[miss] = {"box_size": jnp.ones(dim, dtype=float),"anchor": jnp.zeros(dim, dtype=float)}[miss]

        if mat_table is None:
            matcher   = MaterialMatchmaker.create("linear")
            mat_table = MaterialTable.from_materials([Material.create("elastic", young=1.0e4,  poisson=0.3)], matcher=matcher)

        force_model= ForceModel.create(force_model_type, **force_model_kw)

        domain_kw["box_size"] = jnp.asarray(domain_kw["box_size"],dtype=float)
        domain_kw["anchor"] = jnp.asarray(domain_kw["anchor"],dtype=float)  
        assert domain_kw["box_size"].shape == (dim,), f"box_size={domain_kw['box_size'].shape} shape must match dimension={(dim,)}"
        assert domain_kw["anchor"].shape == (dim,), f"anchor={domain_kw['anchor'].shape} shape must match dimension={(dim,)}"
        _check_material_table(mat_table, force_model.required_material_properties)

        return System(
                integrator = Integrator.create(integrator_type, **integrator_kw),
                collider = Collider.create(collider_type, **collider_kw),
                domain = Domain.create(domain_type, **domain_kw),
                force_model= force_model,
                mat_table = mat_table,
                dt = jnp.asarray(dt, dtype=float),
            )

    @staticmethod
    @partial(jax.jit, static_argnames=("n", "stride"))  
    def trajectory_rollout(state: "State", system: "System", n: int, stride: int = 1) -> Tuple["State", "System", Tuple["State", "System"]]:
        """
        Roll the system forward *n* integrator steps and return:
          - final_state
          - final_system
          - trajectory  (state PyTree with leading axis = n)
        """
        def body(carry, _):
            st, sys = carry
            st, sys = sys.step(st, sys, stride)
            return (st, sys), (st, sys)                 

        (final_state, final_system), traj = jax.lax.scan(body, (state, system), xs=None, length=n)
        return final_state, final_system, traj

    @staticmethod
    @partial(jax.jit, static_argnames=("n"))
    def steps(state: "State", system: "System", n: int) -> Tuple["State", "System"]:
        """
        Advance the simulation `n` steps using `system.integrator.step`
        """
        def body(carry, _):
            st, sys = carry                 
            st, sys = sys.integrator.step(st, sys)
            return (st, sys), None

        (final_state, final_system), _ = jax.lax.scan(body, (state, system), xs=None, length=n)
        return final_state, final_system

    @staticmethod
    @partial(jax.jit, static_argnames=("n"))
    def step(state: "State", system: "System", n: int=1) -> Tuple["State", "System"]:
        return system.integrator.step(state, system) if n == 1 else system.steps(state, system, n)