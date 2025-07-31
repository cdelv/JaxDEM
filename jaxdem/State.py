# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

from dataclasses import dataclass
from typing import Optional, final

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

@final
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class State:
    """Simulation state for N particles in 2- or 3-D."""
    pos:   jax.Array     # (batch?, N, dim)
    vel:   jax.Array     # (batch?, N, dim)
    accel: jax.Array     # (batch?, N, dim)
    rad:   jax.Array     # (batch?, N)
    mass:  jax.Array     # (batch?, N)
    ID:    jax.Array     # (batch?, N)

    @property
    def N(self) -> int:
        """Return number of particles"""
        return self.pos.shape[-2]

    @property
    def dim(self) -> int:
        """Return simulation spatial dimension"""
        return self.pos.shape[-1]

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        return 1 if self.pos.ndim < 3 else self.pos.shape[-3]

    @property
    def is_valid(self) -> bool:
        """Check if the internal representation is inconsistent."""
        valid = self.dim in (2, 3)
        assert valid, f"Simulation dimension (pos.shape[-1]={self.dim}) must be 2 or 3." 

        for name in ("pos", "vel", "accel"):
            arr = getattr(self, name)
            valid = valid and self.pos.shape == arr.shape
            assert valid, f"{name}.shape={arr.shape} is not equal to pos.shape={self.pos.shape}." 

        for name in ("rad", "mass","ID"):
            arr = getattr(self, name)
            valid = valid and self.pos.shape[:-1] == arr.shape
            assert valid, f"{name}.shape={arr.shape} is not equal to pos.shape[:-1]={self.pos.shape[:-1]}."

        return valid 

    # --------------------------------------------------------------------- #
    # Constructors
    # --------------------------------------------------------------------- #
    def __init_subclass__(cls, *args, **kw):
        raise TypeError(f"{State.__name__} is final and cannot be subclassed")
        
    @staticmethod
    def create(pos: ArrayLike,*,
        vel: Optional[ArrayLike] = None,
        accel: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        ID: Optional[ArrayLike] = None,
    ) -> "State":
        """
        Factory constructor.

        Parameters
        ----------
        pos   : (N, dim) or (B, N, dim) array-like – particle positions
        vel   : same shape as `pos`; defaults to zeros
        accel : same shape as `pos`; defaults to zeros
        rad   : shape = `pos.shape[:-1]`; defaults to ones
        mass  : shape = `pos.shape[:-1]`; defaults to ones
        ID    : shape = `pos.shape[:-1]`; defaults to `jnp.arange(N)`

        Returns
        -------
        State
        """
        pos = jnp.asarray(pos, dtype=float)
        N = pos.shape[-2]
        
        vel   = jnp.zeros_like(pos, dtype=float) if vel is None else jnp.asarray(vel, dtype=float)
        accel = jnp.zeros_like(pos, dtype=float) if accel is None else jnp.asarray(accel, dtype=float)
        rad  = jnp.ones(pos.shape[:-1], dtype=float) if rad is None else jnp.asarray(rad, dtype=float)
        mass  = jnp.ones(pos.shape[:-1], dtype=float) if mass is None else jnp.asarray(mass, dtype=float)
        ID    = jnp.broadcast_to(jnp.arange(N, dtype=int), pos.shape[:-1]) if ID is None else jnp.asarray(ID, dtype=int)

        state = State(pos=pos, vel=vel, accel=accel, rad=rad, mass=mass, ID=ID)
        
        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state