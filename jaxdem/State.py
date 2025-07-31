# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM

from dataclasses import dataclass
from typing import Optional, final, Sequence

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
    mat_id:jax.Array     # (batch?, N)
    species_id:jax.Array # (batch?, N)

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

        for name in ("rad", "mass", "ID", "mat_id", "species_id"):
            arr = getattr(self, name)
            valid = valid and self.pos.shape[:-1] == arr.shape
            assert valid, f"{name}.shape={arr.shape} is not equal to pos.shape[:-1]={self.pos.shape[:-1]}."

        return valid 

    def __init_subclass__(cls, *args, **kw):
        raise TypeError(f"{State.__name__} is final and cannot be subclassed")
        
    @staticmethod
    def create(pos: ArrayLike,*,
        vel: Optional[ArrayLike] = None,
        accel: Optional[ArrayLike] = None,
        rad: Optional[ArrayLike] = None,
        mass: Optional[ArrayLike] = None,
        ID: Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
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
        mat_id = jnp.zeros(pos.shape[:-1], dtype=int) if mat_id is None else jnp.asarray(mat_id, dtype=int)
        species_id = jnp.zeros(pos.shape[:-1], dtype=int) if species_id is None else jnp.asarray(species_id, dtype=int)

        state = State(pos=pos, vel=vel, accel=accel, rad=rad, mass=mass, ID=ID, mat_id=mat_id, species_id=species_id)
        
        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state

    @staticmethod
    def merge(state1: "State", state2: "State") -> "State":
        """
        Concatenate two State objects along their particle axis.

        1. Both states must be valid (State.is_valid == True)
        2. Spatial dimension and batch size must match
        3. Particle IDs of `state2` are shifted so that all IDs are unique
        """
        assert state1.is_valid and state2.is_valid, "One of the states is invalid"
        assert state1.dim == state2.dim,           f"dim mismatch: {state1.dim} vs {state2.dim}"
        assert state1.batch_size == state2.batch_size, f"batch_size mismatch: {state1.batch_size} vs {state2.batch_size}"

        # shift IDs of second state so that they stay unique
        state2.ID += state1.N

        # ----------------- tree-wise concatenation --------------------------
        # Arrays that have the same rank as `pos` (`pos`, `vel`, `accel`) are
        # concatenated along axis -2 (particle axis).  Everything else
        # (`rad`, `mass`, `ID`) is concatenated along axis -1.
        pos_ndim = state1.pos.ndim
        def cat(a, b):
            axis = -2 if a.ndim == pos_ndim else -1
            return jnp.concatenate((a, b), axis=axis)

        state = jax.tree_util.tree_map(cat, state1, state2)
        if not state.is_valid:
            raise ValueError(f"State is not valid, state={state}")

        return state             

    @staticmethod
    def add(
        state: "State", pos: ArrayLike, *,
        vel:   Optional[ArrayLike] = None,
        accel: Optional[ArrayLike] = None,
        rad:   Optional[ArrayLike] = None,
        mass:  Optional[ArrayLike] = None,
        ID:    Optional[ArrayLike] = None,
        mat_id: Optional[ArrayLike] = None,
        species_id: Optional[ArrayLike] = None,
    ) -> "State":
        """
        Return a new State that contains all particles in `state` plus the
        additional particle(s) described by the arguments (same conventions as
        `State.create`).  The original `state` is left unchanged.

        Example
        -------
        >>> state  = State.create(jnp.zeros((4, 2)))
        >>> state2 = State.add(state,
        ...                    pos=jnp.array([[1., 1.]]),
        ...                    rad=jnp.array([0.5]),
        ...                    mass=jnp.array([2.0]))
        """
        state2 = State.create(pos, vel=vel, accel=accel, rad=rad, mass=mass, ID=ID, mat_id=mat_id, species_id=species_id)
        return State.merge(state, state2)

    @staticmethod
    def stack(states: Sequence["State"]) -> "State":
        """
        Concatenate several trajectory States along axis 0.

        Every input state must satisfy
          1. `state.is_valid` is True
          2. identical spatial dimension (`dim`)
          3. identical batch size   (`batch_size`)
          4. identical number of particles (`N`)

        No ID shifting is performed because the leading axis represents
        **time**, not new particles.
        """
        states = list(states)
        if not states:
            raise ValueError("State.stack() received an empty list")

        ref = states[0]
        assert ref.is_valid, "first state is invalid"

        # ---------- consistency checks ---------------------------------
        for s in states[1:]:
            assert s.is_valid,                     "one state is invalid"
            assert s.dim == ref.dim,               "dimension mismatch"
            assert s.batch_size == ref.batch_size, "batch size mismatch"
            assert s.N == ref.N,                   "particle count mismatch"

        # ---------- concatenate every leaf -----------------------------
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *states
        )

        if not stacked.is_valid:                   # defensive
            raise ValueError("stacked State is not valid")

        return stacked