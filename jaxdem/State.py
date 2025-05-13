# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax 
import jax.numpy as jnp
from jax.typing import ArrayLike

from dataclasses import dataclass, fields
from typing import Optional

@jax.tree_util.register_dataclass
@dataclass(kw_only=True)
class State:
    pos:   jax.Array = jnp.zeros((1, 3), dtype=float)
    vel:   jax.Array = jnp.zeros((1, 3), dtype=float)
    accel: jax.Array = jnp.zeros((1, 3), dtype=float)
    mass:  jax.Array = jnp.zeros(1     , dtype=float)

    _hash:     jax.Array = jnp.zeros(1 , dtype=int)
    mat_ID:    jax.Array = jnp.zeros(1 , dtype=int)
    shape_ID:  jax.Array = jnp.zeros(1 , dtype=int)

    @property
    def N(self) -> int:
        return self.pos.shape[-2]

    @property
    def dim(self) -> int:
        return self.pos.shape[-1]

    @property
    def is_valid(self) -> int | bool:
        """
        Validate that the state has the expected structure.

        Returns
        -------
        bool
            True if the state is valid; otherwise, False.
        """
        valid = True
        valid *= self.dim in (2, 3)
        assert valid, f"dim must be 2 or 3, got {self.dim}"

        for field_name in ["pos", "vel", "accel"]:
            arr = getattr(self, field_name)
            valid *= arr.shape == self.pos.shape
            assert valid, f"{field_name} shape should be {self.pos.shape}. Got {arr.shape} instead."

        for field_name in ["mass", "_hash", "mat_ID", "shape_ID"]:
            arr = getattr(self, field_name)
            valid *= arr.shape == self.pos.shape[:-1]
            assert valid, f"{field_name} shape should be {self.pos.shape[:-1]}. Got {arr.shape} instead."

        return valid

    @staticmethod
    def create(
            dim: Optional[int] = 3,
            N: Optional[int] = 1,
            pos: Optional[ArrayLike] = None,
            vel: Optional[ArrayLike] = None,
            accel: Optional[ArrayLike] = None,
            mass: Optional[ArrayLike] = None
        ) -> "State":
        """
        Factory method to create and initialize a state instance.

        Parameters
        ----------
        dim : Optional[int], default 3
            The dimensionality of the simulation domain (2 or 3). If not provided,
            dim is inferred from the 'pos' array. Otherwise, defaults to 3.
        N : Optional[int], default 1
            The number of particles. If not provided, N is inferred
            from the 'pos' array. Otherwise, defaults to 1.
        pos : ArrayLike, optional
            The position of the particles with shape (N, dim). If not provided,
            defaults to an array of zeros with shape (N, dim).
        vel : ArrayLike, optional
            The velocity vectors of the particles with shape (N, dim). If not provided,
            defaults to an array of zeros with shape (N, dim).
        accel : ArrayLike, optional
            The acceleration vectors of the particles with shape (N, dim). If not provided,
            defaults to an array of zeros with shape (N, dim).
        mass : ArrayLike, optional
            The masses of the particles with shape (N,). If not provided, defaults to an array
            of ones with shape (N,).

        Returns
        -------
        State
            A new state instance with the provided parameters.
        """
        if pos is not None and N is None:
            pos = jnp.array(pos, dtype=float)
            N = pos.shape[0]
            
        if pos is not None and dim is None:
            pos = jnp.array(pos, dtype=float)
            dim = pos.shape[1]

        if pos is None:
            pos = jnp.zeros((N, dim), dtype=float)
        pos = jnp.array(pos, dtype=float)

        if vel is None:
            vel = jnp.zeros_like(pos, dtype=float)
        vel = jnp.array(vel, dtype=float)

        if accel is None:
            accel = jnp.zeros_like(pos, dtype=float)
        accel = jnp.array(accel, dtype=float)

        if mass is None:
            mass = jnp.ones(pos.shape[:-1], dtype=float)
        mass = jnp.array(mass, dtype=float)

        _hash = jnp.zeros_like(mass, dtype=int)
        mat_ID = jnp.zeros_like(mass, dtype=int)
        shape_ID = jnp.zeros_like(mass, dtype=int)

        s = State(pos=pos, vel=vel, accel=accel, mass=mass, _hash=_hash, mat_ID=mat_ID, shape_ID=shape_ID)
        if not s.is_valid:
            raise ValueError(f"The state is not valid, got {s}")

        return s

    @classmethod
    def combine_states(cls, *states: "State") -> "State":
        """Combine multiple State instances by concatenating all fields along particle dimension"""
        merged_data = {}
        
        for field in fields(cls):
            merged_data[field.name] = jnp.concatenate([getattr(s, field.name) for s in states], axis=0)

        return cls(**merged_data)