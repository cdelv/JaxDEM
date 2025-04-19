# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
import jax
import jax.numpy as jnp

from functools import partial

from .Env import Env

class EnvWrapper:
    """Base class for Env wrappers."""
    def __init__(self, env):
        self._env = env
    
    def __getattr__(self, name):
        return getattr(self._env, name)

class ClipActionEnv(EnvWrapper):
    """
    Environment wrapper that clips continuous actions to a specified range 
    before passing them to the underlying environment.

    Parameters
    ----------
    env : Env
        The base environment to wrap. Must implement the standard Env interface
    low : float, optional
        Minimum value to which each action dimension will be clipped
    high : float, optional
        Maximum value to which each action dimension will be clipped
    """
    def __init__(self, env, low=-1.0, high=1.0):
        self._env = env
        self.low  = low
        self.high = high

    @partial(jax.jit, static_argnums=0)
    def step(self, env_state, action):
        clipped = jnp.clip(action, self.low, self.high)
        return self._env.step(env_state, clipped)