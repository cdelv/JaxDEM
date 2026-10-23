# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

"""Two-dimensional environment where N dynamic gears assemble a tower."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace
from functools import partial
from math import isfinite
from numbers import Integral
from typing import Tuple

from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils import lidar_2d, thermal
from ...utils.linalg import (
    cross,
    cross_3X3D_1X2D,
    dot,
    norm,
    unit,
    unit_and_norm,
)
from . import Environment

# Independent copy of the 260-sphere, 60-tooth Granulabot rim.
# Circular body radius = 1; root-to-tip depth = 0.03.
_GEAR = jnp.array(
    [
        (0.9900000095, 2.743916381e-16, 0.03999999911),
        (0.9999963045, -0.002724410035, 0.02999999933),
        (1.000040174, 0.01291346084, 0.01999999955),
        (0.9586843252, 0.05024251714, 0.03999999911),
        (0.9959117174, 0.09168994427, 0.01999999955),
        (0.9845767021, 0.1034831777, 0.03999999911),
        (0.9932120442, 0.1173753887, 0.01999999955),
        (0.9481807947, 0.1501770914, 0.03999999911),
        (0.9808717966, 0.1952887774, 0.01999999955),
        (0.9683661461, 0.2058325708, 0.03999999911),
        (0.9755020738, 0.2205513269, 0.01999999955),
        (0.9272887707, 0.2484662831, 0.03999999911),
        (0.955085218, 0.2967479825, 0.01999999955),
        (0.9415459633, 0.3059268296, 0.03999999911),
        (0.9471042752, 0.321310848, 0.01999999955),
        (0.8962371945, 0.3440332413, 0.03999999911),
        (0.9188345671, 0.3949559629, 0.01999999955),
        (0.9124339223, 0.4092240036, 0.02999999933),
        (0.9044100046, 0.4026692808, 0.03999999911),
        (0.9083297849, 0.4185500443, 0.01999999955),
        (0.8553662896, 0.4358308911, 0.03999999911),
        (0.8725169301, 0.4888367057, 0.01999999955),
        (0.8573651314, 0.4950000048, 0.03999999911),
        (0.8596035242, 0.5112034678, 0.01999999955),
        (0.8051237464, 0.5228534937, 0.03999999911),
        (0.8166398406, 0.5773617029, 0.01999999955),
        (0.8009268045, 0.5819073915, 0.03999999911),
        (0.8106153607, 0.5855789781, 0.02999999933),
        (0.8014591932, 0.5982561111, 0.01999999955),
        (0.746060133, 0.6041475534, 0.03999999911),
        (0.7518154979, 0.6595609188, 0.01999999955),
        (0.7357133627, 0.6624392867, 0.03999999911),
        (0.734533906, 0.6787540913, 0.01999999955),
        (0.6788225174, 0.6788225174, 0.03999999911),
        (0.6787540913, 0.734533906, 0.01999999955),
        (0.6624392867, 0.7357133627, 0.03999999911),
        (0.6595609188, 0.7518154979, 0.01999999955),
        (0.6041475534, 0.746060133, 0.03999999911),
        (0.5982561111, 0.8014591932, 0.01999999955),
        (0.5819073915, 0.8009268045, 0.03999999911),
        (0.5773617029, 0.8166398406, 0.01999999955),
        (0.5228534937, 0.8051237464, 0.03999999911),
        (0.5112034678, 0.8596035242, 0.01999999955),
        (0.4976387322, 0.8673843741, 0.02999999933),
        (0.4950000048, 0.8573651314, 0.03999999911),
        (0.4888367057, 0.8725169301, 0.01999999955),
        (0.4358308911, 0.8553662896, 0.03999999911),
        (0.4185500443, 0.9083297849, 0.01999999955),
        (0.4026692808, 0.9044100046, 0.03999999911),
        (0.3949559629, 0.9188345671, 0.01999999955),
        (0.3440332413, 0.8962371945, 0.03999999911),
        (0.321310848, 0.9471042752, 0.01999999955),
        (0.3059268296, 0.9415459633, 0.03999999911),
        (0.3116069138, 0.9502111077, 0.02999999933),
        (0.2967479825, 0.955085218, 0.01999999955),
        (0.2484662831, 0.9272887707, 0.03999999911),
        (0.2205513269, 0.9755020738, 0.01999999955),
        (0.2058325708, 0.9683661461, 0.03999999911),
        (0.1952887774, 0.9808717966, 0.01999999955),
        (0.1501770914, 0.9481807947, 0.03999999911),
        (0.1173753887, 0.9932120442, 0.01999999955),
        (0.1034831777, 0.9845767021, 0.03999999911),
        (0.09168994427, 0.9959117174, 0.01999999955),
        (0.05024251714, 0.9586843252, 0.03999999911),
        (0.01291346084, 1.000040174, 0.01999999955),
        (-4.075984746e-16, 0.9900000095, 0.03999999911),
        (-0.01291346084, 1.000040174, 0.01999999955),
        (-0.05024251714, 0.9586843252, 0.03999999911),
        (-0.09168994427, 0.9959117174, 0.01999999955),
        (-0.1072375625, 0.9942334294, 0.02999999933),
        (-0.1034831777, 0.9845767021, 0.03999999911),
        (-0.1173753887, 0.9932120442, 0.01999999955),
        (-0.1501770914, 0.9481807947, 0.03999999911),
        (-0.1952887774, 0.9808717966, 0.01999999955),
        (-0.2058325708, 0.9683661461, 0.03999999911),
        (-0.2205513269, 0.9755020738, 0.01999999955),
        (-0.2484662831, 0.9272887707, 0.03999999911),
        (-0.2967479825, 0.955085218, 0.01999999955),
        (-0.3059268296, 0.9415459633, 0.03999999911),
        (-0.3064247668, 0.9518948793, 0.02999999933),
        (-0.321310848, 0.9471042752, 0.01999999955),
        (-0.3440332413, 0.8962371945, 0.03999999911),
        (-0.3949559629, 0.9188345671, 0.01999999955),
        (-0.4026692808, 0.9044100046, 0.03999999911),
        (-0.4185500443, 0.9083297849, 0.01999999955),
        (-0.4358308911, 0.8553662896, 0.03999999911),
        (-0.4888367057, 0.8725169301, 0.01999999955),
        (-0.4950000048, 0.8573651314, 0.03999999911),
        (-0.5112034678, 0.8596035242, 0.01999999955),
        (-0.5228534937, 0.8051237464, 0.03999999911),
        (-0.5773617029, 0.8166398406, 0.01999999955),
        (-0.5819073915, 0.8009268045, 0.03999999911),
        (-0.5982561111, 0.8014591932, 0.01999999955),
        (-0.6041475534, 0.746060133, 0.03999999911),
        (-0.6595609188, 0.7518154979, 0.01999999955),
        (-0.6711527705, 0.7413190603, 0.02999999933),
        (-0.6624392867, 0.7357133627, 0.03999999911),
        (-0.6787540913, 0.734533906, 0.01999999955),
        (-0.6788225174, 0.6788225174, 0.03999999911),
        (-0.734533906, 0.6787540913, 0.01999999955),
        (-0.7357133627, 0.6624392867, 0.03999999911),
        (-0.7518154979, 0.6595609188, 0.01999999955),
        (-0.746060133, 0.6041475534, 0.03999999911),
        (-0.8014591932, 0.5982561111, 0.01999999955),
        (-0.8009268045, 0.5819073915, 0.03999999911),
        (-0.8074126244, 0.5899871588, 0.02999999933),
        (-0.8166398406, 0.5773617029, 0.01999999955),
        (-0.8051237464, 0.5228534937, 0.03999999911),
        (-0.8596035242, 0.5112034678, 0.01999999955),
        (-0.8573651314, 0.4950000048, 0.03999999911),
        (-0.8725169301, 0.4888367057, 0.01999999955),
        (-0.8553662896, 0.4358308911, 0.03999999911),
        (-0.9083297849, 0.4185500443, 0.01999999955),
        (-0.9044100046, 0.4026692808, 0.03999999911),
        (-0.9188345671, 0.3949559629, 0.01999999955),
        (-0.8962371945, 0.3440332413, 0.03999999911),
        (-0.9471042752, 0.321310848, 0.01999999955),
        (-0.9415459633, 0.3059268296, 0.03999999911),
        (-0.955085218, 0.2967479825, 0.01999999955),
        (-0.9272887707, 0.2484662831, 0.03999999911),
        (-0.9755020738, 0.2205513269, 0.01999999955),
        (-0.978710413, 0.2052460462, 0.02999999933),
        (-0.9683661461, 0.2058325708, 0.03999999911),
        (-0.9808717966, 0.1952887774, 0.01999999955),
        (-0.9481807947, 0.1501770914, 0.03999999911),
        (-0.9932120442, 0.1173753887, 0.01999999955),
        (-0.9845767021, 0.1034831777, 0.03999999911),
        (-0.9959117174, 0.09168994427, 0.01999999955),
        (-0.9586843252, 0.05024251714, 0.03999999911),
        (-1.000040174, 0.01291346084, 0.01999999955),
        (-0.9900000095, -3.578775347e-16, 0.03999999911),
        (-0.9999963045, 0.002724410035, 0.02999999933),
        (-1.000040174, -0.01291346084, 0.01999999955),
        (-0.9586843252, -0.05024251714, 0.03999999911),
        (-0.9959117174, -0.09168994427, 0.01999999955),
        (-0.9845767021, -0.1034831777, 0.03999999911),
        (-0.9932120442, -0.1173753887, 0.01999999955),
        (-0.9481807947, -0.1501770914, 0.03999999911),
        (-0.9808717966, -0.1952887774, 0.01999999955),
        (-0.9683661461, -0.2058325708, 0.03999999911),
        (-0.9755020738, -0.2205513269, 0.01999999955),
        (-0.9272887707, -0.2484662831, 0.03999999911),
        (-0.955085218, -0.2967479825, 0.01999999955),
        (-0.9415459633, -0.3059268296, 0.03999999911),
        (-0.9471042752, -0.321310848, 0.01999999955),
        (-0.8962371945, -0.3440332413, 0.03999999911),
        (-0.9188345671, -0.3949559629, 0.01999999955),
        (-0.9124339223, -0.4092240036, 0.02999999933),
        (-0.9044100046, -0.4026692808, 0.03999999911),
        (-0.9083297849, -0.4185500443, 0.01999999955),
        (-0.8553662896, -0.4358308911, 0.03999999911),
        (-0.8725169301, -0.4888367057, 0.01999999955),
        (-0.8573651314, -0.4950000048, 0.03999999911),
        (-0.8596035242, -0.5112034678, 0.01999999955),
        (-0.8051237464, -0.5228534937, 0.03999999911),
        (-0.8166398406, -0.5773617029, 0.01999999955),
        (-0.8009268045, -0.5819073915, 0.03999999911),
        (-0.8106153607, -0.5855789781, 0.02999999933),
        (-0.8014591932, -0.5982561111, 0.01999999955),
        (-0.746060133, -0.6041475534, 0.03999999911),
        (-0.7518154979, -0.6595609188, 0.01999999955),
        (-0.7357133627, -0.6624392867, 0.03999999911),
        (-0.734533906, -0.6787540913, 0.01999999955),
        (-0.6788225174, -0.6788225174, 0.03999999911),
        (-0.6787540913, -0.734533906, 0.01999999955),
        (-0.6624392867, -0.7357133627, 0.03999999911),
        (-0.6595609188, -0.7518154979, 0.01999999955),
        (-0.6041475534, -0.746060133, 0.03999999911),
        (-0.5982561111, -0.8014591932, 0.01999999955),
        (-0.5819073915, -0.8009268045, 0.03999999911),
        (-0.5773617029, -0.8166398406, 0.01999999955),
        (-0.5228534937, -0.8051237464, 0.03999999911),
        (-0.5112034678, -0.8596035242, 0.01999999955),
        (-0.4976387322, -0.8673843741, 0.02999999933),
        (-0.4950000048, -0.8573651314, 0.03999999911),
        (-0.4888367057, -0.8725169301, 0.01999999955),
        (-0.4358308911, -0.8553662896, 0.03999999911),
        (-0.4185500443, -0.9083297849, 0.01999999955),
        (-0.4026692808, -0.9044100046, 0.03999999911),
        (-0.3949559629, -0.9188345671, 0.01999999955),
        (-0.3440332413, -0.8962371945, 0.03999999911),
        (-0.321310848, -0.9471042752, 0.01999999955),
        (-0.3059268296, -0.9415459633, 0.03999999911),
        (-0.3116069138, -0.9502111077, 0.02999999933),
        (-0.2967479825, -0.955085218, 0.01999999955),
        (-0.2484662831, -0.9272887707, 0.03999999911),
        (-0.2205513269, -0.9755020738, 0.01999999955),
        (-0.2058325708, -0.9683661461, 0.03999999911),
        (-0.1952887774, -0.9808717966, 0.01999999955),
        (-0.1501770914, -0.9481807947, 0.03999999911),
        (-0.1173753887, -0.9932120442, 0.01999999955),
        (-0.1034831777, -0.9845767021, 0.03999999911),
        (-0.09168994427, -0.9959117174, 0.01999999955),
        (-0.05024251714, -0.9586843252, 0.03999999911),
        (-0.01291346084, -1.000040174, 0.01999999955),
        (-1.364402657e-16, -0.9900000095, 0.03999999911),
        (0.01291346084, -1.000040174, 0.01999999955),
        (0.05024251714, -0.9586843252, 0.03999999911),
        (0.09168994427, -0.9959117174, 0.01999999955),
        (0.1072375625, -0.9942334294, 0.02999999933),
        (0.1034831777, -0.9845767021, 0.03999999911),
        (0.1173753887, -0.9932120442, 0.01999999955),
        (0.1501770914, -0.9481807947, 0.03999999911),
        (0.1952887774, -0.9808717966, 0.01999999955),
        (0.2058325708, -0.9683661461, 0.03999999911),
        (0.2205513269, -0.9755020738, 0.01999999955),
        (0.2484662831, -0.9272887707, 0.03999999911),
        (0.2967479825, -0.955085218, 0.01999999955),
        (0.3059268296, -0.9415459633, 0.03999999911),
        (0.3064247668, -0.9518948793, 0.02999999933),
        (0.321310848, -0.9471042752, 0.01999999955),
        (0.3440332413, -0.8962371945, 0.03999999911),
        (0.3949559629, -0.9188345671, 0.01999999955),
        (0.4026692808, -0.9044100046, 0.03999999911),
        (0.4185500443, -0.9083297849, 0.01999999955),
        (0.4358308911, -0.8553662896, 0.03999999911),
        (0.4888367057, -0.8725169301, 0.01999999955),
        (0.4950000048, -0.8573651314, 0.03999999911),
        (0.5112034678, -0.8596035242, 0.01999999955),
        (0.5228534937, -0.8051237464, 0.03999999911),
        (0.5773617029, -0.8166398406, 0.01999999955),
        (0.5819073915, -0.8009268045, 0.03999999911),
        (0.5982561111, -0.8014591932, 0.01999999955),
        (0.6041475534, -0.746060133, 0.03999999911),
        (0.6595609188, -0.7518154979, 0.01999999955),
        (0.6711527705, -0.7413190603, 0.02999999933),
        (0.6624392867, -0.7357133627, 0.03999999911),
        (0.6787540913, -0.734533906, 0.01999999955),
        (0.6788225174, -0.6788225174, 0.03999999911),
        (0.734533906, -0.6787540913, 0.01999999955),
        (0.7357133627, -0.6624392867, 0.03999999911),
        (0.7518154979, -0.6595609188, 0.01999999955),
        (0.746060133, -0.6041475534, 0.03999999911),
        (0.8014591932, -0.5982561111, 0.01999999955),
        (0.8009268045, -0.5819073915, 0.03999999911),
        (0.8074126244, -0.5899871588, 0.02999999933),
        (0.8166398406, -0.5773617029, 0.01999999955),
        (0.8051237464, -0.5228534937, 0.03999999911),
        (0.8596035242, -0.5112034678, 0.01999999955),
        (0.8573651314, -0.4950000048, 0.03999999911),
        (0.8725169301, -0.4888367057, 0.01999999955),
        (0.8553662896, -0.4358308911, 0.03999999911),
        (0.9083297849, -0.4185500443, 0.01999999955),
        (0.9044100046, -0.4026692808, 0.03999999911),
        (0.9188345671, -0.3949559629, 0.01999999955),
        (0.8962371945, -0.3440332413, 0.03999999911),
        (0.9471042752, -0.321310848, 0.01999999955),
        (0.9415459633, -0.3059268296, 0.03999999911),
        (0.955085218, -0.2967479825, 0.01999999955),
        (0.9272887707, -0.2484662831, 0.03999999911),
        (0.9755020738, -0.2205513269, 0.01999999955),
        (0.978710413, -0.2052460462, 0.02999999933),
        (0.9683661461, -0.2058325708, 0.03999999911),
        (0.9808717966, -0.1952887774, 0.01999999955),
        (0.9481807947, -0.1501770914, 0.03999999911),
        (0.9932120442, -0.1173753887, 0.01999999955),
        (0.9845767021, -0.1034831777, 0.03999999911),
        (0.9959117174, -0.09168994427, 0.01999999955),
        (0.9586843252, -0.05024251714, 0.03999999911),
        (1.000040174, -0.01291346084, 0.01999999955),
    ]
)

pos_p, rad = _GEAR[:, :2], _GEAR[:, 2]
rr = 1.0  # circular body radius, excluding teeth
_outer_radius = 1.03
_rad = 0.04  # maximum boundary-sphere radius
N = pos_p.shape[0]
pos = jnp.zeros((N, 2))
volume = 3.236900523999325
inertia = 0.5153936746105543  # polar inertia for a unit-mass gear
q = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
y_min = jnp.min((pos + pos_p - rad[..., None])[:, 1])
one_second = 2.021757e1

# Units used in the simulation
# 1 m' = 24 mm
# 1 kg' = 98 g
# 1 m'/s'^2 = 9.81 m/s^2 (gravity)
#
# This is for training. 1 action every 1/200 seconds and 0.5 second horizon.
# num_steps_epoch = 100
# reset_every = 20
# skip_frames = 50
# max_torque = 28.6 (*mgr)
# attraction force = 5 (*mg)


@partial(jax.named_call, name="two_gears.frictional_floor_force")
def frictional_floor_force(
    pos: jax.Array, state: State, system: System
) -> Tuple[jax.Array, jax.Array]:
    # 1. Wall Definition
    n = jnp.array([0.0, 1.0])
    p = jnp.array([0.0, 1.0])
    surface_arm = -state.rad[..., None] * n
    contact_arm = pos - state.pos_c + surface_arm

    # 2. Material Properties
    k_n = 2e5
    mu = 0.4
    restitution = 0.6
    gamma_n = (-2.0 * jnp.log(restitution) * jnp.sqrt(k_n * state.mass)) / jnp.sqrt(
        jnp.pi**2 + jnp.log(restitution) ** 2
    )

    # 3. Compute normal force
    vc = state.vel + cross_3X3D_1X2D(state.ang_vel, contact_arm)
    vn_scalar = dot(vc, n)
    dist = dot(pos - p, n) - state.rad
    overlap = jnp.maximum(0.0, -dist)
    active = jnp.where(overlap > 0.0, 1.0, 0.0)
    Fn_scalar = k_n * overlap - gamma_n * vn_scalar
    Fn_scalar = jnp.maximum(0.0, Fn_scalar)
    Fn = Fn_scalar[..., None] * n

    # 4. Compute tangential force
    gamma_t = gamma_n
    vt = vc - vn_scalar[..., None] * n
    t, vt_scalar = unit_and_norm(vt)
    Ft_viscous = gamma_t * vt_scalar
    Ft_coulomb = mu * Fn_scalar
    Ft_scalar = jnp.minimum(Ft_viscous, Ft_coulomb)
    Ft = -Ft_scalar[..., None] * t

    # 5. Total Force & Torque
    F_total = Fn + Ft
    # ForceManager adds the constituent-center offset to the clump COM.
    Torque = cross(surface_arm, F_total)

    return F_total * active[..., None], Torque * active[..., None]


def _clump_first_indices(state: State, n: int) -> jax.Array:
    """First particle index of each clump ``0..n-1``. Shape ``(n,)``."""
    return jnp.stack([jnp.argmax(state.clump_id == i) for i in range(int(n))])


def _measure(
    state: State, system: System, env_params: dict
) -> Tuple[jax.Array, jax.Array]:
    """Per-gear distance to its objective and total kinetic energy.

    Gear ``i`` (``clump_id == i``) is paired with ``objective[i]``. Returns
    ``(curr_dist, curr_ke)`` each of shape ``(num_gears,)``.
    """
    n = env_params["objective"].shape[0]
    idx = _clump_first_indices(state, n)
    pos_c = state.pos_c[idx]
    delta = system.domain.displacement(pos_c, env_params["objective"], system)
    curr_dist = norm(delta)

    ke_total = thermal.compute_translational_kinetic_energy_per_particle(
        state
    ) + thermal.compute_rotational_kinetic_energy_per_particle(state)
    cid = state.clump_id
    curr_ke = jnp.stack(
        [jnp.sum(jnp.where(cid == i, ke_total, 0.0)) for i in range(int(n))]
    )
    return curr_dist, curr_ke


@Environment.register("TwoGears")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class TwoGears(Environment):
    r"""Torque-driven gears assembling an assigned vertical tower.

    Gear i targets level i, bottom to top, at a random shared x. All gears
    start on the floor and attract peers within three body radii. Actions
    are externally scaled torques; viscous drag acts on translation and
    rotation. Gear contacts use Cundall--Strack friction.

    Observations follow MultiNavigator: objective direction, displacement
    clipped to [-3, 3], COM velocity, angular velocity, and normalized
    agent/wall LiDAR. Sensors use one center per gear and the actual floor
    at y=1, excluding constituent spheres of the same gear.

    Reward is the action-interval change in
    exp(-(distance / (2 * outer_radius))**4 - beta * K / K_ref).
    K includes the whole gear's translational and rotational energy.
    The optional energy factor rewards settling near the target; its
    default zero coefficient matches MultiNavigator's distance-only reward.
    Reset/checkpoint save the starting potential, and physics preserves it.
    Episodes end at the time limit so agents can keep maintaining a stack.
    """

    num_gears: int = jax.tree.static()
    """Number of gears (agents) that must form the tower."""

    n_lidar_rays: int = jax.tree.static()
    """Number of angular LiDAR bins spanning [-pi, pi)."""

    @classmethod
    @partial(jax.named_call, name="TwoGears.Create")
    def Create(
        cls,
        num_gears: int = 3,
        box_size: float = 20.0,
        max_steps: int = 100000,
        friction: float = 0.2,
        attraction_mag: float = 4.0,
        lidar_range: float = 10.0,
        n_lidar_rays: int = 16,
        kinetic_energy_coeff: float = 0.0,
        kinetic_energy_scale: float = 1.0,
    ) -> TwoGears:
        """Create a tower environment; call reset before use.

        The square box must fit the non-overlapping floor spawn and the
        entire tower: box_size >= 2 * outer_radius * (num_gears + 1).
        friction damps linear and angular motion. attraction_mag is the
        pair force at a center distance of two body radii. Actions remain
        raw torques, with their magnitude controlled externally.

        lidar_range is a center-to-center or center-to-wall distance.
        n_lidar_rays sets the number of angular bins. kinetic_energy_coeff
        is nonnegative, and kinetic_energy_scale is a positive reference
        energy for the complete rigid gear.
        """
        for name, value in (
            ("num_gears", num_gears),
            ("max_steps", max_steps),
            ("n_lidar_rays", n_lidar_rays),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
            ("friction", friction),
            ("attraction_mag", attraction_mag),
            ("kinetic_energy_coeff", kinetic_energy_coeff),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("lidar_range", lidar_range),
            ("kinetic_energy_scale", kinetic_energy_scale),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        n = int(num_gears)
        # This spawn bound also fits the tower above the floor at y=1.
        minimum_box = 2.0 * _outer_radius * (n + 1)
        if not isfinite(box_size) or box_size < minimum_box:
            raise ValueError(
                f"box_size must be finite and at least {minimum_box:g} "
                f"to fit {n} gears and their tower"
            )
        state = State.create(pos=jnp.zeros((n * N, 2)))
        system = System.create(state.shape)

        env_params = {
            "box_size": jnp.asarray(box_size, dtype=float),
            "max_steps": jnp.asarray(max_steps, dtype=int),
            "friction": jnp.asarray(friction, dtype=float),
            "attraction_mag": jnp.asarray(attraction_mag, dtype=float),
            "lidar_range": jnp.asarray(lidar_range, dtype=float),
            "kinetic_energy_coeff": jnp.asarray(kinetic_energy_coeff, dtype=float),
            "kinetic_energy_scale": jnp.asarray(kinetic_energy_scale, dtype=float),
            "objective": jnp.zeros((n, 2)),
            "prev_dist": jnp.zeros((n,)),
            "prev_potential": jnp.zeros((n,)),
        }
        return cls(
            state=state,
            system=system,
            env_params=env_params,
            num_gears=n,
            n_lidar_rays=int(n_lidar_rays),
        )

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="TwoGears.reset")
    def reset(env: TwoGears, key: jax.Array) -> Environment:
        """Sample the tower location and a non-overlapping floor spawn."""
        n = env.num_gears
        _, key_obj, key_x = jax.random.split(key, 3)
        box = jnp.repeat(env.env_params["box_size"], 2)
        y_floor = 1.0

        x_obj = jax.random.uniform(
            key_obj, minval=_outer_radius, maxval=box[0] - _outer_radius
        )
        levels_y = y_floor + 2.0 * _outer_radius * jnp.arange(n) - y_min
        env.env_params["objective"] = jnp.stack(
            [jnp.broadcast_to(x_obj, (n,)), levels_y], axis=1
        )

        # Capped jitter keeps neighboring centers at least two radii apart.
        w = (box[0] - 2.0 * _outer_radius) / n
        centers_x = _outer_radius + (jnp.arange(n) + 0.5) * w
        jitter_amp = jnp.maximum(0.0, (w - 2.0 * _outer_radius) / 2.0)
        xs = centers_x + jax.random.uniform(
            key_x, (n,), minval=-jitter_amp, maxval=jitter_amp
        )

        state = State.create()
        for i in range(n):
            state = State.add_clump(
                state,
                pos=pos + jnp.array([[xs[i], y_floor - y_min]]),
                rad=rad,
                pos_p=pos_p,
                volume=volume,
                inertia=inertia,
                q=q,
            )
        env.state = state

        mat = Material.create(
            "elasticfrict", density=1.0 / volume, young=2e5, poisson=0.3, mu=0.1, e=0.88
        )
        env.system = System.create(
            env.state.shape,
            dt=2e-3,
            domain_type="reflect",
            domain_kw={"box_size": box, "anchor": jnp.zeros(2)},
            force_manager_kw={
                "gravity": [0.0, -1.0],
                "force_functions": (frictional_floor_force,),
            },
            mat_table=MaterialTable.from_materials([mat]),
            force_model_type="cundallstrack",
            collider_type="NeighborList",
            collider_kw={
                "state": env.state,
                "cutoff": 2 * jnp.max(env.state.rad),
                "max_neighbors": min(12, env.state.N),
                "secondary_collider_kw": {"search_range": 1},
            },
        )
        env.state, env.system = System.initialize(env.state, env.system)
        return TwoGears.checkpoint(env, jnp.zeros((n, 1)))

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="TwoGears.step")
    def step(env: TwoGears, action: jax.Array) -> Environment:
        """Apply torque, drag and pair attraction, preserving reward history."""
        action = action.reshape(env.max_num_agents, *env.action_space_shape)
        action_torque = action[env.state.clump_id]
        env.system = env.system.force_manager.add_torque(
            env.state,
            env.system,
            action_torque - env.env_params["friction"] * env.state.ang_vel,
        )
        env.system = env.system.force_manager.add_force(
            env.state,
            env.system,
            -env.env_params["friction"] * env.state.vel,
            is_com=True,
        )

        idx = _clump_first_indices(env.state, env.num_gears)
        centers = env.state.pos_c[idx]
        pair = env.system.domain.displacement(
            centers[:, None, :], centers[None, :, :], env.system
        )
        dist = norm(pair)
        dist = jnp.where(
            jnp.eye(env.num_gears, dtype=bool) | (dist == 0), jnp.inf, dist
        )
        coefficient = env.env_params["attraction_mag"] * (2.0 * rr) ** 3
        magnitude = (coefficient / dist**3) * (dist < 3.0 * rr)
        force = -jnp.sum(magnitude[..., None] * unit(pair), axis=1)
        env.system = env.system.force_manager.add_force(
            env.state, env.system, force[env.state.clump_id], is_com=True
        )
        env.state, env.system = env.system.step(env.state, env.system)
        return env

    @staticmethod
    @jax.jit(inline=True)
    def _potential(env: TwoGears) -> jax.Array:
        """MultiNavigator potential using full-gear distance and energy scales."""
        distance, kinetic_energy = _measure(env.state, env.system, env.env_params)
        energy_cost = (
            env.env_params["kinetic_energy_coeff"]
            * kinetic_energy
            / env.env_params["kinetic_energy_scale"]
        )
        return jnp.exp(-((distance / (2.0 * _outer_radius)) ** 4) - energy_cost)

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="TwoGears.checkpoint")
    def checkpoint(env: TwoGears, action: jax.Array) -> Environment:
        """Capture the live starting potential once before an action interval."""
        idx = _clump_first_indices(env.state, env.num_gears)
        delta = env.system.domain.displacement(
            env.state.pos_c[idx], env.env_params["objective"], env.system
        )
        env.env_params["prev_dist"] = norm(delta)
        env.env_params["prev_potential"] = TwoGears._potential(env)
        return env

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="TwoGears.observation")
    def observation(env: TwoGears) -> jax.Array:
        """Return goal direction/delta, velocity, spin, and normalized LiDAR.

        Layout: direction (2), clipped delta (2), COM velocity (2),
        angular velocity (1), then n_lidar_rays agent/wall proximities.
        LiDAR uses gear centers and the floor at y=1. As in MultiNavigator,
        each angular bin reports its closest detection.
        """
        idx = _clump_first_indices(env.state, env.num_gears)
        centers = env.state.pos_c[idx]
        delta = env.system.domain.displacement(
            centers, env.env_params["objective"], env.system
        )
        # Only the sensing domain moves its lower wall to the contact floor.
        anchor = jnp.array([0.0, 1.0])
        sensing_box = env.system.domain.box_size - anchor
        sensing_system = replace(
            env.system,
            domain=replace(
                env.system.domain,
                anchor=anchor,
                box_size=sensing_box,
                inv_box_size=1.0 / sensing_box,
            ),
        )
        lr = env.env_params["lidar_range"]
        _, _, lidar, _, _ = lidar_2d(
            State.create(pos=centers),
            sensing_system,
            lr,
            env.n_lidar_rays,
            sense_edges=True,
        )
        return jnp.concatenate(
            [
                unit(delta),
                jnp.clip(delta, -3.0, 3.0),
                env.state.vel[idx],
                env.state.ang_vel[idx],
                lidar / lr,
            ],
            axis=-1,
        )

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="TwoGears.reward")
    def reward(env: TwoGears) -> jax.Array:
        """Return the potential change across the complete action interval."""
        return TwoGears._potential(env) - env.env_params["prev_potential"]

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="TwoGears.truncated")
    def truncated(env: TwoGears) -> jax.Array:
        return jnp.asarray(env.system.step_count >= env.env_params["max_steps"])

    @property
    def action_space_size(self) -> int:
        return 1

    @property
    def action_space_shape(self) -> tuple[int]:
        return (1,)

    @property
    def observation_space_size(self) -> int:
        return 7 + self.n_lidar_rays

    @property
    def max_num_agents(self) -> int:
        return self.num_gears
