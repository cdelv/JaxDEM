# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Granulabots with internal magnetic rotors and TwoGears' tower task.

Body radius = robot mass = gravity = 1 (24 mm, 98 g). Each 260-sphere,
60-tooth chassis carries active +z and passive -z magnetic rotors, orbiting
at 0.75. Their solid-cylinder force profile has radius 0.25, length 15.5/24 and axial
spacing 16.5/24; reference Br = 1.45 T. The force uses a beta/six-term blend
with default multiplier 0.25, giving about 2mg of attraction between vertically
touching robots with their same-polarity magnets facing each other.
Mechanical rotor masses and inertias retain the calibrated 98 g robot model.
Actions are normalized motor voltages, with 6 V full scale and 20 ms lag.
The default motor has twice the reference unloaded speed at the same stall
torque; motor_speed_scale=1 restores the paper-calibrated torque-speed law.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap

from ...bonded_forces import BondedForceModel
from ...colliders import Collider
from ...colliders.neighbor_list import NeighborList
from ...colliders._neighbor_cache import check_and_rebuild, forces as contact_forces
from ...forces import ForceModel
from ...forces.cundall_strack import CundallStrackForce
from ...materials import Material, MaterialTable
from ...state import State
from ...system import System
from ...utils.linalg import cross, cross_3X3D_1X2D, norm, norm2, unit
from . import Environment
from ._granulabot_magnets import magnetic_coefficient, magnetic_potential

# Hardcoded (x, y, radius); body/root radius 1, tooth-tip radius 1.03.
_GEAR = jnp.array([
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
])
_OFFSETS, _RADII = _GEAR[:, :2], _GEAR[:, 2]
_N, _STRIDE, _SEARCH, _EXTENT, _ORBIT = 260, 262, 0.04, 1.03, 0.75
_AREA, _BODY_INERTIA = 3.236900523999325, 0.3857475272426203
_MAGNET_MASS, _MAGNET_INERTIA = 0.12577390231447658, 0.004176086600285355
_BODY_MASS = 1 - 2 * _MAGNET_MASS
_TIME = 0.049461936683
_TORQUE = 0.02307312
_STALL, _DAMPING, _FRICTION = .229 * 6 / _TORQUE, .231 / (_TORQUE * _TIME), .024 / _TORQUE
# Chassis ground clearance at its identity orientation, matching TwoGears.
_Y_MIN = jnp.min(_OFFSETS[:, 1]-_RADII)


def _magnetic_curve(r):
    """Solid-cylinder coefficient and potential in simulation units."""
    return magnetic_coefficient(r), magnetic_potential(r)


def rebuild_status(state, system):
    """Return geometry, cutoff, and whether cache maintenance may be needed.

    Geometry changes must reach ``check_and_rebuild`` so it can account for
    image motion before deciding whether the skin budget requires a rebuild.
    """
    col = system.collider
    metric = system.domain.search_geometry_snapshot()
    if metric.shape != (state.dim + 3,):
        raise ValueError(
            f"Domain.search_geometry_snapshot() must have shape {(state.dim + 3,)}"
        )
    search_radii = system.force_model.search_radii(state, system)
    cutoff = jnp.maximum(col.cutoff, 2 * jnp.max(search_radii, initial=0.0))
    displacement = 2.0 * jnp.sqrt(
        jnp.max(norm2(state.pos - col.old_pos), initial=0.0)
    )
    needs_check = (
        (displacement > col.skin)
        | (col.n_build_times == 0)
        | col.invalidated
        | jnp.any(metric != col.metric_snapshot)
        | (cutoff != col.physical_cutoff_snapshot)
        | (col.skin != col.skin_snapshot)
    )
    return metric, cutoff, needs_check


# Keep a scalar outer condition after batching. Ordinary vmap executes both
# branches of each environment's rebuild condition; sequential_vmap avoids that
# work but serializes GPU launches. The outer condition skips the entire batch
# search when all caches are valid, while rebuilds still run in parallel.
_maintain_neighbors = custom_vmap(check_and_rebuild)


@_maintain_neighbors.def_vmap
def _batch_maintain_neighbors(axis_size, in_batched, state, system):
    state, system = jax.tree.map(
        lambda x, batched: x if batched else jnp.broadcast_to(x, (axis_size,)+jnp.shape(x)),
        (state, system), tuple(in_batched),
    )
    rebuild = jax.vmap(lambda s, sy: rebuild_status(s, sy)[2])(state, system)
    updated = jax.lax.cond(
        jnp.any(rebuild),
        lambda args: jax.vmap(check_and_rebuild)(*args),
        lambda args: args[1],
        (state, system),
    )
    return updated, jax.tree.map(lambda _: True, updated)


def _contact_evaluator(advance_history):
    """Evaluate only the occupied prefix, retaining the full storage capacity."""
    def evaluate(state, system, batched=False):
        col = system.collider
        capacity = col.neighbor_list.shape[-1]
        compact = min(1024, capacity)
        occupied = jnp.max(col.row_offsets[..., -1])
        force_fn = partial(contact_forces, advance_history=advance_history)
        if batched:
            force_fn = jax.vmap(force_fn)

        def small(args):
            s, sy = args
            c = sy.collider
            compact_col = replace(c, neighbor_list=c.neighbor_list[..., :compact],
                                  history=c.history[..., :compact, :])
            f, t, h = force_fn(s, replace(sy, collider=compact_col))
            if advance_history:
                h = jnp.concatenate((h, jnp.zeros_like(c.history[..., compact:, :])), axis=-2)
            else:
                h = c.history
            return f, t, h

        def nonempty(args):
            return jax.lax.cond(occupied <= compact, small,
                                lambda pair: force_fn(*pair), args)

        return jax.lax.cond(
            occupied > 0, nonempty,
            lambda args: (jnp.zeros_like(args[0].force), jnp.zeros_like(args[0].torque),
                          jnp.zeros_like(args[1].collider.history) if advance_history else args[1].collider.history),
            (state, system),
        )

    @custom_vmap
    def wrapped(state, system):
        return evaluate(state, system)

    @wrapped.def_vmap
    def batch(axis_size, in_batched, state, system):
        state, system = jax.tree.map(
            lambda x, b: x if b else jnp.broadcast_to(x, (axis_size,)+jnp.shape(x)),
            (state, system), tuple(in_batched),
        )
        result = evaluate(state, system, batched=True)
        return result, jax.tree.map(lambda _: True, result)

    return wrapped


_advance_contacts = _contact_evaluator(True)
_evaluate_contacts = _contact_evaluator(False)


@Collider.register('GranulabotNeighbors')
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _Neighbors(NeighborList):
    @staticmethod
    @partial(jax.jit, static_argnames=('advance_history',), inline=True)
    def compute_force(state, system, *, advance_history=True):
        system = _maintain_neighbors(state, system)
        force, torque, history = (_advance_contacts if advance_history else _evaluate_contacts)(state, system)
        return replace(state, force=force, torque=torque), replace(
            system, collider=replace(system.collider, history=history))

    @staticmethod
    def evaluate_force(state, system):
        return _Neighbors.compute_force(state, system, advance_history=False)


@BondedForceModel.register('GranulabotMagnets')
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MagnetPairs(BondedForceModel):
    pairs: jax.Array
    weights: jax.Array
    particle_count: int = jax.tree.static()

    @classmethod
    def Create(cls, pairs, weights, particle_count):
        return cls(jnp.asarray(pairs), jnp.asarray(weights), particle_count)

    @property
    def force_and_energy_fns(self):
        return self.force, self.energy, True

    @staticmethod
    def force(pos, state, system):
        model = system.bonded_force_model
        i, j = model.pairs[:, 0], model.pairs[:, 1]
        rho = pos[j] - pos[i]
        r = jnp.sqrt(jnp.maximum(jnp.sum(rho*rho, axis=-1), 1e-24))
        c = magnetic_coefficient(r)
        f = (model.weights*c)[:, None] * rho
        return jnp.zeros_like(pos).at[j].add(f).at[i].add(-f), jnp.zeros_like(state.torque)

    @staticmethod
    def energy(pos, state, system):
        return system.bonded_force_model.compute_potential_energy(pos, state, system)

    def compute_potential_energy(self, pos, state, system):
        rho = pos[self.pairs[:, 1]] - pos[self.pairs[:, 0]]
        u = magnetic_potential(jnp.sqrt(jnp.maximum(jnp.sum(rho*rho, axis=-1), 1e-24)))
        return jnp.sum(self.weights*u)

    @staticmethod
    def merge(first, others):
        for other in ([others] if isinstance(others, MagnetPairs) else others):
            first = replace(first, pairs=jnp.concatenate((first.pairs, other.pairs + first.particle_count)),
                            weights=jnp.concatenate((first.weights, other.weights)),
                            particle_count=first.particle_count + other.particle_count)
        return first

    @staticmethod
    def add(model, **kwargs):
        return MagnetPairs.merge(model, MagnetPairs.Create(**kwargs))


@ForceModel.register('GranulabotContacts')
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _Contacts(CundallStrackForce):
    def search_radii(self, state, system):
        return state._rad

    @staticmethod
    def _state(state):
        return replace(state, rad=jnp.where(state.clump_id % 3 == 0, state.rad, state._rad))

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        f, t, h = CundallStrackForce.force(i, j, pos, _Contacts._state(state), system,
                                         history, advance_history=advance_history)
        active = (state.clump_id[i] % 3 == 0) | (state.clump_id[j] % 3 == 0)
        return jnp.where(active, f, 0.), jnp.where(active, t, 0.), jnp.where(active, h, 0.) if advance_history else history

    @staticmethod
    def energy(i, j, pos, state, system):
        active = (state.clump_id[i] % 3 == 0) | (state.clump_id[j] % 3 == 0)
        return jnp.where(active, CundallStrackForce.energy(i, j, pos, _Contacts._state(state), system), 0.)


def _floor(pos, state, system):
    normal = jnp.array([0., 1.])
    arm = -state.rad[:, None] * normal
    velocity = state.vel + cross_3X3D_1X2D(state.ang_vel, pos - state.pos_c + arm)
    overlap = jnp.maximum(1 + state.rad - pos[:, 1], 0.)
    gamma = -2*jnp.log(.6)*jnp.sqrt(2e5*state.mass)/jnp.sqrt(jnp.pi**2 + jnp.log(.6)**2)
    fn = jnp.maximum(2e5*overlap - gamma*velocity[:, 1], 0.)
    ft = -jnp.sign(velocity[:, 0])*jnp.minimum(gamma*jnp.abs(velocity[:, 0]), .4*fn)
    active = (overlap > 0) & (state.clump_id % 3 == 0)
    force = jnp.stack((ft, fn), axis=-1)*active[:, None]
    return force, cross(arm, force)


def _radial(p):
    delta = p[1:] - p[0]
    radius = jnp.sqrt(jnp.maximum(jnp.sum(delta*delta, axis=-1), 1e-20))
    normal = delta / radius[:, None]
    j = jnp.zeros((2, 3, 2), dtype=p.dtype).at[:, 0].set(-normal).at[jnp.arange(2), jnp.arange(1, 3)].set(normal)
    return j.reshape(2, 6), radius, normal


def _solve2(a, b):
    """Small positive-definite constraint solve, avoiding a batched LU kernel."""
    determinant = a[0, 0]*a[1, 1]-a[0, 1]*a[1, 0]
    return jnp.stack((a[1, 1]*b[0]-a[0, 1]*b[1],
                      a[0, 0]*b[1]-a[1, 0]*b[0]))/determinant


def _drive(p0, velocity, omega, mass, inertia, dt, voltage, motor_speed_scale=1.):
    """Two rigid orbits, implicit back-EMF/friction, equal chassis reaction."""
    inv_mass = jnp.repeat(1/mass, 2)
    # Project near the origin: subtracting two rounded world positions and
    # dividing their difference by a small dt amplifies float32 roundoff.
    origin = p0[0]
    local0 = p0-origin

    def project(_, carry):
        p, correction = carry
        j, radius, _ = _radial(p)
        impulse = _solve2((j*inv_mass) @ j.T, radius - _ORBIT)
        delta = -(inv_mass*(j.T @ impulse)).reshape(3, 2)
        return p+delta, correction+delta

    p, correction = jax.lax.fori_loop(0, 8, project, (local0, jnp.zeros_like(local0)))
    v = jnp.concatenate(((velocity + correction/dt).ravel(), omega[0]))
    w = jnp.concatenate((inv_mass, 1/inertia[0]))
    j, radius, normal = _radial(p)
    j = jnp.pad(j, ((0, 0), (0, 1)))
    jw = j*w
    a = jw @ j.T
    v -= jw.T @ _solve2(a, j @ v)
    tangent = jnp.stack((-normal[:, 1], normal[:, 0]), axis=-1)/radius[:, None]
    b = jnp.zeros((2, 3, 2), dtype=p.dtype).at[:, 0].set(-tangent).at[jnp.arange(2), jnp.arange(1, 3)].set(tangent)
    b = jnp.pad(b.reshape(2, 6), ((0, 0), (0, 1))).at[:, -1].set(-1.)
    # The passive joint has zero torque, damping and friction: its impulse is
    # exactly zero. Solve the sole active joint directly, preserving the orbit
    # constraints and equal/opposite chassis reaction without 64 iterations.
    active = b[0]
    response = w*active-jw.T @ _solve2(a, jw @ active)
    mobility, free_speed = active @ response, active @ v
    trial = free_speed+mobility*dt*_STALL*voltage
    damping = _DAMPING/motor_speed_scale
    speed = jnp.sign(trial)*jnp.maximum(jnp.abs(trial)-mobility*dt*_FRICTION, 0.)/(1+mobility*dt*damping)
    v += response*((speed-free_speed)/mobility)
    return p+origin, v[:6].reshape(3, 2), omega.at[0, 0].set(v[-1])


def _indices(n):
    return jnp.arange(n)[:, None]*_STRIDE + jnp.array([0, _N, _N+1])


def _measure(env):
    idx = _indices(env.num_gears)
    state = env.state
    distance = norm(env.system.domain.displacement(state.pos_c[idx[:, 0]], env.env_params['objective'], env.system))
    kinetic = .5*jnp.sum(state.mass[idx]*jnp.sum(state.vel[idx]**2, axis=-1) +
                         jnp.sum(state.inertia[idx]*state.ang_vel[idx]**2, axis=-1), axis=1)
    return distance, kinetic


@Environment.register('Granulabot')
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class Granulabot(Environment):
    """TwoGears-style vertical tower with one voltage action per robot.

    All robots spawn on the floor; robot i targets tower level i at a random
    shared x coordinate. Observations include chassis/neighbor geometry and
    both robots' rotor and motor states. Actions remain normalized motor voltages, not direct
    chassis torques. The two internal rotors keep their constrained mechanics.

    ``attraction_mag`` scales the signed solid-cylinder magnetic interaction.
    Its default 0.25 gives about 2mg net attraction at a chassis separation
    of 2.06 radii with active magnets facing and passive magnets facing away.
    This is an aligned static force calibration, not an all-orientation hold.
    ``motor_speed_scale`` scales unloaded speed at fixed stall torque (default
    2). This is a faster motor model, not a change in timestep or playback rate.
    ``neighbor_capacity`` controls the average contact-pair storage per particle;
    increase it if the collider reports overflow for unusually dense scenes.
    """
    num_gears: int = jax.tree.static()
    neighbor_capacity: int = jax.tree.static()

    @classmethod
    def Create(cls, num_gears=2, box_size=20., max_steps=100000, ke_weight=.1,
               attraction_mag=.25, dt=.0005, neighbor_capacity=32, motor_speed_scale=2.):
        if isinstance(num_gears, bool) or int(num_gears) != num_gears or num_gears < 1:
            raise ValueError("num_gears must be a positive integer")
        n = int(num_gears)
        if not np.isfinite(box_size) or box_size < 2*_EXTENT*(n+1):
            raise ValueError("box_size must be >= 2*1.03*(num_gears+1) to fit the floor spawn and tower")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(motor_speed_scale) or motor_speed_scale <= 0:
            raise ValueError("motor_speed_scale must be finite and positive")
        if int(neighbor_capacity) != neighbor_capacity or neighbor_capacity < 1:
            raise ValueError("neighbor_capacity must be a positive integer")
        state = State.create(pos=jnp.zeros((n*_STRIDE, 2)))
        magnets = (np.arange(n)[:, None]*_STRIDE + np.array([_N, _N+1])).ravel()
        i, j = np.triu_indices(2*n, 1)
        signs = np.tile([1., -1.], n)
        bonds = MagnetPairs.Create(np.stack((magnets[i], magnets[j]), axis=-1),
                                   -attraction_mag*signs[i]*signs[j], state.N)
        system = System.create(state.shape, dt=dt, bonded_force_model=bonds)
        params = dict(box_size=jnp.asarray(box_size), max_steps=jnp.asarray(max_steps),
                      ke_weight=jnp.asarray(ke_weight), action=jnp.zeros((n, 1)),
                      voltage=jnp.zeros(n), objective=jnp.zeros((n, 2)),
                      prev_dist=jnp.zeros(n), prev_ke=jnp.zeros(n),
                      motor_speed_scale=jnp.asarray(motor_speed_scale))
        return cls(state=state, system=system, env_params=params, num_gears=n,
                   neighbor_capacity=int(neighbor_capacity))

    @staticmethod
    @jax.jit
    def reset(env, key):
        n, params = env.num_gears, env.env_params
        _, key_obj, key_x = jax.random.split(key, 3)
        box = jnp.full(2, params['box_size'])
        x_obj = jax.random.uniform(key_obj, minval=_EXTENT, maxval=box[0]-_EXTENT)
        params['objective'] = jnp.stack((jnp.full(n, x_obj),
                                        1-_Y_MIN+2*_EXTENT*jnp.arange(n)), axis=-1)
        width = (box[0]-2*_EXTENT)/n
        jitter = jnp.maximum((width-2*_EXTENT)/2, 0.)
        xs = _EXTENT+(jnp.arange(n)+.5)*width+jax.random.uniform(key_x, (n,), minval=-jitter, maxval=jitter)
        # Build the repeated chassis/rotor topology once with array operations,
        # instead of tracing 3*n add_clump/merge calls on every reset.
        cid = (3*jnp.arange(n)[:, None] +
               jnp.concatenate((jnp.zeros(_N, dtype=int), jnp.array([1, 2])))).ravel()
        local_centers = jnp.zeros((_STRIDE, 2)).at[_N:, 1].set(jnp.array([-_ORBIT, _ORBIT]))
        centers = jnp.stack((xs, jnp.full(n, 1-_Y_MIN)), axis=-1)
        env.state = State.create(
            pos=(centers[:, None, :]+local_centers).reshape(-1, 2),
            pos_p=jnp.tile(jnp.concatenate((_OFFSETS, jnp.zeros((2, 2)))), (n, 1)),
            rad=jnp.tile(jnp.concatenate((_RADII, jnp.full(2, .25))), n),
            _rad=jnp.full(n*_STRIDE, _SEARCH), clump_id=cid,
            volume=jnp.tile(jnp.concatenate((jnp.full(_N, _AREA), jnp.full(2, jnp.pi*.25**2))), n),
            mass=jnp.tile(jnp.concatenate((jnp.full(_N, _BODY_MASS), jnp.full(2, _MAGNET_MASS))), n),
            inertia=jnp.tile(jnp.concatenate((jnp.full(_N, _BODY_INERTIA), jnp.full(2, _MAGNET_INERTIA))), n)[:, None],
        )
        material = Material.create('elasticfrict', density=1/_AREA, young=2e5, poisson=.3, mu=.1, e=.88)
        system = System.create(
            state=env.state, dt=env.system.dt, domain_type='reflect',
            domain_kw=dict(box_size=box, anchor=jnp.zeros(2)),
            force_manager_kw=dict(gravity=[0., -1.], force_functions=(_floor,)),
            mat_table=MaterialTable.from_materials([material]), force_model_type='GranulabotContacts',
            bonded_force_model=env.system.bonded_force_model, collider_type='GranulabotNeighbors',
            collider_kw=dict(cutoff=2*_SEARCH, skin=.02, max_neighbors=env.neighbor_capacity,
                             secondary_collider_kw=dict(cell_size=2*_SEARCH+.02, search_range=1)),
        )
        # Reset changes dynamic state, not the static minimizer configuration.
        # System.create would capture the traced dt in a new FIRE optimizer,
        # making the old and reset pytrees incompatible in reset_if_done.
        env.system = replace(system, minimizer=env.system.minimizer)
        env.state, env.system = System.initialize(env.state, env.system)
        params['action'], params['voltage'] = jnp.zeros((n, 1)), jnp.zeros(n)
        params['prev_dist'], params['prev_ke'] = _measure(env)
        return env

    @staticmethod
    @jax.jit
    def step(env, action):
        """Apply voltage and advance physics; preserve the action-start baseline."""
        p, dt = env.env_params, env.system.dt
        p['action'] = jnp.clip(action.reshape(env.num_gears, 1), -1., 1.)
        p['voltage'] += -jnp.expm1(-dt*_TIME/.020)*(p['action'][:, 0]-p['voltage'])
        env.state, env.system = env.system.step(env.state, env.system)
        s, idx = env.state, _indices(env.num_gears)
        positions, velocity, omega = jax.vmap(_drive, in_axes=(0, 0, 0, 0, 0, None, 0, None))(
            s.pos_c[idx], s.vel[idx], s.ang_vel[idx], s.mass[idx], s.inertia[idx], dt,
            p['voltage'], p['motor_speed_scale'])
        env.state = replace(s, pos_c=positions.reshape(-1, 2)[s.clump_id],
                            vel=velocity.reshape(-1, 2)[s.clump_id],
                            ang_vel=omega.reshape(-1, 1)[s.clump_id])
        return env

    @staticmethod
    @jax.jit
    def checkpoint(env, action):
        """Save per-robot distance and kinetic energy before the next action.

        Parameters
        ----------
        env : Granulabot
            The environment at the start of the action interval.

        action : jax.Array
            The per-agent action about to be applied. Wrappers pass the same
            transformed action to this hook and to each physics step.
            This implementation does not require the action value.

        Returns
        -------
        Environment
            The environment with ``prev_dist`` and ``prev_ke`` captured from the
            live state. Observation and reward accessors compute current values.
        """
        p = env.env_params
        p['prev_dist'], p['prev_ke'] = _measure(env)
        return env

    @staticmethod
    @jax.jit
    def observation(env):
        """Compute per-robot observations from the live physical state."""
        return Granulabot._observe(env)

    @staticmethod
    @jax.jit
    def _observe(env):
        robot_idx = _indices(env.num_gears)
        idx = robot_idx[:, 0]
        s, p = env.state, env.env_params
        centers = s.pos_c[idx]
        pair = env.system.domain.displacement(centers[:, None], centers[None, :], env.system)
        nearest = jnp.argmin(jnp.where(jnp.eye(env.num_gears, dtype=bool), jnp.inf, norm(pair)), axis=1)
        theta = 2*jnp.arctan2(s.q.xyz[idx, 2], s.q.w[idx, 0])
        angle = (theta[nearest]-theta)[:, None]
        target = env.system.domain.displacement(centers, p['objective'], env.system)
        other = env.system.domain.displacement(centers, centers[nearest], env.system)
        # Rotor orientation means orbital position about the chassis, not the
        # local spin quaternion of a freely rotating cylindrical magnet.
        orbit = s.pos_c[robot_idx[:, 1:]]-centers[:, None]
        relative_velocity = s.vel[robot_idx[:, 1:]]-s.vel[idx, None]
        radius2 = jnp.maximum(jnp.sum(orbit*orbit, axis=-1), 1e-20)
        orbital_speed = (orbit[..., 0]*relative_velocity[..., 1]
                         - orbit[..., 1]*relative_velocity[..., 0])/radius2
        motor_speed = orbital_speed-s.ang_vel[idx]
        internal = jnp.concatenate((
            (orbit/jnp.sqrt(radius2)[..., None]).reshape(env.num_gears, 4),
            motor_speed*(_DAMPING/(_STALL*p['motor_speed_scale'])), p['voltage'][:, None],
        ), axis=-1)
        return jnp.concatenate((centers[:, 1:2]+_Y_MIN-1, centers[:, 0:1]/p['box_size'],
                                1-centers[:, 0:1]/p['box_size'], unit(target), target/p['box_size'],
                                unit(other), other/p['box_size'], jnp.sin(angle), jnp.cos(angle),
                                s.vel[idx], s.ang_vel[idx],
                                jnp.sin(theta)[:, None], jnp.cos(theta)[:, None],
                                internal, internal[nearest], s.vel[idx[nearest]], s.ang_vel[idx[nearest]]), axis=-1)

    @staticmethod
    @jax.jit
    def reward(env):
        p = env.env_params
        curr_dist, curr_ke = _measure(env)
        return p['prev_dist']-curr_dist-p['ke_weight']*(curr_ke-p['prev_ke'])

    @staticmethod
    @jax.jit
    def truncated(env):
        return env.system.step_count >= env.env_params['max_steps']

    @property
    def action_space_size(self):
        return 1

    @property
    def action_space_shape(self):
        return (1,)

    @property
    def observation_space_size(self):
        return 35

    @property
    def max_num_agents(self):
        return self.num_gears


__all__ = ['Granulabot']
