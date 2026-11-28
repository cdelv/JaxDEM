# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Materials, Matchmakers, and Material Tables
------------------------------------------------

JaxDEM separates material properties from force laws:

- :py:class:`~jaxdem.materials.Material` stores per-material scalar properties.
- :py:class:`~jaxdem.material_matchmakers.MaterialMatchmaker` mixes two
  materials into an effective pair property.
- :py:class:`~jaxdem.materials.material_table.MaterialTable` stores both scalar
  arrays ``(M,)`` and precomputed pair tables ``(M, M)`` consumed by force laws.

For how force models read these properties, see
:doc:`../auto_examples/force_model_guide`.
"""

# %%
# Built-in and Custom Materials
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create built-in materials with
# :py:meth:`Material.create <jaxdem.factory.Factory.create>`:
#
# * ``"elastic"`` — ``density``, ``young``, ``poisson``
# * ``"elasticfrict"`` — ``density``, ``young``, ``poisson``, ``mu``, ``e``
#   (and optional rolling friction ``mu_r``, default ``0.0``)
# * ``"cundallstrackparams"`` — direct Cundall--Strack stiffness/damping
#   coefficients (``density``, ``k_n``, ``k_t``, ``gamma_n``, ``gamma_t``, ``mu``, ``mu_r``)
# * ``"lj"`` — ``density``, ``epsilon``

from dataclasses import dataclass

import jax.numpy as jnp

import jaxdem as jdem

elastic = jdem.Material.create("elastic", density=2500.0, young=2.0e5, poisson=0.25)
frictional = jdem.Material.create(
    "elasticfrict",
    density=1200.0,
    young=8.0e4,
    poisson=0.35,
    mu=0.6,
    e=0.9,
    mu_r=0.1,
)
lj = jdem.Material.create("lj", density=1.0, epsilon=1.5)
print("Created materials:", elastic.type_name, frictional.type_name, lj.type_name)

# %%
# You can also define a custom :py:class:`~jaxdem.materials.Material` subclass
# whenever a combined or custom force law needs a custom set of scalar fields
# (no factory registration is required if you pass instances directly):


@dataclass
class ElasticLJ(jdem.Material):
    young: float
    poisson: float
    epsilon: float


hybrid = ElasticLJ(density=1.0, young=10.0, poisson=0.3, epsilon=0.5)
print("Custom material fields:", hybrid)

# %%
# Creating a Material Table and Choosing a Matchmaker
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.materials.material_table.MaterialTable.from_materials`
# packs a list of materials into Structure-of-Arrays form and computes effective
# pair arrays (``<prop>_eff``) using a
# :py:class:`~jaxdem.material_matchmakers.MaterialMatchmaker`.
#
# Built-in matchmakers for positive scalar properties :math:`P_i, P_j`:
#
# .. math::
#
#    P_{ij}^{\mathrm{harmonic}} = \frac{2 P_i P_j}{P_i + P_j}, \qquad
#    P_{ij}^{\mathrm{linear}} = \frac{P_i + P_j}{2}.
#
# When one material omits a property defined by another, ``fill`` (default
# ``0.0``) supplies the missing value. With harmonic mixing, a ``0.0`` fill
# produces a zero effective pair value. Force models with their own
# constitutive reduction (such as Hertz's effective modulus) read the scalar
# arrays directly instead.

mat_table = jdem.MaterialTable.from_materials(
    [elastic, frictional],
    matcher=jdem.MaterialMatchmaker.create("harmonic"),
    fill=0.0,
)
mat_table_linear = jdem.MaterialTable.from_materials(
    [elastic, frictional],
    matcher=jdem.MaterialMatchmaker.create("linear"),
)

print("Scalar keys:", sorted(mat_table.props.keys()))
print("Pair keys  :", sorted(mat_table.pair.keys()))
print("young (per material):", mat_table.young)
print("harmonic young_eff[0, 1] =", float(mat_table.young_eff[0, 1]))
print("linear   young_eff[0, 1] =", float(mat_table_linear.young_eff[0, 1]))

# %%
# Using Material IDs in a Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Assign per-particle material indices with :py:attr:`state.mat_id <jaxdem.state.State.mat_id>`.
# Pass ``mat_table`` to **both** :py:meth:`~jaxdem.state.State.create` (if you
# want mass and inertia computed automatically from ``density * volume``) and
# :py:meth:`~jaxdem.system.System.create`.
#
# .. note::
#    When ``mat_table`` is passed to :py:meth:`~jaxdem.state.State.create`, it
#    computes ``mass`` from ``density * volume`` and ignores any explicit
#    ``mass=`` argument. In 2D, ``volume`` is disk area.
#
# Every force model exposes ``required_material_properties`` so you can check
# exactly which keys must exist in ``mat_table``:

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    mat_id=jnp.array([0, 1]),  # particle 0 -> elastic, particle 1 -> frictional
    mat_table=mat_table,
)
system = jdem.System.create(
    state.shape,
    force_model_type="spring",
    mat_table=mat_table,
)
print("Masses from density * area:", state.mass)
print("Required properties for spring:", system.force_model.required_material_properties)
print("Pair (0, 1) effective stiffness:", float(system.mat_table.young_eff[0, 1]))

state, system = jdem.System.initialize(state, system)
state, system = jdem.System.step(state, system)
print("Resulting forces:\n", state.force)

# %%
# Default Material Table
# ~~~~~~~~~~~~~~~~~~~~~~
# Omitting ``mat_table`` in :py:meth:`~jaxdem.system.System.create` builds a
# default single-material elastic table:

default_system = jdem.System.create(state.shape, force_model_type="spring")
print(
    f"Default table: {len(default_system.mat_table)} material, "
    f"matcher={default_system.mat_table.matcher.type_name}, "
    f"keys={sorted(default_system.mat_table.props.keys())}"
)
