r"""Materials, Matchmakers, and Material Tables
------------------------------------------------

JaxDEM separates material definitions from force laws:

- :py:class:`~jaxdem.materials.Material` stores per-material scalar properties.
- :py:class:`~jaxdem.material_matchmakers.MaterialMatchmaker` defines how to mix
  two material properties into an effective pair property.
- :py:class:`~jaxdem.materials.material_table.MaterialTable` stores both scalar
  properties and precomputed pair tables used by force models.

This guide shows how to create materials, choose a matchmaker, and pass a
material table into :py:class:`~jaxdem.system.System`. For how force models
consume these properties, see :doc:`../auto_examples/force_model_guide`.
"""

# %%
# Creating Materials
# ~~~~~~~~~~~~~~~~~~
# You create materials through
# :py:meth:`Material.create <jaxdem.factory.Factory.create>`.
# Each registered material type defines its own required fields.

import jax.numpy as jnp
import numpy as np
import jaxdem as jdem

elastic = jdem.Material.create(
    "elastic",
    density=2500.0,
    young=2.0e5,
    poisson=0.25,
)

frictional = jdem.Material.create(
    "elasticfrict",
    density=1200.0,
    young=8.0e4,
    poisson=0.35,
    mu=0.6,
    e=1.0,
)

lj = jdem.Material.create(
    "lj",
    density=1.0,
    epsilon=1.5,
)

print("elastic type:", elastic.type_name)
print("frictional type:", frictional.type_name)
print("lj type:", lj.type_name)


# %%
# You can inspect available material types in the
# :py:class:`~jaxdem.factory.Factory` registry (the private ``Material._registry`` dictionary):

print("Registered materials:", list(jdem.Material._registry.keys()))


# %%
# Creating a Material Table
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.materials.material_table.MaterialTable.from_materials` converts a list of material objects
# into a
# Structure-of-Arrays representation and computes effective pair properties.
#
# When a material does not define a property, ``fill`` supplies the value.
# With harmonic mixing, a zero-filled property makes the mixed pair value
# zero. Merely storing a property name does not make that interaction active.

harmonic_matcher = jdem.MaterialMatchmaker.create("harmonic")
mat_table = jdem.MaterialTable.from_materials(
    [elastic, frictional],
    matcher=harmonic_matcher,
    fill=0.0,
)

print("Number of materials:", len(mat_table))
print("Stored scalar keys:", sorted(mat_table.props.keys()))
print("Stored pair keys:", sorted(mat_table.pair.keys()))

# Scalar per-material arrays (shape: (M,))
print("young:", mat_table.young)
print("mu (elastic filled with 0.0):", mat_table.mu)

# Effective pair arrays (shape: (M, M))
print("young_eff:\n", mat_table.young_eff)
print("poisson_eff:\n", mat_table.poisson_eff)
print("mu_eff:\n", mat_table.mu_eff)


# %%
# Matchmakers
# ~~~~~~~~~~~
# A :py:class:`~jaxdem.material_matchmakers.MaterialMatchmaker` controls how
# the material table computes effective pair properties.
# For a positive scalar material property P, the two built-in averages are
#
# .. math::
#
#    P_{ij}^{\mathrm{harmonic}}=\frac{2P_iP_j}{P_i+P_j},\qquad
#    P_{ij}^{\mathrm{linear}}=\frac{P_i+P_j}{2}.
#
# Both return P for two identical materials. The harmonic mean is dominated
# by the smaller value and becomes zero when either value is zero, which is
# why missing properties and their fill values affect mixed interactions.
# These are the table's mixing rules; a force model with its own constitutive
# reduction, such as Hertz's effective modulus, can read the per-material
# properties directly instead. Choose a registered mixing rule with
# :py:meth:`MaterialMatchmaker.create <jaxdem.factory.Factory.create>`.

linear_matcher = jdem.MaterialMatchmaker.create("linear")
mat_table_linear = jdem.MaterialTable.from_materials(
    [elastic, frictional],
    matcher=linear_matcher,
)

print("Registered matchmakers:", list(jdem.MaterialMatchmaker._registry.keys()))
print("harmonic young_eff[0,1] =", mat_table.young_eff[0, 1])
print("linear   young_eff[0,1] =", mat_table_linear.young_eff[0, 1])


# %%
# Using Material IDs in a Simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Particles select their material via :py:attr:`state.mat_id <jaxdem.state.State.mat_id>`.
# Force models then query pair values from :py:attr:`system.mat_table <jaxdem.system.System.mat_table>`.
# Pass the table to :py:meth:`State.create <jaxdem.state.State.create>` too if density should determine mass
# and
# inertia. Passing it only to :py:meth:`System.create <jaxdem.system.System.create>` does not update existing
# masses.
# In this 2D demonstration, volume is disk area, so density is interpreted as
# mass per area (or a three-dimensional density with unit out-of-plane depth).
# Use consistent units; the :py:class:`spring law <jaxdem.forces.spring.SpringForce>` reads ``young_eff`` as a
# force/length
# stiffness coefficient rather than converting a bulk modulus automatically.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    mat_id=jnp.array([0, 1]),  # particle 0 uses elastic, particle 1 uses frictional
    mat_table=mat_table,
)

system = jdem.System.create(
    state.shape,
    force_model_type="spring",
    mat_table=mat_table,
)

np.testing.assert_allclose(state.mass, mat_table.density[state.mat_id] * state.volume)
print("Masses from density times area:", state.mass)

print(
    "Required properties for spring:", system.force_model.required_material_properties
)
print("Effective stiffness used by pair (0,1):", system.mat_table.young_eff[0, 1])

state, system = jdem.System.initialize(state, system)
state, system = system.step(state, system)
print("Resulting forces:\n", state.force)


# %%
# Default Material Table in System Creation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you omit ``mat_table``, :py:meth:`~jaxdem.system.System.create`
# builds a default single-material table.

default_system = jdem.System.create(state.shape, force_model_type="spring")
print("Default number of materials:", len(default_system.mat_table))
print("Default matcher:", default_system.mat_table.matcher.type_name)
print("Default scalar keys:", sorted(default_system.mat_table.props.keys()))
