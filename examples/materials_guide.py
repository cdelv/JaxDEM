r"""
Materials, Matchmakers, and Material Tables
-------------------------------------------

JaxDEM separates material definitions from force laws:

- :py:class:`~jaxdem.materials.Material` stores per-material scalar properties.
- :py:class:`~jaxdem.material_matchmakers.MaterialMatchmaker` defines how to mix
  two material properties into an effective pair property.
- :py:class:`~jaxdem.materials.MaterialTable` stores both scalar
  properties and precomputed pair tables used by force models.

This guide shows how to create materials, choose a matchmaker, and pass a
material table into :py:class:`~jaxdem.system.System`. For how force models
consume these properties, see :doc:`../auto_examples/force_model_guide`.
"""

# %%
# Creating Materials
# ~~~~~~~~~~~~~~~~~~
# Materials are created through the factory interface.
# Each registered material type defines its own required fields.

import jax.numpy as jnp
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
# You can inspect available material types in the factory registry:

print("Registered materials:", list(jdem.Material._registry.keys()))


# %%
# Creating a MaterialTable
# ~~~~~~~~~~~~~~~~~~~~~~~~
# :py:meth:`~jaxdem.materials.MaterialTable.from_materials` converts a list of material objects into a
# Structure-of-Arrays representation and computes effective pair properties.
#
# When some materials do not define a property, ``fill`` is used.

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
# effective pair properties are computed.

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
# Particles select their material via ``state.mat_id``.
# Force models then query pair values from ``system.mat_table``.

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    mat_id=jnp.array([0, 1]),  # particle 0 uses elastic, particle 1 uses frictional
)

system = jdem.System.create(
    state.shape,
    force_model_type="spring",
    mat_table=mat_table,
)

print(
    "Required properties for spring:", system.force_model.required_material_properties
)
print("Effective stiffness used by pair (0,1):", system.mat_table.young_eff[0, 1])

state, system = system.step(state, system)
print("Resulting forces:\n", state.force)


# %%
# Default MaterialTable in System.create
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If ``mat_table`` is omitted, :py:meth:`~jaxdem.system.System.create`
# builds a default single-material table.

default_system = jdem.System.create(state.shape, force_model_type="spring")
print("Default number of materials:", len(default_system.mat_table))
print("Default matcher:", default_system.mat_table.matcher.type_name)
print("Default scalar keys:", sorted(default_system.mat_table.props.keys()))
