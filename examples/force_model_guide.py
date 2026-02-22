r"""
Force Models
----------------------------------------

A :py:class:`~jaxdem.forces.ForceModel` defines the pairwise interaction
law between two particles. It is evaluated by the collider for every
interacting pair and returns a force vector and torque.

This guide covers:

- The available (registered) force models.
- How materials supply the parameters each model needs.
- Combining several laws with :py:class:`~jaxdem.forces.law_combiner.LawCombiner`.
- Species-wise interactions with :py:class:`~jaxdem.forces.router.ForceRouter`.
"""

# %%
# Selecting a Force Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# The force model is chosen via the ``force_model_type`` string when
# creating a :py:class:`~jaxdem.system.System`:

import jax
import jax.numpy as jnp
import jaxdem as jdem

state = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
)
system = jdem.System.create(state.shape, force_model_type="spring")
print("Force model:", type(system.force_model).__name__)


# %%
# Available Force Models
# ~~~~~~~~~~~~~~~~~~~~~~~~
# JaxDEM ships with several pairwise force models registered in the
# :py:class:`~jaxdem.forces.ForceModel` factory. You select one via the
# ``force_model_type`` string when calling
# :py:meth:`~jaxdem.system.System.create`:
# Let's create some systems with different force models:

system_spring = jdem.System.create(state.shape, force_model_type="spring")
print("spring →", type(system_spring.force_model).__name__)

# WCA/LJ models require an ``epsilon_eff`` material table.
lj_mat = jdem.MaterialTable.from_materials(
    [jdem.Material.create("lj", density=1.0, epsilon=1.0)]
)
system_wca = jdem.System.create(state.shape, force_model_type="wca", mat_table=lj_mat)
print("wca    →", type(system_wca.force_model).__name__)

system_lj = jdem.System.create(
    state.shape, force_model_type="lennardjones", mat_table=lj_mat
)
print("lj     →", type(system_lj.force_model).__name__)

# %%
# The following ForceModels are registered:

print("ForceModels:", list(jdem.ForceModel._registry.keys()))


# %%
# Materials and Force Requirements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each force model declares the material-pair properties it needs.
# For example, ``"spring"`` requires ``young_eff`` while ``"wca"``
# requires ``epsilon_eff``. These effective properties are automatically
# computed from per-material scalars by a
# :py:class:`~jaxdem.material_matchmakers.MaterialMatchmaker`.
#
# When you pass ``mat_table=None`` (the default), ``System.create`` builds
# a single-material table with sensible defaults. For custom materials,
# build the table yourself:

mat_a = jdem.Material.create("lj", density=1.0, epsilon=1.0)
mat_b = jdem.Material.create("lj", density=1.0, epsilon=2.0)
mat_table = jdem.MaterialTable.from_materials([mat_a, mat_b])

print("epsilon per material:", mat_table.epsilon)
print("epsilon_eff (pair table):\n", mat_table.epsilon_eff)

# %%
# Pass the table to ``System.create``, and assign per-particle material
# IDs via ``mat_id`` in the state:

state_2mat = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    mat_id=jnp.array([0, 1]),  # particle 0 → mat_a, particle 1 → mat_b
)
system_2mat = jdem.System.create(
    state_2mat.shape,
    force_model_type="wca",
    mat_table=mat_table,
)

state_2mat, system_2mat = system_2mat.step(state_2mat, system_2mat)
print("Force model:", type(system_2mat.force_model).__name__)


# %%
# Combining Force Models
# ~~~~~~~~~~~~~~~~~~~~~~~~
# :py:class:`~jaxdem.forces.law_combiner.LawCombiner` sums several
# elementary force laws into one composite model. You pass it a tuple of
# :py:class:`~jaxdem.forces.ForceModel` instances via the ``laws`` field.
# Both forces and energies are added together.

combined = jdem.LawCombiner(
    laws=(jdem.ForceModel.create("spring"), jdem.ForceModel.create("wca"))
)
print("Combined laws:", [type(l).__name__ for l in combined.laws])

# %%
# To use a combined model, pass it directly to ``System.create`` via the
# ``force_model_kw`` argument. Make sure the material table provides all
# the properties required by the child laws:

mat = jdem.Material.create("elastic", density=1.0, young=1e4, poisson=0.3)
mat_lj = jdem.Material.create("lj", density=1.0, epsilon=1.0)

# Build a table that has both young_eff and epsilon_eff.
mat_table_both = jdem.MaterialTable.from_materials([mat, mat_lj])
print("Table has young_eff:", hasattr(mat_table_both, "young_eff"))
print("Table has epsilon_eff:", hasattr(mat_table_both, "epsilon_eff"))

state_combined = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
    rad=jnp.array([1.0, 1.0]),
    mat_id=jnp.array([0, 1]),  # particle 0 → mat_a, particle 1 → mat_b
)
system_combined = jdem.System.create(
    state.shape,
    force_model_type="lawcombiner",
    force_model_kw=dict(
        laws=(jdem.ForceModel.create("spring"), jdem.ForceModel.create("wca"))
    ),
    mat_table=mat_table_both,
)

state_combined, system_combined = system_combined.step(state_combined, system_combined)
print("System force model:", type(system_combined.force_model).__name__)


# %%
# Species-Wise Interactions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# In many simulations different particle *species* should interact under
# different force laws (e.g. soft repulsion between species A-B but a
# stiff spring between A-A). The
# :py:class:`~jaxdem.forces.router.ForceRouter` provides this via a
# species-pair lookup table.
#
# Use :py:meth:`~jaxdem.forces.router.ForceRouter.from_dict` to build
# the table. The keys are ``(species_i, species_j)`` tuples and the
# values are :py:class:`~jaxdem.forces.ForceModel` instances. Pairs not
# specified default to zero interaction.

# Two species: 0 and 1
router = jdem.ForceRouter.from_dict(
    S=2,
    mapping={
        (0, 0): jdem.ForceModel.create("spring"),  # A-A: spring
        (1, 1): jdem.ForceModel.create("wca"),  # B-B: WCA
        (0, 1): jdem.LawCombiner(
            laws=(jdem.ForceModel.create("spring"), jdem.ForceModel.create("wca"))
        ),  # A-B: both
    },
)
print("Router table shape:", len(router.table), "x", len(router.table[0]))

# %%
# Assign species IDs via ``species_id`` in the state, and replace the
# system's force model with the router:

state_species = jdem.State.create(
    pos=jnp.array([[0.0, 0.0], [1.5, 0.0], [3.0, 0.0]]),
    rad=jnp.array([1.0, 1.0, 1.0]),
    species_id=jnp.array([0, 0, 1]),  # first two are species 0, last is species 1
)

system_species = jdem.System.create(
    state_species.shape,
    mat_table=jdem.MaterialTable.from_materials([mat, mat_lj]),
)
system_species.force_model = router

state_species, system_species = system_species.step(state_species, system_species)
print("Active force model:", type(system_species.force_model).__name__)

# %%
# .. note::
#
#    ``species_id`` selects the force *law*, while ``mat_id`` selects the
#    material *parameters* (stiffness, epsilon, etc.). They are independent
#    — you can have species 0 and 1 share the same material but use
#    different force laws, or vice-versa.
