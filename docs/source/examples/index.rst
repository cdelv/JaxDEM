Examples
========

These application recipes build on the component tutorials in the
:doc:`../user_guide/index`. The tutorials explain the API; these recipes
combine it into simulation and analysis workflows. Both sets of pages are
generated from Python sources in the repository's ``examples`` directory.

The scripts use small, reproducible configurations for a short first run.
Their particle counts and iteration limits are teaching defaults, not
production settings or evidence of bulk behavior. Each recipe explains its
observables and any checks needed before interpreting the result.

Construction & packing pipelines
--------------------------------

.. toctree::
   :maxdepth: 1

   ../auto_examples/sphere_construction
   ../auto_examples/general_particle_construction
   ../auto_examples/clump_construction_pipeline
   ../auto_examples/dp_construction_pipeline
   ../auto_examples/minimize_spheres
   ../auto_examples/jam_spheres
   ../auto_examples/sphere_jamming_rattlers_dynamical_matrix
   ../auto_examples/clump_jamming_dynamical_matrix_and_friction
   ../auto_examples/packing_fraction_protocol
   ../auto_examples/pseudo_log_rollout
   ../auto_examples/sasa_calculation
   ../auto_examples/particle_level_friction_calculation

Dynamics & active matter
------------------------

.. toctree::
   :maxdepth: 1

   ../auto_examples/rotating_drum
   ../auto_examples/temperature_density_control
   ../auto_examples/shear_flow_disks
   ../auto_examples/shear_jammed_disks
   ../auto_examples/vicsek_model

Inference, deformation & ensembles
----------------------------------

These short applications connect mechanics to measured outcomes: fitting a
contact parameter, comparing constitutive response, following a rearrangement,
and quantifying preparation dependence across independent simulations.

.. toctree::
   :maxdepth: 1

   ../auto_examples/infer_contact_stiffness
   ../auto_examples/rigid_elastic_viscoplastic_loading
   ../auto_examples/soft_mode_rearrangement
   ../auto_examples/batched_granular_experiments

Reinforcement learning
----------------------

Start with the :doc:`training tutorial <../auto_examples/intro_to_rl>` for
the optional, experimental reinforcement-learning subsystem.

.. toctree::
   :maxdepth: 1

   ../auto_examples/environment_custom_policy
   ../auto_examples/data/index
