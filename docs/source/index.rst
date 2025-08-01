JaxDEM
======

JaxDEM is a GPU-accelerated discrete–element (DEM) framework written in
JAX.
  
- JIT-compiled kernels and vectorisation via JAX  
- Stateless functional core for easy batching, v-map, and p-map  
- First-class support for custom forces, boundary conditions, and
  materials  

.. rubric:: Minimal “hello world” example

.. literalinclude:: ../examples/grid.py
   :language: python
   :lines: 1-50
   :linenos:

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   user_guide/state
   user_guide/system
   user_guide/writer
   user_guide/material
   user_guide/forces
   user_guide/integrator

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/quickstart
   examples/other_example

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   reference/api