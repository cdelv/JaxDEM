JaxDEM
======

JaxDEM is a GPU-accelerated discrete–element framework …

- JIT-compiled kernels via JAX
- Stateless functional core (vmap / pmap friendly)
- Plug-in forces, materials, boundaries

.. rubric:: Minimal “hello-world” simulation

.. literalinclude:: ../../examples/grid.py
   :language: python
   :linenos:

.. toctree::
   :caption: User guide
   :hidden:
   :maxdepth: 2

   user_guide/index

.. toctree::
   :caption: Examples
   :hidden:
   :maxdepth: 2

   examples/index

.. toctree::
   :caption: API reference
   :hidden:
   :maxdepth: 2

   reference/api