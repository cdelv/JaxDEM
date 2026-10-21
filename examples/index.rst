:orphan:

JaxDEM tutorials and examples
=============================

This directory contains the executable sources for two parts of the docs.
The :doc:`../user_guide/index` introduces the API through component tutorials;
the :doc:`../examples/index` combines those components into application
recipes. A source file living here does not make it an application example.

Run a script from the repository root with ``python examples/<name>.py``
after installing JaxDEM and the optional dependencies named by that page.
Defaults use small, reproducible configurations so readers can inspect a
result quickly. Increase sizes and numerical budgets deliberately when
moving to a research calculation, and check convergence before interpreting
its output.
