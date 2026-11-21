Benchmarks
==========

Throughput progression (million particle-steps per second) on the unified
spheres-in-a-box benchmark (``rad = 1``, ``mass = 1``, ``k = 1000``,
``dt = 0.01``, ``steps = 1000`` on GPU and ``100`` for ``naive`` after an equal
warmup phase so measurement begins from a mixed state, ``only_spheres = False``
so performance matches the general clump path):

* **2 fixed report regimes at ``φ = 0.55``:** low velocity (``v = 0.15``) vs.
  high velocity (``v = 0.6``) on a zero-overlap lattice (3D FCC, 2D square).
  External callers can pass any ``φ ∈ [0, 0.7]`` (3D) or ``[0, 0.8]`` (2D),
  ``v ∈ [0, 1]``, and polydispersity ``--alpha >= 1``.
* **2D & 3D × 4 boundary conditions:** ``periodic``, ``leesedwards``,
  ``reflect``, and ``reflectsphere``.
* **All colliders + NeighborList combinations:** ``cell_list``,
  ``cell_list_static``, ``multi_cell_list``, ``multi_cell_list_static``,
  ``naive``, plus ``nl_cell_list``, ``nl_cell_list_static``,
  ``nl_multi_cell_list``, ``nl_multi_cell_list_static``, and ``nl_naive``.
* **Single & Batch modes:** ``single`` at ``N = 65,536`` (``3,000`` for
  ``naive`` / ``nl_naive``) and ``batch`` with ``B = 6`` replicas
  (``6 × 32,768`` spatial / ``6 × 2,000`` naive).
* **Stateless & history force laws:** ``spring`` across all 10 colliders and
  ``cundallstrack`` (with pair history) across the 6 history-capable colliders.

Each hardware target has its own section below, rendered dynamically at docs
build time from the compact JSONL log in
`benchmarks/history.jsonl <https://github.com/cdelv/JaxDEM/blob/main/benchmarks/history.jsonl>`_.

.. jaxdem-benchmarks::

Running Benchmarks Locally
--------------------------

.. code-block:: bash

   # Run everything (warns if power/throttle policy is not high performance)
   python -m benchmarks

   # Run a single case or slice along any axis
   python -m benchmarks --dim 3d --domain periodic --force spring --mode single --collider nl_cell_list --regime high_vel
   python -m benchmarks --dim 3d --domain leesedwards --force cundallstrack
   python -m benchmarks -k nl_cell_list/high_vel

   # Custom packing fraction, velocity, or polydispersity (ad-hoc run; stdout only by default)
   python -m benchmarks --dim 3d --domain periodic --collider cell_list --phi 0.55 --vel 0.3 --alpha 1.2 --steps 200

   # Record this commit into benchmarks/history.jsonl (merges partial runs cleanly)
   python -m benchmarks --dim 3d --domain periodic --record

   # Compare recorded commits or export standalone HTML
   python -m benchmarks compare [REV_A] [REV_B]
   python -m benchmarks report --out /tmp/jaxdem_benchmarks.html
