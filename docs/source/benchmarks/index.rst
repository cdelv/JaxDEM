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

   # Regular benchmark run with Pallas Triton GPU kernels disabled
   python -m benchmarks --dim 3d --domain periodic --collider nl_cell_list --pallas off

   # Record this commit into benchmarks/history.jsonl (merges partial runs cleanly)
   python -m benchmarks --dim 3d --domain periodic --record

   # Compare recorded commits or export standalone HTML
   python -m benchmarks compare [REV_A] [REV_B]
   python -m benchmarks report --out /tmp/jaxdem_benchmarks.html

Throughput Scaling vs. N
------------------------

Use ``python -m benchmarks scaling`` to sweep throughput (million
particle-steps per second) against particle count ``N`` in powers of two (by
default from ``2`` to ``2,097,152``, i.e. ``2^1`` through ``2^21``) and plot
the result in log-log scale.

Every configuration axis accepts either a single value or a comma-separated
list (``--collider``, ``--dim``, ``--domain``, ``--force``, ``--mode``,
``--pallas``, ``--phi``, ``--vel``, ``--alpha``, ``--only-spheres``). Fixed
parameters are automatically summarized in the plot title, while varying
parameters appear next to the collider name in the legend. Each collider keeps
a consistent color and parameter variants are distinguished by line style and
marker shape.

Control the ``N`` range with ``--min-n`` / ``--max-n``, ``--min-exp`` /
``--max-exp``, ``--step-exp`` (exponent stride), or an explicit ``--ns`` list.
By default, ``O(N^2)`` colliders (``naive`` / ``nl_naive``) are capped at
``N = 8,192`` (configurable via ``--max-n-naive``; pass
``--ignore-naive-limits`` to disable both the ``N`` cap and the reduced naive
step count). Use ``--max-seconds`` to stop a curve early once a single point
exceeds a time budget; OOM points stop cleanly without aborting remaining
curves. Results are printed to stdout only by default—pass ``--plot PATH``
(``.png`` / ``.pdf`` / ``.svg`` via matplotlib, or interactive ``.html``)
and/or ``--out PATH`` (re-plottable anytime with ``--from-json PATH``) to save
outputs.

.. code-block:: bash

   # Compare Pallas ON vs OFF for cell_list and nl_cell_list from N=64 to 2M
   python -m benchmarks scaling \
     --collider cell_list,nl_cell_list \
     --pallas on,off \
     --min-n 64 --max-n 2097152 \
     --plot /tmp/pallas_scaling.png

   # Compare 2D vs 3D + Pallas ON vs OFF for a single collider (interactive HTML)
   python -m benchmarks scaling \
     --collider nl_cell_list \
     --dim 2d,3d --pallas on,off \
     --min-exp 1 --max-exp 21 \
     --plot /tmp/nl_2d_3d_pallas.html

   # Compare velocities or polydispersity across colliders
   python -m benchmarks scaling \
     --collider cell_list,nl_cell_list \
     --vel 0.15,0.6 --alpha 1.0,1.5 \
     --min-n 1024 --max-n 524288 \
     --plot /tmp/vel_alpha_scaling.png

   # Save raw scaling JSON and re-plot later without re-running
   python -m benchmarks scaling --collider cell_list,nl_cell_list --out /tmp/scaling.json
   python -m benchmarks scaling --from-json /tmp/scaling.json --plot /tmp/scaling.html
