# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Installation and Setup
--------------------------

Install JaxDEM from PyPI, GitHub, or a local checkout. Python 3.12+ and
JAX 0.8.1+ are required. The core install includes dynamics, analysis, and
energy minimization; optional extras add:

* ``[io]`` — VTK, HDF5, and Orbax checkpointing
* ``[rl]`` — reinforcement-learning environments, models, and trainers
* ``[all]`` — all runtime optional features (``io`` and ``rl``)
* ``[docs]`` — Sphinx documentation build tools
* ``[test]`` — test suite and type-checking dependencies
* ``[dev]`` — all runtime, documentation, and test dependencies combined

Select hardware support (CPU, CUDA, TPU) through JAX's own extras.
"""

# %%
# Create a Virtual Environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    python -m venv .venv
#    source .venv/bin/activate
#    python -m pip install --upgrade pip
#
# On Windows PowerShell, activate with ``.\.venv\Scripts\Activate.ps1``.

# %%
# Install from PyPI
# ~~~~~~~~~~~~~~~~~
# Core simulation tools (CPU backend):
#
# .. code-block:: bash
#
#    python -m pip install JaxDEM
#
# Add the ``[io]`` extra for VTK/HDF5 exports and simulation checkpoints:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[io]"
#
# Install all runtime extras (``io`` and ``rl``) together with JAX's CUDA 13
# backend (use ``jax[cuda12]`` for CUDA 12 or ``jax[tpu]`` on a TPU VM):
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[all]" "jax[cuda13]"
#
# See the `JAX installation guide
# <https://docs.jax.dev/en/latest/installation.html>`_ for platform and driver
# requirements.

# %%
# Install from GitHub
# ~~~~~~~~~~~~~~~~~~~
# To install the latest development commit directly from GitHub:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM @ git+https://github.com/cdelv/JaxDEM.git"
#    python -m pip install "JaxDEM[io] @ git+https://github.com/cdelv/JaxDEM.git"
#    python -m pip install "JaxDEM[all] @ git+https://github.com/cdelv/JaxDEM.git" "jax[cuda13]"

# %%
# Install a Local Checkout
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    git clone https://github.com/cdelv/JaxDEM.git
#    cd JaxDEM
#    python -m pip install -e ".[io]"
#
# Combine extras and a JAX backend as needed:
#
# .. code-block:: bash
#
#    python -m pip install -e ".[all]" "jax[cuda13]"
#    python -m pip install -e ".[dev]" "jax[cuda13]"
#
# Verify the installation and active JAX backend:
#
# .. code-block:: bash
#
#    python -c "import jax, jaxdem; print(jaxdem.__version__, jax.__version__, jax.devices())"

# %%
# GPU Acceleration with Pallas
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# On CUDA (and supported ROCm) GPU backends, JaxDEM automatically accelerates
# spatial colliders and built-in pair force laws using fused Pallas Triton
# kernels dispatched via ``jax.lax.platform_dependent``. All other backends
# (CPU, TPU, and Metal)—as well as custom user-defined force laws, custom
# domains, ``shard_map`` sharding, and automatic differentiation—automatically
# fall back to the Pure-JAX implementation with no code changes required.
#
# To disable Pallas kernels and force the Pure-JAX path on a GPU backend, set
# the ``JAXDEM_DISABLE_PALLAS=1`` environment variable before running Python,
# or toggle them at runtime from Python:
#
# .. code-block:: python
#
#    import jaxdem as jdem
#
#    jdem.set_pallas_enabled(False)  # force Pure JAX (re-enable with True or None)
#
#    with jdem.pallas_enabled(False):
#        ...  # temporarily run Pure JAX inside this block
#
# %%
# Next Steps
# ~~~~~~~~~~
#
# - :doc:`Introduction <../auto_examples/introduction>`
# - :doc:`State Guide <../auto_examples/state_guide>`
# - :doc:`System Guide <../auto_examples/system_guide>`
# - :doc:`Materials Guide <../auto_examples/materials_guide>`
