# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
r"""Installation and Setup
--------------------------

Install JaxDEM from GitHub or a local checkout. Python 3.12+ and JAX 0.8.1+
are required. The core install includes dynamics and energy minimization;
optional extras add:

* ``[io]`` — VTK, HDF5, and Orbax checkpointing
* ``[rl]`` — reinforcement-learning environments, models, and trainers
* ``[docs]`` — Sphinx documentation build tools
* ``[test]`` — test suite dependencies
* ``[all]`` — all optional features combined

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
# Install from GitHub
# ~~~~~~~~~~~~~~~~~~~
# Core simulation tools (CPU backend):
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM @ git+https://github.com/cdelv/JaxDEM.git"
#
# Add the ``[io]`` extra for VTK/HDF5 exports and simulation checkpoints:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[io] @ git+https://github.com/cdelv/JaxDEM.git"
#
# Install all optional features together with JAX's CUDA 13 backend (use
# ``jax[cuda12]`` for CUDA 12 or ``jax[tpu]`` on a TPU VM):
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[all] @ git+https://github.com/cdelv/JaxDEM.git" "jax[cuda13]"
#
# See the `JAX installation guide
# <https://docs.jax.dev/en/latest/installation.html>`_ for platform and driver
# requirements.

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
#    python -m pip install -e ".[io,rl]" "jax[cuda13]"
#    python -m pip install -e ".[all]" "jax[cuda13]"
#
# Verify the installation and active JAX backend:
#
# .. code-block:: bash
#
#    python -c "import jax, jaxdem; print(jax.__version__, jax.devices())"

# %%
# Next Steps
# ~~~~~~~~~~
#
# - :doc:`Introduction <../auto_examples/introduction>`
# - :doc:`State Guide <../auto_examples/state_guide>`
# - :doc:`System Guide <../auto_examples/system_guide>`
# - :doc:`Materials Guide <../auto_examples/materials_guide>`
