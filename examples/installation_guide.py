r"""Installation and Setup
----------------------

Install JaxDEM from GitHub or a local checkout. Python 3.12 or newer and
JAX 0.8.1 or newer are required. The core install includes dynamics and
minimization. Feature extras are ``[io]`` for VTK/HDF5/checkpoints, ``[rl]``
for reinforcement learning, ``[docs]`` for documentation, and ``[test]`` for
testing. ``[all]`` installs all of these features together.

Select hardware support through JAX itself. JaxDEM and its JAX-based
dependencies use the same installed JAX backend.
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
# Core simulation tools:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM @ git+https://github.com/cdelv/JaxDEM.git"
#
# All optional features, with JAX's CUDA 13 backend:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[all] @ git+https://github.com/cdelv/JaxDEM.git" "jax[cuda13]"
#
# The two quoted requirements are separate arguments to pip; there is no ``+``
# between them. JaxDEM extras select features, while JAX extras select its backend.
# Pip resolves one compatible JAX installation for JaxDEM, Optax, Flax, and other
# JAX-based dependencies. Their version requirements may raise the effective
# minimum above JaxDEM's core minimum of 0.8.1.
#
# To pin a particular JAX version, add an exact ``==`` constraint:
#
# .. code-block:: bash
#
#    python -m pip install "JaxDEM[rl] @ git+https://github.com/cdelv/JaxDEM.git" "jax[cuda13]==0.11.0"
#
# Choose a version compatible with your Python version and selected features.
# For CUDA 12, use ``jax[cuda12]``; on a TPU VM use
# ``jax[tpu]``. See the `JAX installation guide
# <https://docs.jax.dev/en/latest/installation.html>`_ for platform and driver
# requirements. TensorFlow GPU support is not required for JaxDEM dynamics or
# JAX-based training; metrics logging is independent of the JAX backend.

# %%
# Install a Local Checkout
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    git clone https://github.com/cdelv/JaxDEM.git
#    cd JaxDEM
#    python -m pip install .
#
# Pick individual features or install everything, together with the desired JAX:
#
# .. code-block:: bash
#
#    python -m pip install ".[io,rl]" "jax[cuda13]"
#    python -m pip install ".[all]" "jax[cuda13]"
#
# Once the package is published on PyPI, the equivalent command is
# ``python -m pip install 'jaxdem[all]' 'jax[cuda13]'``.
# Verify the selected backend with:
#
# .. code-block:: bash
#
#    python -c "import jax; print(jax.__version__, jax.devices())"

# %%
# Next Steps
# ~~~~~~~~~~
#
# - :doc:`Introduction <../auto_examples/introduction>`
# - :doc:`System Guide <../auto_examples/system_guide>`
# - :doc:`Materials Guide <../auto_examples/materials_guide>`
