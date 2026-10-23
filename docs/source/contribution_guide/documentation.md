# Documentation

## Building visual examples

Install the documentation dependencies with `python -m pip install '.[docs]'`.
The visual build also needs ParaView 6.2 and FFmpeg. Set `JAXDEM_PVPYTHON` to
ParaView's `pvpython` executable, then run:

```bash
JAXDEM_DOCS_EXECUTE=1 JAXDEM_DOCS_VISUALS=1 make -C docs html SPHINXOPTS="-W --keep-going"
```

On Linux x86-64, `python tools/install_paraview.py --directory /tmp/jaxdem-paraview`
downloads the pinned official distribution and verifies its checksums. Use the
printed executable path for `JAXDEM_PVPYTHON`. The renderer uses EGL by default;
set `JAXDEM_PARAVIEW_BACKEND=OSMesa` for software offscreen rendering when needed.
An explicit `JAXDEM_DOCS_VISUALS=0` builds text and Matplotlib plots without
ParaView; the published documentation always uses the full visual build.

Every full build executes all examples, including their numerical assertions.
After the final code block, a Sphinx-Gallery scraper exports the current run
through `VTKWriter` and invokes ParaView. It regenerates posters, short movies,
and selected interactive snapshots. Matplotlib handles quantitative diagnostics;
particle layouts, meshes, contacts and displacement arrows belong in ParaView.
No simulation media is checked in or reused from an earlier build. Only the
ParaView installation is cached in CI. The full build clears generated media
first and rejects source changes made while the examples are running, so its
recorded hashes describe the code that produced the output.

Movies select actual saved frames rather than interpolating new configurations.
Quasistatic sequences are labeled by strain instead of physical time. Keep
teaching defaults small, select at most 48 frames for a movie, and use an
interactive snapshot when rotating the scene reveals useful structure. The
[ParaView Glance viewer](https://kitware.github.io/glance/doc/loading_files.html)
uses one shared bundled browser application and a small, freshly exported
``.vtkjs`` scene per example. It loads on demand without a running server.
The bundled ParaView 6.2 viewer references two absent optional modules for ITK
image loading and service workers; the build removes only those dead script
tags and preserves the core application and its license notices.

CI preserves generated media and `render.json` summaries as build artifacts.
The summaries include example and library source hashes, ParaView version,
native timesteps and geometry counts. Compare those records, plots and movies
when a mechanics change alters the result. Numerical assertions provide the
pass/fail checks; images make changes easier to inspect, without assuming
identical pixels across graphics drivers.


## Modules and Submodules

At the beginning of each module, write the copyright notice followed by a brief description of the things defined in the module:

```python
# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
The factory defines and instantiates specific simulation components.
"""
```

## Classes

When writing documentation of a class, everyone should follow the following structure:

```python
class Name:
    """
    Description of the class

    Notes
    -----
    Any special/not obvious behavior the user should be aware of

    Example
    -------
    Instantiation and/or usage example. To write code snippets, use >>> at the beginning of each line:

    >>> @Foo.register("bar") 
    >>> class bar:
    >>>     ...
    """

    class_atribute
    """
    Description
    """

    def method(param1: Type) -> ReturnType:
        """
        Method description.


        Parameters
        ----------
        param1 : Type or None (if applicable), (optional if the parameter is optional or nothing)
            Description
        
        Returns
        -------
        ReturnType
            Description

        Raises
        ------
        ErrorCode
            When does it error

        Example
        -------
        Usage example/s
        """
        ...
```

## Functions

When writing documentation of a function, everyone should follow the same structure as the method documentation above.
