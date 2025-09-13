# Documentation


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

## Examples

## To Do list

* Write a comprehensive list of examples and tests

* Improve documentation

* Fix the VTKWriter.

* Improve tensorboard support and logging for RL.

* Add checkpoints for Rl models. Utilities for saving and loading models. 

* Add VTK rendering for environments and simulations.

* Unify batch and trajectory axis conventions. RL and System.trajectory_rollout differ. We need benchmarks to see which is faster: scan (vmap ()) or vmap (scan ()). Clarify the docs about this.

* Improve RL performance, especially for the LSTM.

* Documentation and tests of forceRouter.py, utils.py

* Improve utils

* Benchmarks

* Unit tests

* How to save interactions to VTK?

* Add more integration methods.

* Created compound force model. Create standard force models—Hertz-Mindilin, Kundall, etc.

* Force reset and gravity.

* Add support for different shapes.

* Add other contact detection strategies (partially ready).

* Separate submodules into directories. Move RL envs to their own file.

* Create the dynamic flag in the state. Use to constrain specific DoFs. 

* Add angular velocity and acceleration arrays to the state. 

* Add facets support.

* Implement deformable particles.

* Implement a way to create only the required elements on the state and system objects and add model-specific data arrays.