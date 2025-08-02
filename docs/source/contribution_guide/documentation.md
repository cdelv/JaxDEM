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

* Write comprenhensive list of examples and tests

* Change the writer docs and operation. Now, all states with ndim > 4 will have its firs dimensions flatened so that ndim == 4. Then, the dim 0 will be treated as a trajectory. Dim 1 can be either a trajectory or a batch. If its a trajectory and ndim == 4, flatten it again? Explore what its more comvenient for batching.

* Documentation of forceRouter.py, utils.py, and writer.py improvements

* Improve utils

* Benchmarks

* Unit tests

* How to save interactions to VTK?

* Add more integration methods

* Created compund force model

* Force resseter and gravity

* Add support for different shapes.

* Add other contact detection strategies

* Separate sub modules into directories

* Create the dynamic flag in state. Use to constrain specific DoFs. 