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

When writing documentation of a function, everyone should follow the same structure as the method documentation above.
