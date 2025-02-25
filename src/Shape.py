# This file is part of the JaxDEM library. For more information and source code
# availability visit https://github.com/cdelv/JaxDEM
#
# JaxDEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions
from abc import ABC, abstractmethod

class Shape(ABC):
    """
    Abstract class for representing particle shapes.
    """
    @abstractmethod
    def __init__(self):
        pass