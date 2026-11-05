"""
Read data from a permanent format into a python variable.
"""
from .json_ import *
from .text import *
from .texsave import *
from .wod import *

__all__ = (json_.__all__ +
           text.__all__ +
           texsave.__all__ +
           wod.__all__)
