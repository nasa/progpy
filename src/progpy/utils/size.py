# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from itertools import chain
import numpy as np

from progpy.sim_result import SimResult
from progpy.utils.containers import DictLikeMatrixWrapper


def dict_handler(d):
    # Handle dictionaries
    return chain.from_iterable(d.items())


def object_handler(o):
    # Handle general objects
    return chain.from_iterable(o.__dict__.items())


# Set of handlers that describe how to estimate the size of the payload of an
# object of a given type in format {type: handler}
all_handlers = {tuple: iter,
                list: iter,
                np.ndarray: lambda a: iter(list(a.flat)),
                dict: dict_handler,
                set: iter,
                frozenset: iter,
                DictLikeMatrixWrapper: dict_handler,
                SimResult: iter
                }


def getsizeof(o):
    """
    Return the size of object in bytes.

    Args:
        o (Any): Object to find size of

    Returns:
        int: Size in bytes of object
    """
    seen = set()  # track which object id's have already been seen

    # Add Custom handler for progpy objects that import this file
    # This is to avoid circular imports
    from progpy import PrognosticsModel
    from progpy.utils.parameters import PrognosticsModelParameters
    all_handlers[PrognosticsModelParameters] = object_handler
    all_handlers[PrognosticsModel] = object_handler

    def sizeof(o):
        """
        Internal (recursive) function to find size of object.

        This is defined in here so it can use the higher-scoped variable `seen`

        Args:
            o (Any): Object to find size of

        Returns:
            int: Size in bytes of object
        """
        if id(o) in seen:
            # Object has already been counted, dont count twice
            # This is to avoid circular references
            return 0
        seen.add(id(o))

        # Get size of object itself
        s = object.__sizeof__(o)

        # Try handlers
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                return s

        # If no handler found, return size of object itself (no recursion)
        # This is mostly for simple types (e.g., int) that don't have a payload
        return s
    return sizeof(o)
