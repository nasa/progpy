# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections import abc


def all_none_iterable(iter, name, loc=None):
    """
    Helper Function to determine if passed in iterable has consistent typings for each index. Used to avoid 
    erroneous arguments and for a more descriptive error message than Python's Interpreter.

    Args:
        iter (Iterable): Iterable
        name (str): The iterable's name
        loc (boolean, Optional): The data point location of the iterable we are validating
    """
    count = sum(isinstance(element, abc.Sequence) for element in iter)  # Number of iterables in each iterable
    if 0 < count < len(iter):
        if loc:
            raise ValueError(f"Some, but not all elements, are iterables for argument {name} at data location {loc}.")
        else:
            raise ValueError(f'Some, but not all elements, are iterables for argument {name}.')
