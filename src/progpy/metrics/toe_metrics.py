# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This file includes functions for calculating metrics specific to Time of Event (ToE) from a single event or multiple events given the same time of prediction
"""
from typing import Iterable
from numpy import isscalar

from ..uncertain_data import UncertainData, UnweightedSamples

def prob_success(toe: UncertainData, time: float, **kwargs) -> float:
    """Calculate probability of success - i.e., probability that event will not occur within a given time (i.e., success)

    Args:
        toe (UncertainData or list[float]): Times of event for a single event (array[float]) or multiple events, output from predictor
        time (float): time for calculation
        **kwargs (optional): Configuration parameters. Supported parameters include:
          * n_samples (int): Number of samples to use for calculating metrics (if ToE is not UnweightedSamples). Defaults to 10,000.
          * keys (list of strings, optional): Keys to calculate metrics for. Defaults to all keys.

    Returns:
        float: Probability of success

    Example:
        ::

            # ToE estimate distribution is returned from a predictor's predict method
            result = predictor.predict(...)
            toe = result.time_of_event 
            
            from progpy.metrics import prob_success
            now = 10 # Current time
            p_success = prob_success(toe, now)
            p_success = prob_success(toe, now, n_samples = 100) # Can also specify number of samples
            p_success = prob_success(toe, now, keys = ['event1']) # Can specify specific keys to consider
    """
    params = {
        'n_samples': 10000,  # Default 
    }
    params.update(kwargs)

    if isinstance(toe, UncertainData):
        # Default to all keys
        keys = params.setdefault('keys', toe.keys())
        if isinstance(keys, str):
            keys = [keys]

        if isinstance(toe, UnweightedSamples):
            samples = toe
        else:
            # Some other distribution besides unweighted samples
            # Generate Samples
            samples = toe.sample(params['n_samples'])

        # If unweighted_samples, calculate metrics for each key
        return {key: prob_success(samples.key(key), 
                time, 
                **kwargs) for key in keys}
    elif isinstance(toe, Iterable):
        if len(toe) == 0:
            raise ValueError('Time of Event must not be empty')
        # Is list or array
        if isscalar(toe[0]) or toe[0] is None:
            # list of numbers - this is the case that we can calculate
            pass
        elif isinstance(toe[0], dict):
            # list of dicts - Supported for backwards compatabilities
            toe = UnweightedSamples(toe)
            return prob_success(toe, time, **kwargs)
        else:
            raise TypeError("ToE must be type Uncertain Data or array of dicts, was {}".format(type(toe)))
    else:
        raise TypeError("ToE must be type Uncertain Data or array of dicts, was {}".format(type(toe)))

    return sum([e is None or e > time for e in toe])/len(toe)
