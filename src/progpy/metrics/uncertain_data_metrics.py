# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This file includes functions for calculating general metrics (i.e. mean, std, percentiles, etc.) on any distribution of type UncertainData (e.g. states, event_states, an EOL distribution, etc.)
"""
from typing import Iterable, Union
from numpy import isscalar, mean, std, array
from scipy import stats
from warnings import warn

from ..uncertain_data import UncertainData, UnweightedSamples

def calc_metrics(data: UncertainData, ground_truth: Union[float, dict] = None, **kwargs) -> dict:
    """Calculate all time of event metrics

    Args:
        data (array[float] or UncertainData): data from a single event
        ground_truth (float, optional): Ground truth value. Defaults to None. dict when data is  of type UncertainData.
        **kwargs (optional): Configuration parameters. Supported parameters include:
          * n_samples (int): Number of samples to use for calculating metrics (if data is not UnweightedSamples). Defaults to 10,000.
          * keys (list of strings, optional): Keys to calculate metrics for. Defaults to all keys.

    Returns:
        dict: collection of metrics
    """
    params = {
        'n_samples': 10000,  # Default is enough to get every percentile
    }
    params.update(kwargs)
    
    if isinstance(data, UncertainData):
        # Default to all keys
        keys = params.setdefault('keys', data.keys())
        if isinstance(keys, str):
            keys = [keys]
        
        if ground_truth and isscalar(ground_truth):
            # If ground truth is scalar, create dict (expected below)
            ground_truth = {key: ground_truth for key in keys}

        if isinstance(data, UnweightedSamples):
            samples = data
        else:
            # Some other distribution besides unweighted samples
            # Generate Samples
            samples = data.sample(params['n_samples'])

        if len(samples) == 0:
            raise ValueError('Data must not be empty')

        # If unweighted_samples, calculate metrics for each key
        result = {key: calc_metrics(samples.key(key), 
                ground_truth if not ground_truth else ground_truth[key],  # If ground_truth is a dict, use key
                **kwargs) for key in keys}

        # Set values specific to distribution
        for key in keys:
            result[key]['mean'] = data.mean[key]
            result[key]['median'] = data.median[key]
            result[key]['percentiles']['50'] = data.median[key]

        return result
    elif isinstance(data, Iterable):
        if len(data) == 0:
            raise ValueError('Data must not be empty')
        # Is list or array
        if isscalar(data[0]) or data[0] is None:
            # list of numbers - this is the case that we can calculate
            pass
        elif isinstance(data[0], dict):
            # list of dicts - Supported for backwards compatabilities
            data = UnweightedSamples(data)
            return calc_metrics(data, ground_truth, **kwargs)
        else:
            raise TypeError("Data must be type Uncertain Data or array of dicts, was {}".format(type(data)))
    else:
        raise TypeError("Data must be type Uncertain Data or array of dicts, was {}".format(type(data)))

    # If we get here then Data is a list of numbers- calculate metrics for numbers
    data_abridged = array([d for d in data if d is not None]) # Must be array
    if len(data_abridged) == 0:
        raise ValueError('All samples were none')
    if len(data_abridged) < len(data):
        warn("Some samples were None, resulting metrics only consider non-None samples. Note: in some cases, this will bias the metrics.")
    data_abridged.sort()
    m = mean(data_abridged)
    median = data_abridged[int(len(data_abridged)/2)]
    metrics = {
        'min': data_abridged[0],
        'percentiles': {
            '0.01': data_abridged[int(len(data_abridged)/10000)] if len(data_abridged) >= 10000 else None,
            '0.1': data_abridged[int(len(data_abridged)/1000)] if len(data_abridged) >= 1000 else None,
            '1': data_abridged[int(len(data_abridged)/100)] if len(data_abridged) >= 100 else None,
            '10': data_abridged[int(len(data_abridged)/10)] if len(data_abridged) >= 10 else None,
            '25': data_abridged[int(len(data_abridged)/4)] if len(data_abridged) >= 4 else None,
            '50': median,
            '75': data_abridged[int(3*len(data_abridged)/4)] if len(data_abridged) >= 4 else None,
        },
        'median': median,
        'mean': m,
        'std': std(data_abridged),
        'max': data_abridged[-1],
        'median absolute deviation': sum([abs(x - median) for x in data_abridged])/len(data_abridged),
        'mean absolute deviation':   sum([abs(x - m)   for x in data_abridged])/len(data_abridged),
        'number of samples': len(data_abridged)
    }

    if ground_truth is not None:
        # Metrics comparing to ground truth
        metrics['mean absolute error'] = sum([abs(x - ground_truth) for x in data_abridged])/len(data_abridged)
        metrics['mean absolute percentage error'] = metrics['mean absolute error']/ ground_truth
        metrics['relative accuracy'] = 1 - abs(ground_truth - metrics['mean'])/ground_truth
        metrics['ground truth percentile'] = stats.percentileofscore(data_abridged, ground_truth)

    return metrics
