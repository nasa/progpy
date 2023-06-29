# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from numpy import array
from typing import List

def plot_hist(samples : array, fig : Figure = None, keys : List[str] = None, **kwargs) -> Figure:
    """Create a histogram

    Args:
        samples (Array(Dict)): A set of samples (e.g., states or eol estimates)
        fig (MatPlotLib Figure, optional): Existing histogram figure to be overritten. Defaults to create new figure.
        keys (List[String], optional): Keys to be plotted. Defaults to All.
    """

    # Input checks
    if len(samples) <= 0:
        raise Exception('Must include atleast one sample to plot')
    
    if keys is not None:
        if isinstance(keys, str):
            keys = [keys]
        try:
            iter(keys)
        except TypeError:
            raise TypeError("Keys should be a list of strings (e.g., ['state1', 'state2'], was {}".format(type(keys)))
        
        for key in keys:
            if key not in samples[0].keys():
                raise TypeError("Key {} was not present in samples (keys: {})".format(key, list(samples[0].keys())))
    else:
        keys = samples[0].keys()
    keys = list(keys)
    
    # Handle input
    parameters = {  # defaults
        'legend': True
    }
    parameters.update(kwargs)        

    if fig is None:
        # If no figure provided, create one
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
    
    # Plot
    for key in keys:
        ax.hist([sample[key] for sample in samples if sample[key] is not None], label=key, **kwargs)

    # Set legend
    if parameters['legend']:
        ax.legend().remove()  # Remove any existing legend - prevents "ghost effect"
        ax.legend(loc='upper right')
    
    return fig
