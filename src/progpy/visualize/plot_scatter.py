# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from math import sqrt
from typing import List

def plot_scatter(samples : List[dict], fig : Figure = None, keys : List[str] = None, legend : str = 'auto', **kwargs) -> Figure:
    """
    Produce a scatter plot for a given list of states

    Args:
        samples ([dict]): Non-empty list of states where each element is a dictionary containing a single sample
        fig (Figure, optional): Existing figure previously used to plot states. If passed a figure argument additional data will be added to the plot. Defaults to creating new figure
        keys (list of strings, optional): Keys to plot. Defaults to all keys.
        legend (optional): When the legend should be shown, options:
            False: Dont show legend
            "auto": Show legend automatically if more than one data set
            True: Always show legend
        **kwargs (optional): Additional keyword arguments passed to scatter function. Includes those supported by scatter

    Returns:
        Figure
    
    Example:
            states = UnweightedSamples([1, 2, 3, 4, 5])
            plot_scatter(states.sample(100)) # With 100 samples
            plot_scatter(states.sample(100), keys=['state1', 'state2']) # only plot those keys
    """
    # Input checks
    if len(samples) <= 0:
        raise Exception('Must include atleast one sample to plot')
    
    if keys is not None:
        try:
            iter(keys)
        except TypeError:
            raise TypeError("Keys should be a list of strings (e.g., ['state1', 'state2'], was {}".format(type(keys)))

    # Handle input
    parameters = {  # defaults
        'alpha': 0.5
    }
    parameters.update(kwargs)

    if keys is None:
        keys = samples[0].keys()
    keys = list(keys)

    n = len(keys)
    if n < 2:
        raise Exception("At least 2 states required for scatter, got {}".format(n))

    if fig is None:
        # If no figure provided, create one
        fig = plt.figure()
        axes = [[fig.add_subplot(n-1, n-1, 1 + i + j*(n-1)) for i in range(n-1)] for j in range(n-1)]
    else:
        # Check size of axes
        if len(fig.axes) != (n-1)*(n-1):
            raise Exception("Cannot use existing figure - Existing figure graphs {} states, data includes {} states".format(sqrt(len(fig.axes))+1, n))

        # Unpack axes
        axes = [[fig.axes[i + j*(n-1)] for i in range(n-1)] for j in range(n-1)]

    for i in range(n-1):
        # For each column
        x_key = keys[i] 

        # Set labels on extremes
        axes[-1][i].set_xlabel(x_key)  # Bottom row
        axes[i][0].set_ylabel(keys[i+1])  # Left column

        # plot 
        for j in range(i, n-1): 
            # for each row
            y_key = keys[j+1]
            x1 = [x[x_key] for x in samples if x is not None]
            x2 = [x[y_key] for x in samples if x is not None]
            axes[j][i].scatter(x1, x2, **parameters)

        # Hide axes not used in plots 
        for j in range(0, i):
            axes[j][i].set_visible(False)

    # Set legend
    if legend == 'auto' or legend:
        labels = [thing.get_label() for thing in axes[0][0].get_children()
            if isinstance(thing, PathCollection)]
        if legend == 'auto' and len(labels) > 0 or legend:
            fig.legend().remove()  # Remove any existing legend - prevents "ghost effect"
            fig.legend(labels=labels, loc='upper right')

    return fig
