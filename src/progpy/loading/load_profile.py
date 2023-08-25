# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import math
from matplotlib import pyplot as plt


class LoadProfile():
    """
    .. versionadded:: 1.6.0

    Superclass for load profiles.
    """

    def plot(self, times, fig: plt.Figure = None):
        """
        Plot a load profile without simulating

        Args:
            times (list[float]): The times for which to plot inputs
        
        Keyword Args:
            fig (plt.Figure, optional): Figure to overwrite (only when compact). Defaults to None.
        """
        values = [self(t) for t in times]
        keys = values[0].keys()
        values = {
            key: [value[key] for value in values] 
            for key in keys}
        if fig is None:
            fig = plt.figure()
        plt.figure(fig)
        for key in keys:
            plt.plot(times, values[key], label=key)
        plt.xlabel('Time (s)')
        plt.legend()
        return fig
