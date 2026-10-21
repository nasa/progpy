# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import matplotlib.pyplot as plt

def show_heatmap(data):
    """
    Generate a heatmap showing correlation between parameters.

    Code from: https://github.com/keras-team/keras-io/blob/13d513d7375656a14698ba4827ebbb4177efcf43/examples/timeseries/timeseries_weather_forecasting.py#L152

    Args:
        data (np.ndarray): Array of data where each column is a variable.
    """
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()