# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This file contains functions for calculating error given a model and some data (times, inputs, outputs). This is used by the PrognosticsModel.calc_error() method.
"""
import math
from typing import List
from warnings import warn
import numpy as np


def MAX_E(m, times: List[float], inputs: List[dict], outputs: List[dict], **kwargs) -> float:
    """
    .. versionadded:: 1.5.0

    Calculate the Maximum Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, optional): Initial state.
        dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to use time between samples.
        stability_tol (float, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of error.

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the error calculated from data up to the instability is returned.

    Returns:
        float: Maximum error between model and data
    """
    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = m.StateContainer(x)

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]

    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0
    t_last = times[0]
    err_max = 0
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = stability_tol * times[-1]

    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            # The none check above is used to cover the case where the model
            # is not able to produce an output for a given input yet
            # For example, in LSTM models, the first few inputs will not 
            # produce an output until the model has received enough data
            # This is true for any window-based model
            if any(np.isnan(z_obs.matrix)):
                if t <= cutoffThreshold:
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. "
                                     f"Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")
                else:
                    warn(f"Model unstable- NaN reached in simulation (t={t})")
                    break
            err_max = max(err_max, np.max(
                np.abs(z.matrix - z_obs.matrix)
            ))
            counter += 1

    if counter == 0:
        return np.nan

    return err_max


def RMSE(m, times: List[float], inputs: List[dict], outputs: List[dict], **kwargs) -> float:
    """
    .. versionadded:: 1.5.0

    Calculate the Root Mean Squared Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, optional): Initial state.
        dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to use time between samples.
        stability_tol (float, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of error.

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the error calculated from data up to the instability is returned.

    Returns:
        float: RMSE between model and data
    """
    return np.sqrt(MSE(m, times, inputs, outputs, **kwargs))


def MSE(m, times: List[float], inputs: List[dict], outputs: List[dict], **kwargs) -> float:
    """
    .. versionadded:: 1.5.0

    Calculate Mean Squared Error (MSE) between simulated and observed

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, optional): Initial state.
        dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to use time between samples.
        stability_tol (float, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of error.

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the error calculated from data up to the instability is returned.

    Returns:
        float: Total error
    """
    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = m.StateContainer(x)

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]

    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)
    cutoffThreshold = stability_tol * times[-1]

    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            # The none check above is used to cover the case where the model
            # is not able to produce an output for a given input yet
            # For example, in LSTM models, the first few inputs will not 
            # produce an output until the model has received enough data
            # This is true for any window-based model
            if any(np.isnan(z_obs.matrix)):
                if t <= cutoffThreshold:
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. "
                                     f"Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")     
                else:
                    warn("Model unstable- NaN reached in simulation (t={})".format(t))
                    break
            err_total += np.sum(np.square(z.matrix - z_obs.matrix), where= ~np.isnan(z.matrix))
            counter += 1
    return err_total/counter


def MAE(m, times: List[float], inputs: List[dict], outputs: List[dict], **kwargs) -> float:
    """
    .. versionadded:: 1.5.0

    Calculate the Mean Absolute Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, optional): Initial state.
        dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to use time between samples.
        stability_tol (float, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of error.

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the error calculated from data up to the instability is returned.

    Returns:
        float: MAE between model and data
    """
    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = m.StateContainer(x)

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]

    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = stability_tol * times[-1]

    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            # The none check above is used to cover the case where the model
            # is not able to produce an output for a given input yet
            # For example, in LSTM models, the first few inputs will not 
            # produce an output until the model has received enough data
            # This is true for any window-based model
            if any(np.isnan(z_obs.matrix)):
                if t <= cutoffThreshold:
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. "
                                     f"Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")
                else:
                    warn(f"Model unstable- NaN reached in simulation (t={t})")
                    break
            err_total += np.sum(
                np.abs(z.matrix - z_obs.matrix))
            counter += 1
    return err_total/counter

def MAPE(m, times: List[float], inputs: List[dict], outputs: List[dict], **kwargs) -> float:
    """
    .. versionadded:: 1.5.0

    Calculate the Mean Absolute Percentage Error between model behavior and some collected data.

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, optional): Initial state.
        dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to use time between samples.
        stability_tol (float, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of error.

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the error calculated from data up to the instability is returned.

    Returns:
        float: MAPE between model and data
    """
    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = m.StateContainer(x)

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]

    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    err_total = 0
    z_obs = m.output(x)
    cutoffThreshold = stability_tol * times[-1]

    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                # Only recalculate if required
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            # The none check above is used to cover the case where the model
            # is not able to produce an output for a given input yet
            # For example, in LSTM models, the first few inputs will not
            # produce an output until the model has received enough data
            # This is true for any window-based model
            if any(np.isnan(z_obs.matrix)):
                if t <= cutoffThreshold:
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. "
                                     f"Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")
                else:
                    warn(f"Model unstable- NaN reached in simulation (t={5})")
                    break
            err_total += np.sum(np.abs(z.matrix - z_obs.matrix) / z.matrix)
            counter += 1
    return err_total/counter


def DTW(m, times, inputs, outputs, **kwargs):
    """
    .. versionadded:: 1.5.0
    
    Dynamic Time Warping Algorithm using FastDTW's package.
    How DTW Works: https://cs.fit.edu/~pkc/papers/tdm04.pdf
    FastDTW Documentation: https://pypi.org/project/fastdtw/

    Args:
        m (PrognosticsModel): Model to use for comparison
        times (list[float], list[list[float]]): Array of times for each sample.
        inputs (list[dict, SimResult]): Array of input dictionaries where input[x] corresponds to time[x].
        outputs (list[dict, SimResult]): Array of output dictionaries where output[x] corresponds to time[x].

    Keyword Args:
        x0 (StateContainer, optional): Initial state.
        dt (float, optional): Maximum time step in simulation. Time step used in simulation is lower of dt and time between samples. Defaults to use time between samples.
        stability_tol (float, optional): Configurable parameter.
            Configurable cutoff value, between 0 and 1, that determines the fraction of the data points for which the model must be stable.
            In some cases, a prognostics model will become unstable under certain conditions, after which point the model can no longer represent behavior. 
            stability_tol represents the fraction of the provided argument `times` that are required to be met in simulation, 
            before the model goes unstable in order to produce a valid estimate of error.

            If the model goes unstable before stability_tol is met, NaN is returned. 
            Else, model goes unstable after stability_tol is met, the error calculated from data up to the instability is returned.

    Returns:
        float: DTW distance between model and data
    """
    x = kwargs.get('x0', m.initialize(inputs[0], outputs[0]))
    dt = kwargs.get('dt', 1e99)
    stability_tol = kwargs.get('stability_tol', 0.95)

    if not isinstance(x, m.StateContainer):
        x = [m.StateContainer(x_i) for x_i in x]

    if not isinstance(inputs[0], m.InputContainer):
        inputs = [m.InputContainer(u_i) for u_i in inputs]

    if not isinstance(outputs[0], m.OutputContainer):
        outputs = [m.OutputContainer(z_i) for z_i in outputs]

    counter = 0  # Needed to account for skipped (i.e., none) values
    t_last = times[0]
    z_obs = m.output(x)  # Initialize
    cutoffThreshold = stability_tol * times[-1]
    simulated = []

    for t, u, z in zip(times, inputs, outputs):
        while t_last < t:
            t_new = min(t_last + dt, t)
            x = m.next_state(x, u, t_new-t_last)
            t_last = t_new
            if t >= t_last:
                z_obs = m.output(x)
        if not (None in z_obs.matrix or None in z.matrix):
            if any(np.isnan(z_obs.matrix)):
                if t <= cutoffThreshold:
                    raise ValueError(f"Model unstable- NAN reached in simulation (t={t}) before cutoff threshold. "
                                     f"Cutoff threshold is {cutoffThreshold}, or roughly {stability_tol * 100}% of the data")     
                else:
                    warn(f"Model unstable- NaN reached in simulation (t={t})")
                    # When model goes unstable after cutoffThreshold, we want to match the last stable observed value with the 
                        # equivalent user-provided output by truncating our user-provided series to match the length of our observed series.
                    percent = counter / len(outputs)
                    index = math.floor(percent * len(outputs))
                    outputs = list(outputs)[:index]
                    break
            simulated.append(z_obs)
            counter += 1

    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    
    def dtw_helper(data):
        """
        Helper function to take in given data (simulated and observed), and then transform it such that the function fastdtw can use it.

        The data would start by looking like:
            [{'x': 2.83}, {'x': 22.83}, {'x': 40.3775}, {'x': 55.4725}, {'x': 68.115}, ...]
        Then would be transformed to something like:
            [[2.83, 0.0], [22.83, 0.5], [40.3775, 1.0], [55.4725, 1.5], [68.115, 2.0], ...]
        
        Note that the helper function also adds 'time' to each of the indices to account for any data that may not necessarily be 
        recorded at a consistent rate.
        """
        transform = []
        for index in data:
            inner_list = []
            for key in index.keys():
                inner_list.append(index.get(key))
            transform.append(inner_list)
        for i, t in enumerate(times):
            transform[i].append(t)
        return transform
    simulated, observed = dtw_helper(simulated), dtw_helper(outputs)
    distance, _ = fastdtw(simulated, observed, dist=euclidean)

    return distance
