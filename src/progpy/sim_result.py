# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from collections import UserList, defaultdict, abc
from copy import deepcopy
from functools import cached_property
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from progpy.exceptions import warn_once
from progpy.utils.containers import DictLikeMatrixWrapper
from progpy.visualize import plot_timeseries
from typing import Dict, List


class SimResult(UserList):
    """
    `SimResult` is a data structure for the results of a simulation, with time. It is returned from the `simulate_to*` methods for :term:`inputs<input>`, :term:`outputs<output>`, :term:`states<state>`, and :term:`event_states<event state>` for the beginning and ending time step of the simulation, plus any save points indicated by the `savepts` and `save_freq` configuration arguments. The class includes methods for analyzing, manipulating, and visualizing the results of the simulation.

    Args:
            times (array[float]): Times for each data point where times[n] corresponds to data[n]
            data (array[Dict[str, float]]): Data points where data[n] corresponds to times[n]
    """

    __slots__ = ["times", "data"]  # Optimization

    def __init__(self, times: list = None, data: list = None, _copy=True):
        if times is None or data is None:
            self.times = []
            self.data = []
        else:
            self.times = times.copy()
            if _copy:
                self.data = deepcopy(data)
            else:
                self.data = data

    # def __getitem__(self, item):
    #     """
    #         created for deprecation warning. [] continues to be handled by parent
    #     """
    #     warn_once('[] for access by row number will be deprecated after version 1.5 of ProgPy. After v1.5, [] will access by column (e.g., data[\'state1\']), Users may use \'iloc\' to access by row number (e.g., data.iloc[10])'
    #         'data by element.', DeprecationWarning, stacklevel=2)
    #     return super().__getitem__(item)

    # def __iter__(self):
    #     """
    #         created for deprecation warning. iteration continues to be handled by parent
    #     """
    #     warn_once(
    #         'iteration will be deprecated after version 1.5 of ProgPy. The function will be renamed, iterrows, '
    #         'and users may begin using it under this name now.',
    #         DeprecationWarning, stacklevel=2)
    #     return super().__iter__()

    def iterrows(self):
        """
        .. versionadded:: 1.5.0

            Iterates -- through keys
        """
        return super().__iter__()

    @cached_property
    def frame(self) -> pd.DataFrame:
        """
        .. versionadded:: 1.5.0

            pd.DataFrame: A pandas DataFrame representing the SimResult data
        """
        # warn_once('frame will be deprecated after version 1.5 of ProgPy.', DeprecationWarning, stacklevel=2)
        if len(self.data) > 0:
            frame = pd.concat(
                [pd.DataFrame(dict(dframe), index=[0]) for dframe in self.data],
                ignore_index=True,
                axis=0,
            )
        else:
            frame = pd.DataFrame()
        if self.times is not None:
            frame.insert(0, "time", self.times)
            frame = frame.set_index("time")
        return frame

    def frame_is_empty(self) -> bool:
        """
        .. versionadded:: 1.5.0

        Returns:
            bool: If the value has been calculated
        """
        return self.frame.empty

    def __setitem__(self, key, value):
        """
        in addition to the normal functionality, updates the _frame if it exists
        """
        super().__setitem__(key, value)
        if "frame" in self.__dict__:  # Has been calculated
            for col in value:
                self.__dict__["frame"].at[key, col] = value[col]

    def __delitem__(self, key):
        """
        in addition to the normal functionality, updates the _frame if it exists
        """
        super().__delitem__(key)
        if "frame" in self.__dict__:
            self.__dict__["frame"].drop([key], inplace=True)

    def insert(self, i: int, item) -> None:
        """
        in addition to the normal functionality, updates the _frame if it exists
        """
        self.insert(i, item)
        if "frame" in self.__dict__:
            for value in item:
                self.__dict__["frame"].insert(i, column=[value], value=item[value])

    @property
    def iloc(self):
        """
        .. versionadded:: 1.5.0

            returns the iloc indexer
        """
        return self.frame.iloc

    def equals(self, other: "SimResult") -> bool:
        """
        .. versionadded:: 1.5.0

        Compare 2 SimResults

        Args:
            other (SimResult)

        Returns:
            bool: If the two SimResults are equal
        """
        return self.times == other.times and self.data == other.data

    def __eq__(self, other) -> bool:
        """
        Compare 2 SimResults

        Args:
            other (SimResult)

        Returns:
            bool: If the two SimResults are equal
        """
        # warn_once(
        #     ' \' == \' will be deprecated after version 1.5 of ProgPy. The function will be available as \' equals() \', '
        #     'and users may begin using it under this name now.',
        #     DeprecationWarning, stacklevel=2)

        return self.equals(other)

    def index_of_data(self, other: dict, *args, **kwargs) -> int:
        """
        Get the index of the first sample where other occurs

        Args:
            other (dict)

        Returns:
            int: Index of first sample where other occurs
        """
        return self.data.index(other, *args, **kwargs)

    def index(self, other: dict, *args, **kwargs) -> int:
        """
        Get the index of the first sample where other occurs

        Args:
            other (dict)

        Returns:
            int: Index of first sample where other occurs
        """
        warn_once(
            "index is deprecated. Use index_of_data instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return self.index_of_data(other, *args, **kwargs)

    def extend(self, other: "SimResult") -> None:
        """
        Extend the SimResult with another SimResult or LazySimResult object

        Args:
            other (SimResult/LazySimResult)

        """
        if isinstance(other, SimResult):
            self.times.extend(other.times)
            self.data.extend(other.data)
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__}")
        if "frame" in self.__dict__:
            del self.__dict__["frame"]

    def pop_by_index(self, index: int = -1) -> dict:
        """Remove and return an element

        Args:
            index (int, optional): Index of element to be removed. Defaults to -1.

        Returns:
            dict: Element Removed
        """
        self.times.pop(index)
        if "frame" in self.__dict__:
            self.__dict__["frame"].drop(
                [self.__dict__["frame"].index.values[index]], inplace=True
            )
        return self.data.pop(index)

    def pop(self, index: int = -1) -> dict:
        """Remove and return an element

        Args:
            index (int, optional): Index of element to be removed. Defaults to -1.

        Returns:
            dict: Element Removed
        """
        warn_once(
            "pop is deprecated. Use pop_by_index instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.pop_by_index(index)

    def remove(self, d: dict = None, t: float = None) -> None:
        """Remove an element

        Args:
            d: Data value to be removed.
            t: Time value to be removed.
        """
        if sum([i is None for i in (d, t)]) != 1:
            raise ValueError(
                "ValueError: Only one named argument (d, t) can be specified."
            )

        if t is not None:
            target_index = self.times.index(t)
        else:
            target_index = self.data.index(d)
        self.pop_by_index(target_index)

    def clear(self) -> None:
        """Clear the SimResult"""
        self.times = []
        self.data = []
        del self.__dict__["frame"]

    def time(self, index: int) -> float:
        """Get time for data point at index `index`

        Args:
            index (int)

        Returns:
            float: Time for which the data point at index `index` corresponds
        """
        return self.times[index]

    def to_numpy(self, keys=None) -> np.ndarray:
        """
        Convert from simresult to numpy array

        Args:
            keys: Subset of keys to return as part of numpy array (by default, all)

        Returns:
            np.ndarray: numpy array representing simresult
        """
        if len(self.data) == 0:
            return np.array([[]], dtype=np.float64)
        if len(self.data[0]) == 0:
            return np.array([[] for _ in self.data], dtype=np.float64)
        if isinstance(self.data[0], DictLikeMatrixWrapper) and keys is None:
            return np.array([u_i.matrix[:, 0] for u_i in self.data], dtype=np.float64)
        if keys is None:
            keys = self.data[0].keys()
        return np.array(
            [[u_i[key] for key in keys] for u_i in self.data], dtype=np.float64
        )

    def plot(self, **kwargs) -> figure:
        """
        Plot the simresult as a line plot

        Keyword Args:
            keys (list[str]): list of keys to plot. If not provided, all keys in the series are plotted.
            figsize (tuple[float, float]): width and height of the figure
            compact (bool): If true, all timeseries are displayed in one plot (multiple colored lines)
            xlabel (str) : label for the x-axis. Default is 'time'
            ylabel (str) : label for the y-axis. Default is 'state'
            title (str) : plot title. Default is no title
            title_fontsize (str or float): plot title fontsize. Default is 'x-large'
            suptitle (str) : plot suptitle. Default is no suptitle
            ticklabel_fontsize (str or float): tick label font sizes. Default is 'small'
            tight_layout (bool): whether to use tight layout (minimize figure blank space around the graph)
            display_labels (str): whether to display x and y-labels in the figure (['no', 'minimal', 'all'])

        Returns:
            Figure
        """
        # warn_once('Behavior of SimResult.plot() will change with version 1.6. New behavior will match that of a pandas data frame.')
        return plot_timeseries(
            self.times, self.data, legend={"display": True}, options=kwargs
        )

    def monotonicity(self) -> Dict[str, float]:
        """
        Calculate monotonicty for a single prediction.
        Given a single simulation result, for each event: go through all predicted states and compare those to the next one.
        Calculates monotonicity for each event key using its associated mean value in UncertainData.

        Where N is number of measurements and sign indicates sign of calculation.

        Coble, J., et. al. (2021). Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society.
        http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404
        Baptistia, M., et. al. (2022). Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Artificial Intelligence, Volume 306.
        https://www.sciencedirect.com/science/article/pii/S0004370222000078

        Args:
            None

        Returns:
            float: Value between [0, 1] indicating monotonicity of a given event for the Prediction.
        """
        # Collect and organize mean values for each event
        by_event = defaultdict(list)
        for uncertaindata in self.data:
            for key, value in uncertaindata.items():
                by_event[key].append(value)

        # For each event, calculate monotonicity using formula
        result = {}
        for key, value in by_event.items():
            mono_sum = 0
            for i in range(len(value) - 1):
                mono_sum += np.sign(value[i + 1] - value[i])
            result[key] = abs(mono_sum / (len(value) - 1))
        return result

    def __not_implemented(self):  # lgtm [py/inheritance/signature-mismatch]
        raise NotImplementedError("Not Implemented")

    # Functions of list not implemented
    # Specified here to stop users from accidentally trying to use them
    # (due to this classes similarity to list)
    append = __not_implemented
    count = __not_implemented
    insert = __not_implemented
    reverse = __not_implemented
    # lgtm [py/missing-equals]


class LazySimResult(SimResult):  # lgtm [py/missing-equals]
    """
    Used to store the result of a simulation, which is only calculated on first request
    """

    def __init__(
        self, fcn: abc.Callable, times: list = None, states: list = None, _copy=True
    ) -> None:
        """
        Args:
            fcn (abc.Callable): function (x) -> z where x is the state and z is the data
            times (array(float)): Times for each data point where times[n] corresponds to data[n]
            data (array(dict)): Data points where data[n] corresponds to times[n]
        """
        self.fcn = fcn
        if times is None or states is None:
            self.times = []
            self.states = []
        else:
            self.times = times.copy()
            if _copy:
                self.states = deepcopy(states)
            else:
                self.states = states

    def __reduce__(self):
        return (self.__class__.__base__, (self.times, self.data))

    def is_cached(self) -> bool:
        """
        Returns:
            bool: If the value has been calculated
        """
        return "data" in self.__dict__

    def clear(self) -> None:
        """
        Clears the times, states, and data cache for a LazySimResult object
        """
        self.times = []
        del self.__dict__["data"]
        self.states = []

    def extend(self, other: "LazySimResult", _copy=True) -> None:
        """
        Extend the LazySimResult with another LazySimResult object
        Raise ValueError if SimResult is passed
        Function fcn of other LazySimResult MUST match function fcn of LazySimResult object to be extended

        Args:
            other (LazySimResult)

        """
        if isinstance(other, self.__class__):
            self.times.extend(other.times)
            if _copy:
                self.states.extend(deepcopy(other.states))
            else:
                self.states.extend(other.states)
            if "data" in self.__dict__:  # self is cached
                if not other.is_cached():
                    # Either are not cached
                    if "data" in self.__dict__:
                        del self.__dict__["data"]
                else:
                    self.__dict__["data"].extend(other.data)
        elif isinstance(other, SimResult):
            raise ValueError(
                f"ValueError: {self.__class__} cannot be extended by SimResult. First convert to SimResult using to_simresult() method."
            )
        else:
            raise ValueError(f"ValueError: Argument must be of type {self.__class__}.")

    def pop(self, index: int = -1) -> dict:
        """Remove an element. If data hasn't been cached, remove the state - so it wont be calculated

        Args:
            index (int, optional): Index of element to be removed. Defaults to -1.

        Returns:
            dict: Element Removed
        """
        self.times.pop(index)
        x = self.states.pop(index)
        if "data" in self.__dict__:  # is cached
            return self.__dict__["data"].pop(index)
        return self.fcn(x)

    def remove(self, d: float = None, t: float = None, s=None) -> None:
        """
        Remove an element

        Args:
            d: Data value to be removed
            t: Time value to be removed
            s: State value to be removed
        """
        if sum([i is None for i in (d, t, s)]) != 2:
            raise ValueError(
                "ValueError: Only one named argument (d, t, s) can be specified."
            )

        if t is not None:
            target_index = self.times.index(t)
        elif s is not None:
            target_index = self.states.index(s)
        else:
            target_index = self.data.index(d)
        self.pop(target_index)

    def to_simresult(self) -> SimResult:
        return SimResult(self.times, self.data)

    @cached_property
    def data(self) -> List[dict]:
        """
        Get the data (elements of list). Only calculated on first request

        Returns:
            array(dict): data
        """
        return [self.fcn(x) for x in self.states]
