# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from matplotlib.figure import Figure
from numpy import array

from ..utils.table import print_table_recursive
from ..visualize import plot_scatter, plot_hist
from progpy.utils.containers import DictLikeMatrixWrapper


class UncertainData(ABC):
    """
    Abstract base class for data with uncertainty. Any new uncertainty type must implement this class
    """

    def __init__(self, _type=dict):
        self._type = _type
    
    @abstractmethod
    def sample(self, nSamples: int = 1):
        """Generate samples from data

        Args:
            nSamples (int, optional): Number of samples to generate. Defaults to 1.

        Returns:
            samples (UnweightedSamples): Array of nSamples samples

        Example:
            ::
            
                samples = data.samples(100)
        """

    @property
    @abstractproperty
    def median(self) -> dict:
        """The median of the UncertainData distribution or samples 

        Returns:
            dict[str, float]: Median value. e.g., {'key1': 23.2, ...}

        Example:
            ::
            
                median_value = data.median
        """

    @property
    @abstractproperty
    def mean(self) -> dict:
        """The mean of the UncertainData distribution or samples 

        Returns:
            dict[str, float]: Mean value. e.g., {'key1': 23.2, ...}

        Example:
            ::
                
                mean_value = data.mean
        """

    @property
    @abstractproperty
    def cov(self) -> array:
        """The covariance matrix of the UncertiantyData distribution or samples in order of keys (i.e., cov[1][1] is the standard deviation for key keys()[1])

        Returns:
            np.array[np.array[float]]: Covariance matrix

        Example:
            ::

                covariance_matrix = data.cov
        """

    def relative_accuracy(self, ground_truth: dict) -> dict:
        """The relative accuracy is how close the mean of the distribution is to the ground truth, on relative terms
        
        :math:`RA = 1 - \dfrac{\| r-p \|}{r}`

        Where r is ground truth and p is mean of predicted distribution [0]_

        Returns:
            dict[str, float]: Relative accuracy for each event where value is relative accuracy between [0,1]

        Example:
            ::

                ra = data.relative_accuracy({'key1': 22, 'key2': 57})

        References:
            .. [0] Prognostics: The Science of Making Predictions (Goebel et al, 239)
        """
        # if this check isn't here, goes to divide by zero check and raises AttributeError instead of TypeError. Keep? There are unittests checking for type
        if not (isinstance(ground_truth, dict) or isinstance(ground_truth, DictLikeMatrixWrapper)):
            raise TypeError("Ground truth must be passed as a dictionary or *.container argument.")
        if not all(ground_truth.values()):
            raise ZeroDivisionError("Ground truth values must be non-zero in calculating relative accuracy.")
        return {k:1 - (abs(ground_truth[k] - v)/ground_truth[k]) for k,v in self.mean.items()}

    @abstractmethod
    def keys(self):
        """Get the keys for the property represented

        Returns:
            list[str]: keys

        Example:
            ::
                
                keys = data.keys()
        """

    def __contains__(self, key):
        return key in self.keys()

    def percentage_in_bounds(self, bounds: tuple, keys: list = None, n_samples: int = 1000) -> dict:
        """Calculate percentage of dist is within specified bounds

        Args:
            bounds (tuple[float, float] or dict): Lower and upper bounds. \n
                if tuple: (lower, upper)\n
                if dict: {key: (lower, upper), ...}
            keys (list[str], optional): UncertainData keys to consider when calculating. Defaults to all keys.
            n_samples (int, optional): Number of samples to use when calculating

        Returns:
            dict: Percentage within bounds for each key in keys (where 0.5 = 50%). e.g., {'key1': 1, 'key2': 0.75}

        Example:
            ::
            
                data.percentage_in_bounds((1025, 1075))
                data.percentage_in_bounds({'key1': (1025, 1075), 'key2': (2520, 2675)})
                data.percentage_in_bounds((1025, 1075), keys=['key1', 'key3'])
        """
        return self.sample(n_samples).percentage_in_bounds(bounds, keys=keys)

    def metrics(self, **kwargs) -> dict:
        """Calculate Metrics for this dist

        Keyword Args:
            ground_truth (int or dict, optional): Ground truth value. Defaults to None.
            n_samples (int, optional): Number of samples to use for calculating metrics (if not UnweightedSamples)
            keys (list[str], optional): Keys to calculate metrics for. Defaults to all keys.

        Returns:
            dict: Dictionary of metrics

        Example:
            ::

                print(data.metrics())
                m = data.metrics(ground_truth={'key1': 200, 'key2': 350})
                m = data.metrics(keys=['key1', 'key3'])
        """
        from ..metrics import calc_metrics
        return calc_metrics(self, **kwargs)

    def plot_scatter(self, fig : Figure = None, keys : list = None, num_samples : int = 100, **kwargs) -> Figure:
        """
        Produce a scatter plot

        Args:
            fig (Figure, optional): Existing figure previously used to plot states. If passed a figure argument additional data will be added to the plot. Defaults to creating new figure
            keys (list[str], optional): Keys to plot. Defaults to all keys.
            num_samples (int, optional): Number of samples to plot. Defaults to 100
            **kwargs (optional): Additional keyword arguments passed to scatter function.

        Returns:
            Figure

        Example:
            ::

                m = [5, 7, 3]
                c = [[0.3, 0.5, 0.1], [0.6, 0.7, 1e-9], [1e-9, 1e-10, 1]]
                d = MultivariateNormalDist(['a', 'b', 'c'], m, c)
                d.plot_scatter() # With 100 samples
                states.plot_scatter(num_samples=5) # Specifying the number of samples to plot
                states.plot_scatter(keys=['a', 'b']) # only plot those keys
        """
        if keys is None:
            keys = self.keys()
        samples = self.sample(num_samples)
        return plot_scatter(samples, fig=fig, keys=keys, **kwargs)

    def plot_hist(self, fig = None, keys = None, num_samples = 100, **kwargs):
        """
        Create a histogram

        Args:
            fig (MatPlotLib Figure, optional): Existing histogram figure to be overritten. Defaults to create new figure.
            num_samples (int, optional): Number of samples to plot. Defaults to 100
            keys (list(String), optional): Keys to be plotted. Defaults to None.

        Example:
            ::
                
                m = [5, 7, 3]
                c = [[0.3, 0.5, 0.1], [0.6, 0.7, 1e-9], [1e-9, 1e-10, 1]]
                d = MultivariateNormalDist(['a', 'b', 'c'], m, c)
                d.plot_hist() # With 100 samples
                states.plot_hist(num_samples=20) # Specifying the number of samples to plot
                states.plot_hist(keys=['a', 'b']) # only plot those keys
        """
        if keys is None:
            keys = self.keys()
        samples = self.sample(num_samples)
        return plot_hist(samples, fig=fig, keys=keys, **kwargs)

    def describe(self, title : str = "UncertainData Metrics", print : bool = True) -> defaultdict:
        """
        Print and view basic statistical information about this UncertainData object in a text-based printed table.

        Args:
            title : str
                Title of the table, printed before data rows.
            print : bool = True 
                Optional argument specifying whether to print or not; default true.

        Returns:
            defaultdict
                Dictionary of lists used to print metrics.
        
        Example:
            ::

                data.describe()
        """
        recursive_metrics_table = print_table_recursive(self.metrics(), title, print)
        return recursive_metrics_table

    @abstractmethod
    def __add__(self, other : int) -> "UncertainData":
        """Overriding __add__ (+ operator) for UncertainData. 

        Args:
            other (int): Integer value to be applied to class where appropriate.
        """

    @abstractmethod
    def __radd__(self, other : int) -> "UncertainData":
        """Overriding __radd__ (+ operator right) for UncertainData. 

        Args:
            other (int): Integer value to be applied to class where appropriate.
        """

    @abstractmethod
    def __iadd__(self, other : int) -> "UncertainData":
        """Overriding __iadd__ (+= operator) for UncertainData. 

        Args:
            other (int): Integer value to be applied to class where appropriate.
        """

    @abstractmethod
    def __sub__(self, other : int) -> "UncertainData":
        """Overriding __sub__ (- operator) for UncertainData. 

        Args:
            other (int): Integer value to be applied to class where appropriate.
        """

    @abstractmethod
    def __rsub__(self, other : int) -> "UncertainData":
        """Overriding __rsub__ (- operator right) for UncertainData. 

        Args:
            other (int): Integer value to be applied to class where appropriate.
        """

    @abstractmethod
    def __isub__(self, other : int) -> "UncertainData":
        """Overriding __isub__ (-= operator) for UncertainData. 

        Args:
            other (int): Integer value to be applied to class where appropriate.
        """
