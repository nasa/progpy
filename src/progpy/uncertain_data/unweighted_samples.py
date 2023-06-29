# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from collections import UserList
from collections.abc import Iterable
from numpy import array, cov, random
from warnings import warn

from progpy.utils.containers import DictLikeMatrixWrapper

from . import UncertainData


class UnweightedSamples(UncertainData, UserList):
    """
    Uncertain Data represented by a set of samples. Objects of this class can be treated like a list where samples[n] returns the nth sample (Dict). 

    Args:
        samples (array, dict, or model.*Container, optional): array of samples. Defaults to empty array.\n
            If dict, must be of the form of {key: [value, ...], ...}\n
            If list, must be of the form of [{key: value, ...}, ...]\n
            If InputContainer, OutputContainer, or StateContainer, must be of the form of *Container({'key': value, ...})
    """
    def __init__(self, samples: list = [], _type=dict):
        super().__init__(_type)
        if isinstance(samples, dict) or isinstance(samples, DictLikeMatrixWrapper):
            # Is in form of {key: [value, ...], ...}
            # Convert to array of samples
            if len(samples.keys()) == 0:
                self.data = []  # is empty
                return
            n_samples = len(list(samples.values())[0])  # Number of samples
            self.data = [{key: value[i] for key, value in samples.items()} for i in range(n_samples)]
        elif isinstance(samples, Iterable):
            # is in form of [{key: value, ...}, ...]
            self.data = samples
        else:
            raise ValueError('Invalid input. Must be list or dict, was {}'.format(type(samples)))

    def __eq__(self, other):
        return isinstance(other, UnweightedSamples) and self.data == other.data

    def __getitem__(self, n):
        datem = self.data[n]
        return self._type(datem) if datem is not None else None

    def __add__(self, other: int) -> "UncertainData":
        if other == 0:
            return self
        result = []
        for i in range(len(self.data)):
            new_dict = {}
            for k,v in self.data[i].items():
                new_dict[k] = v + other
            result.append(new_dict)
        return UnweightedSamples(result)

    def __radd__(self, other: int) -> "UncertainData":
        return self.__add__(other)

    def __iadd__(self, other: int) -> "UncertainData":
        if other != 0:
            for i in range(len(self.data)):
                for k,v in self.data[i].items():
                    self.data[i][k] += other
        return self

    def __sub__(self, other: int) -> "UncertainData":
        if other == 0:
            return self
        result = []
        for i in range(len(self.data)):
            new_dict = {}
            for k,v in self.data[i].items():
                new_dict[k] = v - other
            result.append(new_dict)
        return UnweightedSamples(result)

    def __rsub__(self, other: int) -> "UncertainData":
        return self.__sub__(other)

    def __isub__(self, other: int) -> "UncertainData":
        if other != 0:
            for i in range(len(self.data)):
                for k,v in self.data[i].items():
                    self.data[i][k] -= other
        return self

    def __reduce__(self):
        return (UnweightedSamples, (self.data, ))

    def sample(self, num_samples: int = 1, replace: bool = True) -> "UnweightedSamples":
        # Completely random resample
        indices = random.choice(len(self.data), int(num_samples), replace=replace)
        return UnweightedSamples([self.data[i] for i in indices], _type=self._type)

    def keys(self) -> list:
        if len(self.data) == 0:
            return []  # is empty
        for sample in self:
            if sample is not None:
                return sample.keys()
        return []  # Every element is none

    def key(self, key) -> list:
        """Return samples for given key

        Args:
            key (str): key

        Returns:
            list: list of values for given key
        """
        return [sample[key] for sample in self.data if sample is not None]

    @property
    def median(self) -> dict:
        # Calculate Geometric median of all samples
        min_value = float('inf')
        none_flag = False
        for i, datem in enumerate(self.data):
            if datem is None:
                continue
            p1 = array([d for d in datem.values() if d is not None])
            if not none_flag and len(p1) < len(datem):
                none_flag = True
                warn("Some samples were None, resulting median is of all non-None samples. Note: in some cases, this will bias the median result.")
            total_dist = sum(
                sum((p1 - array([di for di in d.values() if di is not None]))**2)  # Distance between 2 points
                for d in self.data if d is not None)  # For each point
            if total_dist < min_value:
                min_index = i
                min_value = total_dist
        return self._type(self[min_index])

    @property
    def mean(self) -> dict:
        mean = {}
        for key in self.keys():
            values = array([x[key] for x in self.data if x is not None and x[key] is not None])
            if len(values) < len(self.data):
                warn("Some samples were None, resulting mean is of all non-None samples. Note: in some cases, this will bias the mean result.")
            mean[key] = values.mean()
        return self._type(mean)

    @property
    def cov(self) -> dict:
        if len(self.data) == 0:
            return [[]]
        unlabeled_samples = array([[x[key] for x in self.data if x is not None and x[key] is not None] for key in self.keys()])
        if len(unlabeled_samples) < len(self.data):
            warn("Some samples were None, resulting covariance is of all non-None samples. Note: in some cases, this will bias the covariance result.")
        return cov(unlabeled_samples)

    def __str__(self):
        return 'UnweightedSamples({})'.format(self.data)

    @property
    def size(self) -> int:
        """Get the number of samples. Note: kept for backwards compatibility, prefer using len() instead.

        Returns:
            int: Number of samples
        """
        return len(self)

    def percentage_in_bounds(self, bounds, keys: list = None) -> dict:
        if not keys:
            keys = self.keys()
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(bounds, list):
            bounds = {key: bounds for key in self.keys()}
        if not isinstance(bounds, dict) or all([isinstance(b, list) and len(b) == 2 for b in bounds]):
            raise TypeError("Bounds must be list [lower, upper] or dict (key: [lower, upper]), was {}".format(type(bounds)))
        n_elements = len(self.data)
        return {key: sum([x is not None and x < bounds[key][1] and x > bounds[key][0] for x in self.key(key)])/n_elements for key in keys}
