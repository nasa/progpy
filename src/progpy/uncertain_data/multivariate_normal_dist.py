# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from numpy import array
from numpy.random import multivariate_normal

from . import UncertainData, UnweightedSamples


class MultivariateNormalDist(UncertainData):
    """
    Data represented by a multivariate normal distribution with mean and covariance matrix

    Args:
            labels (list[str]): Labels for states, in order of mean values
            mean (array[float]): Mean values for state in the same order as labels
            covar (array[array[float]]): Covariance matrix for state
    """
    def __init__(self, labels, mean: array, covar : array, _type = dict):
        self.__labels = list(labels)
        self.__mean = array(list(mean))
        self.__covar = array(list(covar))
        super().__init__(_type)

    def __reduce__(self):
        return (MultivariateNormalDist, (self.__labels, self.__mean, self.__covar))

    def __eq__(self, other: "MultivariateNormalDist") -> bool:
        return isinstance(other, MultivariateNormalDist) and self.keys() == other.keys() and self.mean == other.mean and (self.cov == other.cov).all()

    def __add__(self, other: int) -> "UncertainData":
        if other == 0:
            return self
        return MultivariateNormalDist(self.__labels, array([i+other for i in self.__mean]), self.__covar)

    def __radd__(self, other: int) -> "UncertainData":
        return self.__add__(other)

    def __iadd__(self, other: int) -> "UncertainData":
        if not isinstance(other, (int, float)):
            raise TypeError(f" unsupported operand type(s) for +: '{type(other)}' and '{type(self.__mean[0])}'")
        if other != 0:
            self.__mean = array([i+other for i in self.__mean])
        return self

    def __sub__(self, other: int) -> "UncertainData":
        if other == 0:
            return self
        return MultivariateNormalDist(self.__labels, array([i-other for i in self.__mean]), self.__covar)

    def __rsub__(self, other: int) -> "UncertainData":
        return self.__sub__(other)

    def __isub__(self, other: int) -> "UncertainData":
        if not isinstance(other, (int, float)):
            raise TypeError(f" unsupported operand type(s) for -: '{type(other)}' and '{type(self.__mean[0])}'")
        if other != 0:
            self.__mean = array([i-other for i in self.__mean])
        return self

    def sample(self, num_samples: int = 1) -> UnweightedSamples:
        if len(self.__mean) != len(self.__labels):
            raise Exception("labels must be provided for each value")
    
        samples = multivariate_normal(self.__mean, self.__covar, num_samples)
        samples = [{key: value for (key, value) in zip(self.__labels, x)} for x in samples]
        return UnweightedSamples(samples, _type = self._type)

    def keys(self) -> list:
        return self.__labels

    @property
    def median(self) -> float:
        # For normal distribution medain = mean
        return self.mean

    @property
    def mean(self) -> dict:
        return self._type({key: value for (key, value) in zip(self.__labels, self.__mean)})

    def __str__(self) -> str:
        return 'MultivariateNormalDist(mean: {}, covar: {})'.format(self.__mean, self.__covar)     

    @property
    def cov(self) -> array:
        return self.__covar
