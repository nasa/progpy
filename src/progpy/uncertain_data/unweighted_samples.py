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

    def __init__(self, samples: list = None, _type=dict):
        super().__init__(_type)
        if samples is None:
            self.data = []
            return
        if isinstance(samples, (dict, DictLikeMatrixWrapper)):
            # Is in form of {key: [value, ...], ...}
            # Convert to array of samples
            if len(samples.keys()) == 0:
                self.data = []  # is empty
                return
            n_samples = len(next(iter(samples.values())))  # Number of samples
            self.data = [
                {key: value[i] for key, value in samples.items()}
                for i in range(n_samples)
            ]
        elif isinstance(samples, Iterable):
            # is in form of [{key: value, ...}, ...]
            self.data = samples
        else:
            raise TypeError(
                f"Invalid input. Must be list or dict, was {type(samples)}"
            )

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
            for k, v in self.data[i].items():
                new_dict[k] = v + other
            result.append(new_dict)
        return UnweightedSamples(result)

    def __radd__(self, other: int) -> "UncertainData":
        return self.__add__(other)

    def __iadd__(self, other: int) -> "UncertainData":
        if other != 0:
            for i in range(len(self.data)):
                for k in self.data[i]:
                    self.data[i][k] += other
        return self

    def __sub__(self, other: int) -> "UncertainData":
        if other == 0:
            return self
        result = []
        for i in range(len(self.data)):
            new_dict = {}
            for k, v in self.data[i].items():
                new_dict[k] = v - other
            result.append(new_dict)
        return UnweightedSamples(result)

    def __rsub__(self, other: int) -> "UncertainData":
        return self.__sub__(other)

    def __isub__(self, other: int) -> "UncertainData":
        if other != 0:
            for i in range(len(self.data)):
                for k in self.data[i]:
                    self.data[i][k] -= other
        return self

    def __reduce__(self):
        return (UnweightedSamples, (self.data,))

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
        """Return the sample minimizing total squared distance to all samples.

        This is a representative observed sample, not a coordinate-wise or
        geometric median. Ties select the first sample in input order.

        Whole missing samples (None) are ignored. Partially missing or
        non-finite samples are rejected rather than silently misaligned.
        """
        import numpy as np

        samples = [sample for sample in self.data if sample is not None]
        if not samples:
            raise ValueError("Cannot calculate median without valid samples.")

        keys = tuple(samples[0].keys())
        if not keys:
            raise ValueError("Cannot calculate median for samples with no variables.")

        expected_keys = set(keys)
        for sample in samples:
            if set(sample.keys()) != expected_keys:
                raise ValueError("All samples must contain the same keys.")
            if any(sample[key] is None for key in keys):
                raise ValueError(
                    "Cannot calculate median with partially missing samples."
                )

        # Use a consistent key order across samples.
        values = np.asarray(
            [[sample[key] for key in keys] for sample in samples],
            dtype=np.float64,
        )
        if values.ndim != 2:
            raise ValueError("Each sample variable must be a scalar.")
        if not np.isfinite(values).all():
            raise ValueError("Median requires finite sample values.")

        if len(samples) != len(self.data):
            warn(
                "Some samples were None; median uses only non-None samples. "
                "This may bias the result.",
                stacklevel=2,
            )

        centered = values - values.mean(axis=0)
        squared_distances = np.einsum("ij,ij->i", centered, centered)
        index = int(np.argmin(squared_distances))

        return self._type(samples[index])

    @property
    def mean(self) -> dict:
        mean = {}
        for key in self.keys():
            values = array(
                [x[key] for x in self.data if x is not None and x[key] is not None]
            )
            if len(values) < len(self.data):
                warn(
                    "Some samples were None, resulting mean is of all non-None samples. Note: in some cases, this will bias the mean result."
                )
            mean[key] = values.mean()
        return self._type(mean)

    @property
    def cov(self) -> dict:
        if len(self.data) == 0:
            return [[]]
        unlabeled_samples = array(
            [
                [x[key] for x in self.data if x is not None and x[key] is not None]
                for key in self.keys()
            ]
        )
        if len(unlabeled_samples) < len(self.data):
            warn(
                "Some samples were None, resulting covariance is of all non-None samples. Note: in some cases, this will bias the covariance result."
            )
        return cov(unlabeled_samples)

    def __str__(self):
        return f"UnweightedSamples({self.data})"

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
        if not isinstance(bounds, dict) or all(
            isinstance(b, list) and len(b) == 2 for b in bounds
        ):
            raise TypeError(
                f"Bounds must be list [lower, upper] or dict (key: [lower, upper]), was {type(bounds)}"
            )
        n_elements = len(self.data)
        return {
            key: sum(
                [
                    x is not None and x < bounds[key][1] and x > bounds[key][0]
                    for x in self.key(key)
                ]
            )
            / n_elements
            for key in keys
        }
        
    def to_json(self) -> dict:
        """Encode UnweightedSamples to JSON serializable dictionary

        Returns:
            dict: JSON serializable dictionary
        """
        return {
            "type": "UnweightedSamples",
            'samples': self.data
        }
        
    @classmethod
    def from_json(cls, data: dict) -> "UnweightedSamples":
        """Decode UnweightedSamples from JSON serializable dictionary

        Args:
            data (dict): JSON serializable dictionary
        Returns:
            UnweightedSamples: Decoded UnweightedSamples object
        """
        if data.get("type") != "UnweightedSamples":
            raise ValueError(
                f"Invalid type for UnweightedSamples decoding, expected 'UnweightedSamples', got {data.get('type')}"
            )
        return cls(samples=data.get("samples", []))
