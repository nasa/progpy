# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from collections import UserList, defaultdict, namedtuple
import json
from typing import Dict, List

from numpy import sign

from ..sim_result import SimResult
from ..uncertain_data import UnweightedSamples, UncertainData, from_json

PredictionResults = namedtuple(
    "PredictionResults",
    ["times", "inputs", "states", "outputs", "event_states", "time_of_event"],
)


class Prediction:
    """
    Class for the result of a prediction. Is returned by the predict method of a predictor.

    Args:
        times (list[float]):
            Times for each data point where times[n] corresponds to data[n]
        data (list[UncertainData]):
            Data points for each time in times
    """

    def __init__(self, times: list, data: list):
        self.times = times
        self.data = data

    def to_dict(self) -> dict:
        """Serialize this prediction and its uncertain data to a dictionary."""
        return {
            "type": type(self).__name__,
            "times": self.times,
            "data": [entry.to_json() for entry in self.data],
        }
        
    def to_json(self) -> str:
        """Serialize this prediction and its uncertain data to a JSON string.

        The serialized data contains the prediction type, times, and encoded
        uncertain-data entries. It does not include the ``PredictionResults``
        returned alongside predictions by predictors.
        """
        return json.dumps(
            self.to_dict(), sort_keys=True, indent=4, separators=(",", ": ")
        )
        
        
    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            raise TypeError("Invalid prediction data: expected an object.")

        prediction_classes = {
            "Prediction": Prediction,
            "UnweightedSamplesPrediction": UnweightedSamplesPrediction,
        }
        prediction_type = data.get("type")
        if (
            not isinstance(prediction_type, str)
            or prediction_type not in prediction_classes
        ):
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        prediction_class = prediction_classes[prediction_type]
        if cls is not Prediction and cls is not prediction_class:
            raise ValueError(
                f"Invalid prediction type for {cls.__name__} decoding: "
                f"expected '{cls.__name__}', got '{prediction_type}'."
            )
        if "times" not in data or "data" not in data:
            raise ValueError("Invalid prediction data: missing 'times' or 'data'.")
        if not isinstance(data["times"], list) or not isinstance(data["data"], list):
            raise TypeError(
                "Invalid prediction data: 'times' and 'data' must be lists."
            )
        if prediction_class is UnweightedSamplesPrediction:
            return prediction_class.from_dict(data)
        if not all(isinstance(entry, dict) for entry in data["data"]):
            raise TypeError(
                "Invalid prediction data: each data entry must be an object."
            )
        return prediction_class(
            data["times"], [from_json(entry) for entry in data["data"]]
        )

    @classmethod
    def from_json(cls, data: str) -> "Prediction":
        """Reconstruct a prediction from a string produced by :meth:`to_json`.

        Calling this method on ``Prediction`` dispatches to the concrete
        prediction class recorded in the JSON. Calling it on a concrete
        subclass requires that the serialized type matches that subclass.

        Raises:
            ValueError: If the JSON envelope or prediction type is invalid.
            TypeError: If the JSON envelope contains values of invalid types.
        """
        return cls.from_dict(json.loads(data))

    def __eq__(self, other: "Prediction") -> bool:
        """Compare 2 Predictions

        Args:
            other (Prediction):

        Returns:
            bool: If the two Predictions are equal
        """
        return self.times == other.times and self.data == other.data

    def snapshot(self, time_index: int) -> UncertainData:
        """Get all samples from a specific timestep

        Args:
            index (int):
                Timestep (index number from times)

        Returns:
            UncertainData: Distribution for time corresponding to times[timestep]
        """
        return self.data[time_index]

    @property
    def mean(self) -> List[dict]:
        """Estimate of the mean value of the prediction at each time

        Returns:
            list[dict]:
                Mean value of the prediction at each time where mean[n] corresponds to the mean value of the prediction at time times[n].\n
                The mean value at each time is a dictionary. \n
                e.g., [{'state1': 1.2, 'state2': 1.3, ...}, ...]

        Example:
            mean_value = data.mean
        """
        return [dist.mean for dist in self.data]

    def monotonicity(self) -> Dict[str, float]:
        """Calculate monotonicty for a single prediction.
        Given a single prediction, for each event: go through all predicted states and compare those to the next one.
        Calculates monotonicity for each event key using its associated mean value in UncertainData.

        :math:`monotonoicity = \| \Sigma \dfrac{sign(i+1 - i)}{N-1}\|`

        Where N is number of measurements and sign indicates sign of calculation [0]_ [1]_.

        Returns:
            dict (str, float): Value between [0, 1] indicating monotonicity of a given event for the Prediction.

        References:
            .. [0] Coble, J., et. al. (2021). Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society. http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404
            .. [1] Baptistia, M., et. al. (2022). Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Aritifical Intelligence, Volume 306. https://www.sciencedirect.com/science/article/pii/S0004370222000078

        """
        # Collect and organize mean values for each event
        by_event = defaultdict(list)
        for uncertaindata in self.data:
            for key, value in uncertaindata.mean.items():
                by_event[key].append(value)

        # For each event, calculate monotonicity using formula
        result = {}
        for key, l in by_event.items():
            mono_sum = []
            for i in range(len(l) - 1):
                mono_sum.append(sign(l[i + 1] - l[i]))
            result[key] = abs(sum(mono_sum) / (len(l) - 1))
        return result


class UnweightedSamplesPrediction(Prediction, UserList):
    """
    Immutable data class for the result of a prediction, where the predictions are stored as UnweightedSamples. Is returned from the predict method of a sample based prediction class (e.g., MonteCarlo). Objects of this class can be iterated and accessed like a list (e.g., prediction[0]), where prediction[n] represents a profile for sample n.

    Args:
        times (list[float]):
            Times for each data point where times[n] corresponds to data[:][n]
        data (list[SimResult]):
            Data points where data[n] is a SimResult for sample n
    """

    def __init__(self, times: list, data: list):
        super(UnweightedSamplesPrediction, self).__init__(times, data)
        self.__transformed = False  # If transform has been calculated

    def to_dict(self) -> dict:
        """Serialize each sample profile as a SimResult dictionary."""
        profiles = []
        for profile in self.data:
            if isinstance(profile, SimResult):
                serialized_profile = profile.to_dict()
            else:
                serialized_profile = profile.to_json()
            if isinstance(serialized_profile, str):
                serialized_profile = json.loads(serialized_profile)
            profiles.append(serialized_profile)

        return {
            "type": type(self).__name__,
            "times": self.times,
            "data": profiles,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnweightedSamplesPrediction":
        """Reconstruct a prediction from dictionaries produced by :meth:`to_dict`."""
        if not isinstance(data, dict):
            raise TypeError("Invalid prediction data: expected an object.")
        if data.get("type") != cls.__name__:
            raise ValueError(
                f"Invalid prediction type for {cls.__name__} decoding: "
                f"expected '{cls.__name__}', got '{data.get('type')}'."
            )
        if "times" not in data or "data" not in data:
            raise ValueError("Invalid prediction data: missing 'times' or 'data'.")
        if not isinstance(data["times"], list) or not isinstance(data["data"], list):
            raise TypeError(
                "Invalid prediction data: 'times' and 'data' must be lists."
            )
        if not all(isinstance(profile, (dict, str)) for profile in data["data"]):
            raise TypeError(
                "Invalid prediction data: each sample profile must be an object "
                "or JSON string."
            )
        profiles = [
            json.loads(profile) if isinstance(profile, str) else profile
            for profile in data["data"]
        ]
        return cls(
            data["times"],
            [
                SimResult.from_dict(profile)
                if profile.get("type") == "SimResult"
                else from_json(profile)
                for profile in profiles
            ],
        )

    def __calculate_tranform(self):
        """
        Calculate tranform of the data from data[sample_id][time_id] to data[time_id][sample_id]. Result is cached as self.__transform and is used in methods which look at a snapshot for a specific time
        """
        # Lazy calculation of tranform - only if needed
        # Note: prediction stops when event is reached, so for the length of all will not be the same.
        # If the prediction doesn't go this far, then the value is set to None
        self.__transform = [
            UnweightedSamples(
                [
                    sample[time_index] if len(sample) > time_index else None
                    for sample in self.data
                ]
            )
            for time_index in range(len(self.times))
        ]
        self.__transformed = True

    def __str__(self) -> str:
        return "UnweightedSamplesPrediction with {} savepoints".format(len(self.times))

    @property
    def mean(self) -> list:
        if not self.__transformed:
            self.__calculate_tranform()
        return [dist.mean for dist in self.__transform]

    def snapshot(self, time_index: int) -> UnweightedSamples:
        """Get all samples from a specific timestep

        Args:
            index (int): Timestep (index number from times)

        Returns:
            UnweightedSamples: Samples for time corresponding to times[timestep]
        """
        if not self.__transformed:
            self.__calculate_tranform()
        return self.__transform[time_index]

    def __not_implemented(self, *args, **kw):
        """
        This function is not implemented. Calling this will raise an error. Is is only included to make the class immutable.

        Raises:
            ValueError:
        """
        raise ValueError("UnweightedSamplesPrediction is immutable (i.e., read only)")

    append = __not_implemented
    extend = __not_implemented
    clear = __not_implemented
    reverse = __not_implemented
    remove = __not_implemented
    insert = __not_implemented
    pop = __not_implemented
    __setitem__ = __not_implemented
    __setslice__ = __not_implemented
    __delitem__ = __not_implemented
