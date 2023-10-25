# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from collections import UserList, defaultdict, namedtuple
from typing import Dict, List
from numpy import sign
from warnings import warn

from ..uncertain_data import UnweightedSamples, UncertainData

PredictionResults = namedtuple('PredictionResults', ["times", "inputs", "states", "outputs", "event_states", "time_of_event"])


class Prediction():
    """
    Class for the result of a prediction. Is returned by the predict method of a predictor.

    Args:
        times (list[float]):
            Times for each data point where times[n] corresponds to data[n]
        data (list[UncertainData]):
            Data points for each time in times 
    """

    def __init__(self, times : list, data : list):
        self.times = times
        self.data = data

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
            for key,value in uncertaindata.mean.items():
                by_event[key].append(value)

        # For each event, calculate monotonicity using formula
        result = {}
        for key,l in by_event.items():
            mono_sum = []
            for i in range(len(l)-1): 
                mono_sum.append(sign(l[i+1] - l[i])) 
            result[key] = abs(sum(mono_sum) / (len(l)-1))
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

    def __calculate_tranform(self):
        """
        Calculate tranform of the data from data[sample_id][time_id] to data[time_id][sample_id]. Result is cached as self.__transform and is used in methods which look at a snapshot for a specific time
        """
        # Lazy calculation of tranform - only if needed
        # Note: prediction stops when event is reached, so for the length of all will not be the same. 
        # If the prediction doesn't go this far, then the value is set to None
        self.__transform = [UnweightedSamples([sample[time_index] if len(sample) > time_index else None for sample in self.data]) for time_index in range(len(self.times))]
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
