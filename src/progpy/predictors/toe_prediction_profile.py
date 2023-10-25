# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.
import matplotlib.pyplot as plt
from collections import UserDict
from typing import Dict
import numpy as np

from progpy.uncertain_data import UncertainData 

class ToEPredictionProfile(UserDict):
    """
    Data structure for storing the result of multiple predictions, including time of prediction. This data structure can be treated as a dictionary of time of prediction to Time of Event (ToE) prediction. Iteration of this data structure is in order of increasing time of prediction
    """
    def add_prediction(self, time_of_prediction: float, toe_prediction: UncertainData):
        """Add a single prediction to the profile

        Args:
            time_of_prediction (float): Time that the prediction was made
            toe_prediction (UncertainData): Distribution of predicted ToEs
        """
        self[time_of_prediction] = toe_prediction

    # Functions below are defined to ensure that any iteration is in order of increasing time of prediction
    def __iter__(self):
        return iter(sorted(super(ToEPredictionProfile, self).__iter__()))

    def items(self):
        """
        Get iterators for the items (time_of_prediction, toe_prediction) of the prediction profile
        """
        return iter((k, self[k]) for k in self)

    def keys(self):
        """
        Get iterator for the keys (i.e., time_of_prediction) of the prediction profile
        """
        return sorted(super(ToEPredictionProfile, self).keys())

    def values(self):
        """
        Get iterator for the values (i.e., toe_prediction) of the prediction profile
        """
        return [self[k] for k in self.keys()]

    def alpha_lambda(self, ground_truth: Dict[str, float], lambda_value: float, alpha: float, beta: float, **kwargs) -> Dict[str, bool]:
        """Calculate Alpha lambda metric for the prediction profile

        Args:
            ground_truth (dict[str, float]):
                Ground Truth time of event for each event (e.g., {'event1': 748, 'event2', 2233, ...})
            lambda_value (float):
                Prediction time at or after which metric is evaluated. Evaluation occurs at this time (if a prediction exists) or the next prediction following.
            alpha (float): 
                percentage bounds around time to event (where 0.2 allows 20% error TtE)
            beta (float):
                portion of prediction that must be within those bounds

        Keyword Args:
            keys (list[str], optional): 
                list of keys to use. If not provided, all keys are used.
            print (bool, optional)
                If True, print the results to the screen. Default is False.

        Returns:
            dict[str, bool]: If alpha lambda was met for each key (e.g., {'event1': True, 'event2', False, ...})
        """
        from ..metrics import alpha_lambda
        return alpha_lambda(self, ground_truth, lambda_value, alpha, beta, **kwargs)

    def prognostic_horizon(self, criteria_eqn, ground_truth, **kwargs) -> Dict[str, float]:
        """
        Compute prognostic horizon metric, defined as the difference between a time ti, when the predictions meet specified performance criteria, and the time corresponding to the true Time of Event (ToE), for each event.

        :math:`PH = ToE - ti`

        Args:
            toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
            criteria_eqn (Callable function): A function (toe: UncertainData, ground_truth: dict[str, float]) -> dict[str, bool] calculating whether a prediction in ToEPredictionProfile meets some criteria. \n
                | Args: 
                |  * toe (UncertainData): A single prediction of Time of Event (ToE)
                |  * ground truth (dict[str, float]): Ground truth passed into prognostics_horizon
                | Returns: Map of event names to boolean representing if the event has been met. 
                |   e.g., {'event1': True, 'event2': False}
            ground_truth (dict): Dictionary containing ground truth; specified as key, value pairs for event and its value. E.g, {'event1': 47.3, 'event2': 52.1, 'event3': 46.1}
        
        Keyword Args:
            print (bool): 
                Boolean specifying whether the prognostic horizon metric should be printed.

        Returns:
            dict: Dictionary containing prognostic horizon calculations (value) for each event (key). e.g., {'event1': 12.3, 'event2': 15.1}
        """
        from ..metrics import prognostic_horizon
        return prognostic_horizon(self, criteria_eqn, ground_truth, **kwargs)

    def cumulative_relative_accuracy(self, ground_truth, **kwargs) -> Dict[str, float]:
        r"""
        Compute cumulative relative accuracy for a given profile, defined as the normalized sum of relative prediction accuracies at specific time instances.
        
        :math:`CRA = \Sigma \left( \dfrac{RA}{N} \right)` for each event

        Where :math:`\Sigma` is summation of all relative accuracies for a given profile and N is the total count of profiles [0]_

        Args:
            ground_truth (dict): Dictionary containing ground truth; specified as key, value pairs for event and its value. E.g, {'event1': 47.3, 'event2': 52.1, 'event3': 46.1}

        Returns:
            dict: Dictionary containing cumulative relative accuracy (value) for each event (key). e.g., {'event1': 12.3, 'event2': 15.1}

        References:
            .. [0] Journal Prognostics Health Management, Saxena et al.
        """
        from ..metrics import cumulative_relative_accuracy
        return cumulative_relative_accuracy(self, ground_truth, **kwargs)

    def monotonicity(self, **kwargs) -> Dict[str, float]:
        r"""Calculate monotonicty for a prediction profile. 
        Given a prediction profile, for each prediction: go through all predicted states and compare those to the next one.
        Calculates monotonicity for each prediction key using its associated mean value in :py:class:`progpy.uncertain_data.UncertainData`.
        
        :math:`monotonoicity = \|\Sigma \left( \dfrac{sign(i+1 - i)}{N-1} \right) \|`

        Where N is number of measurements and sign indicates sign of calculation. [0]_ [1]_

        Args:
            toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        
        Returns:
            dict (str, dict): Dictionary where keys represent a profile and dict is a subdictionary representing an event and its respective monotonicitiy value between [0, 1].

        References:
            .. [1] Coble, J., et. al. (2021). Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society. http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404
            .. [2] Baptistia, M., et. al. (2022). Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Aritifical Intelligence, Volume 306. https://www.sciencedirect.com/science/article/pii/S0004370222000078
        """
        from ..metrics import monotonicity
        return monotonicity(self, **kwargs)
    
    def plot(self, ground_truth: dict = None , alpha: float = None, show: bool = True) -> dict: # use ground truth, alpha if given,
        """Produce an alpha-beta plot depicting the TtE distribution by time of prediction for each event.

        Args:
            ground_truth (dict):
                Optional dictionary argument containing event and its respective ground truth value; none by default and plotted if specified
            alpha (float):
                Optional alpha value; none by default and plotted if specified
            show (bool):
                Optional bool value; specify whether to display generated plots. Default is true
        
        Returns:
            dict[str, Figure] :
                Collection of generated matplotlib figures for each event in profile\n
                e.g., {'event1': Fig, 'event2': Fig}
        
        Example:
            ::

                gt = {'event1': 3442, 'event2': 175} # Ground Truth
                figs = profile.plot(gt) # Figure with ground truth line
                figs = profile.plot(gt, alpha = 0.2) # Figure with ground truth line and 20% alpha bounds
                figs = profile.plot(gt, alpha = 0.2, show=False) # Dont display figure
        """
        result_figs = {}
        for t,v in self.items():
            raw_samples = v.sample(100) # sample distribution (red scatter plot)
            for key in v.keys():
                if key not in result_figs:
                    # Prepare Figure for Plot
                    fig_window = plt.figure() # Create new figure for this event key
                    fig_sub = fig_window.subplots()
                    fig_sub.grid()
                    fig_sub.set_title(f"{key} Event")
                    fig_sub.set_xlabel('Time of Prediction (s)') # time to prediction
                    fig_sub.set_ylabel('Time to Event (s)') # time to event
                    result_figs[key] = fig_window
                # Create scatter plot for this event
                samples = [e[key]-t for e in raw_samples]
                result_figs[key].get_axes()[0].scatter([t]*len(samples), samples, color='red') # Adding single distribution of estimates

        if ground_truth: # If ground_truth is specified, add ground_truth to each event plot (green line)
            for key, val in ground_truth.items():
                gt_x = range(int(val))
                gt_y = range(int(val), 0, -1)
                result_figs[key].get_axes()[0].plot(gt_x, gt_y, color='green')
                if alpha: # if ground_truth and alpha are specified, add alpha bounds (faded green highlight)
                    result_figs[key].get_axes()[0].fill_between(gt_x, np.array(gt_y)*(1-alpha), np.array(gt_y)*(1+alpha), color='green', alpha=0.2)
                result_figs[key].get_axes()[0].set_xlim(0, val+1)

        if show: # Optionally not display plots and just return plot objects
            plt.show()
        return result_figs 
