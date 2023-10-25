# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

"""
This file includes functions for calculating metrics given a Time of Event (ToE) profile (i.e., ToE's calculated at different times of prediction resulting from running prognostics multiple times, e.g., on playback data). The metrics calculated here are specific to multiple ToE estimates (e.g. alpha-lambda metric)
"""
from numpy import sign
from collections import defaultdict
from typing import Callable, Dict

from ..predictors import ToEPredictionProfile

def alpha_lambda(toe_profile: ToEPredictionProfile, ground_truth: dict, lambda_value: float, alpha: float, beta: float, **kwargs) -> dict: 
    """
    Compute alpha lambda metric, a common metric in prognostics. Alpha-Lambda is met if alpha % of the Time to Event (TtE) distribution is within beta % of the ground truth at prediction time lambda.

    Args:
        toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        ground_truth (dict): Ground Truth time of event for each event (e.g., {'event1': 748, 'event2', 2233, ...})
        lambda_value (float): Prediction time at or after which metric is evaluated. Evaluation occurs at this time (if a prediction exists) or the next prediction following.
        alpha (float): percentage bounds around time to event (where 0.2 allows 20% error TtE)
        beta (float): portion of prediction that must be within those bounds
        kwargs (optional): configuration arguments. Accepted args include:
            * keys (list[string], optional): list of keys to use. If not provided, all keys are used.

    Returns:
        dict: dictionary containing key value pairs for each key and whether the alpha-lambda was met.
    """
    params = {
        'print': False
    }
    params.update(kwargs)

    for (t_prediction, toe) in toe_profile.items():
        if (t_prediction >= lambda_value):
            # If keys not provided, use all
            keys = params.setdefault('keys', toe.keys())

            bounds = {key : [gt - alpha*(gt-t_prediction), gt + alpha*(gt-t_prediction)] for key, gt in ground_truth.items()}
            pib = toe.percentage_in_bounds(bounds)
            result = {key: pib[key] >= beta for key in keys}
            if params['print']:
                for key in keys:
                    print('\n', key)
                    print('\ttoe:', toe.key(key))
                    print('\tBounds: [{} - {}]({}%)'.format(bounds[key][0], bounds[key][1], pib[key]))
            return result

def prognostic_horizon(toe_profile: ToEPredictionProfile, criteria_eqn: Callable, ground_truth: dict, **kwargs) -> dict:
    """
    Compute prognostic horizon metric, defined as the difference between a time ti, when the predictions meet specified performance criteria, and the time corresponding to the true Time of Event (ToE), for each event.
    PH = ToE - ti
    Args:
        toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        criteria_eqn (Callable function): A function (tte: UncertainData, ground_truth_tte: dict[str, float]) -> dict[str, bool] calculating whether a prediction in ToEPredictionProfile meets some criteria. \n
            | Args: 
            |  * tte (UncertainData): A single prediction of Time of Event (ToE)
            |  * ground truth tte (dict[str, float]): Ground truth passed into prognostics_horizon
            | Returns: Map of event names to boolean representing if the event has been met. 
            |   e.g., {'event1': True, 'event2': False}
        ground_truth (dict): Dictionary containing ground truth; specified as key, value pairs for event and its value. E.g, {'event1': 47.3, 'event2': 52.1, 'event3': 46.1}
        kwargs (optional): configuration arguments. Accepted args include:
            * print (bool): Boolean specifying whether the prognostic horizon metric should be printed.

    Returns:
        dict: Dictionary containing prognostic horizon calculations (value) for each event (key). e.g., {'event1': 12.3, 'event2': 15.1}
    """
    params = {
        'print': False
    }
    params.update(kwargs)

    ph_result = {k:None for k in ground_truth.keys()}  # False means not yet met; will be either a numerical value or None if met
    for (t_prediction, toe) in toe_profile.items():
        # Convert to TtE for toe and ground_truth
        tte = toe - t_prediction
        ground_truth_tte = {}
        ground_truth_tte = {k:v-t_prediction for k,v in ground_truth.items()}
        # Pass to criteria_eqn
        criteria_eqn_dict = criteria_eqn(tte, ground_truth_tte) # -> dict[event_names as str, bool]
        for k,v in criteria_eqn_dict.items():
            if v and (ph_result[k] == None):
                ph_calc = ground_truth[k] - t_prediction
                if ph_calc > 0:
                    ph_result[k] = ph_calc  # PH = EOL - ti # ground truth is a dictionary {'EOD': 3005.2} should be ph_result[k] = g_truth[key] - t_prediction
                if (all(v != None for v in ph_result.values())):
                    # Return PH once all criteria are met
                    return ph_result
    # Return PH when criteria not met for at least one event key
    return ph_result

def cumulative_relative_accuracy(toe_profile: ToEPredictionProfile, ground_truth: dict, **kwargs) -> Dict[str, float]:
    """
    Compute cumulative relative accuracy for a given profile, defined as the normalized sum of relative prediction accuracies at specific time instances.
    
    CRA = Σ(RA)/N for each event
    Where Σ is summation of all relative accuracies for a given profile and N is the total count of profiles (Journal Prognostics Health Management, Saxena et al.)
    Args:
        toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        ground_truth (dict): Dictionary containing ground truth; specified as key, value pairs for event and its value. E.g, {'event1': 47.3, 'event2': 52.1, 'event3': 46.1}
        kwargs (optional): configuration arguments. Accepted args include:

    Returns:
        dict: Dictionary containing cumulative relative accuracy (value) for each event (key). e.g., {'event1': 12.3, 'event2': 15.1}
    """
    ra_sums = defaultdict(int)
    for uncertaindata in toe_profile.values():
        for event,value in uncertaindata.relative_accuracy(ground_truth).items():
            ra_sums[event] += value
    return {event:ra_sum/len(toe_profile) for event, ra_sum in ra_sums.items()}

def monotonicity(toe_profile: ToEPredictionProfile, **kwargs) -> Dict[str, float]:
        """Calculate monotonicty for a prediction profile. 
        Given a prediction profile, for each prediction: go through all predicted events and compare those to the next one.
        Calculates monotonicity for each prediction key using its associated mean value in UncertainData.
        
        monotonoicity = |Σsign(i+1 - i) / N-1|
        Where N is number of measurements and sign indicates sign of calculation.
        Coble, J., et. al. (2021). Identifying Optimal Prognostic Parameters from Data: A Genetic Algorithms Approach. Annual Conference of the PHM Society.
        http://www.papers.phmsociety.org/index.php/phmconf/article/view/1404
        Baptistia, M., et. al. (2022). Relation between prognostics predictor evaluation metrics and local interpretability SHAP values. Aritifical Intelligence, Volume 306.
        https://www.sciencedirect.com/science/article/pii/S0004370222000078

        Args:
            toe_profile (ToEPredictionProfile): A profile of predictions, the combination of multiple predictions
        Returns:
            dict (str, float): Dictionary where keys represent an event and values are float representing its respective monotonicitiy value between [0, 1].
        """
        result = dict()
        by_event = defaultdict(list)
        for time,uncertaindata in toe_profile.items():
            # Collect and organize mean values for each event in the individual prediction v
            for event,value in uncertaindata.mean.items():
                by_event[event].append(value - time)
        # For each event of this prediction v, calculate monotonicity using formula
        for key,l in by_event.items():
            mono_sum = []
            for i in range(len(l)-1): 
                mono_sum.append(sign(l[i+1] - l[i])) 
            result[key] = abs(sum(mono_sum) / (len(l)-1))
        return result
