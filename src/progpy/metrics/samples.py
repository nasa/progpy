# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

# This file is kept for backwards compatability
from numpy import mean, sqrt
from warnings import warn

from .uncertain_data_metrics import calc_metrics as eol_metrics
from .toe_metrics import prob_success
from .toe_profile_metrics import alpha_lambda

def mean_square_error(values: list, ground_truth: float) -> float:
    """Mean Square Error
    Args:
        values (List[float]): Times of Event (ToE) for a single event, output from predictor
        ground_truth (float): Ground truth ToE
    Returns:
        float: mean square error of ToE predictions
    """
    return sum([(mean(x) - ground_truth) ** 2 for x in values]) / len(values)


def root_mean_square_error(values, ground_truth):
    """Root Mean Square Error
    Args:
        values (List[float]): Times of Event (ToE) for a single event, output from predictor
        ground_truth (float): Ground truth ToE
    Returns:
        float: root mean square error of ToE predictions
    """
    return sqrt(sum([(mean(x) - ground_truth)**2 for x in values])/len(values))
