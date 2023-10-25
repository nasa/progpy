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
    return sum([(mean(x) - ground_truth)**2 for x in values])/len(values)

def root_mean_square_error(values, ground_truth):
    """Root Mean Square Error
    Args:
        values (List[float]): Times of Event (ToE) for a single event, output from predictor
        ground_truth (float): Ground truth ToE
    Returns:
        float: root mean square error of ToE predictions
    """
    return sqrt(sum([(mean(x) - ground_truth)**2 for x in values])/len(values))

def percentage_in_bounds(toe: list, bounds: tuple) -> float:
    """Calculate percentage of ToE dist is within specified bounds

    Args:
        toe (List[float]): Times of Event (ToE) for a single event, output from predictor
        bounds ((float, float)): Lower and upper bounds

    Returns:
        float: Percentage within bounds (where 1 = 100%)
    """
    warn('percentage_in_bounds has been deprecated in favor of UncertainData.percentage_in_bounds(bounds). This function will be removed in a future release')
    return sum([x < bounds[1] and x > bounds[0] for x in toe])/ len(toe)
