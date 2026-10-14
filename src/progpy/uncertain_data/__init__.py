# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All Rights Reserved.

from .uncertain_data import UncertainData
from .unweighted_samples import UnweightedSamples
from .scalar_data import ScalarData
from .multivariate_normal_dist import MultivariateNormalDist

__all__ = ["UncertainData", "UnweightedSamples", "ScalarData", "MultivariateNormalDist"]

def from_json(data: dict) -> UncertainData:
    """Decode JSON serializable representation of UncertainData object.

    Args:
        data (dict): JSON serializable representation of UncertainData object.

    Returns:
        UncertainData: Decoded UncertainData object.
    """
    if "type" not in data:
        raise ValueError("Invalid JSON data: missing 'type' key.")
    if data["type"] == "UnweightedSamples":
        return UnweightedSamples.from_json(data)
    elif data["type"] == "ScalarData":
        return ScalarData.from_json(data)
    elif data["type"] == "MultivariateNormalDist":
        return MultivariateNormalDist.from_json(data)
    else:
        raise ValueError(f"Unknown UncertainData type: {data['type']}")
