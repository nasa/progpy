# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from .data_model import DataModel
from .dmd import DMDModel
from .lstm_model import LSTMStateTransitionModel
from .pce import PolynomialChaosExpansion, PCE

SURROGATE_METHOD_LOOKUP = {
    'dmd': DMDModel.from_model,
    'lstm': LSTMStateTransitionModel.from_model,
    'pce': PolynomialChaosExpansion.from_model,
}
