# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable

from progpy.predictors.prediction import PredictionResults
from ..uncertain_data import UncertainData


class Predictor(ABC):
    """
    Interface class for predictors

    Abstract base class for creating predictors that perform prediction. Predictor subclasses must implement this interface. Equivilant to "Predictors" in NASA's Matlab Prognostics Algorithm Library

    Parameters
    ----------
    model : PrognosticsModel
        See: :py:mod:`progpy` package\n
        A prognostics model to be used in prediction
    kwargs : optional, keyword arguments
    """
    default_parameters = {}

    def __init__(self, model, **kwargs):
        if not hasattr(model, 'output'):
            raise NotImplementedError("model must have `output` method")
        if not hasattr(model, 'next_state'):
            raise NotImplementedError("model must have `next_state` method")
        if not hasattr(model, 'inputs'):
            raise NotImplementedError("model must have `inputs` property")
        if not hasattr(model, 'outputs'):
            raise NotImplementedError("model must have `outputs` property")
        if not hasattr(model, 'states'):
            raise NotImplementedError("model must have `states` property")
        if not hasattr(model, 'simulate_to_threshold'):
            raise NotImplementedError("model must have `simulate_to_threshold` property")
        self.model = model

        self.parameters = deepcopy(self.default_parameters)
        self.parameters.update(kwargs)

    @abstractmethod
    def predict(self, state: UncertainData, future_loading_eqn: Callable, **kwargs) -> PredictionResults:
        """
        Perform a single prediction

        Parameters
        ----------
        state : UncertainData
            Distribution representing current state of the system
        future_loading_eqn : function (t, x) -> z
            Function to generate an estimate of loading at future time t, and state x
        options (optional, kwargs): configuration options\n
        Any additional configuration values. Additional keyword arguments may be supported that are are specific to the predictor, but include \n
            * t0: Starting time (s)
            * dt : Step size (s)
            * horizon : Prediction horizon (s)
            * events : List of events to be predicted (subset of model.events, default is all events)
            * event_strategy: str, optional
                Strategy for stopping evaluation. Default is 'first'. One of:\n
                * *first*: Will stop when first event in `events` list is reached.\n
                * *all*: Will stop when all events in `events` list have been reached

        Return
        ----------
        result from prediction, including: NameTuple
            * times (List[float]): Times for each savepoint such that inputs.snapshot(i), states.snapshot(i), outputs.snapshot(i), and event_states.snapshot(i) are all at times[i]            
            * inputs (Prediction): Inputs at each savepoint such that inputs.snapshot(i) is the input distribution (type UncertainData) at times[i]
            * states (Prediction): States at each savepoint such that states.snapshot(i) is the state distribution (type UncertainData) at times[i]
            * outputs (Prediction): Outputs at each savepoint such that outputs.snapshot(i) is the output distribution (type UncertainData) at times[i]
            * event_states (Prediction): Event states at each savepoint such that event_states.snapshot(i) is the event state distribution (type UncertainData) at times[i]
            * time_of_event (UncertainData): Distribution of predicted Time of Event (ToE) for each predicted event, represented by some subclass of UncertaintData (e.g., MultivariateNormalDist)
        """
        pass

    def __getitem__(self, arg):
        return self.parameters[arg]

    def __setitem__(self, key, value):
        self.parameters[key] = value
