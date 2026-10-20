# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from copy import deepcopy

from progpy.predictors import Predictor, Prediction, PredictionResults

# Replace the following with whatever form of UncertainData you would like to use to represent ToE
from progpy.uncertain_data import ScalarData


class TemplatePredictor(Predictor):
    """
    Template class for performing model-based prediction
    """

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    default_parameters = {  # Default Parameters, used as config for UKF
        "Example Parameter": 0.0
    }

    def __init__(self, model, **kwargs):
        """
        Constructor (optional)
        """
        super().__init__(model, **kwargs)
        # ADD PARAMETER CHECKS HERE
        # e.g., self.parameters['some_value'] < 0

        # INITIALIZE PREDICTOR

    def predict(self, state, future_loading_eqn, **kwargs):
        """
        Perform a single prediction

        Parameters
        ----------
        state : UncertainData
            Estimate of the state at the time of prediction, reprecented by UncertainData
        future_loading_eqn : function (t, x) -> z
            Function to generate an estimate of loading at future time t and state z
        options : dict, optional
            Dictionary of any additional configuration values. See default parameters, above

        Returns (namedtuple)
        -------
        times : List[float]
            Times for each savepoint such that inputs.snapshot(i), states.snapshot(i), outputs.snapshot(i), and event_states.snapshot(i) are all at times[i]
        inputs : Prediction
            Inputs at each savepoint such that inputs.snapshot(i) is the input distribution (type UncertainData) at times[i]
        states : Prediction
            States at each savepoint such that states.snapshot(i) is the state distribution (type UncertainData) at times[i]
        outputs : Prediction
            Outputs at each savepoint such that outputs.snapshot(i) is the output distribution (type UncertainData) at times[i]
        event_states : Prediction
            Event states at each savepoint such that event_states.snapshot(i) is the event state distribution (type UncertainData) at times[i]
        time_of_event : UncertainData
            Distribution of predicted Time of Event (ToE) for each predicted event, represented by some subclass of UncertaintData (e.g., MultivariateNormalDist)
        """
        params = deepcopy(self.parameters)  # copy default parameters
        params.update(kwargs)

        # PERFORM PREDICTION HERE, REPLACE THE FOLLOWING LISTS

        # Times of each savepoint (specified by savepts and save_freq)
        times = []  # array of float (e.g., [0.0, 0.5, 1.0, ...])

        # Inputs, State, Outputs, and Event States at each savepoint are stored by type Prediction
        # Replace [] with estimates of the appropriate property in the form of a subclass of UncertainData (e.g, ScalarData)
        inputs = Prediction(times, [])
        states = Prediction(times, [])
        outputs = Prediction(times, [])
        event_states = Prediction(times, [])

        # Time of event is represented by some type of UncertainData (e.g., MultivariateNormalDist)
        time_of_event = ScalarData({"event1": 748, "event2": 300})
        # Save the final state when each event occurs like this, with each final state represented by an UncertainData object (e.g., MultivariateNormalDist)
        # time_of_event.final_state = {'event1': ScalarData({'state1': 10, 'state2': 20}), 'event2': ScalarData({'state1': 12, 'state2': 18})}

        return PredictionResults(
            times, inputs, states, outputs, event_states, time_of_event
        )
