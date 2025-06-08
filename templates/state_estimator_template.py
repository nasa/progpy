# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from progpy import state_estimators

# Replace the following with whatever form of UncertainData you would like to use to represent state
from progpy.uncertain_data import UncertainData, ScalarData


class TemplateStateEstimator(state_estimators.StateEstimator):
    """
    Template for State Estimator
    """

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    default_parameters = {  # Default Parameters, used as config
        "Example Parameter": 0.0,
        "t0": 0.0,  # Initial timestamp
    }

    def __init__(self, model, x0, **kwargs):
        """
        Constructor (optional)
        """
        super().__init__(model, x0, **kwargs)
        # ADD PARAMETER CHECKS HERE
        # e.g., self.parameters['some_value'] < 0

        # ADD ANY STATE ESTIMATOR INITIALIZATION LOGIC

    def estimate(self, t, u, z) -> None:
        """
        Perform one state estimation step (i.e., update the state estimate)

        Parameters
        ----------
        t : double
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        u : dict
            Applied inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Measured outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """
        # REPLACE WITH UPDATE STATE ESTIMATION

        # Note, returns none. State is accessed using the property state_estimator.x

    @property
    def x(self) -> UncertainData:
        """
        Getter for property 'x', the current estimated state.

        Example
        -------
        state = observer.x
        """
        # REPLACE THE FOLLOWING WITH THE LOGIC TO CONSTRUCT/RETURN THE STATE

        # Here we're using ScalarData, but the state could be represented by any other type of UncertainData (e.g., MultivariateNormalDist)
        x = ScalarData(
            self.model.StateContainer({key: 0.0 for key in self.model.states})
        )

        return x
