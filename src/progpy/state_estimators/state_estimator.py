# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy

from ..uncertain_data import UncertainData


class StateEstimator(ABC):
    """
    Interface class for state estimators

    Abstract base class for creating state estimators that perform state estimation. Subclasses must implement this interface. Equivalent to "Observers" in NASA's Matlab Prognostics Algorithm Library

    Args:
        model (PrognosticsModel):
            A prognostics model to be used in state estimation
            See: Prognostics Model Package
        x0 (UncertainData, model.StateContainer, or dict):
            Initial (starting) state, with keys defined by model.states \n
            e.g., x = ScalarData({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    Keyword Args:
        t0 (float):
            Initial time at which prediction begins, e.g., 0
        dt (float):
            Maximum timestep for prediction in seconds. By default, the timestep dt is the difference between the last and current call of .estimate(). Some models are unstable at larger dt. Setting a smaller dt will force the model to take smaller steps; resulting in multiple prediction steps for each estimate step. Default is the parameters['dt']
            e.g., dt = 1e-2
        **kwargs: 
            See state-estimator specific documentation for specific keyword arguments.
    """

    default_parameters = {
        't0': -1e-10,
        'dt': float('inf')
    }

    def __init__(self, model, x0, **kwargs):
        # Check model
        if not hasattr(model, 'output'):
            raise NotImplementedError("model must have `output` method")
        if not hasattr(model, 'next_state'):
            raise NotImplementedError("model must have `next_state` method")
        if not hasattr(model, 'outputs'):
            raise NotImplementedError("model must have `outputs` property")
        if not hasattr(model, 'states'):
            raise NotImplementedError("model must have `states` property")
        self.model = model

        # Check x0
        for key in model.states:
            if key not in x0:
                raise KeyError("x0 missing state `{}`".format(key))
        
        # Process kwargs (configuration)
        self.parameters = deepcopy(StateEstimator.default_parameters)
        self.parameters.update(self.default_parameters)
        self.parameters.update(kwargs)

        if isinstance(self.parameters['t0'], int):
            self.parameters['t0'] = float(self.parameters['t0'])
        if isinstance(self.parameters['dt'], int):
            self.parameters['dt'] = float(self.parameters['dt'])

        if not isinstance(self.parameters['t0'], float):
            raise TypeError(f"t0 must be float, was {type(self.parameters['t0'])}")
        if not isinstance(self.parameters['dt'], float):
            raise TypeError(f"dt must be float, was {type(self.parameters['dt'])}")
        if self.parameters['dt'] <= 0:
            raise ValueError(f"dt must be positive, was {self.parameters['dt']}")

        self.t = self.parameters['t0']  # Initial Time

    @abstractmethod
    def estimate(self, t: float, u, z, **kwargs) -> None:
        """
        Perform one state estimation step (i.e., update the state estimate, filt.x)

        Args
        ----------
        t : float
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        u : InputContainer
            Measured inputs, with keys defined by model.inputs.
            e.g., u = m.InputContainer({'i':3.2}) given inputs = ['i']
        z : OutputContainer
            Measured outputs, with keys defined by model.outputs.
            e.g., z = m.OutputContainer({'t':12.4, 'v':3.3}) given outputs = ['t', 'v']

        Keyword Args
        -------------
            dt : float, optional
                Maximum timestep for prediction in seconds. By default, the timestep dt is the difference between the last and current call of .estimate(). Some models are unstable at larger dt. Setting a smaller dt will force the model to take smaller steps; resulting in multiple prediction steps for each estimate step. Default is the parameters['dt']
                e.g., dt = 1e-2
            **kwargs: 
                See state-estimator specific documentation for specific keyword arguments.

        Note
        ----
        This method updates the state estimate stored in filt.x, but doesn't return the updated estimate. Call filt.x to get the updated estimate.
        """

    @property
    @abstractproperty
    def x(self) -> UncertainData:
        """
        The current estimated state. 

        Example
        -------
        state = filt.x
        """
    
    def __getitem__(self, arg):
        return self.parameters[arg]

    def __setitem__(self, key, value):
        self.parameters[key] = value
