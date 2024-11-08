# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from filterpy import kalman
from numpy import diag, array
from warnings import warn, catch_warnings, simplefilter

from progpy.state_estimators import state_estimator
from progpy.uncertain_data import MultivariateNormalDist, UncertainData

class UnscentedKalmanFilter(state_estimator.StateEstimator):
    """
    An Unscented Kalman Filter (UKF) for state estimation

    This class defines logic for performing an unscented kalman filter with a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to generate a state estimate and covariance matrix. 

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Args:
        model (PrognosticsModel):
            A prognostics model to be used in state estimation
            See: Prognostics Model Package
        x0 (UncertainData, model.StateContainer, or dict):
            Initial (starting) state, with keys defined by model.states \n
            e.g., x = ScalarData({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    Keyword Args:
        alpha (float, optional):
            UKF Scaling parameter
        beta (float, optional):
            UKF Scaling parameter
        kappa (float, optional):
            UKF Scaling parameter
        t0 (float, optional):
            Starting time (s)
        dt (float, optional):
            Maximum timestep for prediction in seconds. By default, the timestep dt is the difference between the last and current call of .estimate(). Some models are unstable at larger dt. Setting a smaller dt will force the model to take smaller steps; resulting in multiple prediction steps for each estimate step. Default is the parameters['dt']
            e.g., dt = 1e-2
        Q (list[list[float]], optional):
            Process Noise Matrix 
        R (list[list[float]], optional):
            Measurement Noise Matrix 
    """
    default_parameters = {
        'alpha': 1, 
        'beta': 0, 
        'kappa': -1,
    } 

    def __init__(self, model, x0, **kwargs):
        super().__init__(model, x0, **kwargs)

        self.__input = None
        self.x0 = x0
        # Saving for reduce pickling

        def measure(x):
            # Disable deprecation warnings for internal progpy code.
            x = model.StateContainer({key: value for (key, value) in zip(x0.keys(), x)})
            R_err = model.parameters['measurement_noise'].copy()
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                model.parameters['measurement_noise'] = dict.fromkeys(R_err, 0)
                z = model.output(x)
                model.parameters['measurement_noise'] = R_err
                return array(list(z.values())).ravel()

        if 'Q' not in self.parameters:
            self.parameters['Q'] = diag([1.0e-3 for _ in x0.keys()])

        def state_transition(x, dt):
            # Disable deprecation warnings for internal progpy code.
            x = model.StateContainer({key: value for (key, value) in zip(x0.keys(), x)})
            Q_err = model.parameters['process_noise'].copy()
            with catch_warnings():
                simplefilter("ignore", DeprecationWarning)
                model.parameters['process_noise'] = dict.fromkeys(Q_err, 0)
                x = model.next_state(x, self.__input, dt)
                return array(list(x.values())).ravel()

        num_states = len(x0.keys())
        num_measurements = model.n_outputs
        points = kalman.MerweScaledSigmaPoints(num_states, alpha=self.parameters['alpha'], beta=self.parameters['beta'], kappa=self.parameters['kappa'])
        self.filter = kalman.UnscentedKalmanFilter(num_states, num_measurements, self.parameters['dt'], measure, state_transition, points)
        
        if isinstance(x0, dict) or isinstance(x0, model.StateContainer):
            warn("Use UncertainData type if estimating filtering with uncertain data.")
            self.filter.x = array(list(x0.values()))
            self.filter.P = self.parameters['Q'] / 10
        elif isinstance(x0, UncertainData):
            x_mean = x0.mean
            self.filter.x = array(list(x_mean.values()))
            self.filter.P = x0.cov
        else:
            raise TypeError("TypeError: x0 initial state must be of type {{dict, UncertainData}}")

        if 'R' not in self.parameters:
            # Size of what's being measured (not output) 
            # This is determined by running the measure function on the first state
            self.parameters['R'] = diag([1.0e-3 for i in range(len(measure(self.filter.x)))])
        self.filter.Q = self.parameters['Q']
        self.filter.R = self.parameters['R']

    def estimate(self, t: float, u, z, **kwargs):
        """
        Perform one state estimation step (i.e., update the state estimate)

        Parameters
        ----------
        t : double
            Current timestamp in seconds (≥ 0.0)
            e.g., t = 3.4
        u : dict
            Measured inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : dict
            Measured outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']

        Keyword Args
        ------------
        dt : float, optional
            Maximum timestep for prediction in seconds. By default, the timestep dt is the difference between the last and current call of .estimate(). Some models are unstable at larger dt. Setting a smaller dt will force the model to take smaller steps; resulting in multiple prediction steps for each estimate step. Default is the parameters['dt']
            e.g., dt = 1e-2
        """
        assert t > self.t, "New time must be greater than previous"
        dt = kwargs.get('dt', self.parameters['dt'])
        dt = min(t - self.t, dt)
        self.__input = u
        while self.t < t:
            self.filter.predict(dt=dt)
            self.t += dt
        self.filter.update(array(list(z.values())))
    
    @property
    def x(self) -> MultivariateNormalDist:
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return MultivariateNormalDist(self.x0.keys(), self.filter.x, self.filter.P, _type=self.model.StateContainer)
