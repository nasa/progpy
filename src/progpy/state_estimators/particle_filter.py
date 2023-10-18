# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from filterpy.monte_carlo import residual_resample
import numpy as np
from numpy import array, empty, take, exp, max, take, float64
from scipy.stats import norm
from warnings import warn

from progpy.utils.containers import DictLikeMatrixWrapper

from . import state_estimator
from ..uncertain_data import UnweightedSamples, ScalarData, UncertainData


class ParticleFilter(state_estimator.StateEstimator):
    """
    Estimates state using a Particle Filter (PF) algorithm.

    This class defines logic for a PF using a Prognostics Model (see Prognostics Model Package). This filter uses measurement data with noise to estimate the state of the system using a particles. At each step, particles are predicted forward (with noise). Particles are resampled with replacement from the existing particles according to how well the particles match the observed measurements.

    The supported configuration parameters (keyword arguments) for UKF construction are described below:

    Args:
        model (PrognosticsModel):
            A prognostics model to be used in state estimation
            See: Prognostics Model Package
        x0 (UncertainData, model.StateContainer, or dict):
            Initial (starting) state, with keys defined by model.states \n
            e.g., x = ScalarData({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']

    Keyword Args:
        t0 (float, optional):
            Starting time (s)
        dt (float, optional): 
            Maximum timestep for prediction in seconds. By default, the timestep dt is the difference between the last and current call of .estimate(). Some models are unstable at larger dt. Setting a smaller dt will force the model to take smaller steps; resulting in multiple prediction steps for each estimate step. Default is the parameters['dt']
            e.g., dt = 1e-2
        num_particles (int, optional):
            Number of particles in particle filter
        resample_fcn (function, optional):
            Resampling function ([weights]) -> [indexes] e.g., filterpy.monte_carlo.residual_resample
    """
    default_parameters = {
            't0': -1e-99,  # practically 0, but allowing for a 0 first estimate
            'num_particles': None, 
            'resample_fcn': residual_resample,
        }

    def __init__(self, model, x0, **kwargs):
        super().__init__(model, x0, **kwargs)
        
        self._measure = model.output

        # Build array inplace
        if isinstance(x0, DictLikeMatrixWrapper) or isinstance(x0, dict):
            x0 = ScalarData(x0)
        elif not isinstance(x0, UncertainData):
            raise TypeError(f"x0 must be of type UncertainData or StateContainer, was {type(x0)}.")

        if self.parameters['num_particles'] is None and isinstance(x0, UnweightedSamples):
            sample_gen = x0  # Directly use samples passed in
            self.parameters['num_particles'] = len(x0)
        else:
            if self.parameters['num_particles'] is None:
                # Default to 100 particles
                self.parameters['num_particles'] = 100
            else:
                # Added to avoid float/int issues
                self.parameters['num_particles'] = int(self.parameters['num_particles'])
            sample_gen = x0.sample(self.parameters['num_particles'])
        samples = {k: array(sample_gen.key(k), dtype=float64) for k in x0.keys()}
        self.particles = model.StateContainer(samples)

        if 'R' in self.parameters:
            # For backwards compatibility
            warn("'R' is deprecated. Use 'measurement_noise' instead.", DeprecationWarning)
            self.parameters['measurement_noise'] = self.parameters['R']
        elif 'measurement_noise' not in self.parameters:
            self.parameters['measurement_noise'] = {key: 0.0 for key in model.outputs}
    
    def __str__(self):
        return "{} State Estimator".format(self.__class__)
        
    def estimate(self, t : float, u, z, dt = None):
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
        ------------
        dt : float, optional
            Maximum timestep for prediction in seconds. By default, the timestep dt is the difference between the last and current call of .estimate(). Some models are unstable at larger dt. Setting a smaller dt will force the model to take smaller steps; resulting in multiple prediction steps for each estimate step. Default is the parameters['dt']
            e.g., dt = 1e-2

        Note
        ----
        This method updates the state estimate stored in filt.x, but doesn't return the updated estimate. Call filt.x to get the updated estimate.
        """
        assert t > self.t, "New time must be greater than previous"
        if dt is None:
            dt = min(t - self.t, self.parameters['dt'])

        # Check Types
        if isinstance(u, dict):
            u = self.model.InputContainer(u)
        if isinstance(z, dict):
            z = self.model.OutputContainer(z)

        # Optimization
        particles = self.particles
        next_state = self.model.next_state
        apply_process_noise = self.model.apply_process_noise
        apply_limits = self.model.apply_limits
        output = self._measure
        # apply_measurement_noise = self.model.apply_measurement_noise
        noise_params = self.parameters['measurement_noise']
        num_particles = self.parameters['num_particles']
        # Check which output keys are present (i.e., output of measurement function)
        measurement_keys = output(self.model.StateContainer({key: particles[key][0] for key in particles.keys()})).keys()

        if self.model.is_vectorized:
            # Propagate particles state
            while self.t < t:
                dt_i = min(dt, t-self.t)
                particles = apply_process_noise(next_state(particles, u, dt_i), dt_i)
                self.particles = apply_limits(particles)
                self.t += dt_i

            # Get particle measurements
            zPredicted = output(self.particles)
        else:
            # Reserve space (for efficiency)
            zPredicted = {key: empty(num_particles) for key in measurement_keys}
            # Propagate and calculate weights
            for i in range(num_particles):
                t_i = self.t  # Used to mark time for each particle
                x = self.model.StateContainer({key: particles[key][i] for key in particles.keys()})
                while t_i < t:
                    dt_i = min(dt, t-t_i)
                    x = next_state(x, u, dt_i) 
                    x = apply_process_noise(x, dt_i)
                    x = apply_limits(x)
                    t_i += dt_i
                for key in particles.keys():
                    self.particles[key][i] = x[key]
                z = output(x)
                for key in measurement_keys:
                    zPredicted[key][i] = z[key]
            self.t = t

        # Calculate pdf values
        pdfs = array([norm(zPredicted[key], noise_params[key]).logpdf(z[key])
                      for key in zPredicted.keys()])

        # Calculate log weights
        log_weights = pdfs.sum(0)

        # Scale
        # We subtract the max log weights for numerical stability. 
        # Sometimes log weights can be a large negative value
        # when you exponentiate that value the computer will round the result to 0 for most of the weights (sometimes all of them) 
        # this causes problems when trying to sample from the particles. 
        # We shift them up by the max log weight (essentially making the max log weight 0) to help avoid that problem. 
        # When we normalize the weights by dividing by the sum of all the weights, that constant cancels out.
        max_log_weight = max(log_weights)
        scaled_weights = log_weights - max_log_weight

        # Convert to weights
        unnorm_weights = exp(scaled_weights)
        
        # Normalize
        total_weight = sum(unnorm_weights)
        self.weights = unnorm_weights / total_weight

        # Resample indices
        indexes = self.parameters['resample_fcn'](self.weights)

        # Resampled particles
        samples = [take(self.particles[state], indexes)
                   for state in self.particles.keys()]

        # Particles as a dictionary
        self.particles = self.model.StateContainer(array(samples))

    @property
    def x(self) -> UnweightedSamples:
        """
        Getter for property 'x', the current estimated state. 

        Example
        -------
        state = observer.x
        """
        return UnweightedSamples(self.particles, _type = self.model.StateContainer)
