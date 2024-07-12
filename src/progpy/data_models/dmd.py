# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from numbers import Number
import numpy as np
from scipy.interpolate import interp1d
import types
from warnings import warn

from progpy.exceptions import warn_once
from progpy.sim_result import SimResult, LazySimResult
from progpy import LinearModel, PrognosticsModel
from progpy.data_models import DataModel


class DMDModel(LinearModel, DataModel):
    """
    .. versionadded:: 1.3.0

    A subclass of :py:class:`progpy.LinearModel` and :py:class:`progpy.data_models.DataModel` that uses Dynamic Mode Decomposition to simulate a system throughout time.
    
    Given an initial :term:`state` of the system and the expected :term:`input` throughout time, this class defines a model that can approximate the dynamics of the system throughout time until threshold is met. This model can be fully data-driven (using from_data) or a :term:`surrogate` of another model (using from_model) where internal states of a high-fidelity model augment the purely data-driven method. 

    Args
    ---------
        dmd_matrix : np.ndarray
            Matrix used by DMD

    Keyword args
    ------------
        input_keys : list[str]
            List of input keys
        dt : float
            Time step
        output_keys : list[str]
            List of output keys
        x0 : StateContainer
            Initial state of the system
        state_keys : list[str]
            List of state keys
        event_keys : list[str]
            List of event keys

    See Also
    ---------

        :py:class:`LinearModel`

        :py:class:`DataModel`

    Note
    -------
    DMD does not generate accurate approximations for all models, especially highly non-linear sections, and can be sensitive to the training data time step. 

    In general, the approximation is less accurate if the DMD matrix is unstable. 
    Additionally, this implementation does not yet include all functionalities of DMD (e.g. reducing the system's dimensions through SVD). Further functionalities will be included in future releases. \n
    """

    A = None
    B = None
    C = None
    F = None

    def __getnewargs__(self):
        return (self.dmd_matrix, )

    def __init__(self, dmd_matrix, *_, **kwargs):
        if isinstance(dmd_matrix, PrognosticsModel):
            # Initialized in __new__
            # Remove in future version
            return

        if 'input_keys' not in kwargs:
            raise ValueError('input_keys must be specified')
        if 'dt' not in kwargs:
            raise ValueError('dt must be specified')
        if 'output_keys' not in kwargs:
            raise ValueError('output_keys must be specified')
        if 'x0' not in kwargs:
            raise ValueError('x0 must be specified')

        params = {
            'state_keys': [],
            'event_keys': []
        }
        params.update(**kwargs)

        self.inputs = params['input_keys']
        n_inputs = len(params['input_keys'])
        
        self.states = params['state_keys'] + params['output_keys'] + params['event_keys']
        n_states = len(params['state_keys'])
        
        self.outputs = params['output_keys']
        n_outputs = len(params['output_keys'])
        
        has_overwritten_threshold_eqn = hasattr(self, 'events')
        if has_overwritten_threshold_eqn:
             # To support overridding to add custom threshold equation
            n_events = 0
        else:
            self.events = params['event_keys']
            n_events = len(self.events)


        n_total = n_states + n_outputs + n_events

        self.A = dmd_matrix[:, 0:n_total]
        self.B = np.vstack(dmd_matrix[:, n_total:n_total+n_inputs])
        self.C = np.zeros((n_outputs, n_total))
        for iter1 in range(n_outputs):
            self.C[iter1, n_states+iter1] = 1
        if not has_overwritten_threshold_eqn:
            self.F = np.zeros((n_events, n_total))
        for iter2 in range(n_events):
            self.F[iter2, n_states+n_outputs+iter2] = 1

        super().__init__(**params)
        
        self.dt = params['dt']
        self.dmd_matrix = dmd_matrix
        self.parameters['dmd_matrix'] = dmd_matrix  # This simplifies pickling (all data in parameters)
        
        if 'x0' in self.parameters and not isinstance(self.parameters['x0'], self.StateContainer):
            self.parameters['x0'] = self.StateContainer(params['x0'])

    @classmethod
    def from_data(cls, times, inputs, outputs, states=None, event_states=None, **kwargs):
        """
        Create a DMD model from data

        Args:
            times (list[list]):
                list of input data for use in data. Each element is the times for a single run of size (n_times)
            inputs (list[np.array]):
                list of :term:`input` data for use in data. Each element is the inputs for a single run of size (n_times, n_inputs)
            outputs (list[np.array]):
                list of :term:`output` data for use in data. Each element is the outputs for a single run of size (n_times, n_outputs)
            states (list[np.array], optional):
                list of :term:`state` data for use in data. Each element is the states for a single run of size (n_times, n_states)
            event_states (list[np.array], optional):
                list of :term:`event state` data for use in data. Each element is the event states for a single run of size (n_times, n_event_states)

        Keyword Args:
            trim_data_to (float, optional):
                Fraction (0-1) of data resulting from :py:func:`progpy.PrognosticsModel.simulate_to_threshold` used to train DMD surrogate model
                e.g. if trim_data_to = 0.7 and the simulated data spans from t=0 to 100, the surrogate model is trained on the data from t=0 to 70 \n   
                Note: To trim data to a set time, use the 'horizon' parameter
            stability_tol (float, optional):
                Value that determines the tolerance for DMD matrix stability
            training_noise (float, optional):
                Noise added to the training data sampled from a standard normal distribution with standard deviation of training_noise. Adding noise to the training data results in a slight perturbation that removes any linear dependencies among the data
            input_keys (list[str], optional):
                List of :term:`input` keys
            state_keys (list[str], optional):
                List of :term:`state` keys
            output_keys (list[str], optional):
                List of :term:`output` keys
            event_keys (list[str], optional):
                List of :term:`event` keys
        
        Additionally, other keyword arguments from :py:func:`progpy.PrognosticsModel.simulate_to_threshold`

        Attributes:
            dmd_matrix (np.array): Dynamic Mode Decomposition Matrix
            dt (float): Time step of data

        Returns:
            DMDModel: Model generated from data
        """
        # Configure
        config = {  # Defaults
            'trim_data_to': 1,
            'stability_tol': 1e-05,
            'training_noise': 1e-05,
            'input_keys': None,
            'state_keys': None,
            'output_keys': None,
            'event_keys': None,
            'dt': None
        }
        config.update(kwargs)

        # Input validation
        if not isinstance(config['trim_data_to'], Number) or config['trim_data_to'] > 1 or config['trim_data_to'] <= 0:
            raise ValueError("Invalid 'trim_data_to' input value, must be between 0 and 1.")
        if not isinstance(config['stability_tol'], Number) or config['stability_tol'] < 0:
            raise ValueError(f"Invalid 'stability_tol' input value {config['stability_tol']}, must be a positive number.")
        if not isinstance(config['training_noise'], Number) or config['training_noise'] < 0:
            raise ValueError(f"Invalid 'training_noise' input value {config['training_noise']}, must be a positive number.")
        DataModel.check_data_format(inputs, outputs, states, event_states)
        if isinstance(config['dt'], list):
            # This means one dt for each run
            # Use mean
            config['dt'] = sum(config['dt'])/len(config['dt'])
        elif config['dt'] is None:
            # Use times from data - calculate mean dt
            dts = [t[j+1] - t[j] for t in times for j in range(len(t)-1)]
            config['dt'] = sum(dts)/len(dts)
        if 'save_freq' not in config:
            config['save_freq'] = config['dt']
        elif isinstance(config['save_freq'], list):
            # This means one save_freq for each run
            # Use mean
            config['save_freq'] = sum(config['save_freq'])/len(config['save_freq'])

        # Handle Keys
        if config['input_keys'] is None:
            for u in inputs:
                if isinstance(u, SimResult):
                    config['input_keys'] = list(u[0].keys())
                    break
            if config['input_keys'] is None:
                # Wasn't able to fill it in
                config['input_keys'] = [f'u{i}' for i in range(len(inputs[0][0]))]
        if config['state_keys'] is None:
            if states is None:
                config['state_keys'] = []
            else:
                for x in states:
                    if isinstance(x, SimResult):
                        config['state_keys'] = list(x[0].keys())
                        break
                if config['state_keys'] is None:
                    # Wasn't able to fill it in
                    config['state_keys'] = [f'x{i}' for i in range(len(states[0][0]))]
        if config['output_keys'] is None:
            for z in outputs:
                if isinstance(z, SimResult):
                    config['output_keys'] = list(z[0].keys())
                    break
            if config['output_keys'] is None:
                # Wasn't able to fill it in
                config['output_keys'] = [f'z{i}' for i in range(len(outputs[0][0]))]
        if config['event_keys'] is None:
            if event_states is None:
                config['event_keys'] = []
            else:
                for es in event_states:
                    if isinstance(es, SimResult):
                        config['event_keys'] = list(es[0].keys())
                        break
                if config['event_keys'] is None:
                    # Wasn't able to fill it in
                    config['event_keys'] = [f'es{i}' for i in range(len(event_states[0][0]))]

        for state_key in config['state_keys']:
            if state_key in config['input_keys'] or state_key in config['output_keys'] or state_key in config['event_keys']:
                config['state_keys'].remove(state_key)
                warn(f"State value '{state_key}' is duplicated in inputs, outputs, or events; duplicate has been removed.")

        for input_key in config['input_keys']:
            if input_key in config['output_keys'] or input_key in config['event_keys']:
                warn(f"Input value '{input_key}' is duplicated in outputs or events")

        for output_key in config['output_keys']:
            if output_key in config['event_keys']:
                warn(f"Output value '{output_key}' is duplicated in events")

        n_inputs = len(config['input_keys'])
        n_states = len(config['state_keys'])
        n_outputs = len(config['output_keys'])
        n_events = len(config['event_keys'])

        # Initialize lists to hold individual matrices
        x_list = []
        xprime_list = []
        time_list = []

        # Train DMD model
        n_runs = len(inputs)
        for run in range(n_runs):
            t = times[run]
            u = inputs[run]
            z = outputs[run]

            if len(u) != len(z):
                raise ValueError(f"Must have same number of steps for inputs and outputs in a single run. Not true for run {run}")
            if len(u) == 0:
                raise ValueError(f"Each run must have at least one timestep, not true for Run {run}")

            if isinstance(u, SimResult):
                u = u.to_numpy(config['input_keys'])

            if states != None:
                x = states[run]
                if len(x) != len(u):
                    raise ValueError(f"Must have same number of steps for inputs, states, and outputs in a single run. Not true for states in run {run}")
                if isinstance(x, SimResult):
                    x = x.to_numpy(config['state_keys'])
            else:
                x = np.array([[] for _ in u])

            if isinstance(z, SimResult):
                z = z.to_numpy(config['output_keys'])

            if event_states is not None:
                es = event_states[run]
                if len(es) != len(u):
                    raise ValueError(f"Must have same number of steps for inputs, event_states, and outputs in a single run. Not true for event_states in run {run}")
                if isinstance(es, SimResult):
                    es = es.to_numpy(config['event_keys'])
            else:
                es = np.array([[] for _ in u])

            # Trim
            if config['trim_data_to'] != 1:
                index = int(len(u)*config['trim_data_to'])
                t = t[:index]
                u = u[:index]
                x = x[:index]
                z = z[:index]
                es = es[:index]
            
            # Interpolate
            # This is done when dt and save_freq are not equal. 
            # dt is the frequency at which simulation is performed to generate data,
            # while save_freq is the frequency for the dmd_matrix
            if config['save_freq'] != config['dt'] or config['dt'] is None:
                if isinstance(config['save_freq'], (tuple, list, np.ndarray)):
                    # Tuple used to specify start and frequency
                    t_step = config['save_freq'][1]
                    # Use starting time or the next multiple
                    t_start = config['save_freq'][0]
                    start = max(t_start, t[0])
                else:
                    # Otherwise - start is t0
                    t_step = config['save_freq']
                    start = t[0]
                    
                t_new = np.arange(start, t[-1], t_step)
                u = interp1d(t, u, axis=0)(t_new)
                if n_states != 0:
                    x = interp1d(t, x, axis=0)(t_new)
                z = interp1d(t, z, axis=0)(t_new)
                if n_events != 0:
                    # Optimization - avoid interpolation of empty array
                    es = interp1d(t, es, axis=0)(t_new)
                t = t_new  # Reset time to new timestep

            # Apply training noise
            if config['training_noise'] != 0:
                n_steps = len(u)
                u += np.random.randn(n_steps, n_inputs)*config['training_noise']
                if n_states != 0:
                    x += np.random.randn(n_steps, n_states)*config['training_noise']
                z += np.random.randn(n_steps, n_outputs)*config['training_noise']
                if n_events != 0:  
                    # Optimization - avoid generation for empty array
                    es += np.random.randn(n_steps, n_events)*config['training_noise']

            # Initialize DMD matrices
            time_list.append(t)
            x_mat = np.hstack((x, z, es, u)).T
            x_list.append(x_mat[:, :-1])
            xprime_list.append(x_mat[:-n_inputs if n_inputs != 0 else None, 1:])

        # Format training data for DMD and solve for matrix A, in the form X' = AX
        print('Generate DMD Surrogate Model')

        if 'x0' not in config:
            config['x0'] = np.mean(np.array([x[:-n_inputs if n_inputs != 0 else None, 0] for x in x_list]), axis=0)[np.newaxis].T
     
        # Convert lists of datasets into arrays, sequentially stacking data in the horizontal direction
        x_mat = np.hstack((x_list[:]))
        xprime_mat = np.hstack((xprime_list[:]))

        # Calculate DMD matrix using the Moore-Penrose pseudo-inverse:
        dmd_matrix = np.dot(xprime_mat, np.linalg.pinv(x_mat))

        # Check for stability of dmd_matrix
        eig_val, _ = np.linalg.eig(dmd_matrix[:, 0:-n_inputs if n_inputs > 0 else None])
        
        if sum(eig_val > 1) != 0:
            for eig_val_i in eig_val:
                if eig_val_i > 1 and (eig_val_i-1) > config['stability_tol']:
                    warn("The DMD matrix is unstable, may result in poor approximation.")

        del config['dt']

        # Build Model
        return cls(
            dmd_matrix,
            dt=config['save_freq'],
            **config)
        
    @classmethod
    def from_model(cls, m, load_functions, **kwargs):
        process_noise_temp = {key: 0 for key in m.events}
        config = {
            'add_dt': False,
            'process_noise': {**m.parameters['process_noise'], **m.parameters['measurement_noise'], **process_noise_temp},
            'measurement_noise': m.parameters['measurement_noise'],
            'process_noise_dist': m.parameters.get('process_noise_dist', 'normal'),
            'measurement_noise_dist': m.parameters.get('measurement_noise_dist', 'normal')
        }  # Defaults specific to this class
        config.update(kwargs)
        # Build Model
        m_dmd = super().from_model(m, load_functions, **config)

        # Override initialize for dmd to use base model (since we have it in the original model)
        def init_dmd(self, u=None, z=None):
            x = m.initialize(u,z)
            x.update(m.output(x))
            x.update(m.event_state(x))

            return self.StateContainer(x)
        m_dmd.initialize = types.MethodType(init_dmd, m_dmd)

        return m_dmd

    def next_state(self, x, u, _):
        x.matrix = np.matmul(self.A, x.matrix) + self.E
        if self.B.shape[1] != 0:
           x.matrix += np.matmul(self.B, u.matrix)
        
        return x   

    def simulate_to_threshold(self, future_loading_eqn, first_output=None, events=None, **kwargs):
        # Save keyword arguments same as DMD training for approximation
        kwargs_sim = kwargs.copy()
        kwargs_sim['save_freq'] = self.dt
        kwargs_sim['dt'] = self.dt

        # Simulate to threshold at DMD time step
        results = super().simulate_to_threshold(future_loading_eqn, first_output, events, **kwargs_sim)
        
        # Interpolate results to be at user-desired time step
        if 'dt' in kwargs:
            warn_once("dt is not used in DMD approximation")

        # Default parameters
        config = {
            'dt': None,
            'save_freq': None,
            'save_pts': []
        }
        config.update(kwargs)

        if (config['save_freq'] == self.dt or
            (isinstance(config['save_freq'], tuple) and
                config['save_freq'][0] % self.dt < 1e-9 and
                config['save_freq'][1] == self.dt)
            ) and config['save_pts'] == []:
            # In this case, the user wants what the DMD approximation returns
            return results

        # In this case, the user wants something different than what the DMD 
        # approximation retuns, so we must interpolate.
        # Define time vector based on user specifications
        time_basic = [results.times[0], results.times[-1]]
        time_basic.extend(config['save_pts'])
        if config['save_freq'] is not None:
            if isinstance(config['save_freq'], tuple):
                # Tuple used to specify start and frequency
                t_step = config['save_freq'][1]
                # Use starting time or the next multiple
                t_start = config['save_freq'][0]
                start = max(t_start, results.times[0] - (results.times[0]-t_start) % t_step)
                time_array = np.arange(start+t_step, results.times[-1], t_step)
            else:
                time_array = np.arange(results.times[0]+config['save_freq'], results.times[-1], config['save_freq'])
            time_basic.extend(time_array.tolist())
        time_interp = sorted(time_basic)

        # Interpolate States
        states_dict_temp = {}
        for states_name in self.states:
            states_list_temp = [results.states[iter1a][states_name] for iter1a in range(len(results.states))]
            states_dict_temp[states_name] = interp1d(results.times, states_list_temp)(time_interp)
        states_interp = [
            self.StateContainer({key: state[i] for key, state in states_dict_temp.items()})
            for i in range(len(time_interp))
        ]
            
        # Interpolate Inputs
        inputs_dict_temp = {}
        for inputs_name in self.inputs:
            inputs_list_temp = [results.inputs[iter1a][inputs_name] for iter1a in range(len(results.inputs))]
            inputs_dict_temp[inputs_name] = interp1d(results.times, inputs_list_temp)(time_interp)
        inputs_interp = [
            self.InputContainer({key: input[i] for key, input in inputs_dict_temp.items()}) for i in range(len(time_interp))
        ]

        states = SimResult(time_interp, states_interp)
        inputs = SimResult(time_interp, inputs_interp)
        outputs = LazySimResult(self.output, time_interp, states_interp)
        event_states = LazySimResult(self.event_state, time_interp, states_interp)

        return self.SimulationResults(
            time_interp,
            inputs,
            states,
            outputs,
            event_states
        )

# Kept for backwards compatability
SurrogateDMDModel = DMDModel
