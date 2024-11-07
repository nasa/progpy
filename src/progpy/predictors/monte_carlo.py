# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from collections import abc
from copy import deepcopy
from typing import Callable
from progpy.sim_result import SimResult, LazySimResult
from progpy.uncertain_data import UnweightedSamples, UncertainData

from .prediction import UnweightedSamplesPrediction, PredictionResults
from .predictor import Predictor


class MonteCarlo(Predictor):
    """
    Class for performing a monte-carlo model-based prediction.

    A Predictor using the monte carlo algorithm. The provided initial states are simulated until either a specified time horizon is met, or the threshold for all simulated events is reached for all samples. A provided future loading equation is used to compute the inputs to the system at any given time point. 

    The following configuration parameters are supported (as kwargs in constructor or as parameters in predict method):
    
    Configuration Parameters
    ------------------------------
    n_samples : int, optional
        Default number of samples to use. If not specified, a default value is used. If state is type UnweightedSamples and n_samples is not provided, the provided unweighted samples will be used directly.
    save_freq : float, optional
        Default frequency at which results are saved (s).
    """

    __DEFAULT_N_SAMPLES = 100 # Default number of samples to use, if none specified and not UncertainData

    default_parameters = { 
        'n_samples': None,
        'event_strategy': 'all',
        'constant_noise': False
    }

    def predict(self, state: UncertainData, future_loading_eqn: Callable=None, events=None, **kwargs) -> PredictionResults:
        """
        Perform a single prediction

        Parameters
        ----------
        state : UncertainData 
            Distribution representing current state of the system
        future_loading_eqn : function (t, x=None) -> z, optional
            Function to generate an estimate of loading at future time t, and state x

        Keyword Arguments
        ------------------
        t0 : float, optional
            Initial time at which prediction begins, e.g., 0
        dt : float, optional
            Simulation step size (s), e.g., 0.1
        events : list[str], optional
            Events to predict (subset of model.events) e.g., ['event1', 'event2']
        event_strategy: str, optional
            Strategy for stopping evaluation. Default is 'first'. One of:\n
            * *first*: Will stop when first event in `events` list is reached.\n
            * *all*: Will stop when all events in `events` list have been reached
        horizon : float, optional
            Prediction horizon (s)
        n_samples : int, optional
            Number of samples to use. If not specified, a default value is used. If state is type UnweightedSamples and n_samples is not provided, the provided unweighted samples will be used directly.
        save_freq : float, optional
            Frequency at which results are saved (s)
        save_pts : list[float], optional
            Any additional savepoints (s) e.g., [10.1, 22.5]
        constant_noise : bool, optional
            If the same noise should be applied every step. Default: False

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
        if isinstance(state, dict) or isinstance(state, self.model.StateContainer):
            from progpy.uncertain_data import ScalarData
            state = ScalarData(state, _type=self.model.StateContainer)
        elif isinstance(state, UncertainData):
            state._type = self.model.StateContainer
        else:
            raise TypeError("state must be UncertainData, dict, or StateContainer")

        if future_loading_eqn is None:
            future_loading_eqn = lambda t, x=None: self.model.InputContainer({})

        params = deepcopy(self.parameters)  # copy parameters
        params.update(kwargs)  # update for specific run
        params['print'] = False
        params['progress'] = False
        # Remove event_strategy from params to not confuse simulate_to method call
        event_strategy = params.pop('event_strategy')

        if not isinstance(state, UnweightedSamples) and params['n_samples'] is None:
            # if not unweighted samples, some sample number is required, so set to default.
            params['n_samples'] = MonteCarlo.__DEFAULT_N_SAMPLES
        elif isinstance(state, UnweightedSamples) and params['n_samples'] is None:
            params['n_samples'] = len(state)  # number of samples is from provided state

        if events is None:
            if 'events' in params and params['events'] is not None:
                # Set at a predictor construction
                events = params['events']
            else:
                # Otherwise, all events
                events = self.model.events
        
        if not isinstance(events, (abc.Iterable)) or isinstance(events, (dict, bytes)):
            # must be string or list-like (list, tuple, set)
            # using abc.Iterable adds support for custom data structures
            # that implement that abstract base class
            raise TypeError(f'`events` must be a single event string or list of events. Was unsupported type {type(events)}.')
        if len(events) == 0 and 'horizon' not in params:
            raise ValueError("If specifying no event (i.e., simulate to time), must specify horizon")
        if isinstance(events, str):
            # A single event
            events = [events]
        if not all([key in self.model.events for key in events]):
            raise ValueError("`events` must be event names")
        if not isinstance(events, list):
            # Change to list because of the limits of jsonify
            events = list(events)

        if 'events' in params: 
            # Params is provided as a argument in construction
            # Remove it so it's not passed to simulate_to*
            del params['events']

        # Sample from state if n_samples specified or state is not UnweightedSamples (Case 2)
        # Or if is Unweighted samples, but there are the wrong number of samples (Case 1)
        if (
            (isinstance(state, UnweightedSamples) and len(state) != params['n_samples'])  # Case 1
            or not isinstance(state, UnweightedSamples)):  # Case 2
            state = state.sample(params['n_samples'])

        es_eqn = self.model.event_state
        tm_eqn = self.model.threshold_met
        simulate_to_threshold = self.model.simulate_to_threshold

        time_of_event_all = []
        last_states = []
        times_all = []
        inputs_all = []
        states_all = []
        outputs_all = []
        event_states_all = []

        if params['constant_noise']:
            # Save loads
            process_noise = self.model['process_noise']
            process_noise_dist = self.model.parameters.get('process_noise_dist', 'normal')

        # Perform prediction
        t0 = params.get('t0', 0)
        HORIZON = params.get('horizon', float('inf'))  # Save the horizon to be used later
        for x in state:
            if params['constant_noise']:
                # Calculate process noise
                x_noise = self.model.apply_process_noise(x.copy(), 1)
                x_noise = self.model.StateContainer({key: x_noise[key] - x[key] for key in x.keys()})

                self.model['process_noise'] = x_noise
                self.model['process_noise_dist'] = 'constant'

            first_output = self.model.output(x)
            
            time_of_event = {}
            last_state = {}

            params['t0'] = t0
            params['x'] = x
            params['horizon'] = HORIZON  # reset to initial horizon

            if 'save_freq' in params and not isinstance(params['save_freq'], tuple):
                params['save_freq'] = (params['t0'], params['save_freq'])
            
            if len(events) == 0:  # Predict to time
                (times, inputs, states, outputs, event_states) = simulate_to_threshold(
                    future_loading_eqn,
                    first_output,
                    events=[],
                    **params
                )
            else:
                events_remaining = events.copy()

                times = []
                inputs = SimResult(_copy=False)
                states = SimResult(_copy=False)
                outputs = LazySimResult(fcn=self.model.output, _copy=False)
                event_states = LazySimResult(fcn=es_eqn, _copy=False)

                # Non-vectorized prediction
                while len(events_remaining) > 0:  # Still events to predict
                    # Since horizon is relative to t0 (the simulation starting point),
                    # we must subtract the difference in current t0 from the initial (i.e., prediction t0)
                    # each subsequent simulation
                    params['horizon'] = HORIZON - (params['t0'] - t0)
                    (t, u, xi, z, es) = simulate_to_threshold(
                        future_loading_eqn,
                        first_output,
                        events=events_remaining,
                        **params
                    )

                    # Add results
                    times.extend(t)
                    inputs.extend(u)
                    states.extend(xi)
                    outputs.extend(z, _copy=False)
                    event_states.extend(es, _copy=False)

                    # Get which event occurs
                    t_met = tm_eqn(states[-1])
                    t_met = {key: t_met[key] for key in events_remaining}  # Only look at remaining keys

                    try:
                        event = list(t_met.keys())[list(t_met.values()).index(True)]
                    except ValueError:
                        # no event has occured - hit horizon
                        for event in events_remaining:
                            time_of_event[event] = None
                            last_state[event] = None
                        break

                    # An event has occured
                    time_of_event[event] = times[-1]
                    if event_strategy == 'all':
                        events_remaining.remove(event)  # No longer an event to predict to
                    elif event_strategy in ('first', 'any'):
                        events_remaining = []
                    else:
                        raise ValueError(f"Invalid value for `event_strategy`: {event_strategy}. Should be either 'all' or 'first'")

                    # Remove last state (event)
                    params['t0'] = times.pop()
                    inputs.pop()
                    params['x'] = states.pop()
                    last_state[event] = params['x'].copy()
                    outputs.pop()
                    event_states.pop()
            
            # Add to "all" structures
            if len(times) > len(times_all):  # Keep longest
                times_all = times
            inputs_all.append(inputs)
            states_all.append(states)
            outputs_all.append(outputs)
            event_states_all.append(event_states)
            time_of_event_all.append(time_of_event)
            last_states.append(last_state)

            # Reset noise
            if params['constant_noise']:
                self.model['process_noise'] = process_noise
                self.model['process_noise_dist'] = process_noise_dist
              
        inputs_all = UnweightedSamplesPrediction(times_all, inputs_all)
        states_all = UnweightedSamplesPrediction(times_all, states_all)
        outputs_all = UnweightedSamplesPrediction(times_all, outputs_all)
        event_states_all = UnweightedSamplesPrediction(times_all, event_states_all)
        time_of_event = UnweightedSamples(time_of_event_all)

        # Transform final states:
        time_of_event.final_state = {
            key: UnweightedSamples(
                    [sample[key] for sample in last_states],
                    _type=self.model.StateContainer
                ) for key in time_of_event.keys()
        }

        return PredictionResults(
            times_all,
            inputs_all,
            states_all,
            outputs_all,
            event_states_all,
            time_of_event
        )
