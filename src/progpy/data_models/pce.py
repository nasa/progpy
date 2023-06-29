# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
# This ensures that the directory containing examples is in the python search directories 

import chaospy as cp
import numpy as np
import scipy as sp

from progpy.data_models import DataModel


class PolynomialChaosExpansion(DataModel):
    """
    .. versionadded:: 1.5.0

    Polynomial Chaos Expansion direct data model. Uses chaospy to generate a polynomial chaos expansion from data or a model to learn the relationship between future inputs and time of event.

    Generally this is used as a :term:`surrogate` for a :term:`model` that is too expensive to simulate. This is done using the :meth:`PolynomialChaosExpansion.from_model` method. The model is used to generate data, which is then used to train the polynomial chaos expansion. The polynomial chaos expansion can then be used to predict the time of event for future inputs.

    Args:
        models (List[chaospy.Poly]):
            Polynomial chaos expansion models (one for each event)
        times (list[float]):
            List of times to use for the polynomial chaos expansion
        input_keys (list[str]):
            List of input keys for the inputs

    Keyword Args:
        event_keys (list[str], optional):
            List of event keys for the events. If not provided, will be generated as e0, e1, e2, etc.

    See Also:
        :class:`progpy.data_models.DataModel`
        PolynomialChaosExpansion.from_data
        PolynomialChaosExpansion.from_model

    Note:
        The generated model is only valid for the intial state at which the data was generated. If the initial state is different, the model will not be valid.
    """
    def __init__(self, models, times, input_keys, **kwargs):
        self.inputs = input_keys
        self.states = []
        self.outputs = []
        self.events = kwargs.get('event_keys', [f'e{i}' for i in range(len(models))])

        super().__init__(**kwargs)
        self.parameters['models'] = models
        self.parameters['times'] = times

    def time_of_event(self, x, future_loading_eqn, **kwargs) -> dict:
        loading = np.reshape(np.array([future_loading_eqn(t, x).matrix for t in self.parameters['times']]).T, (len(self.inputs * len(self.parameters['times']))))
        return {key: model(*loading) for key, model in zip(self.events, self.parameters['models'])}

    @classmethod
    def from_data(cls, times, inputs, time_of_event, input_keys, **kwargs):
        """
        Create a PolynomialChaosExpansion from data.

        Args:
            times (list[float]):
                list of times data for use in data. Each element is the time such that inputs[i] is the inputs at time[i]
            inputs (np.array):
                list of :term:`input` data for use in data. Each  eelement is the inputs for a single run of size (n_samples, n_inputs*n_times)
            time_of_event (np.array):
                Array of time of event data for use in data. Each element is the time of event for a single run of size (n_samples, n_events)
            input_keys (list[str]):
                List of input keys for the inputs

        Keyword Args:
            J (chaospy.Distribution, optional):
                Joint distribution to sample from. Must include distribution for each timepoint for each input [u0_t0, u0_t1, ..., u1_t0, ...]. If not included, input_dists must be provided
            input_dists (list[chaospy.Distribution], optional):
                List of chaospy distributions for each input for each timepoint
            order (int, optional):
                Order of the polynomial chaos expansion
        """
        default_params = {
            'J': None,
            'input_dists': None,
            'order': 2,
        }
        params = default_params.copy()
        params.update(kwargs)
        
        if len(input_keys) == 0:
            raise ValueError('Must provide at least one input key, was empty')
        if params['J'] is None and params['input_dists'] is None:
            raise ValueError('Either J or input_dists must be provided')
        if params['J'] is None:
            params['J'] = cp.J(*params['input_dists'])
        if params['order'] < 1:
            raise ValueError(f'order must be greater than 0, was {params["order"]}')
        if len(time_of_event) == 0:
            raise ValueError('Time of event must include at least one run')
        if len(times) == 0:
            raise ValueError('Times must include at least one time')
        if len(inputs) == 0:
            raise ValueError('Inputs must include at least one run')
        if len(time_of_event) != len(inputs):
            raise ValueError('There must be the same number of runs for inputs and time of event')

        n_events = len(time_of_event[0])
        if n_events == 0:
            raise ValueError('There must be at least one event to train an PCE model')

        # Train
        inputs = inputs.T
        expansion = cp.generate_expansion(order=params['order'], dist=params['J'])  # Order=2 is the only hyperparameter
        surrogates = [
            cp.fit_regression(expansion, inputs, toe_i) for toe_i in time_of_event.T
        ]

        return cls(surrogates, times=times, input_keys=input_keys, **params)

    @classmethod
    def from_model(cls, m, x, input_dists, times, **kwargs):
        """
        Create a PolynomialChaosExpansion from a model.

        Args:
            m (Model):
                Model to create PolynomialChaosExpansion from
            x (StateContainer):
                Initial state to use for simulation
            input_dists (dict[key, chaospy.Distribution]):"
                List of chaospy distributions for each input
            times (list[float]):
                List of coordinates along the time axis used to estimate the expansion coefficients (collocation points).

        Keyword Args:
            N (int, optional):
                Number of samples to use for training
            dt (float, optional):
                Time step to use for simulation
            order (int, optional):
                Order of the polynomial chaos expansion
        """
        default_params = {
            'N': 1000,
            'dt': 0.1,
        }
        params = default_params.copy()
        params.update(kwargs)

        if params['N'] < 1:
            raise ValueError(f'N must be greater than 0, was {params["N"]}. At least one sample required')
        if len(m.events) < 1:
            raise ValueError('Model must have at least one event')
        if len(m.inputs) < 1:
            raise ValueError('Model must have at least one input')

        # ChaosPy doesn't support copying distributions.
        # As a workaround we create a new UserDistribution for each timepoint for each input
        # The UserDistribution is functionally the same as the original distribution
        input_dists = [cp.UserDistribution(
                cdf=input_dists[key].cdf,
                pdf=input_dists[key].pdf,
                ppf=input_dists[key].ppf
            )
            for key in m.inputs
            for _ in range(len(times))
            ]
        J = cp.J(*input_dists)  # Joint distribution to sample from
        
        # Simulate to collect time_of_event data
        time_of_event = np.empty((params['N'], len(m.events)), dtype=np.float64)

        def future_loading(t, x=None):
            nonlocal interpolator
            return m.InputContainer(interpolator(t)[np.newaxis].T)
        
        all_samples = J.sample(size=params['N'], rule='latin_hypercube')
        for i in range(params['N']):
            # Sample
            inputs = np.reshape(all_samples[:, i], (len(m.inputs), len(times)))
            interpolator = sp.interpolate.interp1d(times, inputs, bounds_error=False, fill_value=inputs[:, -1])

            # Simulate to get data
            time_of_event_i = m.time_of_event(x, future_loading, dt=params['dt'])

            # Add to list
            time_of_event[i] = [time_of_event_i[key] for key in m.events]
        
        params['input_keys'] = m.inputs
        params['x'] = x
        params['times'] = times
        return cls.from_data(
            inputs=all_samples.T,
            time_of_event=time_of_event,
            event_keys=m.events,
            J=J,
            **params)

PCE = PolynomialChaosExpansion
