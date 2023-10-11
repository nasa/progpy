# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Sequence, Iterable
import numpy as np

from progpy import PrognosticsModel, CompositeModel

DIVIDER = '.'


class MixtureOfExpertsModel(CompositeModel):
    """
    .. versionadded:: 1.6.0

    T

    Args:
        models (list[PrognosticsModel]): List of at least 2 models that form the ensemble
    """

    default_parameters = {
        'max_score_step': 0.1
    }

    def initialize(self, u={}, z={}):
        if u is None:
            u = {}
        if z is None:
            z = {}

        x_0 = {}
        # Initialize the models
        for (name, m) in self.parameters['models']:
            u_i = {key: u.get(name + DIVIDER + key, None) for key in m.inputs}
            z_i = {key: z.get(name + DIVIDER + key, None) for key in m.outputs}
            x_i = m.initialize(u_i, z_i)
            for key, value in x_i.items():
                x_0[name + DIVIDER + key] = value
            x_0[name + DIVIDER + "_score"] = 0.5 # Initialize to half
        return self.StateContainer(x_0)

    def __init__(self, models: list, **kwargs):
        # Run initializer in ComositeModel
        # Note: Input validation is done there
        super().__init__(models, **kwargs)

        # Re-Initialize (overriding CompositeModel) for all except state
        # This is because state will work like composite model, but all others will be more like ensemble model
        self.inputs = set()
        self.outputs = set()
        self.events = set()
        self.performance_metric_keys = set()
        
        for (_, m) in self.parameters['models']:
            self.inputs |= set(m.inputs)
            self.outputs |= set(m.outputs)
            self.events |= set(m.events)
            self.performance_metric_keys |= set(m.performance_metric_keys)

        self.inputs = list(self.inputs)
        self.outputs = list(self.outputs)
        self.states = list(self.states)
        self.events = list(self.events)
        self.performance_metric_keys = list(self.performance_metric_keys)

        # Add last output to inputs
        self.inputs.extend(self.outputs)

        # Add model scores
        self.states.extend([model[0] + DIVIDER + "_score" for model in self.parameters['models']])

        # Finish initialization with prognostics model
        # To reset statecontainer, etc.
        # First reset noise (the double initialization doesnt work for that)
        self.parameters = {key: value for key, value in self.parameters.items()}  # Convert to dict
        self.parameters['process_noise'] = kwargs.get('process_noise', 0)
        self.parameters['measurement_noise'] = kwargs.get('process_noise', 0)
        PrognosticsModel.__init__(self, **self.parameters)

    def next_state(self, x, u, dt):
        x = super().next_state(x, u, dt)

        # If z is not none
        if not np.any(np.isnan([u[key] for key in self.inputs])):
            # If none in not u, that means that we have an updated output, so update the scores
            # u excluded when there is not update
            mses = []
            # calculate mse on predicted output
            for name, m in self.parameters['models']:
                gt = [u[z_key] for z_key in m.outputs]
                x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
                pred = [m.output(x_i)[z_key] for z_key in m.outputs]
                mses.append(np.square(np.subtract(gt, pred)).mean())
                
            min_mse = min(mses)
            max_mse = max(mses)
            diff_mse = max_mse-min_mse

            # Score delta - +self.parameters['max_score_step'] for best, -self.parameters['max_score_step'] for worse
            score_delta = [(min_mse-mse)/diff_mse*0.2+0.1 for mse in mses]
            for i, (key, _) in enumerate(self.parameters['models']):
                score_key = key + DIVIDER + "_score"
                x[score_key] += score_delta[i]

                # Apply lower limit
                x[score_key] = np.maximum(x[score_key], 0)
                
                # Apply upper limit
                if x[score_key] > 1:
                    x[score_key] -= score_delta[i] # undo application
                    # scale all to be <0.8
                    # This is needed to prevent one outlier bad model 
                    # From causing the other models to become saturated at 1
                    for j, (key_i, _) in enumerate(self.parameters['models']):
                        score_key_i = key_i + DIVIDER + "_score"
                        x[score_key_i] *= 0.8
                        # Also scale the 
                        score_delta[j] *= 0.8 # Also needs to be scaled

                    x[score_key] += score_delta[i] # Redo application

        return x

    def output(self, x):
        # Identify best model
        best_value = -1
        for i, (key, _) in enumerate(self.parameters['models']):
            score_key = key + DIVIDER + "_score"
            if x[score_key] > best_value:
                best_value = x[score_key]
                best_index = i

        # Prepare state
        name, m = self.parameters['models'][best_index]
        x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
        return m.output(x_i)

    def event_state(self, x):
        # Identify best model
        best_value = -1
        for i, (key, _) in enumerate(self.parameters['models']):
            score_key = key + DIVIDER + "_score"
            if x[score_key] > best_value:
                best_value = x[score_key]
                best_index = i

        name, m = self.parameters['models'][best_index]
        x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
        return m.event_state(x_i)

    def threshold_met(self, x):
        # Identify best model
        best_value = -1
        for i, (key, _) in enumerate(self.parameters['models']):
            score_key = key + DIVIDER + "_score"
            if x[score_key] > best_value:
                best_value = x[score_key]
                best_index = i

        name, m = self.parameters['models'][best_index]
        x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
        return m.threshold_met(x_i)

    def performance_metrics(self, x):
        # Identify best model
        best_value = -1
        for i, (key, _) in enumerate(self.parameters['models']):
            score_key = key + DIVIDER + "_score"
            if x[score_key] > best_value:
                best_value = x[score_key]
                best_index = i

        name, m = self.parameters['models'][best_index]
        x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
        return m.performance_metrics(x_i)
