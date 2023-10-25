# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np

from progpy import PrognosticsModel, CompositeModel

DIVIDER = '.'


class MixtureOfExpertsModel(CompositeModel):
    """
    .. versionadded:: 1.6.0

    Mixture of Experts (MoE) models combine multiple models of the same system, similar to Ensemble models. Unlike Ensemble Models, the aggregation is done by selecting the "best" model. That is the model that has performed the best over the past. Each model will have a 'score' that is tracked in the state, and this determines which model is best. 

    The MoE model's inputs include the inputs and outputs of the individual models making up the model. If the output values are provided as an input to the model then the model will update the score during state transition. The outputs are required to update the score, since this allows for the calculation of how "good" the model is. If outputs are not provided, state transition will continue as normal, without updating model scores. 

    At a state transition when outputs are provided, the score for the best model will increase by max_score_step for the best fitting model (i.e., lowest error in output) and decrease by max_score_step for the worst. All other models will be scaled between these, based on the error.
    
    Typically, outputs are provided in the MoE model input when performing a state estimation step (i.e. when there is measured data) but not when predicting forward (i.e. when the output is unknown).

    When calling output, event_state, threshold_met, or performance_metrics, only the model with the best score will be called, and those results returned. In case of a tie, the first model (in the order provided by the constructor) of the tied models will be used. If not every outputs, event, or performance metric has been identified, the next best model will be used to fill in the blanks, and so on.

    Args:
        models (list[PrognosticsModel]): List of at least 2 models that form the ensemble

    Keyword Args:
        process_noise : Optional, float or dict[str, float]
          :term:`Process noise<process noise>` (applied at dx/next_state). 
          Can be number (e.g., .2) applied to every state, a dictionary of values for each 
          state (e.g., {'x1': 0.2, 'x2': 0.3}), or a function (x) -> x
        process_noise_dist : Optional, str
          distribution for :term:`process noise` (e.g., normal, uniform, triangular)
        measurement_noise : Optional, float or dict[str, float]
          :term:`Measurement noise<measurement noise>` (applied in output eqn).
          Can be number (e.g., .2) applied to every output, a dictionary of values for each
          output (e.g., {'z1': 0.2, 'z2': 0.3}), or a function (z) -> z
        measurement_noise_dist : Optional, str
          distribution for :term:`measurement noise` (e.g., normal, uniform, triangular)
        max_score_step : Optional, float
          The maximum step in the score. This is the value by which the score of the best model increases, and the worst model decreases.

    Example:
    
        >> m = MixtureOfExpertsModel([model1, model2])
    """

    default_parameters = {
        'max_score_step': 0.01
    }

    def __init__(self, models: list, **kwargs):
        # Run initializer in CompositeModel
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

    def initialize(self, u={}, z={}):
        if u is None:
            u = {}
        if z is None:
            z = {}
        
        # Initialize the models
        x_0 = {}
        for (name, m) in self.parameters['models']:
            u_i = {key: u.get(name + DIVIDER + key, None) for key in m.inputs}
            z_i = {key: z.get(name + DIVIDER + key, None) for key in m.outputs}
            x_i = m.initialize(u_i, z_i)
            for key, value in x_i.items():
                x_0[name + DIVIDER + key] = value
            x_0[name + DIVIDER + "_score"] = 0.5  # Initialize to half
        return self.StateContainer(x_0)

    def next_state(self, x, u, dt):

        # Update state
        for (name, m) in self.parameters['models']:
            # Prepare inputs
            u_i = {key: u.get(key, None) for key in m.inputs}
            u_i = m.InputContainer(u_i)
            
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Propagate state
            x_next_i = m.next_state(x_i, u_i, dt)

            # Save to super state
            for key, value in x_next_i.items():
                x[name + '.' + key] = value

        # If z is present and not none - update score
        if (len(set(self.outputs)- set(u.keys())) == 0 and # Not missing an output
            not np.any(np.isnan([u[key] for key in self.outputs]))): # Not case where Output is NaN
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

            # Score delta: +self.parameters['max_score_step'] for best, -self.parameters['max_score_step'] for worse
            score_delta = [(min_mse-mse)/diff_mse*(2*self.parameters['max_score_step'])+self.parameters['max_score_step'] for mse in mses]
            for i, (key, _) in enumerate(self.parameters['models']):
                score_key = key + DIVIDER + "_score"
                x[score_key] += score_delta[i]

                # Apply lower limit
                x[score_key] = np.maximum(x[score_key], 0)
                # Note: lower limit saturation is acceptable
                
                # Apply upper limit
                if x[score_key] > 1:
                    x[score_key] -= score_delta[i] # undo application
                    # scale all to be <0.8
                    # This is needed to prevent one outlier bad model 
                    # From causing the other models to become saturated at 1
                    for j, (key_i, _) in enumerate(self.parameters['models']):
                        score_key_i = key_i + DIVIDER + "_score"
                        x[score_key_i] *= 0.8
                        # Also scale the score deltas
                        score_delta[j] *= 0.8

                    x[score_key] += score_delta[i] # Redo application

        return x

    def best_model(self, x, _excepting=[]):
        """
        Get the best-performing model according to the scores

        Args:
            x (StateContainer): System state

        Returns:
            tuple[str, PrognosticsModel]: The name and model for the model with the highest score. If two models are tied, the first one will be returned in the order they were passed to the constructor.
        """
        # Identify best model
        best_value: float = -1
        for i, (key, _) in enumerate(self.parameters['models']):
            if key in _excepting:
                continue # Skip excepting
            score_key = key + DIVIDER + "_score"
            if x[score_key] > best_value:
                best_value = x[score_key]
                best_index = i
        return self.parameters['models'][best_index]

    def output(self, x):
        excepting = []
        outputs_seen = set()
        z = {}
        while outputs_seen != set(self.outputs):
            # Not all outputs have been calculated
            name, m = self.best_model(x, _excepting=excepting)
            excepting.append(name)

            new_outputs = set(m.outputs) - outputs_seen
            if len(new_outputs) > 0:
                # Has an output that hasn't been seen

                # Prepare state
                x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
                z_i = m.output(x_i)
                
                # Merge in new outputs
                for key in new_outputs:
                    z[key] = z_i[key]

                # Add new outputs
                outputs_seen |= new_outputs

        return self.OutputContainer(z)

    def event_state(self, x):
        excepting = []
        events_seen = set()
        es = {}
        while events_seen != set(self.events):
            # Not all outputs have been calculated
            name, m = self.best_model(x, _excepting=excepting)
            excepting.append(name)

            new_events = set(m.events) - events_seen
            if len(new_events) > 0:
                # Has an event that hasn't been seen

                # Prepare state
                x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
                es_i = m.event_state(x_i)
                
                # Merge in new events
                for key in new_events:
                    es[key] = es_i[key]

                # Add new events
                events_seen |= new_events

        return es

    def threshold_met(self, x):
        excepting = []
        events_seen = set()
        tm = {}
        while events_seen != set(self.events):
            # Not all outputs have been calculated
            name, m = self.best_model(x, _excepting=excepting)
            excepting.append(name)

            new_events = set(m.events) - events_seen
            if len(new_events) > 0:
                # Has an event that hasn't been seen

                # Prepare state
                x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
                tm_i = m.threshold_met(x_i)
                
                # Merge in new events
                for key in new_events:
                    tm[key] = tm_i[key]

                # Add new events
                events_seen |= new_events

        return tm

    def performance_metrics(self, x):
        excepting = []
        performance_metrics_seen = set()
        pm = {}
        while performance_metrics_seen != set(self.performance_metric_keys):
            # Not all outputs have been calculated
            name, m = self.best_model(x, _excepting=excepting)
            excepting.append(name)

            new_performance_metrics = set(m.performance_metric_keys) - performance_metrics_seen
            if len(new_performance_metrics) > 0:
                # Has an performance metrics that hasn't been seen

                # Prepare state
                x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})
                pm_i = m.performance_metrics(x_i)
                
                # Merge in new events
                for key in new_performance_metrics:
                    pm[key] = pm_i[key]

                # Add new events
                performance_metrics_seen |= new_performance_metrics

        return pm
