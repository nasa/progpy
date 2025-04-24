# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

__all__ = ['predictors', 'uncertain_data', 'state_estimators', 'run_prog_playback', 'metrics']

from progpy.prognostics_model import PrognosticsModel
from progpy.ensemble_model import EnsembleModel
from progpy.composite_model import CompositeModel
from progpy.mixture_of_experts import MixtureOfExpertsModel
from progpy.linear_model import LinearModel
from progpy import predictors, state_estimators, uncertain_data

import warnings

__version__ = '1.7.1'

def run_prog_playback(obs, pred, future_loading, output_measurements, **kwargs):
    warnings.warn("Depreciated in 1.2.0, will be removed in a future release.", DeprecationWarning)
    config = {# Defaults
        'predict_rate': 0, # Default- predict every step
        'num_samples': 10,
        'predict_config': {}
    }
    config.update(kwargs)

    next_predict = output_measurements[0][0] + config['predict_rate']
    times = []
    inputs = []
    states = []
    outputs = []
    event_states = []
    toes = []
    index = 0
    for (t, measurement) in output_measurements:
        obs.estimate(t, future_loading(t), measurement)
        if t >= next_predict:
            pred_results = pred.predict(obs.x.sample(config['num_samples']), future_loading, **config['predict_config'])
            times.append(pred_results.times)
            inputs.append(pred_results.inputs)
            states.append(pred_results.states)
            outputs.append(pred_results.outputs)
            event_states.append(pred_results.event_states)
            toes.append(pred_results.time_of_event)
            index += 1
            next_predict += config['predict_rate']
    return predictors.PredictionResults(
        times, 
        inputs, 
        states, 
        outputs, 
        event_states, 
        toes
    ) 
