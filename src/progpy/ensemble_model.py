# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Sequence
import numpy as np

from progpy import PrognosticsModel


class EnsembleModel(PrognosticsModel):
    """
    .. versionadded:: 1.5.0

    An Ensemble Model is a collection of models which run together. The results of each model are aggregated using the aggregation_method function. This is generally done to improve the accuracy of prediction when you have multiple models that each represent part of the behavior, or represent a distribution of different behaviors. 

    .. role:: python(code)
        :language: python
        :class: highlight
    
    Ensemble Models are constructed from a set of other models (e.g., :python:`m = EnsembleModel((m1, m2, m3))`). The models then operate functionally as one prognostic model.

    Args:
        models (list[PrognosticsModel]): List of at least 2 models that form the ensemble

    Keyword Arguments:
        aggregation_method (function): Function that aggregates the outputs of the models in the ensemble. Default is np.mean
    """

    default_parameters = {
        'aggregation_method': np.mean,
    }

    def __init__(self, models: list, **kwargs):
        if not isinstance(models, Sequence):
            raise TypeError(f'EnsembleModel must be initialized with a list of models, got {type(models)}')
        if len(models) < 1:
            raise ValueError('EnsembleModel requires at least one model')
        for i, m in enumerate(models):
            if not isinstance(m, PrognosticsModel):
                raise TypeError(f'EnsembleModel requires all models to be PrognosticsModel instances. models[{i}] was {type(m)}')

        inputs = set()
        states = set()
        outputs = set()
        events = set()
        for m in models:
            inputs |= set(m.inputs)
            states |= set(m.states)
            outputs |= set(m.outputs)
            events |= set(m.events)
        self.inputs = list(inputs)
        self.states = list(states)
        self.outputs = list(outputs)
        self.events = list(events)

        super().__init__(**kwargs)
        self.parameters['models'] = models

    def initialize(self, u=None, z=None):
        xs = [
            m.initialize(
                m.InputContainer(u) if u is not None else None, 
                m.OutputContainer(z) if z is not None else None
            ) for m in self.parameters['models']]
        x0 = {}
        for x in xs:
            for key in x.keys():
                if key in x0:
                    x0[key].append(x[key])
                else:
                    x0[key] = [x[key]]
        for key in x0.keys():
            x0[key] = self.parameters['aggregation_method'](x0[key]) 
        return self.StateContainer(x0)

    def next_state(self, x, u, dt):
        xs = [m.next_state(m.StateContainer(x), m.InputContainer(u), dt) for m in self.parameters['models']]
        xs_final = {}
        for x in xs:
            for key in x.keys():
                if key in xs_final:
                    xs_final[key].append(x[key])
                else:
                    xs_final[key] = [x[key]]
        for key in xs_final.keys():
            xs_final[key] = self.parameters['aggregation_method'](xs_final[key])

        return self.StateContainer(xs_final)
        
    def output(self, x):
        zs = [m.output(m.StateContainer(x)) for m in self.parameters['models']]
        zs_final = {}
        for z in zs:
            for key in z.keys():
                if key in zs_final:
                    zs_final[key].append(z[key])
                else:
                    zs_final[key] = [z[key]]
        for key in zs_final.keys():
            zs_final[key] = self.parameters['aggregation_method'](zs_final[key])

        return self.OutputContainer(zs_final)
        
    def event_state(self, x) -> dict:
        es = [m.event_state(m.StateContainer(x)) for m in self.parameters['models']]
        es_final = {}
        for es_i in es:
            for key in es_i.keys():
                if key in es_final:
                    es_final[key].append(es_i[key])
                else:
                    es_final[key] = [es_i[key]]
        for key in es_final.keys():
            es_final[key] = self.parameters['aggregation_method'](es_final[key])

        return es_final

    def performance_metrics(self, x) -> dict:
        pms = [m.performance_metrics(x) for m in self.parameters['models']]
        pms_final = {}
        for pm in pms:
            for key in pm.keys():
                if key in pms_final:
                    pms_final[key].append(pm[key])
                else:
                    pms_final[key] = [pm[key]]
        for key in pms_final.keys():
            pms_final[key] = self.parameters['aggregation_method'](pms_final[key])

        return pms_final

    def time_of_event(self, *args, **kwargs):
        toes = [m.time_of_event(*args, **kwargs) for m in self.parameters['models']]
        toe_final = {}
        for toe in toes:
            for key in toe.keys():
                if key in toe_final:
                    toe_final[key].append(toe[key])
                else:
                    toe_final[key] = [toe[key]]
        for key in toe_final.keys():
            toe_final[key] = self.parameters['aggregation_method'](toe_final[key])

        return toe_final
