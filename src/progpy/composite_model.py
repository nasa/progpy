# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable

from progpy import PrognosticsModel

DIVIDER = '.'


class CompositeModel(PrognosticsModel):
    """
    .. versionadded:: 1.5.0

    A CompositeModel is a PrognosticsModel that is composed of multiple PrognosticsModels. This is a tool for modeling system-of-systems. I.e., interconnected systems, where the behavior and state of one system effects the state of another system. The composite prognostics models are connected using defined connections between the output or state of one model, and the input of another model. The resulting CompositeModel behaves as a single model.

    Args:
        models (list[PrognosticsModel] or list[tuple[str, PrognosticsModel]]):
            A list of PrognosticsModels to be combined into a single model.
            Provided in one of two forms:

            1. A list of PrognosticsModels. The name of each model will be the class name. A number will be added for duplicates

            2. A list of tuples where the first element is the model name and the second element is the model

            Note: Order provided will be the order that models are executed
        connections (list[tuple[str, str]], optional):
            A list of tuples where the first element is the name of the output, state, or performance metrics of one model and the second element is the name of the input of another model.
            The first element of the tuple must be of the form "model_name.output_name", "model_name.state_name", or "model_name.performance_metric_key".
            The second element of the tuple must be of the form "model_name.input_name".
            For example, if you have two models, "Batt1" and "Batt2", and you want to connect the output of "Batt1" to the input of "Batt2", you would use the following connection: ("Batt1.output", "Batt2.input")

    Keyword Args:
        outputs (list[str]):
            Model outputs in format "model_name.output_name". Must be subset of all outputs from models. If not provided, all outputs will be included.
    """

    def __init__(self, models: list, connections: list = [], **kwargs):
        # General Input Validation
        if not isinstance(models, Iterable):
            raise ValueError('The models argument must be a list')
        if len(models) <= 1:
            raise ValueError('The models argument must contain at least two models')
        if not isinstance(connections, Iterable):
            raise ValueError('The connections argument must be a list')

        # Initialize
        self.inputs = set()
        self.states = set()
        self.outputs = set()
        self.events = set()
        self.performance_metric_keys = set()
        self.model_names = set()
        duplicate_names = {}
        kwargs['models'] = []

        # Handle models
        for m in models:
            if isinstance(m, Iterable):
                if len(m) != 2:
                    raise ValueError('Each model tuple must be of the form (name: str, model). For example ("Batt1", BatteryElectroChem())')
                if not isinstance(m[0], str):
                    raise ValueError('The first element of each model tuple must be a string')
                if not isinstance(m[1], PrognosticsModel):
                    raise ValueError('The second element of each model tuple must be a PrognosticsModel')
                if m[0] in self.model_names:
                    duplicate_names[m[0]] = duplicate_names.get(m[0], 1) + 1
                    m = (m[0] + '_' + str(duplicate_names[m[0]]), m[1])
                self.model_names.add(m[0])
                kwargs['models'].append(m)
            elif isinstance(m, PrognosticsModel):
                m = (m.__class__.__name__, m)
                if m[0] in self.model_names:
                    duplicate_names[m[0]] = duplicate_names.get(m[0], 1) + 1
                    m = (m[0] + '_' + str(duplicate_names[m[0]]), m[1])
                self.model_names.add(m[0])
                kwargs['models'].append(m)
            else:
                raise ValueError(f'Each model must be a PrognosticsModel or tuple (name: str, PrognosticsModel), was {type(m)}')

        for (name, m) in kwargs['models']:
            self.inputs |= set([name + DIVIDER + u for u in m.inputs])
            self.states |= set([name + DIVIDER + x for x in m.states])
            self.outputs |= set([name + DIVIDER + z for z in m.outputs])
            self.events |= set([name + DIVIDER + e for e in m.events])
            self.performance_metric_keys |= set([name + DIVIDER + p for p in m.performance_metric_keys])
        
        # Handle outputs
        if 'outputs' in kwargs:
            if isinstance(kwargs['outputs'], str):
                kwargs['outputs'] = [kwargs['outputs']]
            if not isinstance(kwargs['outputs'], Iterable):
                raise ValueError('The outputs argument must be a list[str]')
            if not set(kwargs['outputs']).issubset(self.outputs):
                raise ValueError('The outputs of the composite model must be a subset of the outputs of the models')
            self.outputs = kwargs['outputs']
        
        # Handle Connections
        kwargs['connections'] = []
        self.__to_input_connections = {m_name: [] for m_name in self.model_names}
        self.__to_state_connections = {m_name: [] for m_name in self.model_names}
        self.__to_state_from_pm_connections = {m_name: [] for m_name in self.model_names}

        for connection in connections:
            # Input validation
            if not isinstance(connection, Iterable) or len(connection) != 2:
                raise ValueError('Each connection must be a tuple of the form (input: str, output: str)')
            if not isinstance(connection[0], str) or not isinstance(connection[1], str):
                raise ValueError('Each connection must be a tuple of the form (input: str, output: str)')

            in_key, out_key = connection
            # Validation
            if out_key not in self.inputs:
                raise ValueError(f'The output key, {out_key}, must be an input to one of the composite models. Options include {self.inputs}')

            # Remove the out_key from inputs
            # These no longer are an input to the composite model
            # as they are now satisfied internally
            self.inputs.remove(out_key)
                
            # Split the keys into parts (model, key_part)
            (in_model, in_key_part) = in_key.split('.')
            (out_model, out_key_part) = out_key.split('.')

            # Validate parts
            if in_model == out_model:
                raise ValueError('The input and output models must be different')
            if in_model not in self.model_names:
                raise ValueError('The input model must be one of the models in the composite model')
            if out_model not in self.model_names:
                raise ValueError('The output model must be one of the models in the composite model')
            
            # Add to connections
            if in_key in self.states:
                self.__to_input_connections[out_model].append((in_key, out_key_part))
            elif in_key in self.outputs:
                # In output
                self.__to_input_connections[out_model].append((in_key, out_key_part))

                # Add to state (to preserve last between runs)
                self.__to_state_connections[in_model].append((in_key_part, in_key))
                self.states.add(in_key)
            elif in_key in self.performance_metric_keys:
                # In performance metric
                self.__to_input_connections[out_model].append((in_key, out_key_part))

                # Add to state (to preserve last between runs)
                self.__to_state_from_pm_connections[out_model].append((in_key_part, in_key))
                self.states.add(in_key)
            else:
                raise ValueError('The input key must be an output or state of one of the composite models')
        
        # Finish initialization
        super().__init__(**kwargs)

    def initialize(self, u={}, z={}):
        if u is None:
            u = {}
        if z is None:
            z = {}

        x_0 = {}
        # Initialize the models
        for (name, m) in self.parameters['models']:
            u_i = {key: u.get(name + '.' + key, None) for key in m.inputs}
            z_i = {key: z.get(name + '.' + key, None) for key in m.outputs}
            x_i = m.initialize(u_i, z_i)
            for key, value in x_i.items():
                x_0[name + '.' + key] = value
        
            # Process connections
            # This initializes the states that are connected to outputs
            for (in_key_part, in_key) in self.__to_state_connections[name]:
                if in_key in z.keys():
                    x_0[in_key] = z[in_key]
                else:  # Missing from z, so estimate using initial state
                    z_ii = m.output(x_i)
                    x_0[in_key] = z_ii.get(in_key_part, None)

            # This initializes the states that are connected to performance metrics
            for (in_key_part, in_key) in self.__to_state_from_pm_connections[name]:
                pm = m.performance_metrics(x_i)
                x_0[in_key] = pm.get(in_key_part, None)
                
        return self.StateContainer(x_0)

    def next_state(self, x, u, dt):
        for (name, m) in self.parameters['models']:
            # Prepare inputs
            u_i = {key: u.get(name + '.' + key, None) for key in m.inputs}
            # Process connections
            # This updates the inputs that are connected to states/outputs
            for (in_key, out_key_part) in self.__to_input_connections[name]:
                u_i[out_key_part] = x[in_key]

            u_i = m.InputContainer(u_i)
            
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Propagate state
            x_next_i = m.next_state(x_i, u_i, dt)

            # Save to super state
            for key, value in x_next_i.items():
                x[name + '.' + key] = value
            
            # Process connections
            # This updates the states that are connected to outputs
            if len(self.__to_state_connections[name]) > 0:
                # From Outputs
                z_i = m.output(x_next_i)
                for (in_key_part, in_key) in self.__to_state_connections[name]:
                    x[in_key] = z_i.get(in_key_part, None)

            if len(self.__to_state_from_pm_connections) > 0:
                # From Performance Metrics
                pm_i = m.performance_metrics(x_next_i)
                for (in_key_part, in_key) in self.__to_state_from_pm_connections[name]:
                    x[in_key] = pm_i.get(in_key_part, None)

        return x

    def output(self, x):
        z = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            z_i = m.output(x_i)

            # Save to super outputs
            for key, value in z_i.items():
                z[name + '.' + key] = value
        return self.OutputContainer(z)

    def performance_metrics(self, x):
        metrics = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            metrics_i = m.performance_metrics(x_i)

            # Save to super outputs
            for key, value in metrics_i.items():
                metrics[name + '.' + key] = value
        return metrics

    def event_state(self, x):
        e = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            e_i = m.event_state(x_i)

            # Save to super outputs
            for key, value in e_i.items():
                e[name + '.' + key] = value
        return e

    def threshold_met(self, x):
        tm = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({key: x[name + '.' + key] for key in m.states})

            # Get outputs
            tm_i = m.threshold_met(x_i)

            # Save to super outputs
            for key, value in tm_i.items():
                tm[name + '.' + key] = value
        return tm
