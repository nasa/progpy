# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from collections.abc import Iterable
from inspect import signature

from progpy import PrognosticsModel

DIVIDER = '.'


class CompositeModel(PrognosticsModel):
    """
    .. versionadded:: 1.5.0

    A CompositeModel is a PrognosticsModel that is composed of multiple PrognosticsModels. This is a tool for modeling system-of-systems. I.e., interconnected systems, where the behavior and state of one system effects the state of another system. The composite prognostics models are connected using defined connections between the output or state of one model, and the input of another model. The resulting CompositeModel behaves as a single model.

    Args:
        models (list[PrognosticsModel or function] or list[tuple[str, PrognosticsModel or function]]):
            A list of PrognosticsModels to be combined into a single model.
            Provided in one of two forms:

            1. A list of PrognosticsModels or functions. The name of each model will be the class name for models or 'function' for functions. A number will be added for duplicates

            2. A list of tuples where the first element is the model/function name and the second element is the model/function

            Note: Order provided will be the order that models are executed
        connections (list[tuple[str, str]], optional):
            A list of tuples where the first element is the name of the output, state, or performance metrics of one model or function return and the second element is the name of the input of another model or argument of a function.
            The first element of the tuple must be of the form "model_name.output_name", "model_name.state_name", or "model_name.performance_metric_key".
            The second element of the tuple must be of the form "model_name.input_name".
            For example, if you have two models, "Batt1" and "Batt2", and you want to connect the output of "Batt1" to the input of "Batt2", you would use the following connection: ("Batt1.output", "Batt2.input")

    Keyword Args:
        outputs (list[str]):
            Model outputs in format "model_name.output_name". Must be subset of all outputs from models. If not provided, all outputs will be included.

    Example:
        >>> m = SomeModel()
        >>> m2 = SomeOtherModel()
        >>> def kelvin_to_celcius(temp_in_kelvin):
        >>>     return temp_in_kelvin - 273.15
        >>> connections = [
        >>>     ('m1.temp', 'kelvin_to_celcius.temp_in_kelvin')
        >>>     ('kelvin_to_celcius.return', 'm2.temp')
        >>> ]
        >>> m_composite = CompositeModel(
        >>>     (('m1', m), ('kelvin_to_celcius', kelvin_to_celcius), ('m2', m2)),  # models
        >>>     connections=connections
        >>> )

    .. note:: Model parameters can be set and accessed using the '[model].[param]' format. For example, for composite model m, m['foo.bar'] would set the parameter 'bar' for the model 'foo'.
    """

    def __init__(self, models: list, connections: list = [], **kwargs):
        # General Input Validation
        if not isinstance(models, Iterable):
            raise ValueError('The models argument must be a list')
        if len(models) <= 1:
            raise ValueError(
                'The models argument must contain at least two models')
        if not isinstance(connections, Iterable):
            raise ValueError('The connections argument must be a list')

        # Initialize
        kwargs['model_names'] = set()
        kwargs['models'] = []
        kwargs['functions'] = []
        kwargs['connections'] = connections
        duplicate_names = {}

        # Handle models
        for m in models:
            # Ensure tuple format
            if isinstance(m, Iterable):  # Already a tuple
                if len(m) != 2:
                    raise ValueError(
                        'Each model tuple must be of the form '
                        '(name: str, model). For example '
                        '("Batt1", BatteryElectroChem())')
                if not isinstance(m[0], str):
                    raise ValueError(
                        'The first element of each model tuple must'
                        ' be a string')
            elif isinstance(m, PrognosticsModel):
                m = (m.__class__.__name__, m)
            elif callable(m):
                m = ('function', m)
            else:
                raise ValueError(f'Each model must be a PrognosticsModel or tuple (name: str, PrognosticsModel), was {type(m)}')

            # Check for duplicate names
            if m[0] in kwargs['model_names']:
                duplicate_names[m[0]] = duplicate_names.get(m[0], 1) + 1
                m = (m[0] + '_' + str(duplicate_names[m[0]]), m[1])
            kwargs['model_names'].add(m[0])

            # Handle model/function
            if isinstance(m[1], PrognosticsModel):
                kwargs['models'].append(m)
            elif callable(m[1]):
                kwargs['functions'].append(m)
            else:
                raise ValueError(
                    'The second element of each model tuple must be a'
                    ' PrognosticsModel')

        self.__setstate__(kwargs)

        # Finish initialization
        super().__init__(**kwargs)

    def __setstate__(self, params: dict) -> None:
        """
        Setup inputs, outputs, connections from models, functions. Needed to fix copying/pickling

        Args:
            params (dict): kwargs (either parameters or kwargs into constructor)
        """
        self.inputs = set()
        self.states = set()
        self.outputs = set()
        self.events = set()
        self.performance_metric_keys = set()

        # update inputs, states, outputs, etc.
        for (name, m) in params['models']:
            self.inputs |= set([name + DIVIDER + u for u in m.inputs])
            self.states |= set([name + DIVIDER + x for x in m.states])
            self.outputs |= set([name + DIVIDER + z for z in m.outputs])
            self.events |= set([name + DIVIDER + e for e in m.events])
            self.performance_metric_keys |= set([name + DIVIDER + p for p in m.performance_metric_keys])

        for (name, fcn) in params['functions']:
            self.inputs |= set([name + DIVIDER + u for u in signature(fcn).parameters.keys()])
            self.states.add(name + DIVIDER + 'return')
            self.outputs.add(name + DIVIDER + 'return')

        # Handle outputs
        if 'outputs' in params:
            if isinstance(params['outputs'], str):
                params['outputs'] = [params['outputs']]
            if not isinstance(params['outputs'], Iterable):
                raise ValueError('The outputs argument must be a list[str]')
            if not set(params['outputs']).issubset(self.outputs):
                raise ValueError(
                    'The outputs of the composite model must be a '
                    'subset of the outputs of the models')
            self.outputs = params['outputs']

        # Handle Connections
        self.__to_input_connections = {
            m_name: [] for m_name in params['model_names']}
        self.__to_state_connections = {
            m_name: [] for m_name in params['model_names']}
        self.__to_state_from_pm_connections = {
            m_name: [] for m_name in params['model_names']}

        for connection in params['connections']:
            # Input validation
            if not isinstance(connection, Iterable) or len(connection) != 2:
                raise ValueError(
                    'Each connection must be a tuple of the form'
                    ' (input: str, output: str)')
            if not isinstance(connection[0], str) or not isinstance(connection[1], str):
                raise ValueError(
                    'Each connection must be a tuple of the form'
                    ' (input: str, output: str)')

            in_key, out_key = connection
            # Validation
            if out_key not in self.inputs:
                raise ValueError(
                    f'The output key, {out_key}, must be an input'
                    ' to one of the composite models. Options '
                    f'include {self.inputs}')

            # Remove the out_key from inputs
            # These no longer are an input to the composite model
            # as they are now satisfied internally
            self.inputs.remove(out_key)

            # Split the keys into parts (model, key_part)
            (in_model, in_key_part) = in_key.split('.')
            (out_model, out_key_part) = out_key.split('.')

            # Validate parts
            if in_model == out_model:
                raise ValueError(
                    'The input and output models must be different')
            if in_model not in params['model_names']:
                raise ValueError(
                    'The input model must be one of the models'
                    ' in the composite model')
            if out_model not in params['model_names']:
                raise ValueError(
                    'The output model must be one of the models'
                    ' in the composite model')

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
                raise ValueError(
                    f'The input key {in_key} must be an output or state')

        # Setup callbacks
        # These callbacks will enable setting of parameters in composed models.
        # E.g., composite.parameters['abc.def'] will set parameter 'def' for composed model 'abc'
        class PassthroughParams():
            def __init__(self, models, model_name, key):
                self.models = models
                self.model_name = model_name
                self.key = key
                i = 0
                for (name, _) in models:
                    if name == model_name:
                        break
                    i += 1
                self.model_index = i
                self.combined_key = self.model_name + '.' + self.key

            def __call__(self, params: dict) -> dict:
                params['models'][self.model_index][1].parameters[self.key] = params[self.combined_key]
                return {}
        
        for (name, m) in params['models']:
            # TODO(CT): TRY JUST SAVING NAME
            for key in m.parameters.keys():
                combined_key = name + '.' + key
                if combined_key in self.param_callbacks:
                    self.param_callbacks[combined_key].append(PassthroughParams(params['models'], name, key))
                else:
                    self.param_callbacks[combined_key] = [PassthroughParams(params['models'], name, key)]

        return super().__setstate__(params)

    def initialize(self, u=None, z=None):
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
            # This initializes the states connected to outputs
            for (in_key_part, in_key) in self.__to_state_connections[name]:
                if in_key in z.keys():
                    x_0[in_key] = z[in_key]
                else:  # Missing from z, so estimate using initial state
                    z_ii = m.output(x_i)
                    x_0[in_key] = z_ii.get(in_key_part, None)

            # This initializes the states connected to performance metrics
            for (in_key_part, in_key) in self.__to_state_from_pm_connections[name]:
                pm = m.performance_metrics(x_i)
                x_0[in_key] = pm.get(in_key_part, None)

        # Initialize functions
        for (name, fcn) in self.parameters['functions']:
            keys = set(signature(fcn).parameters.keys())
            fcn_in = {
                fcn_key: x_0.get(out_key, z.get(out_key, None))
                for (out_key, fcn_key) in self.__to_input_connections[name]}
            for key in (keys - set(fcn_in.keys())):
                # For remaining keys - get from input
                fcn_in[key] = u.get(name + DIVIDER + key, None)
            x_0[name + DIVIDER + 'return'] = fcn(**fcn_in)

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
            x_i = m.next_state(x_i, u_i, dt)

            # Save to super state
            for key, value in x_i.items():
                x[name + '.' + key] = value
            
            # Process connections
            # This updates the states that are connected to outputs
            if len(self.__to_state_connections[name]) > 0:
                # From Outputs
                z_i = m.output(x_i)
                for (in_key_part, in_key) in self.__to_state_connections[name]:
                    x[in_key] = z_i.get(in_key_part, None)

            if len(self.__to_state_from_pm_connections) > 0:
                # From Performance Metrics
                pm_i = m.performance_metrics(x_i)
                for (in_key_part, in_key) in self.__to_state_from_pm_connections[name]:
                    x[in_key] = pm_i.get(in_key_part, None)

        # Process functions
        for (name, fcn) in self.parameters['functions']:
            keys = set(signature(fcn).parameters.keys())
            fcn_in = {
                fcn_key: x.get(out_key, None)
                for (out_key, fcn_key) in self.__to_input_connections[name]}
            for key in (keys - set(fcn_in.keys())):
                # For remaining keys - get from input
                fcn_in[key] = u.get(name + DIVIDER + key, None)
            x[name + DIVIDER + 'return'] = fcn(**fcn_in)

        return x

    def output(self, x):
        z = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({
                key: x[name + '.' + key] for key in m.states})

            # Get outputs
            z_i = m.output(x_i)

            # Save to super outputs
            for key, value in z_i.items():
                z[name + '.' + key] = value

        for (name, _) in self.parameters['functions']:
            z[name + DIVIDER + 'return'] = x[name + DIVIDER + 'return']
        return self.OutputContainer(z)

    def performance_metrics(self, x) -> dict:
        metrics = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({
                key: x[name + '.' + key] for key in m.states})

            # Get outputs
            metrics_i = m.performance_metrics(x_i)

            # Save to super outputs
            for key, value in metrics_i.items():
                metrics[name + '.' + key] = value
        return metrics

    def event_state(self, x) -> dict:
        e = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({
                key: x[name + '.' + key] for key in m.states})

            # Get outputs
            e_i = m.event_state(x_i)

            # Save to super outputs
            for key, value in e_i.items():
                e[name + '.' + key] = value
        return e

    def threshold_met(self, x) -> dict:
        tm = {}
        for (name, m) in self.parameters['models']:
            # Prepare state
            x_i = m.StateContainer({
                key: x[name + '.' + key] for key in m.states})

            # Get outputs
            tm_i = m.threshold_met(x_i)

            # Save to super outputs
            for key, value in tm_i.items():
                tm[name + '.' + key] = value
        return tm
