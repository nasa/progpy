# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from abc import ABC, abstractmethod
import numpy as np
from progpy import PrognosticsModel


class LinearModel(PrognosticsModel, ABC):
    r"""
    A linear prognostics :term:`model`. Used when behavior can be described using a simple linear time-series model defined by the following equations:
    .. math::
        \dfrac{dx}{dt} = Ax + Bu + E

        z = Cx + D

        es = Fx + G

    where x is :term:`state`, u is :term:`input`, z is :term:`output` and es is :term:`event state`

    Linear Models must inherit from this class and define the following properties:
        * A: 2-d np.array[float], dimensions: n_states x n_states
        * B: 2-d np.array[float], optional (zeros by default), dimensions: n_states x n_inputs
        * C: 2-d np.array[float], dimensions: n_outputs x n_states
        * D: 1-d np.array[float], optional (zeros by default), dimensions: n_outputs x 1
        * E: 1-d np.array[float], optional (zeros by default), dimensions: n_states x 1
        * F: 2-d np.array[float], dimensions: n_es x n_states
        * G: 1-d np.array[float], optional (zeros by default), dimensions: n_es x 1
        * inputs:  list[str] - :term:`input` keys
        * states:  list[str] - :term:`state` keys
        * outputs: list[str] - :term:`output` keys
        * events:  list[str] - :term:`event` keys
    """

    # Default Values are set to None
    default_parameters = {
        '_B': None,
        '_D': None,
        '_E': None,
        '_G': None
    }

    def __init__(self, **kwargs):
        params = LinearModel.default_parameters.copy()
        params.update(self.default_parameters)
        params.update(kwargs)
        super().__init__(**params)

        # Set each property to itself
        # This triggers the default value logic in the setter
        # for cases where the property has not been overwritten
        self.B = self.B
        self.D = self.D
        self.E = self.E
        self.G = self.G

        if self.F is None and type(self).event_state == LinearModel.event_state:
            raise AttributeError(
                'LinearModel must define F if event_state is not defined. Either override event_state or define F.')
        
        self.matrixCheck()

# check to see if attributes are different and if functions within the models are overriden as well (i.e threshold_met and event_state)
    def __eq__(self, other):
        return isinstance(other, LinearModel) \
            and np.all(self.A == other.A) \
            and np.all(self.B == other.B) \
            and np.all(self.C == other.C) \
            and np.all(self.D == other.D) \
            and np.all(self.E == other.E) \
            and np.all(self.F == other.F) \
            and np.all(self.G == other.G) \
            and self.inputs == other.inputs \
            and self.outputs == other.outputs \
            and self.events == other.events \
            and self.states == other.states \
            and self.performance_metric_keys == other.performance_metric_keys \
            and self.parameters == other.parameters \
            and self.state_limits == other.state_limits \
            and type(self).threshold_met == type(other).threshold_met \
            and type(self).event_state == type(other).event_state \
            and type(self).dx == type(other).dx \
            and type(self).next_state == type(other).next_state \
            and type(self).output == type(other).output \
            and type(self).performance_metrics == type(other).performance_metrics

    def matrixCheck(self) -> None:
        """
        Public class method for checking matrices dimensions across all properties of the model.
        """
        self._propertyCheck(self.n_states, self.n_states,
                            ["A", "states", "states"])
        self._propertyCheck(self.n_states, self.n_inputs,
                            ["B", "states", "inputs"])
        self._propertyCheck(self.n_outputs, self.n_states,
                            ["C", "outputs", "states"])
        self._propertyCheck(self.n_outputs, 1,
                            ["D", "outputs", "1"])
        self._propertyCheck(self.n_states, 1,
                            ["E", "states", "1"])
        self._propertyCheck(self.n_events, 1,
                            ["G", "events", "1"])

        if self.F is not None:
            self._propertyCheck(self.n_events, self.n_states, [
                                "F", "events", "states"])

    def _propertyCheck(self, rowsCount: int, colsCount: int, notes: list) -> None:
        """
        matrix: Input matrix to check dimensions of (e.g. self.A, self.B, etc)
        rowsCount: Row count to check matrix against
        colsCount: Column count to check matrix against
        notes: List of strings containing information for exception message debugging
        """
        target_property = getattr(self, notes[0])
        if isinstance(target_property, list):
            setattr(self, notes[0], np.array(target_property))
        matrix = getattr(self, notes[0])
        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                "Matrix type check failed: @property {} dimensions is not of type list or NumPy array.".format(notes[0]))

        matrixShape = matrix.shape

        if (matrix.ndim != 2 or  # Checks to see if matrix is two-dimensional
            matrixShape[0] != rowsCount or  # checks if matrix has correct row count
            matrixShape[1] != colsCount):  # check all rows are equal to correct column count
            raise AttributeError(
                "Matrix size check failed: @property {} dimensions improperly formed along {} x {}.".format(notes[0], notes[1], notes[2]))

    @property
    @abstractmethod
    def A(self):
        pass

    @property
    def B(self):
        return self.parameters['_B']

    @B.setter
    def B(self, value):
        if (value is None):
            self.parameters['_B'] = np.zeros((self.n_states, self.n_inputs))
        else:
            prev_value = self.parameters['_B']
            self.parameters['_B'] = value
            try:
                self._propertyCheck(self.n_states, self.n_inputs, [
                                    "B", "states", "inputs"])
            except (TypeError, AttributeError) as ex:
                # Unacceptable value, reset and re-raise
                self.parameters['_B'] = prev_value
                raise ex

    @property
    @abstractmethod
    def C(self):
        pass

    @property
    def D(self):
        return self.parameters['_D']

    @D.setter
    def D(self, value):
        if (value is None):
            self.parameters['_D'] = np.zeros((self.n_outputs, 1))
        else:
            prev_value = self.parameters['_D']
            self.parameters['_D'] = value
            try:
                self._propertyCheck(self.n_outputs, 1, ["D", "outputs", "1"])
            except (TypeError, AttributeError) as ex:
                # Unacceptable value, reset and re-raise
                self.parameters['_D'] = prev_value
                raise ex

    @property
    def E(self):
        return self.parameters['_E']

    @E.setter
    def E(self, value):
        if (value is None):
            self.parameters['_E'] = np.zeros((self.n_states, 1))
        else:
            prev_value = self.parameters['_E']
            self.parameters['_E'] = value
            try:
                self._propertyCheck(self.n_states, 1, ["E", "states", "1"])
            except (TypeError, AttributeError) as ex:
                # Unacceptable value, reset and re-raise
                self.parameters['_E'] = prev_value
                raise ex

    @property
    @abstractmethod
    def F(self):
        pass

    @property
    def G(self):
        return self.parameters['_G']

    @G.setter
    def G(self, value):
        if (value is None):
            self.parameters['_G'] = np.zeros((self.n_events, 1))
        else:
            prev_value = self.parameters['_G']
            self.parameters['_G'] = value
            try:
                self._propertyCheck(self.n_events, 1, ["G", "events", "1"])
            except (TypeError, AttributeError) as ex:
                # Unacceptable value, reset and re-raise
                self.parameters['_G'] = prev_value
                raise ex

    def dx(self, x, u):
        dx_array = np.matmul(self.A, x.matrix) + self.E
        if len(u.matrix) > 0:
            dx_array += np.matmul(self.B, u.matrix)
        return self.StateContainer(dx_array)

    def output(self, x):
        z_array = np.matmul(self.C, x.matrix) + self.D
        return self.OutputContainer(z_array)

    def event_state(self, x):
        es_array = np.matmul(self.F, x.matrix) + self.G
        return {key: value[0] for key, value in zip(self.events, es_array)}

    def threshold_met(self, x):
        es_array = np.matmul(self.F, x.matrix) + self.G
        return {key: value[0] <= 0 for key, value in zip(self.events, es_array)}
