# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from warnings import catch_warnings, simplefilter
from collections import UserDict, abc
from copy import deepcopy
import json
from numbers import Number
import numpy as np
from scipy.integrate import OdeSolver
import types


from progpy.utils.containers import DictLikeMatrixWrapper
from progpy.utils.next_state import next_state_functions, SciPyIntegrateNextState
from progpy.utils.noise_functions import measurement_noise_functions, process_noise_functions
from progpy.utils.serialization import CustomEncoder, custom_decoder
from progpy.utils.size import getsizeof

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Fix circular import issue in PrognosticsModelParameters init
    from progpy.prognostics_model import PrognosticsModel


class PrognosticsModelParameters(UserDict):
    """
    Prognostics Model Parameters - this class replaces a standard dictionary.
    It includes the extra logic to process the different supported manners of defining noise.

    Args:
        model: PrognosticsModel for which the params correspond
        dict_in: Initial parameters
        callbacks: Any callbacks for derived parameters f(parameters): updates (dict)
    """
    def __init__(self, model: "PrognosticsModel", dict_in: dict = {}, callbacks: dict = {}, _copy: bool = True):
        super().__init__()
        self._m = model
        self.callbacks = {}
        # Note: Callbacks are set to empty to prevent calling callbacks with a
        # partial or empty dict on line 32.
        for (key, value) in dict_in.items():
            self.__setitem__(key, value, _copy=_copy)

        # Add and run callbacks
        # Has to be done here so the base parameters are all set
        self.callbacks = callbacks
        for key in callbacks:
            if key in self:
                for callback in callbacks[key]:
                    changes = callback(self)
                    self.update(changes)
    
    def __sizeof__(self):
        return getsizeof(self)

    def __eq__(self, other):
        if set(self.data.keys()) != set(other.data.keys()):
            return False
        for key, value in self.data.items():
            if not np.all(value == other[key]):
                # Note: np.all is used to handle numpy array elements
                # Otherwise value == other[key] would return a numpy array of
                # bools for each element
                return False
        return True
    
    def copy(self):
        return self.__class__(self._m, self.data, self.callbacks, _copy=False)

    def __copy__(self):
        return self.__class__(self._m, self.data, self.callbacks, _copy=False)
    
    def __deepcopy__(self, memo):
        result = self.__class__(self._m, self.data, self.callbacks, _copy=True)
        memo[id(self)] = result
        return result

    def __setitem__(self, key: str, value: float, _copy: bool = False) -> None:
        """Set model configuration, overrides dict.__setitem__()

        Args:
            key (str): configuration key to set
            value: value to set that configuration value to

        Raises:
            TypeError: Improper configuration for a model
        """
        # Deepcopy is needed here to force copying when value is an object (e.g., dict)
        if _copy:
            value = deepcopy(value)
        
        super().__setitem__(key, value)

        if key in self.callbacks:
            for callback in self.callbacks[key]:
                changes = callback(self)
                self.update(changes)  # Merge in changes

        # Handle setting integration_method.
        # This will override the next_state method
        if key == 'integration_method':
            if self._m.is_discrete and self._m.is_state_transition_model:
                raise TypeError(
                    "Cannot set integration method for discrete model (where next_state is overridden)")
            if isinstance(value, type) and issubclass(value, OdeSolver):
                # the integration_method is a SciPy Integrator
                fcn = SciPyIntegrateNextState(self._m, value)
                self._m.next_state = types.MethodType(
                    fcn,
                    self._m)
                return
            method = value.lower()
            if method in next_state_functions.keys():
                self._m.next_state = types.MethodType(
                    next_state_functions[method],
                    self._m)
                return
            raise ValueError(
                    f"Unsupported integration method {method}")
        
        if key == 'process_noise' or key == 'process_noise_dist':
            if callable(self['process_noise']):  # Provided a function
                self._m.apply_process_noise = types.MethodType(self['process_noise'], self._m)
            else:  # Not a function
                if key == 'process_noise' and isinstance(self['process_noise'], DictLikeMatrixWrapper):
                    # If it's already a DictLikeMatrixWrapper- convert to dict,
                    # so then it will be treated as a dictionary and missing keys will be filled in
                    # this way we're also sure that the final result is the "right" kind of container
                    self.data['process_noise'] = dict(self['process_noise'])

                if isinstance(self['process_noise'], Number):
                    # Process noise is single number - convert to dict
                    self.data['process_noise'] = self._m.StateContainer({key: self['process_noise'] for key in self._m.states})
                elif isinstance(self['process_noise'], dict):
                    noise = self['process_noise']
                    for key in self._m.states:
                        # Set any missing keys to 0
                        if key not in noise.keys():
                            noise[key] = 0
                            
                    self.data['process_noise'] = self._m.StateContainer(noise)
                
                # Process distribution type
                if 'process_noise_dist' in self and self['process_noise_dist'].lower() not in process_noise_functions:
                    raise ValueError(f"Unsupported process noise distribution {self['process_noise_dist']}")
                
                # Disable deprecation warnings for internal progpy code.
                with catch_warnings():
                    simplefilter("ignore", DeprecationWarning)
                    
                    if all(value == 0 for value in self['process_noise'].values()):
                        # No noise, use none function
                        fcn = process_noise_functions['none']
                        self._m.apply_process_noise = types.MethodType(fcn, self._m)
                    elif 'process_noise_dist' in self:
                        fcn = process_noise_functions[self['process_noise_dist'].lower()]
                        self._m.apply_process_noise = types.MethodType(fcn, self._m)
                    else:
                        # Default to gaussian
                        fcn = process_noise_functions['gaussian']
                        self._m.apply_process_noise = types.MethodType(fcn, self._m)
                
                #resetwarnings()

                # Make sure every key is present
                # (single value already handled above)
                for key in self._m.states:
                    if key not in self['process_noise']:
                        self['process_noise'][key] = 0

        elif key == 'measurement_noise' or key == 'measurement_noise_dist':
            if callable(self['measurement_noise']):
                self._m.apply_measurement_noise = types.MethodType(self['measurement_noise'], self._m)
            else:
                if key == 'measurement_noise' and isinstance(self['measurement_noise'], DictLikeMatrixWrapper):
                    # If it's already a DictLikeMatrixWrapper- convert to dict,
                    # so then it will be treated as a dictionary and missing keys will be filled in
                    # this way we're also sure that the final result is the "right" kind of container
                    self.data['measurement_noise'] = dict(self['measurement_noise'])

                if isinstance(self['measurement_noise'], Number):
                    # Process noise is single number - convert to dict
                    self.data['measurement_noise'] = self._m.OutputContainer({key: self['measurement_noise'] for key in self._m.outputs})
                elif isinstance(self['measurement_noise'], dict):
                    noise = self['measurement_noise']
                    for key in self._m.outputs:
                        # Set any missing keys to 0
                        if key not in noise.keys():
                            noise[key] = 0
                    self.data['measurement_noise'] = self._m.OutputContainer(noise)
                
                # Process distribution type
                if 'measurement_noise_dist' in self and self['measurement_noise_dist'].lower() not in measurement_noise_functions:
                    raise ValueError(f"Unsupported measurement noise distribution {self['measurement_noise_dist']}")

                # Disable deprecation warnings for internal progpy code.
                with catch_warnings():
                    simplefilter("ignore", category=DeprecationWarning)

                    if all(value == 0 for value in self['measurement_noise'].values()):
                        # No noise, use none function
                        fcn = measurement_noise_functions['none']
                        self._m.apply_measurement_noise = types.MethodType(fcn, self._m)
                    elif 'measurement_noise_dist' in self:
                        fcn = measurement_noise_functions[self['measurement_noise_dist'].lower()]
                        self._m.apply_measurement_noise = types.MethodType(fcn, self._m)
                    else:
                        # Default to gaussian
                        fcn = measurement_noise_functions['gaussian']
                        self._m.apply_measurement_noise = types.MethodType(fcn, self._m)

                #resetwarnings()
    
                # Make sure every key is present
                # (single value already handled above)
                if not all([key in self['measurement_noise'] for key in self._m.outputs]):
                    raise ValueError("Measurement noise must have ever key in model.outputs")

    def register_derived_callback(self, key: str, callback: abc.Callable) -> None:
        """Register a new callback for derived parameters

        Args:
            key (str): key for which the callback is triggered
            callback (function): callback function f(parameters) -> updates (dict)
        """
        if key in self.callbacks:
            self.callbacks[key].append(callback)
        else:
            self.callbacks[key] = [callback]

        # Run new callback
        if key in self:
            updates = callback(self[key])
            self.update(updates)

    def to_json(self):
        """
        Serialize parameters as JSON objects
        """
        return json.dumps(self.data, cls=CustomEncoder)
    
    @classmethod
    def from_json(cls, data):
        """
        Create a new parameters object from parameters that were serialized as a JSON object

        Args:
            data:
                JSON serialized parameters necessary to build parameters
                See to_json method

        Returns:
            Parameters: Parameters generated from serialized parameters
        """

        extract_parameters = json.loads(data, object_hook=custom_decoder)
 
        return cls(**extract_parameters)
