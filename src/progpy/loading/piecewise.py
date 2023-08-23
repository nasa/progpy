# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

class Piecewise():
    """
    .. versionadded:: 1.5.0
    
    This is a simple piecewise future loading class. It takes a list of times and values and returns the value that corresponds to the current time. The object replaces the future_loading_eqn for simulate_to and simulate_to_threshold.

    Args
    -----
        InputContainer : class
            The InputContainer class for the model (model.InputContainer)
        times : list[float]
            A list of times (s)
        values : dict[str, list[float]]
            A dictionary with keys matching model inputs. Dictionary contains list of value for that input at until time in times (i.e., index 0 is the load until time[0], then it's index 1). Values dictionary should have the same or one more value than times. If values has one more value than times, then the last value is the default and will be applied after the last time has passed

    Example
    -------
    >>> from progpy.loading import Piecewise
    >>> m = SomeModel()
    >>> future_load = Piecewise(m.InputContainer, [0, 10, 20], {'input0': [0, 1, 0, 0.2]})
    >>> m.simulate_to_threshold(future_load)
    """
    def __init__(self, InputContainer, times, values):
        self.InputContainer = InputContainer

        n_values = None
        for key in values.keys():
            if n_values is None:
                n_values = len(values[key])
            elif n_values != len(values[key]):
                diff = len(values[key]) - n_values
                raise ValueError(
                    f"All elements in values must have "
                    "the same number of elements. {key} had "
                    "{f'{diff} more' if diff > 0 else f'{-diff} less'}")
        if n_values is not None and (n_values != len(times) and n_values != (len(times) + 1)):
            raise ValueError(
                f"Elements in values must have the same or "
                "one more element than times")

        self.times = times
        self.values = values

        if n_values is not None and (n_values == len(times) + 1):
            # Last is the default (i.e, the value after the last time)
            self.times.append(float('inf')) 

    def __call__(self, t, x=None):
        """
        Return the value that corresponds to the current time

        Args:
            t (float): Time (s)
            x (StateContainer, optional): Current state. Defaults to None.

        Returns:
            InputContainer: The value that corresponds to the current time
        """
        return self.InputContainer({
            key: next(self.values[key][i] for i in range(len(self.times)) if self.times[i] > t)
            for key in self.values})
