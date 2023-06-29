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
            A dictionary with keys matching model inputs. Dictionary contains list of value for that input at until time in times (i.e., index 0 is the load until time[0], then it's index 1)

    Example
    -------
    >>> from progpy.loading import Piecewise
    >>> m = SomeModel()
    >>> future_load = Piecewise(m.InputContainer, [0, 10, 20], {'input0': [0, 1, 0]})
    >>> m.simulate_to_threshold(future_load)
    """
    def __init__(self, InputContainer, times, values):
        self.InputContainer = InputContainer
        self.times = times
        self.values = values

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
