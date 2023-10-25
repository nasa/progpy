# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np


class MovingAverage():
    """
    .. versionadded:: 1.5.0
    
    This is a simple moving average future loading class. It takes input values and stores them in a window. When called, it returns the average of the values in the window. The object replaces the future_loading_eqn for simulate_to and simulate_to_threshold.

    Parameters
    ----------
    InputContainer : class
        The InputContainer class for the model (model.InputContainer)
    window : int, optional
        The size of the window to use for the moving average, by default 10

    Example
    -------
    >>> from progpy.loading import MovingAverage
    >>> m = SomeModel()
    >>> future_load = MovingAverage(m.InputContainer)
    >>> for _ in range(WINDOW_SIZE):
    >>>     load = load_source.get_load()
    >>>     future_load.add_load(load)
    >>> m.simulate_to_threshold(future_load)
    """
    def __init__(self, InputContainer: type, window: int = 10):
        self.window = window
        self.InputContainer = InputContainer
        self.values = {}
        self.index = 0

    def add_load(self, input):
        """
        Add a load to the moving average

        Args:
            input (InputContainer): One load value to add to the moving average
        """
        for key, value in input.items():
            if key not in self.values:
                self.values[key] = np.zeros(self.window)
            self.values[key][self.index] = value
        self.index = (self.index + 1) % self.window

    def __call__(self, t: float, x=None):
        """
        Return the average of the values in the window

        Args:
            t (float): Time (s)
            x (StateContaienr, optional): Current state. Defaults to None.

        Returns:
            InputContainer: The average of the values in the window
        """
        return self.InputContainer({key: np.mean(self.values[key]) for key in self.values})
