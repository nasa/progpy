Loading
=========

The loading subpackage includes some classes for complex load estimation algorithms. See :download:`examples.future_loading <../../../../progpy/examples/future_loading.py>` for more details.

Load Estimator Class interface
------------------------------
The key aspect of a load estimator is that it needs to be able to be called with either time or time and state. The most common way of accomplishing this is with a function, described in the dropdown below.

.. dropdown:: Functional Load Estimator

    .. code-block:: python

        >>> def load_estimator(t, x = None):
        >>>    # Calculate loading as function of time (t) and state (x)
        >>>    return load

The second approach for load estimators is a load estimation class. This is used to represent complex behavior. The interface for this is described in the dropdown below.

.. dropdown:: Class Load Estimator

    .. code-block:: python

        >>> class LoadEstimator:
        >>>    def __init__(self, *args, **kwargs):
        >>>        # Initialize the load estimator
        >>>        pass
        >>>    def __call__(self, t, x = None):
        >>>        # Calculate loading as function of time (t) and state (x)
        >>>        return load

Load Estimator Classes
----------------------

.. autoclass:: progpy.loading.Piecewise

.. autoclass:: progpy.loading.MovingAverage

.. autoclass:: progpy.loading.GaussianNoiseLoadWrapper

Controllers
------------------

.. autoclass:: progpy.loading.controllers.LQR

.. autoclass:: progpy.loading.controllers.LQR_I
