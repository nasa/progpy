Load Estimators
================

Load estimators are functions that describe the expected future load. The specific load estimator is specified by class name (e.g., Const) by the `load_est` key when starting a new session. Each class has specific configuration parameters to be specified in `load_est_cfg`. By default, MovingAverage is used.

Here's an example setting the load estimator and config:

    >>> from prog_client import Session
    >>> s = Session('BatteryCircuit', load_est='Const', load_est_cfg={'load': 1.0})

The following load estimators are supported:

.. |br| raw:: html

     <br>

.. autofunction:: prog_server.models.load_ests.Const

|br|

.. autofunction:: prog_server.models.load_ests.Variable

|br|

.. autofunction:: prog_server.models.load_ests.MovingAverage
