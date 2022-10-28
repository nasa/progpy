Included Models
=================== 
The :ref:`prog_models<prog_models Guide>` package is distributed with a few pre-constructed models that can  be used in simulation or prognostics (with the :ref:`prog_algs <prog_algs Guide>` package). These models are summarized in the following sections.

.. ..  contents:: 
..     :backlinks: top

Battery Model
-------------------------------------------------------------

.. tabs::

    .. tab:: ElectroChem (EOD)

        .. autoclass:: prog_models.models.BatteryElectroChemEOD

    .. tab:: ElectroChem (EOL)

        .. autoclass:: prog_models.models.BatteryElectroChemEOL

    .. tab:: ElectroChem (Combo)

        .. autoclass:: prog_models.models.BatteryElectroChem

        .. autoclass:: prog_models.models.BatteryElectroChemEODEOL

    .. tab:: Circuit

        .. autoclass:: prog_models.models.BatteryCircuit


Pump Model
-------------------------------------------------------------

There are two variants of the pump model based on if the wear parameters are estimated as part of the state. The models are described below

.. tabs::

    .. tab:: Base Model

        .. autoclass:: prog_models.models.CentrifugalPumpBase

    .. tab:: With Wear As State

        .. autoclass:: prog_models.models.CentrifugalPump

        .. autoclass:: prog_models.models.CentrifugalPumpWithWear

Pneumatic Valve
-------------------------------------------------------------

There are two variants of the valve model based on if the wear parameters are estimated as part of the state. The models are described below

.. tabs::

    .. tab:: Base Model

        .. autoclass:: prog_models.models.PneumaticValveBase

    .. tab:: With Wear As State

        .. autoclass:: prog_models.models.PneumaticValve

        .. autoclass:: prog_models.models.PneumaticValveWithWear

DC Motor
-------------------------------------------------------------

.. tabs:: 

    .. tab:: Single Phase

        .. autoclass:: prog_models.models.DCMotorSP

    .. tab:: Triple Phase

        .. autoclass:: prog_models.models.DCMotor

ESC
-------------------------------------------------------------
.. autoclass:: prog_models.models.ESC

Powertrain
-------------------------------------------------------------
.. autoclass:: prog_models.models.Powertrain

ThrownObject
-------------------------------------------------------------
.. autoclass:: prog_models.models.ThrownObject
