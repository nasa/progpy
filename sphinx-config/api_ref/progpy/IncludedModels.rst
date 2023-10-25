Included Models
=================== 
The progpy package is distributed with a few pre-constructed models that can  be used in simulation or prognostics. These models are summarized in the following sections.

.. ..  contents:: 
..     :backlinks: top

Battery Model
-------------------------------------------------------------

.. tabs::

    .. tab:: ElectroChem (EOD)

        .. autoclass:: progpy.models.BatteryElectroChemEOD

    .. tab:: ElectroChem (EOL)

        .. autoclass:: progpy.models.BatteryElectroChemEOL

    .. tab:: ElectroChem (Combo)

        .. autoclass:: progpy.models.BatteryElectroChem

        .. autoclass:: progpy.models.BatteryElectroChemEODEOL

    .. tab:: Circuit

        .. autoclass:: progpy.models.BatteryCircuit


Pump Model
-------------------------------------------------------------

There are two variants of the pump model based on if the wear parameters are estimated as part of the state. The models are described below

.. tabs::

    .. tab:: Base Model

        .. autoclass:: progpy.models.CentrifugalPumpBase

    .. tab:: With Wear As State

        .. autoclass:: progpy.models.CentrifugalPump

        .. autoclass:: progpy.models.CentrifugalPumpWithWear

Pneumatic Valve
-------------------------------------------------------------

There are two variants of the valve model based on if the wear parameters are estimated as part of the state. The models are described below

.. tabs::

    .. tab:: Base Model

        .. autoclass:: progpy.models.PneumaticValveBase

    .. tab:: With Wear As State

        .. autoclass:: progpy.models.PneumaticValve

        .. autoclass:: progpy.models.PneumaticValveWithWear

DC Motor
-------------------------------------------------------------

.. tabs:: 

    .. tab:: Single Phase

        .. autoclass:: progpy.models.DCMotorSP

    .. tab:: Triple Phase

        .. autoclass:: progpy.models.DCMotor

ESC
-------------------------------------------------------------
.. autoclass:: progpy.models.ESC

Powertrain
-------------------------------------------------------------
.. autoclass:: progpy.models.Powertrain

PropellerLoad
-------------------------------------------------------------
.. autoclass:: progpy.models.PropellerLoad

Aircraft Models
-------------------------------------------------------------
Aircraft model simulate the flight of an aircraft. All aircraft models inherit from :py:class:`progpy.models.aircraft_model.AircraftModel`. Included models are listed below:

.. autoclass:: progpy.models.aircraft_model.SmallRotorcraft

ThrownObject
-------------------------------------------------------------
.. autoclass:: progpy.models.ThrownObject
