Included Models
=================== 
The prog_models is distributed with a few pre-constructed models that can  be used in simulation or prognostics (with the prog_algs package). These models are summarized in the following sections.

..  contents:: 
    :backlinks: top

Battery Model - Circuit
-------------------------------------------------------------

.. autoclass:: prog_models.models.BatteryCircuit

Battery Model - Electro Chemistry
-------------------------------------------------------------

There are three different flavors of Electro Chemistry Battery Models distributed with the package, described below

End of Discharge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEOD

End of Life (i.e., InsufficientCapacity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChemEOL

End of Discharge, End of Life (i.e., InsufficientCapacity & EOD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.BatteryElectroChem

.. autoclass:: prog_models.models.BatteryElectroChemEODEOL


Pump Model
-------------------------------------------------------------

There are two variants of the pump model based on if the wear parameters are estimated as part of the state. The models are described below

Pump Model (Base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.CentrifugalPumpBase

Pump Model (With Wear)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.CentrifugalPump

.. autoclass:: prog_models.models.CentrifugalPumpWithWear

Pneumatic Valve
-------------------------------------------------------------

There are two variants of the valve model based on if the wear parameters are estimated as part of the state. The models are described below

Pneumatic Valve (Base)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.PneumaticValveBase

Pneumatic Valve (With Wear)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: prog_models.models.PneumaticValve

.. autoclass:: prog_models.models.PneumaticValveWithWear

DC Motor
-------------------------------------------------------------

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
