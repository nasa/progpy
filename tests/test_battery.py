# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest
import numpy as np

from progpy.models import (
    BatteryCircuit,
    BatteryElectroChem,
    BatteryElectroChemEOL,
    BatteryElectroChemEOD,
    BatteryElectroChemEODEOL,
    SimplifiedBattery,
)
from progpy.loading import Piecewise

# Variable (piece-wise) future loading scheme
future_loading = Piecewise(
    dict, [600, 900, 1800, 3000, float("inf")], {"i": [2, 1, 4, 2, 3]}
)

future_loading_power = Piecewise(
    dict, [600, 900, 1800, 3000, float("inf")], {"P": [25, 12, 50, 25, 33]}
)


class TestBattery(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_battery_circuit(self):
        batt = BatteryCircuit()
        result = batt.simulate_to(200, future_loading, {"t": 18.95, "v": 4.183})

    def test_battery_electrochem(self):
        batt = BatteryElectroChem()
        result = batt.simulate_to(200, future_loading, {"t": 18.95, "v": 4.183})
        self.assertEqual(BatteryElectroChem, BatteryElectroChemEODEOL)

    """
    Test that the current combined model has the same results as the initial implementation. 
    
    Note that this initial implementation is no longer in ProgPy. While we have confidence in 
    the quantitative results, a development challenge related to different results if print is True or False
    forced the need for an updated EODEOL model (see https://github.com/nasa/progpy/issues/199). In the 
    below test, some values from this initial implementation are hard-coded for testing purposes.

    The combined model changes can be found here: https://github.com/nasa/progpy/pull/207 
    Note that states are not compared since the updated implementation uses different states and parameters.
    """

    def test_battery_electrochem_results(self):
        config = {"save_freq": 1000, "dt": 2, "events": "InsufficientCapacity"}

        def future_loading(t, x=None):
            load = 1

            if x is not None:
                event_state = batt.event_state(x)
                if event_state["EOD"] > 0.95:
                    load = 1  # Discharge
                elif event_state["EOD"] < 0.05:
                    load = -1  # Charge

            return batt.InputContainer({"i": load})

        batt = BatteryElectroChem()
        result = batt.simulate_to_threshold(future_loading, **config)

        # Check a middle result
        self.assertEqual(result.times[-10], 219000.0)
        self.assertEqual(result.inputs[-10], {"i": np.float64(-1.0)})
        self.assertEqual(
            result.outputs[-10],
            {"t": np.float64(18.775450787889213), "v": np.float64(3.0165983707625728)},
        )
        self.assertEqual(
            result.event_states[-10],
            {
                "EOD": np.float64(0.16598370762572756),
                "InsufficientCapacity": np.float64(0.03947368418956007),
            },
        )

        # Check the last result
        self.assertEqual(result.times[-1], 228000.0)
        self.assertEqual(result.inputs[-1], {"i": np.float64(-1.0)})
        self.assertEqual(
            result.outputs[-1],
            {"t": np.float64(18.76922976507518), "v": np.float64(3.0087596988706986)},
        )
        self.assertEqual(
            result.event_states[-1],
            {"EOD": np.float64(0.08759698870698607), "InsufficientCapacity": 0.0},
        )

    """
    Test that the combined model has the same result if print is True or False.

    This check is necessary since the event_state and output are calculated during the simulation 
    when print is True but only as needed when print is False due to LazySim optimization.
    For more details, refer to https://github.com/nasa/progpy/issues/199
    """

    def test_battery_electrochem_printed(self):
        config = {
            "save_freq": 1000,
            "dt": 2,
            "events": "InsufficientCapacity",
            "print": True,
        }

        config2 = {
            "save_freq": 1000,
            "dt": 2,
            "events": "InsufficientCapacity",
            "print": False,
        }

        def future_loading(t, x=None):
            load = 1

            if x is not None:
                event_state = batt.event_state(x)
                if event_state["EOD"] > 0.95:
                    load = 1  # Discharge
                elif event_state["EOD"] < 0.05:
                    load = -1  # Charge

            return batt.InputContainer({"i": load})

        def future_loading2(t, x=None):
            load = 1

            if x is not None:
                event_state = batt2.event_state(x)
                if event_state["EOD"] > 0.95:
                    load = 1  # Discharge
                elif event_state["EOD"] < 0.05:
                    load = -1  # Charge

            return batt2.InputContainer({"i": load})

        batt = BatteryElectroChem()
        batt2 = BatteryElectroChem()

        result = batt.simulate_to_threshold(future_loading, **config)
        result2 = batt2.simulate_to_threshold(future_loading2, **config2)

        self.assertEqual(result.times, result2.times)
        self.assertEqual(result.states, result2.states)
        self.assertEqual(result.event_states, result2.event_states)
        self.assertEqual(result.outputs, result2.outputs)

    def test_battery_electrochem_EOD(self):
        batt = BatteryElectroChemEOD()
        result = batt.simulate_to(200, future_loading, {"t": 18.95, "v": 4.183})

    def test_battery_simplified(self):
        batt = SimplifiedBattery()
        result = batt.simulate_to(200, future_loading_power, {"v": 4.183})

    def test_battery_electrochem_EOL(self):
        batt = BatteryElectroChemEOL()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(
            200, future_loading, {"t": 18.95, "v": 4.183}
        )

    def test_batt_namedtuple_access(self):
        batt = BatteryElectroChemEOL()
        named_results = batt.simulate_to(200, future_loading, {"t": 18.95, "v": 4.183})
        # Can't test for equality, sim values different each run. Test assignment
        times = named_results.times
        inputs = named_results.inputs
        states = named_results.states
        outputs = named_results.outputs
        event_states = named_results.event_states


# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Battery models")
    result = runner.run(load_test.loadTestsFromTestCase(TestBattery)).wasSuccessful()

    if not result:
        raise Exception("Failed test")


if __name__ == "__main__":
    main()
