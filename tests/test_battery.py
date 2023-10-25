# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from progpy.models import BatteryCircuit, BatteryElectroChem, BatteryElectroChemEOL, BatteryElectroChemEOD, BatteryElectroChemEODEOL
from progpy.loading import Piecewise

# Variable (piece-wise) future loading scheme 
future_loading = Piecewise(
    dict,
    [600, 900, 1800, 3000, float('inf')],
    {'i': [2, 1, 4, 2, 3]})


class TestBattery(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__
    
    def test_battery_circuit(self):
        batt = BatteryCircuit()
        result = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_battery_electrochem(self):
        batt = BatteryElectroChem()
        result = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})
        self.assertEqual(BatteryElectroChem, BatteryElectroChemEODEOL)

        # check warning raised when changing overwritten parameter
        with self.assertWarns(UserWarning):
            batt.parameters['Ro'] = 10
        
        with self.assertWarns(UserWarning):
            batt.parameters['qMobile'] = 10

        with self.assertWarns(UserWarning):
            batt.parameters['tDiffusion'] = 10

    def test_battery_electrochem_EOD(self):
        batt = BatteryElectroChemEOD()
        result = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_battery_electrochem_EOL(self):
        batt = BatteryElectroChemEOL()
        (times, inputs, states, outputs, event_states) = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})

    def test_batt_namedtuple_access(self):
        batt = BatteryElectroChemEOL()
        named_results = batt.simulate_to(200, future_loading, {'t': 18.95, 'v': 4.183})
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

if __name__ == '__main__':
    main()
