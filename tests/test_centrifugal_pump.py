# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest

from progpy.models.centrifugal_pump import CentrifugalPump, CentrifugalPumpWithWear, CentrifugalPumpBase


class TestCentrifugalPump(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_centrifugal_pump_base(self):
        pump = CentrifugalPumpBase(process_noise=0)

        cycle_time = 3600

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time / 2.0:
                V = 471.2389
            elif t < cycle_time / 2 + 100:
                V = 471.2389 + (t - cycle_time / 2)
            elif t < cycle_time - 100:
                V = 571.2389
            else:
                V = 471.2398 - (t - cycle_time)

            return {
                'Tamb': 290,
                'V': V,
                'pdisch': 928654,
                'psuc': 239179,
                'wsync': V * 0.8
            }

        x0 = pump.initialize(future_loading(0))
        # Same as with wear except without wA, wRadial, wThrust
        x0_test = {
            'w': 376.991118431,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'Q': 0.0,
            'A': 12.7084,
            'QLeak': -8.303463132934355e-08
        }
        for key in pump.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)

        x = pump.next_state(x0, future_loading(0), 1)
        x_test = {
            'w': 372.68973081274,
            'Q': 0.017417462,
            'Tt': 290.027256332,
            'Tr': 290.1065917275,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'QLeak': -8.303463132934355e-08
        }
        for key in pump.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = pump.output(x)
        z_test = {
            'Qout': 0.0174,
            'To': 290,
            'Tr': 290.1066,
            'Tt': 290.0273,
            'w': 372.6897
        }

        for key in pump.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 4)

        # Wear rates are parameters instead of states
        pump.parameters['wA'] = 1e-2
        pump.parameters['wThrust'] = 1e-10
        (times, inputs, states, outputs, event_states) = pump.simulate_to_threshold(future_loading, pump.output(
            pump.initialize(future_loading(0), {})))
        self.assertAlmostEqual(times[-1], 23892)

    def test_centrifugal_pump_with_wear(self):
        pump = CentrifugalPumpWithWear(process_noise=0)

        cycle_time = 3600

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time / 2.0:
                V = 471.2389
            elif t < cycle_time / 2 + 100:
                V = 471.2389 + (t - cycle_time / 2)
            elif t < cycle_time - 100:
                V = 571.2389
            else:
                V = 471.2398 - (t - cycle_time)

            return {
                'Tamb': 290,
                'V': V,
                'pdisch': 928654,
                'psuc': 239179,
                'wsync': V * 0.8
            }

        x0 = pump.initialize(future_loading(0))
        x0_test = {
            'w': 376.991118431,
            'wA': 0.00,
            'wRadial': 0.0,
            'wThrust': 0,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'Q': 0.0,
            'A': 12.7084,
            'QLeak': -8.303463132934355e-08
        }
        for key in pump.states:
            self.assertAlmostEqual(x0[key], x0_test[key], 7)
        x = pump.next_state(x0, future_loading(0), 1)
        x_test = {
            'w': 372.68973081274,
            'Q': 0.017417462,
            'Tt': 290.027256332,
            'Tr': 290.1065917275,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'wA': 0.0,
            'wThrust': 0,
            'wRadial': 0,
            'QLeak': -8.303463132934355e-08
        }
        for key in pump.states:
            self.assertAlmostEqual(x[key], x_test[key], 7)

        z = pump.output(x)
        z_test = {
            'Qout': 0.0174,
            'To': 290,
            'Tr': 290.1066,
            'Tt': 290.0273,
            'w': 372.6897
        }

        for key in pump.outputs:
            self.assertAlmostEqual(z[key], z_test[key], 4)

        pump.parameters['x0']['wA'] = 1e-2
        pump.parameters['x0']['wThrust'] = 1e-10
        (times, inputs, states, outputs, event_states) = pump.simulate_to_threshold(future_loading, pump.output(
            pump.initialize(future_loading(0), {})))
        self.assertAlmostEqual(times[-1], 23892)

        # Check warning when changing overwritten Parameters
        with self.assertWarns(UserWarning):
            pump.parameters['wA'] = 1e-2

        with self.assertWarns(UserWarning):
            pump.parameters['wRadial'] = 1e-2

        with self.assertWarns(UserWarning):
            pump.parameters['wThrust'] = 1e-10

    def test_centrifugal_pump(self):
        self.assertEqual(CentrifugalPump, CentrifugalPumpWithWear)
        # CentrifugalPump is alias for "with wear"

    def test_centrifugal_pump_namedtuple_access(self):
        pump = CentrifugalPumpWithWear(process_noise=0)

        cycle_time = 3600

        def future_loading(t, x=None):
            t = t % cycle_time
            if t < cycle_time / 2.0:
                V = 471.2389
            elif t < cycle_time / 2 + 100:
                V = 471.2389 + (t - cycle_time / 2)
            elif t < cycle_time - 100:
                V = 571.2389
            else:
                V = 471.2398 - (t - cycle_time)

            return {
                'Tamb': 290,
                'V': V,
                'pdisch': 928654,
                'psuc': 239179,
                'wsync': V * 0.8
            }

        x0 = pump.initialize(future_loading(0))
        x0_test = {
            'w': 376.991118431,
            'wA': 0.00,
            'wRadial': 0.0,
            'wThrust': 0,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'Tt': 290,
            'Tr': 290,
            'To': 290,
            'Q': 0.0,
            'A': 12.7084,
            'QLeak': -8.303463132934355e-08
        }

        x = pump.next_state(x0, future_loading(0), 1)
        x_test = {
            'w': 372.68973081274,
            'Q': 0.017417462,
            'Tt': 290.027256332,
            'Tr': 290.1065917275,
            'To': 290,
            'A': 12.7084,
            'rThrust': 1.4e-6,
            'rRadial': 1.8e-6,
            'wA': 0.0,
            'wThrust': 0,
            'wRadial': 0,
            'QLeak': -8.303463132934355e-08
        }

        z = pump.output(x)
        z_test = {
            'Qout': 0.0174,
            'To': 290,
            'Tr': 290.1066,
            'Tt': 290.0273,
            'w': 372.6897
        }

        pump.parameters['x0']['wA'] = 1e-2
        pump.parameters['x0']['wThrust'] = 1e-10
        named_results = pump.simulate_to_threshold(future_loading, pump.output(pump.initialize(future_loading(0), {})))
        times = named_results.times
        inputs = named_results.inputs
        states = named_results.states
        outputs = named_results.outputs
        event_states = named_results.event_states


# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Centrifugal Pump Model")
    result = runner.run(load_test.loadTestsFromTestCase(TestCentrifugalPump)).wasSuccessful()

    if not result:
        raise Exception("Failed test")


if __name__ == '__main__':
    main()
