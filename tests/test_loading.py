# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import unittest

from progpy.loading import Piecewise, GaussianNoiseWrapper


class Testloading(unittest.TestCase):
    def test_piecewise_construction(self):
        """
        Test a series of  cases when constructing a Piecewise Loading object
        """

        # One too many loads
        with self.assertRaises(ValueError):
            Piecewise({}, [1, 2], {'a': [1, 2, 3, 4]})

        # One too many loads
        with self.assertRaises(ValueError):
            Piecewise({}, [1, 2], {'a': [1]})

        # Should work with same number of loads
        Piecewise({}, [1, 2], {'a': [1, 2]})

        # Should work with one more load
        Piecewise({}, [1, 2], {'a': [1, 2, 3]})
    
    def test_gaussian_seed(self):
        def loading(t, x=None):
            return {'a': 10}
        
        # Default: two values should be different (because of randomness)
        loading_with_noise = GaussianNoiseWrapper(loading, 10)
        load1 = loading_with_noise(10)

        loading_with_noise = GaussianNoiseWrapper(loading, 10)
        load2 = loading_with_noise(10)

        self.assertNotEqual(load1['a'], load2['a'])

        # Setting seed, two values should be the same now
        loading_with_noise = GaussianNoiseWrapper(loading, 10, seed=550)
        load1 = loading_with_noise(10)

        loading_with_noise = GaussianNoiseWrapper(loading, 10, seed=550)
        load2 = loading_with_noise(10)
        self.assertEqual(load1['a'], load2['a'])
    
    def test_wedge_gaussian_load(self):
        def loading(t, x=None):
            return {'a': 10}
        
        loading_with_noise = GaussianNoiseWrapper(loading, 0, std_slope=10)

        # Standard deviation at t=0 should be 0.
        self.assertEqual(
            loading_with_noise(0),
            loading_with_noise(0))

        # At any greater time, the standard deviation is not 0
        self.assertNotEqual(
            loading_with_noise(10),
            loading_with_noise(10))

        # Check that standard deviation increases with time. 
        values = [loading_with_noise(10)['a'] for _ in range(20)]
        std = np.std(values)

        values = [loading_with_noise(1e4)['a'] for _ in range(20)]
        std2 = np.std(values)
        self.assertGreater(std2, std)

        # Check t0 functionality
        loading_with_noise.t0 = 10
        self.assertEqual(  # Case t<t0
            loading_with_noise(5),
            loading_with_noise(5))
        self.assertEqual(  # Case t==t0
            loading_with_noise(10),
            loading_with_noise(10))
        self.assertNotEqual(  # Case t > t0
            loading_with_noise(11),
            loading_with_noise(11))

# This allows the module to be executed directly    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Loading")
    result = runner.run(l.loadTestsFromTestCase(Testloading)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()