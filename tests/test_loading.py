# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest

from progpy.loading import Piecewise

class Testloading(unittest.TestCase):
    def test_piecewise_construction(self):
        """
        Test a series of  cases when constructing a Piecewise Loading object
        """

        # One too many loads
        with self.assertRaises(ValueError):
            Piecewise(dict, [1, 2], {'a': [1, 2, 3, 4]})

        # One too many loads
        with self.assertRaises(ValueError):
            Piecewise(dict, [1, 2], {'a': [1]})

        # Should work with same number of loads
        l = Piecewise(dict, [1, 2], {'a': [1, 2]})
        fig = l.plot([0, 1, 1.9])
        # Plot shouldn't work if provided a time after the end
        with self.assertRaises(StopIteration):
            l.plot([0, 1, 2])

        # Should work with one more load
        l = Piecewise(dict, [1, 2], {'a': [1.1, 2.1, 1.6]})
        # Plot (using same plot (to test that feature))
        # This time with a time after the last time, now it should work
        l.plot([0, 1, 1.9, 3], fig=fig)

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