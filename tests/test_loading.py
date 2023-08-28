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
            Piecewise({}, [1, 2], {'a': [1, 2, 3, 4]})

        # One too many loads
        with self.assertRaises(ValueError):
            Piecewise({}, [1, 2], {'a': [1]})

        # Should work with same number of loads
        Piecewise({}, [1, 2], {'a': [1, 2]})

        # Should work with one more load
        Piecewise({}, [1, 2], {'a': [1, 2, 3]})

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