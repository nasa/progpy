# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import unittest
from progpy.visualize import plot_scatter


class TestVisualize(unittest.TestCase):
    def test_scatter(self):
        # Nominal 
        data = [{'x': 1, 'y': 2, 'z': 3}, {'x': 1.5, 'y': 2.2, 'z': -1}, {'x': 0.9, 'y': 2.1, 'z': 7}]
        fig = plot_scatter(data)
        fig = plot_scatter(data, fig=fig)  # Add to figure
        plot_scatter(data, fig=fig, keys=['x', 'y', 'z'])  # All keys
        plot_scatter(data, keys=['y', 'z'])  # Subset of keys

        # Incorrect keys
        try:
            plot_scatter(data, keys=7)  # Not iterable
            self.fail()
        except Exception:
            pass

        try:
            plot_scatter(data, keys=['x', 'i'])  # Not present
            self.fail()
        except Exception:
            pass

        # Changing number of keys
        fig = plot_scatter(data)
        try:
            plot_scatter(data, fig=fig, keys=['y', 'z'])  # Different number of keys
            self.fail()
        except Exception:
            pass

        # Too few keys
        try:
            plot_scatter(data, keys=['y'])  # Only one key
            self.fail()
        except Exception:
            pass

# This allows the module to be executed directly    
def main():
    l = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Visualize")
    result = runner.run(l.loadTestsFromTestCase(TestVisualize)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()