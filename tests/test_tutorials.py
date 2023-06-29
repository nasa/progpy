# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import importlib.util
from os.path import dirname, join
import sys
import unittest
import warnings

sys.path.append(join(dirname(__file__), ".."))  # needed to access tutorial

class TestTutorials(unittest.TestCase):
    def test_tutorial_ipynb(self):
        if importlib.util.find_spec('testbook') is None:
            warnings.warn('testbook not installed')
        else:
            from testbook import testbook
            with testbook('./tutorial.ipynb', execute=True) as tb:
                self.assertEqual(tb.__class__.__name__, "TestbookNotebookClient")

def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(load_test.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
    