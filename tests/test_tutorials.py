# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from os.path import dirname, join
import sys
import unittest
from testbook import testbook

sys.path.append(join(dirname(__file__), ".."))

class TestTutorials(unittest.TestCase):
    def run_notebook_test(self, notebook_path):
        with testbook(notebook_path, execute=True, timeout=300) as tb:
            self.assertEqual(tb.__class__.__name__, "TestbookNotebookClient")
        
    def test_notebook_tutorials(self):
        notebook_paths = [
            './examples/00_Intro.ipynb',
            './examples/01_Simulation.ipynb',
            './examples/02_Parameter Estimation.ipynb',
            './examples/03_Existing Models.ipynb',
            './examples/04_New Models.ipynb',
            './examples/05_Data Driven.ipynb',
        ]
        for notebook_path in notebook_paths:
                self.run_notebook_test(notebook_path)

def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials - Part 1")
    result = runner.run(load_test.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
    