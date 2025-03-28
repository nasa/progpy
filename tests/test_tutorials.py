# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

import importlib.util
from os.path import dirname, join
import sys
import unittest
import warnings
from testbook import testbook

sys.path.append(join(dirname(__file__), ".."))  # needed to access tutorial

class TestTutorials(unittest.TestCase):
    def run_notebook_test(self, notebook_path):
        with testbook(notebook_path, execute=True, timeout=300) as tb:
            self.assertEqual(tb.__class__.__name__, "TestbookNotebookClient")
        
    def test_notebook_tutorials(self):
        notebook_paths = [
            './tutorial.ipynb',
            './examples/00_Intro.ipynb',
            './examples/01_Simulation.ipynb',
            './examples/02_Parameter Estimation.ipynb',
            './examples/03_Existing Models.ipynb',
            './examples/04_New Models.ipynb',
            './examples/05_Data Driven.ipynb',
            './examples/06_Combining Models.ipynb',
            './examples/07_State Estimation.ipynb',
            './examples/08_Prediction.ipynb',
            './examples/09_Prognostic Example.ipynb',
            './examples/10_Prognostics Server.ipynb',
            './examples/2024PHMTutorial.ipynb'
        ]
        for notebook_path in notebook_paths:
                self.run_notebook_test(notebook_path)

def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials")
    result = runner.run(load_test.loadTestsFromTestCase(TestTutorials)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
    