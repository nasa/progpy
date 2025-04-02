# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from os.path import dirname, join
import sys
import unittest
from testbook import testbook

sys.path.append(join(dirname(__file__), ".."))

class TestTutorialsPartTwo(unittest.TestCase):
    def run_notebook_test(self, notebook_path):
        with testbook(notebook_path, execute=True, timeout=1200) as tb:
            self.assertEqual(tb.__class__.__name__, "TestbookNotebookClient")
        
    def test_notebook_tutorials(self):
        notebook_paths = [
            './examples/08_Prediction.ipynb',
            './examples/09_Prognostic Example.ipynb'
        ]
        for notebook_path in notebook_paths:
                self.run_notebook_test(notebook_path)

def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Tutorials - Part 3")
    result = runner.run(load_test.loadTestsFromTestCase(TestTutorialsPartTwo)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
    