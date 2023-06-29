# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import unittest


class TestDatasets(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    # Bad URL tests
    def test_nasa_battery_bad_url_download(self):
        from progpy.datasets import nasa_battery
        nasa_battery.urls = {'RW1': "https://BADURLTEST"}
        with self.assertRaises(ConnectionError):
            (desc, data) = nasa_battery.load_data(1)
        # Legit website, but it's not the repos
        nasa_battery.urls = {'RW1': "https://github.com/nasa/progpy"}
        with self.assertRaises(ConnectionError):
            (desc, data) = nasa_battery.load_data(1)

    def test_nasa_cmapss_bad_url_download(self):
        from progpy.datasets import nasa_cmapss
        BAD_URL = "https://"+"BADURLTEST"
        nasa_cmapss.URL = BAD_URL
        with self.assertRaises(ConnectionError):
            (train, test, results) = nasa_cmapss.load_data(1)
        # Legit website, but it's not the repos
        nasa_cmapss.URL = "https://github.com/nasa/progpy"
        with self.assertRaises(ConnectionError):
            (train, test, results) = nasa_cmapss.load_data(1)
    # Testing for successful download located in manual testing files; test_manual.py

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Datasets")
    result = runner.run(load_test.loadTestsFromTestCase(TestDatasets)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
