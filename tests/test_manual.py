# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
from os.path import dirname, join
from matplotlib import pyplot as plt
from numpy import array
from numpy.testing import assert_array_equal
import pandas as pd
import sys
import unittest
from unittest.mock import patch

sys.path.append(join(dirname(__file__), ".."))  # Needed to access examples
from examples import dataset, sim_battery_eol, custom_model, playback

from progpy.datasets import nasa_cmapss, nasa_battery

"""
This file includes tests that are too long to be run as part of the automated tests. Instead, these tests are run manually as part of the release process.
"""


class TestManual(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_playback_example(self):
        playback.run_example()

    def test_nasa_battery_download(self):
        (desc, data) = nasa_battery.load_data(1)
        
        # Verifying desc
        self.assertEqual(desc['procedure'], "Uniform random walk discharge at room temperature with variable recharge duration")
        self.assertEqual(desc['description'], "Experiment consisting of repeated iteration of a randomized series of discharging pulses followed by a recharging period of variable length. Batteries are charged and discharged at room temperature")
        self.assertDictEqual(desc['runs'][0], {'type': 'D', 'desc': 'low current discharge at 0.04A', 'date': '30-Dec-2013 15:53:29'})
        self.assertDictEqual(desc['runs'][8532], {'type': 'R', 'desc': 'rest (random walk)', 'date': '22-Feb-2014 07:45:49'})
        self.assertDictEqual(desc['runs'][-1], {'type': 'D', 'desc': 'discharge (random walk)', 'date': '02-Jun-2014 16:43:48'})

        # Verifying data
        assert_array_equal(data[0].columns, pd.core.indexes.base.Index(['relativeTime', 'current', 'voltage', 'temperature'], dtype='object'))
        
        self.assertEqual(data[0]['current'][15], 0.04)
        assert_array_equal(data[0].iloc[-1], array([1.8897668e+05, 4.0000000e-02, 3.2000000e+00, 1.7886300e+01]))
        assert_array_equal(data[8532].iloc[0], array([1.000000e-02, 0.000000e+00, 3.645000e+00, 3.124247e+01]))
        assert_array_equal(data[8532].iloc[-1], array([0.54, 0, 3.716, 31.24247]))
        assert_array_equal(data[-1].iloc[0], array([0.04, 3.004, 3.647, 28.08937]))
        assert_array_equal(data[-1].iloc[-1], array([178.38, 3, 3.2, 32.53947]))

    def test_nasa_cmapss_download(self):
        (train, test, results) = nasa_cmapss.load_data(1)
        
        # Testing train data
        assert_array_equal(train.iloc[0], array([1.00000e+00, 1.00000e+00, 2.30000e-03, 3.00000e-04, 1.00000e+02, 5.18670e+02, 6.43020e+02, 1.58529e+03, 1.39821e+03, 1.46200e+01, 2.16100e+01, 5.53900e+02, 2.38804e+03, 9.05017e+03, 1.30000e+00, 4.72000e+01, 5.21720e+02, 2.38803e+03, 8.12555e+03, 8.40520e+00, 3.00000e-02, 3.92000e+02, 2.38800e+03, 1.00000e+02, 3.88600e+01, 2.33735e+01]))
        assert_array_equal(train.iloc[-1], array([1.00000e+02, 1.98000e+02, 1.30000e-03, 3.00000e-04, 1.00000e+02, 5.18670e+02, 6.42950e+02, 1.60162e+03, 1.42499e+03, 1.46200e+01, 2.16100e+01, 5.52480e+02, 2.38806e+03, 9.15503e+03, 1.30000e+00, 4.78000e+01, 5.21070e+02, 2.38805e+03, 8.21464e+03, 8.49030e+00, 3.00000e-02, 3.96000e+02, 2.38800e+03, 1.00000e+02, 3.87000e+01, 2.31855e+01]))
        assert_array_equal(train.iloc[6548], array([5.20000e+01,  6.60000e+01, -1.90000e-03, -0.00000e+00, 1.00000e+02,  5.18670e+02,  6.42070e+02,  1.58397e+03, 1.39125e+03,  1.46200e+01,  2.16100e+01,  5.54590e+02, 2.38804e+03,  9.05261e+03,  1.30000e+00,  4.71200e+01, 5.22480e+02,  2.38803e+03,  8.13633e+03,  8.39150e+00, 3.00000e-02,  3.92000e+02,  2.38800e+03,  1.00000e+02, 3.90500e+01,  2.34304e+01]))

        # Testing test data
        assert_array_equal(test.iloc[0], array([ 1.00000e+00,  1.00000e+00, -7.00000e-04, -4.00000e-04, 1.00000e+02,  5.18670e+02,  6.41820e+02,  1.58970e+03, 1.40060e+03,  1.46200e+01,  2.16100e+01,  5.54360e+02, 2.38806e+03,  9.04619e+03,  1.30000e+00,  4.74700e+01, 5.21660e+02,  2.38802e+03,  8.13862e+03,  8.41950e+00, 3.00000e-02,  3.92000e+02,  2.38800e+03,  1.00000e+02, 3.90600e+01,  2.34190e+01]))
        assert_array_equal(test.iloc[-1], array([ 1.00000e+02,  2.00000e+02, -3.20000e-03, -5.00000e-04, 1.00000e+02,  5.18670e+02,  6.43850e+02,  1.60038e+03, 1.43214e+03,  1.46200e+01,  2.16100e+01,  5.50790e+02, 2.38826e+03,  9.06148e+03,  1.30000e+00,  4.82000e+01, 5.19300e+02,  2.38826e+03,  8.13733e+03,  8.50360e+00, 3.00000e-02,  3.96000e+02,  2.38800e+03,  1.00000e+02, 3.83700e+01,  2.30522e+01]))
        assert_array_equal(test.iloc[6548], array([3.30000e+01, 1.37000e+02, 1.70000e-03, 2.00000e-04, 1.00000e+02, 5.18670e+02, 6.42380e+02, 1.58655e+03, 1.41089e+03, 1.46200e+01, 2.16100e+01, 5.53960e+02, 2.38807e+03, 9.06359e+03, 1.30000e+00, 4.74500e+01, 5.21950e+02, 2.38805e+03, 8.14151e+03, 8.43050e+00, 3.00000e-02, 3.91000e+02, 2.38800e+03, 1.00000e+02, 3.90000e+01, 2.33508e+01]))
        
        # Testing results
        assert_array_equal(results, array([112.,  98.,  69.,  82.,  91.,  93.,  91.,  95., 111.,  96.,  97.,
       124.,  95., 107.,  83.,  84.,  50.,  28.,  87.,  16.,  57., 111.,
       113.,  20., 145., 119.,  66.,  97.,  90., 115.,   8.,  48., 106.,
         7.,  11.,  19.,  21.,  50., 142.,  28.,  18.,  10.,  59., 109.,
       114.,  47., 135.,  92.,  21.,  79., 114.,  29.,  26.,  97., 137.,
        15., 103.,  37., 114., 100.,  21.,  54.,  72.,  28., 128.,  14.,
        77.,   8., 121.,  94., 118.,  50., 131., 126., 113.,  10.,  34.,
       107.,  63.,  90.,   8.,   9., 137.,  58., 118.,  89., 116., 115.,
       136.,  28.,  38.,  20.,  85.,  55., 128., 137.,  82.,  59., 117.,
        20.]))

    def test_dataset_example(self):
        with patch('matplotlib.pyplot.show'):
            dataset.run_example()

    def test_sim_battery_eol_example(self):
        with patch('matplotlib.pyplot.show'):
            sim_battery_eol.run_example()

    def test_custom_model_example(self):
        with patch('matplotlib.pyplot.show'):
            custom_model.run_example()

# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Manual")
    with patch('matplotlib.pyplot.show'):
        result = runner.run(load_test.loadTestsFromTestCase(TestManual)).wasSuccessful()
    plt.close('all')

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
