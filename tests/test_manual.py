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
        self.assertEqual(
            desc["procedure"],
            "Uniform random walk discharge at room temperature with variable recharge duration",
        )
        self.assertEqual(
            desc["description"],
            "Experiment consisting of repeated iteration of a randomized series of discharging pulses followed by a recharging period of variable length. Batteries are charged and discharged at room temperature",
        )
        self.assertDictEqual(
            desc["runs"][0],
            {
                "type": "D",
                "desc": "low current discharge at 0.04A",
                "date": "30-Dec-2013 15:53:29",
            },
        )
        self.assertDictEqual(
            desc["runs"][8532],
            {"type": "R", "desc": "rest (random walk)", "date": "22-Feb-2014 07:45:49"},
        )
        self.assertDictEqual(
            desc["runs"][-1],
            {
                "type": "D",
                "desc": "discharge (random walk)",
                "date": "02-Jun-2014 16:43:48",
            },
        )

        # Verifying data
        assert_array_equal(
            data[0].columns,
            pd.core.indexes.base.Index(
                ["relativeTime", "current", "voltage", "temperature"], dtype="object"
            ),
        )

        self.assertEqual(data[0]["current"][15], 0.04)
        assert_array_equal(
            data[0].iloc[-1],
            array([1.8897668e05, 4.0000000e-02, 3.2000000e00, 1.7886300e01]),
        )
        assert_array_equal(
            data[8532].iloc[0],
            array([1.000000e-02, 0.000000e00, 3.645000e00, 3.124247e01]),
        )
        assert_array_equal(data[8532].iloc[-1], array([0.54, 0, 3.716, 31.24247]))
        assert_array_equal(data[-1].iloc[0], array([0.04, 3.004, 3.647, 28.08937]))
        assert_array_equal(data[-1].iloc[-1], array([178.38, 3, 3.2, 32.53947]))

    def test_nasa_cmapss_download(self):
        (train, test, results) = nasa_cmapss.load_data(1)

        # Testing train data
        assert_array_equal(
            train.iloc[0],
            array(
                [
                    1.00000e00,
                    1.00000e00,
                    2.30000e-03,
                    3.00000e-04,
                    1.00000e02,
                    5.18670e02,
                    6.43020e02,
                    1.58529e03,
                    1.39821e03,
                    1.46200e01,
                    2.16100e01,
                    5.53900e02,
                    2.38804e03,
                    9.05017e03,
                    1.30000e00,
                    4.72000e01,
                    5.21720e02,
                    2.38803e03,
                    8.12555e03,
                    8.40520e00,
                    3.00000e-02,
                    3.92000e02,
                    2.38800e03,
                    1.00000e02,
                    3.88600e01,
                    2.33735e01,
                ]
            ),
        )
        assert_array_equal(
            train.iloc[-1],
            array(
                [
                    1.00000e02,
                    1.98000e02,
                    1.30000e-03,
                    3.00000e-04,
                    1.00000e02,
                    5.18670e02,
                    6.42950e02,
                    1.60162e03,
                    1.42499e03,
                    1.46200e01,
                    2.16100e01,
                    5.52480e02,
                    2.38806e03,
                    9.15503e03,
                    1.30000e00,
                    4.78000e01,
                    5.21070e02,
                    2.38805e03,
                    8.21464e03,
                    8.49030e00,
                    3.00000e-02,
                    3.96000e02,
                    2.38800e03,
                    1.00000e02,
                    3.87000e01,
                    2.31855e01,
                ]
            ),
        )
        assert_array_equal(
            train.iloc[6548],
            array(
                [
                    5.20000e01,
                    6.60000e01,
                    -1.90000e-03,
                    -0.00000e00,
                    1.00000e02,
                    5.18670e02,
                    6.42070e02,
                    1.58397e03,
                    1.39125e03,
                    1.46200e01,
                    2.16100e01,
                    5.54590e02,
                    2.38804e03,
                    9.05261e03,
                    1.30000e00,
                    4.71200e01,
                    5.22480e02,
                    2.38803e03,
                    8.13633e03,
                    8.39150e00,
                    3.00000e-02,
                    3.92000e02,
                    2.38800e03,
                    1.00000e02,
                    3.90500e01,
                    2.34304e01,
                ]
            ),
        )

        # Testing test data
        assert_array_equal(
            test.iloc[0],
            array(
                [
                    1.00000e00,
                    1.00000e00,
                    -7.00000e-04,
                    -4.00000e-04,
                    1.00000e02,
                    5.18670e02,
                    6.41820e02,
                    1.58970e03,
                    1.40060e03,
                    1.46200e01,
                    2.16100e01,
                    5.54360e02,
                    2.38806e03,
                    9.04619e03,
                    1.30000e00,
                    4.74700e01,
                    5.21660e02,
                    2.38802e03,
                    8.13862e03,
                    8.41950e00,
                    3.00000e-02,
                    3.92000e02,
                    2.38800e03,
                    1.00000e02,
                    3.90600e01,
                    2.34190e01,
                ]
            ),
        )
        assert_array_equal(
            test.iloc[-1],
            array(
                [
                    1.00000e02,
                    2.00000e02,
                    -3.20000e-03,
                    -5.00000e-04,
                    1.00000e02,
                    5.18670e02,
                    6.43850e02,
                    1.60038e03,
                    1.43214e03,
                    1.46200e01,
                    2.16100e01,
                    5.50790e02,
                    2.38826e03,
                    9.06148e03,
                    1.30000e00,
                    4.82000e01,
                    5.19300e02,
                    2.38826e03,
                    8.13733e03,
                    8.50360e00,
                    3.00000e-02,
                    3.96000e02,
                    2.38800e03,
                    1.00000e02,
                    3.83700e01,
                    2.30522e01,
                ]
            ),
        )
        assert_array_equal(
            test.iloc[6548],
            array(
                [
                    3.30000e01,
                    1.37000e02,
                    1.70000e-03,
                    2.00000e-04,
                    1.00000e02,
                    5.18670e02,
                    6.42380e02,
                    1.58655e03,
                    1.41089e03,
                    1.46200e01,
                    2.16100e01,
                    5.53960e02,
                    2.38807e03,
                    9.06359e03,
                    1.30000e00,
                    4.74500e01,
                    5.21950e02,
                    2.38805e03,
                    8.14151e03,
                    8.43050e00,
                    3.00000e-02,
                    3.91000e02,
                    2.38800e03,
                    1.00000e02,
                    3.90000e01,
                    2.33508e01,
                ]
            ),
        )

        # Testing results
        assert_array_equal(
            results,
            array(
                [
                    112.0,
                    98.0,
                    69.0,
                    82.0,
                    91.0,
                    93.0,
                    91.0,
                    95.0,
                    111.0,
                    96.0,
                    97.0,
                    124.0,
                    95.0,
                    107.0,
                    83.0,
                    84.0,
                    50.0,
                    28.0,
                    87.0,
                    16.0,
                    57.0,
                    111.0,
                    113.0,
                    20.0,
                    145.0,
                    119.0,
                    66.0,
                    97.0,
                    90.0,
                    115.0,
                    8.0,
                    48.0,
                    106.0,
                    7.0,
                    11.0,
                    19.0,
                    21.0,
                    50.0,
                    142.0,
                    28.0,
                    18.0,
                    10.0,
                    59.0,
                    109.0,
                    114.0,
                    47.0,
                    135.0,
                    92.0,
                    21.0,
                    79.0,
                    114.0,
                    29.0,
                    26.0,
                    97.0,
                    137.0,
                    15.0,
                    103.0,
                    37.0,
                    114.0,
                    100.0,
                    21.0,
                    54.0,
                    72.0,
                    28.0,
                    128.0,
                    14.0,
                    77.0,
                    8.0,
                    121.0,
                    94.0,
                    118.0,
                    50.0,
                    131.0,
                    126.0,
                    113.0,
                    10.0,
                    34.0,
                    107.0,
                    63.0,
                    90.0,
                    8.0,
                    9.0,
                    137.0,
                    58.0,
                    118.0,
                    89.0,
                    116.0,
                    115.0,
                    136.0,
                    28.0,
                    38.0,
                    20.0,
                    85.0,
                    55.0,
                    128.0,
                    137.0,
                    82.0,
                    59.0,
                    117.0,
                    20.0,
                ]
            ),
        )

    def test_dataset_example(self):
        with patch("matplotlib.pyplot.show"):
            dataset.run_example()

    def test_sim_battery_eol_example(self):
        with patch("matplotlib.pyplot.show"):
            sim_battery_eol.run_example()

    def test_custom_model_example(self):
        with patch("matplotlib.pyplot.show"):
            custom_model.run_example()


# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Manual")
    with patch("matplotlib.pyplot.show"):
        result = runner.run(load_test.loadTestsFromTestCase(TestManual)).wasSuccessful()
    plt.close("all")

    if not result:
        raise Exception("Failed test")


if __name__ == "__main__":
    main()
