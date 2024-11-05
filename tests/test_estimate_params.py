# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import numpy as np
import unittest

from progpy.models import ThrownObject
from progpy.models.test_models.other_models import OneInputTwoOutputsOneEvent


class TestEstimateParams(unittest.TestCase):
    def test_estimate_params_works(self):
        """
        Base Cases
        """
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()

        self.assertEqual(m.parameters, gt)

        # Now lets incorrectly set some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed']
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys)
        for key in keys: 
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # Incorrectly set parameters, but provide a bounds for what said values should be.
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 4), (20, 42)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        # Checking to see if our estimated parameters are within the bounds.
        self.assertLessEqual(m.parameters['thrower_height'], 4)
        self.assertGreaterEqual(m.parameters['thrower_height'], 0)
        self.assertLessEqual(m.parameters['throwing_speed'], 42)
        self.assertGreaterEqual(m.parameters['throwing_speed'], 20)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 4), (40, 40)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 1)
        self.assertLessEqual(m.parameters['thrower_height'], 4)
        self.assertGreaterEqual(m.parameters['thrower_height'], 0)
        self.assertEqual(m.parameters['throwing_speed'], 40)

        # Testing initial parameters that are not within the defined bounds; estimated parameters are not a good fit
        m.parameters['thrower_height'] = 10
        m.parameters['throwing_speed'] = -10
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 4), (20, 42)))
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 2)
        # These returned values are still within the bounds, they are just not close to the original value at all.
        # This results in our estimate parameters to result in the upper extremes
        self.assertAlmostEqual(m.parameters['thrower_height'], 4)
        self.assertAlmostEqual(m.parameters['throwing_speed'], 42)

        # Testing convergence with the same initial parameters but no given bounds were provided        
        m.parameters['thrower_height'] = 4
        m.parameters['throwing_speed'] = 24
        m.parameters['g'] = -20
        keys = ['thrower_height', 'throwing_speed', 'g']
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

    def test_layered_params(self):
        """
        Testing case where optimized state is layered (e.g., ['x0]['a']). I.e., stored in a dict or other structure.
        """
        m = OneInputTwoOutputsOneEvent()

        ts = [0, 1, 2]
        us = [{'u0': 1}]*3
        zs = [ # corresponds to x0 = 1 and a = 2
            {'x0+b': np.float64(2.0), 'x0+c': np.float64(2.0)},
            {'x0+b': np.float64(4.0), 'x0+c': np.float64(4.0)},
            {'x0+b': np.float64(6.0), 'x0+c': np.float64(6.0)}]

        keys = ['a', ('x0', 'x0')]

        m.estimate_params(times=ts, inputs=us, outputs=zs, keys=keys)

        self.assertAlmostEqual(m['x0']['x0'], 1, delta=1e-4)
        self.assertAlmostEqual(m['a'], 2, delta=1e-4)

    def test_estimate_params(self):
        """
        General estimate_params testing
        """
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()

        # Reset some parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed', 'g']

        # Need at least one data point
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[], inputs=[], outputs=[])
        self.assertEqual(
            'Times, inputs, and outputs must have at least one element',
            str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=None, inputs=None, output=None)
        self.assertEqual(
            'Missing keyword arguments times, inputs, outputs',
            str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times='', inputs='', outputs='')
        self.assertEqual(
            'Times, inputs, and outputs must have at least one element',
            str(cm.exception) 
        )

        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[[]], inputs=[[]], outputs=[[]])
        self.assertEqual(
            'Times, inputs, and outputs for Run 0 must have at least one element',
            str(cm.exception)
        )

        # Checking when one parameter is valid but others are not.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[], inputs=[], outputs=results.outputs)
        self.assertEqual(
             'Times, inputs, and outputs must be same length. Length of times: 0, Length of inputs: 0, Length of outputs: 9',
             str(cm.exception)
        )
        
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=results.times, inputs=None, outputs=None)
        self.assertEqual(
            'Missing keyword arguments inputs, outputs',
            str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=None)
        self.assertEqual(
            'Missing keyword arguments outputs',
            str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=None, inputs=[None], outputs=[])
        self.assertEqual(
            'Missing keyword arguments times',
            str(cm.exception)
        )

        # Strings are iterables by definition so this would pass as well, regardless if they are ints
        # Later tests will ensure they must be ints.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times= None, inputs = 'Testing', outputs='Testing')
        self.assertEqual(
            'Missing keyword arguments times',
            str(cm.exception)
        )

        # Now with bounds that don't include the true values
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 4), (20, 37), (-20, 0)))
        # Checking each key to see if they are not equal to the original parameters
        for key in keys:
            self.assertNotAlmostEqual(m.parameters[key], gt[key], 1)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Now with bounds that do include the true values
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 8), (20, 42), (-20, -5)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # Lower Bound Greater than Upper Bound
        # Error called by minimize function.
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((30, 7), (20, 42), (-20, 0)))

        # Testing all bounds are incorrect
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((30, 7), (40, 20), (0, -20)))

        # Implement different variations of lists and tuples and see if they work as intended
        # Try incomplete list:
        with self.assertRaises(ValueError):
            # Missing bound
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 4), (20, 42)))
        with self.assertRaises(ValueError):
            # Extra bound
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((0, 4), (20, 42), (-20, 0), (-20, 10)))

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Dictionary bounds
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42), 'g': (-20, 0)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Dictionary bounds - missing
        # Will fill with (-inf, inf)
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Dictionary bounds - extra & garbage key
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42), 'g': (-20, 0), 'dummy': (-50, 0)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Dictionary bounds - extra & not garbage key
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'thrower_height': (0, 4), 'throwing_speed': (20, 42), 'g': (-20, 0), 'rho': (-100, 100)})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        self.assertEqual(m.parameters['rho'], gt['rho'])

        # Bounds - wrong type
        with self.assertRaises(ValueError):
            # bounds isn't tuple or dict
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=0)
        with self.assertRaises(ValueError):
            # bounds isn't tuple or dict
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds='a')
        with self.assertRaises(ValueError):
            # Item isn't a tuple
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'g': 7})

        # Passing in bounds as a dictionary with tuples
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'g': (7, 14)})

        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=None)
        with self.assertRaises(ValueError):
            # Tuple isn't of size 2, more specifically, tuple is size less than 2
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'g': (7,)})
        with self.assertRaises(ValueError):
            # Tuple is a size greater than 2
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'g': (7, 8, 9)})
        with self.assertRaises(ValueError):
            # Item is a list of length 1
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds={'g': [7]})

        # With inputs, outputs, and times
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=[results.times], inputs=[results.inputs], outputs=[results.outputs], keys=keys)
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # When values are swapped between parameters
        with self.assertRaises(TypeError):
            m.estimate_params(times=[results.inputs], inputs=[results.times], outputs=[results.outputs], keys=keys)

        # Does not include required inputs 
        with self.assertRaises(ValueError):
            m.estimate_params(times=[results.times], outputs=[results.outputs], keys=keys)

        # No keys
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs)
        # Note keys in this case is still ['thrower_height', 'throwing_speed'], however, estimate_params
        # isn't given 'keys' explicitly, thus it is optimizing every parameter and not just the incorrectly
        # initialized ones. Therefore, we also have to ensure calc_error returns a small number.
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        self.assertLess(m.calc_error(results.times, results.inputs, results.outputs), 1)

        # No Data
        with self.assertRaises(ValueError):
            m.estimate_params()

        # Testing with Arrays
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[(0, 4), (20, 42), (-20, 15)])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Too little bounds given in wrapper array
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[(0, 4), (20, 42)])
        
        # Too many bounds given in wrapper array
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[(0, 4), (20, 42), (-4, 15), (-8, 8)])

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Regardless of type of wrapper type around bounds, it should work
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[0, 4], [20, 42], [-12, 15]])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Testing tuples passed in for bounds
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[(0, 4), (20, 42), (-20, 15)])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Testing tuples and arrays in for bounds
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[0, 4], (20, 42), [-12, 15]])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        # Bounds wrapped by np.array
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([[0,4], [20, 42], [-12, 15]]))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Bounds wrapped around with an extra wrapper
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[[0, 4], (20, 42), [-4, 15]]])

        # This should error as outputs is already a dictionary and we cannot place a hashable type within another.
        with self.assertRaises(TypeError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[-1, 5], (20, 40), {-5, 15}])

        # Lower Bound greater than Upper Bound
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[4, 0], [-20, 20], [0, 40]])

        # Incorrect length given for bounds
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[-15, 15], [-20, 20, 32], [0, 4]])
    
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[[-15], [-20, 20], [0, 4]])

        # Testing with np arrays
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([(0, 4), (-1000, 42), (-900, 0)]))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # One of the bounds has three values.
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([(1, 2), (2, 3, 4), (4, 5)]))
        
        # 4 Bounds provided for 3 keys
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([(1, 2), (2, 3), (4,5), (-1, 20)]))

        # 2 Bounds provided for 3 keys
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([(1, 2), (2, 3)]))

        # Passing a np.array of string value that is a letter
        with self.assertRaises(TypeError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array('a'))
        
        # Passing in string values in the correct 'bounds' format
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([('a', 'b'),('a', 'c'), ('d', 'e')]))

        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('a', 'b'),('a', 'c'), ('d', 'e')))

        # Passing in bool value.
        with self.assertRaises(TypeError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array(True))
        
        # True and False values are converted to their integer counterparts. So this does not fail.
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bound=np.array([(False, True), (False, True), (False, True)]))

        # Having np array defined with one list and two tuples
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([[1, 2], (2, 3), (4,5)]))

        # Correct number of bounds with np.array as their inner wrapper
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array([1, 2]), np.array([2, 3]), np.array([4, 5])])

        # Four bounds passed with three keys where inner wrapper is np.arrays
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array([1, 2]), np.array([2, 3]), np.array([4, 5]), np.array([-1, 20])])
        
        # Two bounds passed with three keys where inner wrapper is np.arrays
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array([1, 2]), np.array([2, 3])])

        # Passing in strings gets converted automatically, even inside an np.array
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['4', '9']), np.array(['2', '3']), np.array(['4', '5'])])
        
        # Passing in strings and integers into bounds
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['4', '9']), np.array([2, 3]), np.array([4, 5])])

        # Passing in strings, integers, and floating values in bounds
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['4', '9']), np.array([2, 3]), np.array([4.123, 5.346])])

        # Error due to upper bound being less than lower bound
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([(True, False), (False, True), (False, True)]))

        # Testing bounds where one bound has additional wrappers
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(((([-3, 4]))), (1, 400), (-20, 30)))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        value1 = m.calc_error(results.times, results.inputs, results.outputs)

        # Testing different formats of inner wrapper types works
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=([-3, 12], (1, 400), np.array([-20, 30])))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        value2 = m.calc_error(results.times, results.inputs, results.outputs)
        self.assertAlmostEqual(value1, value2)

        # Testing passing in strings with standard bounds
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -8
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('-3', '12'), ('1', '42'), ('-12', '30')))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        check = m.calc_error(results.times, results.inputs, results.outputs)


        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -8
        # Testing with np.array
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=np.array([('-3', '12'), ('1', '42'), ('-12', '30')]))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        check2 = m.calc_error(results.times, results.inputs, results.outputs)

        # Checking if passing in different inner wrapper types for bounds deviates between calls.
        self.assertEqual(check, check2)

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -8
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('-3.12', '12'), ('1', '42.125381'), ('-12', '30')))
        check3 = m.calc_error(results.times, results.inputs, results.outputs)

        # Checking if passing in different inner wrapper types for bounds deviates between calls,
        # but this time we are passing in floating values into our bounds
        self.assertEqual(check, check3)

        # Passing in an integer for a singular test
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('a', '12'), ('1', '20'), ('-5', '30')))
        
        # Incorrect length of a singular bound
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('-3'), ('1', '20'), ('-5', '30')))
        
        # Upper bound greater than lower bound and they are string values.
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('9', '12'), ('30', '20'), ('-5', '30')))
        
        # Both string literals and upper bound being less than lower bound
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('a', 'b'), ('30', '20'), ('-5', '30')))
        
        # Having an incorrect bound length of three
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('-3', '12'), ('20', '30', '40'), ('-5', '30')))

        # Passing in too many bounds
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('-3', '12'), ('20', '30'), ('-5', '30'), ('-20, 20')))
        
        # Passing in not enough bounds
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=(('-3', '12'), ('20', '30')))
        
        # Using np.array for each bound. Different typing for each
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['a', '9']), np.array(['2', '3']), np.array(['4', '5'])])

        # Using np.array for each bound specifically. Different typings for them
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['a', 's']), np.array([2, 3]), np.array([4, 5])])

        # Too many bounds given in np.array for each bound
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['4', '9']), np.array(['2', '3']), np.array(['4', '5']), np.array(['-2', '4'])])
        
        # Lower Bound greater than Upper Bound with np.array wrapper around each bound 
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['9', '4']), np.array(['2', '3']), np.array(['4', '5'])])
        
        # Checking comparing between a string literal integer and an ints
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['1', 9]), np.array([2, 3]), np.array([4, 5])])

        # Testing bounds equality
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=((1, 1), (2, 4), (-1, 24)))

        # Testing bounds equality with strings in np.array
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=[np.array(['9', '9']), np.array(['2', '3']), np.array(['4', '5'])])

        # Resetting parameters
        m2 = ThrownObject()
        m.parameters['thrower_height'] = m2.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = m2.parameters['throwing_speed'] = 25
        m.parameters['g'] = m2.parameters['g'] = 8
        m_result = m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys=keys, error_method = 'MSE')
        m2_result = m2.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys=keys, error_method = 'DTW')
        # Checking if two different error_methods would result in a different final_simplex. A different final_simplex returned value would indicate
        # that the optimized result is different. Since all other parameters are the same, and because we are using different error_methods, the only
        # explanation for having different final_simplex values between the two results is because of the different error_methods, thus showing we have
        # successfully passed in the error_methods to estimate_params()!
        self.assertFalse(np.array_equal(m_result['final_simplex'], m2_result['final_simplex']))

        # Resetting parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -8
        times = np.array([0.0, 0.5, 1.0])
        inputs = np.array([{}, {}, {}])
        outputs = np.array([{'x': 1.83}, {'x': 21.83}, {'x': 38.78612068965517}])
        # Testing estimate_params properly handles when results are passed in as np.arrays
        m.estimate_params(times = times, inputs = inputs, outputs = outputs)

    def test_keys(self):
        """
        Testing features where keys are not parameters in the model
        """
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Setting keys to incorrect values
        keys = ['x', 'y', 'g']
        bound=((0, 8), (20, 42), (-20, -5))
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        # Incorrect key length
        keys = ['x', 'y']
        bound=((0, 8), (20, 42), (-20, -5))
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        # Checking a parameter that does not exist.
        # gives bounds error
        keys = ['thrower_height', 'throwing_speed', 1]
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        
        # Keys within a tuple should not error
        keys = ('thrower_height', 'throwing_speed', 'g')
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Reset parameters after successful estimate_params call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Keys are a Set. Should throw an exception.
        keys = {'thrower_height', 'throwing_speed', 'g'}
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        # Keys are an array
        keys = np.array(['thrower_height', 'throwing_speed', 'g'])
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Reset parameters after successful estimate_params call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
  
        # Testing keys within tuples that are a length of one
        keys = [('thrower_height', ), ('throwing_speed', ), ('g', )]
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        # Testing keys within tuples that are broken
        keys = [('thrower_height', 'a'), ('throwing_speed', 'b'), ('g', 'c')]
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        # Testing lists within lists
        keys = [['thrower_height'], ['throwing_speed'], ['g']]
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

    def test_parameters(self):
        """
        Testing if passing in other keyword arguments works as intended
        """

        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        data = [(results.times, results.inputs, results.outputs)]
        gt = m.parameters.copy()

        # Now lets reset some parameters with a method call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -5
        keys = ['thrower_height', 'throwing_speed', 'g']
        bound=((0, 8), (20, 42), (-20, -5))
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, method='TNC')

        # Saving the calc_error value from our first call
        saveError = m.calc_error(results.times, results.inputs, results.outputs)

        # Testing that default method passed in works as intended
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -5
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound)

        # Checking to make sure that passing in a different method results in a different error value.
        self.assertNotEqual(saveError, m.calc_error(results.times, results.inputs, results.outputs))

        # Passing 'TNC' method along with 'maxfun 1000' as our options
        # to see what happens if they are changed when we pass in 
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        m.parameters['g'] = -5
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, method='TNC', options={'maxfun': 1000, 'disp': False})
        
        self.assertGreater(saveError, m.calc_error(results.times, results.inputs, results.outputs))

        # Passing in Method that does not exist
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, method='madeUpName')

        # Reset incorrect parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25
        keys = ['thrower_height', 'throwing_speed']
        bound=((0, 8), (20, 42))
                     
        # Not setting up 'maxiter' and/or 'disp'
        # Needs to be str: int format.
        with self.assertRaises(TypeError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, method='Powell', options= {1:2, True:False})
        with self.assertRaises(TypeError):
            m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, method='Powell', options={'maxiter': '9999', 'disp': False})

        # Key values for options that do not exist are heavily method dependent
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, options={'1':2, '2':2, '3':3})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        saveError = m.calc_error(results.times, results.inputs, results.outputs)

        # Resetting parameters
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # maxiter is set to -1 here, thus, we are not going to get to a close answer
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, options={'maxiter': -1, 'disp': False})

        self.assertNotEqual(saveError, m.calc_error(results.times, results.inputs, results.outputs))

        #Resetting parameters for next estimate_params function call
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        # Passing in arbitrary options should not affect estimate_params
        m.estimate_params(times=results.times, inputs=results.inputs, outputs=results.outputs, keys=keys, bounds=bound, options= {'1':3, 'disp':1})
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        self.assertAlmostEqual(saveError, m.calc_error(results.times, results.inputs, results.outputs), delta = 0.00001)

    def test_multiple_runs(self):
        """
        In this test, we are examining the behavior of estimate_params when there are multiple runs.
        """
        m = ThrownObject()

        # The value of time1, time2, inputs, and outputs are arbitrary values

        time1 = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        time2 = [0, 1, 2, 3]

        inputs = [[{}]*9, [{}]*4]
        outputs = [[{'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
            {'x': 7.91},], 
            [
                {'x': 1.83},
                {'x': 36.95},
                {'x': 62.36},
                {'x': 77.81},
            ]]
        
        
        # Checking to see if multiple runs can exist.
        # time1 and time2 are explicitly being passed in into a parent wrapper list.
        # See definitions of variables to understand format.
        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

        # Checking to see if wrapping in tuple works.
        m.estimate_params(times=(time1, time2), inputs=inputs, outputs=outputs)

        # Adding another wrapper list around outputs. List error, 2, 2, 1 will result.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[time1, time2], inputs=inputs, outputs=[outputs])
        self.assertEqual(
            'Times, inputs, and outputs must be same length. Length of times: 2, Length of inputs: 2, Length of outputs: 1',
            str(cm.exception)
        )

        # Adding another wrapper list around times. List error, 1, 2, 2 will result.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[[time1, time2]], inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length. Length of times: 1, Length of inputs: 2, Length of outputs: 2',
            str(cm.exception)
        )

        incorrectTimesRunsLen = [[0, 1, 2, 4, 5, 6, 7, 8, 9]]

        # Passing in only one run for Times whereas inputs and outputs have two runs
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=incorrectTimesRunsLen, inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length. Length of times: 1, Length of inputs: 2, Length of outputs: 2',
            str(cm.exception)
        )

        incorrectTimesLen = [[0, 1, 2, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]

        # Passing in correct amount of runs, however at run 1, Times has an incorrect length of 5 whereas inputs and outputs have a length of 4
        # Test is also validating if run 1 raises the exception
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=incorrectTimesLen, inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 1. Length of times: 5, Length of inputs: 4, Length of outputs: 4',
            str(cm.exception)
        )

        # Incorrect lengths for times.
        with self.assertRaises(ValueError):
            m.estimate_params(times=[time1, [time2]], inputs=inputs, outputs=outputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 1. Length of times: 5, Length of inputs: 4, Length of outputs: 4',
            str(cm.exception)
        )
        # Wrapper list becomes a set.
        timesSet = [time1, time2]
        # Unhashable type for outputs
        with self.assertRaises(TypeError):
            m.estimate_params(times=set(timesSet), inputs=inputs, outputs=set(outputs))
        
        time1 = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9])
        time2 = [0, 1, 2, 3]

        # Confirming estimate_params works when different runs are passed in as different data types.
        # Passing in time1 as a np.array datatype.
        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

        # Changing keyword arguments
        time1 = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        outputs = [[{'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
            {'x': 7.91},], 
            np.array([
                {'x': 1.83},
                {'x': 36.95},
                {'x': 62.36},
                {'x': 77.81},
            ])]

        # Passing in an np.array structure as a runs for outputs.
        m.estimate_params(times=[time1, time2], inputs=inputs, outputs=outputs)

        # Test case that would be fixed with future changes to Containers
        # with self.assertRaises(ValueError):
        #     m.estimate_params(times=[incorrectTimesLen], inputs=[inputs], outputs=[outputs])


    def test_incorrect_lens(self):
        """
        Goal of this test is to compare the results of passing in incorrect lengths into our estimate_params call.

        Furthermore, checks incorrect lengths within multiple runs.
        """
        # Initializing our model
        m = ThrownObject(process_noise = 0, measurement_noise = 0)
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()

        # Define Keys for estimate_params
        keys = ['thrower_height', 'throwing_speed']
        
        # Defining wrongInputLen to test parameter length tests.
        wrongInputLen = [{}]*8
        
        # Wrong Input Length and Values
        with self.assertRaises(ValueError):
            m.estimate_params(times=[results.times], inputs=[wrongInputLen], outputs=[results.outputs])
        
        # Same as last example but times not in wrapper list
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=[wrongInputLen], outputs=[results.outputs])
        
        # Passing in Runs directly
        with self.assertRaises(ValueError):
            m.estimate_params(times=results.times, inputs=wrongInputLen, outputs=results.outputs)

        # Defining wrongOutputs to test parameter length tests.
        # Arbitrary Outputs created here.
        wrongOutputs = [
            {'x': 1.83},
            {'x': 36.95},
            {'x': 62.36},
            {'x': 77.81},
            {'x': 83.45},
            {'x': 79.28},
            {'x': 65.3},
            {'x': 41.51},
        ]

        # Wrong outputs parameter length
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], inputs=[results.inputs], outputs=[wrongOutputs])
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 9, Length of inputs: 9, Length of outputs: 8',
            str(cm.exception)
        )

        # Both inputs and outputs with incorrect lengths
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], inputs=[wrongInputLen], outputs=[wrongOutputs])
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 9, Length of inputs: 8, Length of outputs: 8',
            str(cm.exception)
        )

        # Without wrapper
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=results.times, inputs=wrongInputLen, outputs= wrongOutputs)
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 9, Length of inputs: 8, Length of outputs: 8',
            str(cm.exception)
        )

        # Length error expected, 1, 9, 1.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[[results.times]], inputs=[results.inputs], outputs=[[results.outputs]])
        self.assertEqual(
            'Times, inputs, and outputs must be same length for the run at index 0. Length of times: 1, Length of inputs: 9, Length of outputs: 1',
            str(cm.exception)
        )

        # Passing in incorrect Runs
        with self.assertRaises(ValueError):
            m.estimate_params(times= results.times, inputs= results.inputs, outputs=wrongOutputs)

        # Different types of wrappers should not affect function
        # Testing functionality works with having only a few wrapper sequences
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        m.estimate_params(times=results.times, inputs=(results.inputs), outputs=(results.outputs))
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)
        
        # Same as last test, but inputs have tuple wrapper sequence instead.
        m.parameters['thrower_height'] = 1.5
        m.parameters['throwing_speed'] = 25

        m.estimate_params(times=results.times, inputs=(results.inputs), outputs=[results.outputs])
        for key in keys:
            self.assertAlmostEqual(m.parameters[key], gt[key], 2)

        # Cannot pass in Sets
        with self.assertRaises(TypeError):
            m.estimate_params(times=set(results.times), inputs=results.inputs, outputs=[results.outputs])

        # Missing inputs
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], outputs=[results.outputs])
        self.assertEqual(
            'Missing keyword arguments inputs',
            str(cm.exception)
        )

        # 'input' is not a keyword argument, so technically not defining the inputs.
        with self.assertRaises(ValueError) as cm:
            m.estimate_params(times=[results.times], input=[results.inputs], outputs=[results.outputs]) 
        self.assertEqual(
            'Missing keyword arguments inputs',
            str(cm.exception)
        )

        # Will work in future case, but not at the current moment
        # with self.assertRaises(ValueError)
            # m.estimate_params(times=[[times]], inputs=[[inputs]], outputs=[[outputs]])

    def test_tolerance(self):
        """
        Test which specifically targets adding tolerance as a keyword argument.
        """
        m = ThrownObject()
        results = m.simulate_to_threshold(save_freq=0.5)
        gt = m.parameters.copy()
        keys = ['thrower_height', 'throwing_speed', 'g']
        bound = ((-15, 15), (24, 42), (-20, 10))

        # Testing that high tolerance results in high error
        # Now lets reset some parameters
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10

        # High tolerance would result in a higher calc_error
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                          bounds=bound, keys = keys, tol = 10)
        if not all(abs(m.parameters[key] - gt[key]) > 0.02 for key in keys):
            raise ValueError("m.parameter shouldn't be too close to the original parameters")

        # Note that tolerance does not convert here.
        with self.assertRaises(TypeError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
                              bounds=bound, keys = keys , tol = "12")    

        # When tolerance is empty list
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [])
        hold1 = m.calc_error(results.times, results.inputs, results.outputs)

        # Tolerance works as intended as long as it is within a sequence of length 1
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10  
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [15])
        hold2 = m.calc_error(results.times, results.inputs, results.outputs)
        # Making sure that it has a different value.
        self.assertNotEqual(hold1, hold2)

        # Confirming that passing tolerance as a sequence of length 1 outputs the same error as setting tolerance to the value in the original sequence.
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 15)
        hold1 = m.calc_error(results.times, results.inputs, results.outputs)
        self.assertEqual(hold1, hold2)

        # Now testing with a different type of sequence
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = (15))
        hold2 = m.calc_error(results.times, results.inputs, results.outputs)
        self.assertEqual(hold1, hold2)

        # Cannot pass Sets into tolerance
        with self.assertRaises(TypeError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = {1})

        # Works when tolerance is a floating point value.
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 1.123543297)
        if not all(abs(m.parameters[key] - gt[key]) > 0.02 for key in keys):
            self.fail("m.parameter shouldn't be too close to the original parameters")

        # When tolerance is a list of many integers
        with self.assertRaises(ValueError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [1, 2, 3])

        # When tolerance is a list of mixed values (integers and strings)
        with self.assertRaises(TypeError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = [1, '2', '3'])
        
        # When tolerance is a tuple
        with self.assertRaises(ValueError):
            m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = (1, 2, 3))
        
        # Passing Tolerance and Method together should work
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys=keys, method = 'TNC', tol = 1e-9)
        track1 = m.calc_error(results.times, results.inputs, results.outputs)

        # Using different methods should result in different errors.
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, tol = 1e-9)
        track2 = m.calc_error(results.times, results.inputs, results.outputs)
        self.assertNotAlmostEqual(track1, track2)

        # Tests checking very small tolerance values
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
                          bounds=bound, keys=keys, method = 'TNC', tol = 1e-14)
        track1 = m.calc_error(results.times, results.inputs, results.outputs)

        # Now testing what occurs with even smaller tolerance
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, 
                            bounds=bound, keys=keys, method='TNC', tol=1e-20)
        track2 = m.calc_error(results.times, results.inputs, results.outputs)

        # Note that at some point, estimate_params will converge to some number regardless of how small the tol is.
        self.assertEqual(track1, track2)

        # Tests to check how tolerance and options work together.
        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                           bounds=bound, keys=keys, tol = 1e-2)
        override1 = m.calc_error(results.times, results.inputs, results.outputs)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                            bounds=bound, keys=keys, tol = 1e-5)
        override2 = m.calc_error(results.times, results.inputs, results.outputs)

        self.assertNotEqual(override1, override2)

        m.parameters['thrower_height'] = 3.1
        m.parameters['throwing_speed'] = 29
        m.parameters['g'] = 10
        m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs,
                           bounds = bound, keys=keys, tol = 1e-2, options={'xatol': 1e-5})
        override3 = m.calc_error(results.times, results.inputs, results.outputs)

        # Passed in options properly overrides the tolerance that is placed. 
        self.assertNotEqual(override1, override3)
        self.assertEqual(override2, override3)

def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting EstimateParams Feature")
    result = runner.run(load_test.loadTestsFromTestCase(TestEstimateParams)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
