# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import numpy as np
import pickle
import sys
import unittest

from progpy.utils.containers import DictLikeMatrixWrapper


class TestDictLikeMatrixWrapper(unittest.TestCase):
    def setUp(self):
        # set stdout (so it won't print)
        sys.stdout = StringIO()

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def _checks(self, c1):
        self.assertListEqual(c1.keys(), ['a', 'b'])
        self.assertListEqual(list(c1.values()), [1., 2.])
        self.assertListEqual(list(c1.items()), [('a', 1.), ('b', 2.)])
        self.assertEqual(c1['a'], 1)
        self.assertEqual(c1['b'], 2)
        self.assertTrue(np.array_equal(c1.matrix, np.array([[1.], [2.]])))
        # Membership
        self.assertIn('a', c1)
        self.assertIn('b', c1)
        self.assertNotIn('c', c1)
        self.assertEqual(len(c1), 2)
        # Setting by dict
        c1['a'] = -1.0
        self.assertTrue((c1.matrix == np.array([[-1.], [2.]])).all())
        self.assertEqual(c1['a'], -1.0)
        # Setting by matrix
        c1.matrix[1][0] = -2.0
        self.assertTrue(np.all(c1.matrix == np.array([[-1.], [-2.]])))
        self.assertEqual(c1['b'], -2.0)
        # Pickling
        c2 = pickle.loads(pickle.dumps(c1))
        self.assertTrue(np.all((c2.matrix == np.array([[-1.], [-2.]]))))
        self.assertEqual(c2['a'], -1.0)
        self.assertEqual(c2['b'], -2.0)
        # Equality
        c2 = DictLikeMatrixWrapper(['a', 'b'], {'a': -1.0, 'b': -2.0})
        self.assertEqual(c1, c2)
        # update
        c1.update({'c': 3.0, 'b': 2.0})
        self.assertTrue((c1.matrix == np.array([[-1], [2], [3]])).all())
        self.assertEqual(c1['c'], 3)
        self.assertListEqual(c1.keys(), ['a', 'b', 'c'])
        other = DictLikeMatrixWrapper(['c', 'd'], np.array([[5], [7]]))
        c1.update(other)
        self.assertTrue((c1.matrix == np.array([[-1], [2], [5], [7]])).all())
        self.assertEqual(c1['d'], 7)
        self.assertListEqual(c1.keys(), ['a', 'b', 'c', 'd'])
        # deleting items
        del c1['a']
        self.assertTrue((c1.matrix == np.array([[2], [5], [7]])).all())
        self.assertListEqual(c1.keys(), ['b', 'c', 'd'])
        del c1['c']
        self.assertTrue((c1.matrix == np.array([[2], [7]])).all())
        self.assertListEqual(c1.keys(), ['b', 'd'])
        del c1['d']
        del c1['b']
        self.assertTrue((c1.matrix == np.array([[]])).all())
        self.assertListEqual(c1.keys(), [])


    def test_dict_init(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], {'a': 1, 'b': 2})
        self._checks(c1)
    
    def test_array_init(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], np.array([[1], [2]]))
        self._checks(c1)

    def test_matrix_init(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], np.matrix([[1], [2]]))
        self._checks(c1)

    def test_broken_init(self):
        with self.assertRaises(TypeError):
            DictLikeMatrixWrapper(['a', 'b'], [1, 2])

    def test_pickle(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], {'a': 1, 'b': 2})
        c2 = pickle.loads(pickle.dumps(c1))
        self.assertTrue((c2.matrix == np.array([[1], [2]])).all())

    def test_dot(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], {'a': 1, 'b': 2})
        c2 = DictLikeMatrixWrapper(['a', 'b'], {'a': 4, 'b': 5})
        c1_dot_c2 = c1.dot(c2.matrix)  # should be, (1*4) + (2*5) = 14
        self.assertTrue((c1_dot_c2[0] == 14).all())

    def test_matrix(self):
        c1 = DictLikeMatrixWrapper(['a', 'b'], {'a': 1, 'b': 2})
        self.assertTrue((c1.matrix == np.array([[1], [2]])).all())


# This allows the module to be executed directly
def main():
    load_test = unittest.TestLoader()
    runner = unittest.TextTestRunner()
    print("\n\nTesting Containers")
    result = runner.run(load_test.loadTestsFromTestCase(TestDictLikeMatrixWrapper)).wasSuccessful()

    if not result:
        raise Exception("Failed test")

if __name__ == '__main__':
    main()
