import faulthandler
faulthandler.enable()

import numpy as np

from cc_model.load_datasets import *
from cc_model.fast_wl import *

import unittest


def create_line_graph(n):
    edges = np.empty((2*(n-1), 2), dtype=np.uint64)

    edges[:n-1,0]= np.arange(n-1)
    edges[:n-1,1]= np.arange(1,n)
    edges[n-1:,1]= np.arange(n-1)
    edges[n-1:,0]= np.arange(1,n)
    return edges

class TestFastWLMethods(unittest.TestCase):

    def test_to_in_neighbors(self):
        edges = np.array([[0,1,0], [1,2,2]], dtype=np.uint32).T
        arr1, arr2, _ = to_in_neighbors(edges)
        np.testing.assert_array_equal(arr1, [0,0,1,3])
        np.testing.assert_array_equal(arr2, [0,1,0])


    def test_wl_line(self):
        n=8


        out = WL_fast(create_line_graph(n))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 4)
        arr0, arr1, arr2, arr3 = out

        np.testing.assert_array_equal(arr0, np.zeros(n, dtype=np.uint32))
        np.testing.assert_array_equal(arr1, [0, 1, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(arr2, [0, 1, 2, 2, 2, 2, 1, 0])
        np.testing.assert_array_equal(arr3, [0, 1, 2, 3, 3, 2, 1, 0])

    def test_wl_line2(self):
        n=7

        out = WL_fast(create_line_graph(n))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 4)
        arr0, arr1, arr2, arr3 = out

        np.testing.assert_array_equal(arr0, np.zeros(n, dtype=np.uint32))
        np.testing.assert_array_equal(arr1, [0, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(arr2, [0, 1, 2, 2, 2, 1, 0])
        np.testing.assert_array_equal(arr3, [0, 1, 2, 3, 2, 1, 0])

    def test_wl_line3(self):
        """Now with imperfection"""
        n=7
        starting_labels = np.array([0,0,0,100,0,0,0], dtype=np.uint32)
        out = WL_fast(create_line_graph(n), starting_labels )
        print(out)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out),2)

        arr0, arr1 = out
        self.assertEqual(arr0.dtype, starting_labels.dtype)
        self.assertEqual(arr1.dtype, starting_labels.dtype)
        np.testing.assert_array_equal(arr0, [0,0,0,1,0,0,0])
        np.testing.assert_array_equal(arr1, [0, 1, 2, 3, 2, 1, 0])
        #np.testing.assert_array_equal(arr3, [0, 1, 2, 3, 2, 1, 0])

    def test_wl_4(self):
        edges = np.array([[0, 3],
                [1, 2],
                [2, 4],
                [2, 5],
                [3, 6],
                [3, 7],
                [4, 8],
                [5, 8],
                [6, 7],
                [3, 0],
                [2, 1],
                [4, 2],
                [5, 2],
                [6, 3],
                [7, 3],
                [8, 4],
                [8, 5],
                [7, 6]], dtype=np.uint32)

        out = WL_fast(edges)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out),6)

        results = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 2, 2, 2, 2, 2],
                   [0, 0, 1, 1, 2, 2, 2, 2, 3],
                   [0, 0, 1, 1, 2, 2, 3, 3, 4],
                   [0, 0, 1, 2, 3, 3, 4, 4, 5],
                   [0, 1, 2, 3, 4, 4, 5, 5, 6]]
        for arr, arr_expected in zip(out, results):
            np.testing.assert_array_equal(arr, arr_expected)



if __name__ == '__main__':
    unittest.main()