import unittest
import unittest.mock
import os
import math
from miniflow import *


class BasicTest(unittest.TestCase):
    # Called for each test
    def setUp(self):
        pass

    # Called for each test
    def tearDown(self):
        pass

    def test_add(self):
        x, y, z = Input(), Input(), Input()
        feed_dict = {x: 10, y: 5, z: 8}
        self.assertEqual(compute(Add([x,y,z]), feed_dict), 23)

    def test_mult(self):
        x, y, z = Input(), Input(), Input()
        feed_dict = {x: 10, y: 5, z: 8}
        self.assertEqual(compute(Mult([x, y, z]), feed_dict), 400,"Mult passed")

    def test_linear(self):
        inputs, weights, bias = Input(), Input(), Input()
        f = Linear([inputs,weights, bias])

        feed_dict = {
                 inputs: [6, 14, 3],
                 weights: [0.5, 0.25, 1.4],
                 bias: 2
             }

        self.assertEqual(compute(f, feed_dict), 12.7)



def compute(node, feed):
        # NOTE: because topological_sort set the values for the `Input` nodes we could also access
        # the value for x with x.value (same goes for y).
        sorted_nodes = topological_sort(feed)
        return forward_pass(node, sorted_nodes)

# class MockUnitTest(unittest.TestCase):
#     # Called for each test
#     m = unittest.mock.Mock()
#     m.append('x')
#     m.assert_any_call()


