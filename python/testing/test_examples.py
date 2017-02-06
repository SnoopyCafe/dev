import unittest
import unittest.mock
import os
import math



class BasicTest(unittest.TestCase):
    # Called for each test
    def setUp(self):
        pass

    # Called for each test
    def tearDown(self):
        pass

    def test_addition(self):
        self.assertEqual(2 + 2, 4)

    def test_sqrt(self):
        self.assertAlmostEqual(math.sqrt(7), 2.6457,3)

# class MockUnitTest(unittest.TestCase):
#     # Called for each test
#     m = unittest.mock.Mock()
#     m.append('x')
#     m.assert_any_call()


