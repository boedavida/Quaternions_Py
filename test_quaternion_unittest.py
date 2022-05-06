# >python -m unittest test_quaternion_unittest
# >python -m unittest discover

import unittest
import numpy as np
from quaternions import Quaternion


class TestQuaternionClass(unittest.TestCase):

    def test_mul(self):
        # Ref: Yan-Bin Jia (2018)
        # p*q ans: 8 + -9i + -2j + 11k
        p = Quaternion(a=3., b=1., c=-2., d=1.)
        q = Quaternion(a=2., b=-1., c=2., d=3.)
        test_result = p*q
        expected_result = Quaternion(a=8., b=-9., c=-2., d=11.)
        self.assertTrue(test_result == expected_result)

    def test_angle(self):
        t = 2 * np.pi / 3
        im = np.array([1., 1., 1.]) * (1 / np.sqrt(3))
        q = Quaternion(a=t, imag = im)
        self.assertEqual(q.angle(), t)

    def test_axis(self):
        t = 2 * np.pi / 3
        im = np.array([1., 1., 1.]) * (1 / np.sqrt(3))
        q = Quaternion(a=t, imag = im)
        self.assertTrue(np.array_equal(q.axis(), im))

    def test_rotatev(self):
        t = 2 * np.pi / 3
        im = np.array([1., 1., 1.]) * (1 / np.sqrt(3))
        q = Quaternion(a=t, imag=im)
        v = (1, 0, 0)
        test_result = q.rotatev(v)
        expected_result = (0, 1, 0)
        self.assertTrue(np.allclose(test_result, expected_result, atol=1e-15, equal_nan=False))

    def test_rotateframe(self):
        t = 2 * np.pi / 3
        im = np.array([1., 1., 1.]) * (1 / np.sqrt(3))
        q = Quaternion(a=t, imag=im)
        v = (0, 1, 0)
        test_result = q.rotateframe(v)
        expected_result = (1, 0, 0)
        self.assertTrue(np.allclose(test_result, expected_result, atol=1e-15, equal_nan=False))


if __name__ == '__main__':
    unittest.main()
