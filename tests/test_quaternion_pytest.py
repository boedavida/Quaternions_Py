"""Tests for the Quaternion class"""
import pytest
import numpy as np
from quaternions import Quaternion


@pytest.fixture(name="quaternion_list")
def fixture_quaternion_list():
    p = Quaternion(a=3.0, b=1.0, c=-2.0, d=1.0)
    q = Quaternion(a=2.0, b=-1.0, c=2.0, d=3.0)   
    p_q_product = Quaternion(a=8.0, b=-9.0, c=-2.0, d=11.0)
    # Ref: Yan-Bin Jia (2018): p*q ans: 8 + -9i + -2j + 11k
    return [p, q, p_q_product] 

def test_mul(quaternion_list):
    """Test the multiplication method"""
    p = quaternion_list[0]
    q = quaternion_list[1]
    expected_result = quaternion_list[2]
    test_result = p * q
    assert test_result == expected_result

def test_angle():
    """Test the angle based instantiation"""
    t = 2 * np.pi / 3
    im = np.array([1.0, 1.0, 1.0]) * (1 / np.sqrt(3))
    q = Quaternion(a=t, imag=im)
    assert t == q.angle()


def test_axis():
    """Test the axis method"""
    t = 2 * np.pi / 3
    im = np.array([1.0, 1.0, 1.0]) * (1 / np.sqrt(3))
    q = Quaternion(a=t, imag=im)
    assert np.array_equal(q.axis(), im)


def test_rotatev():
    """Test the rotate vector method"""
    t = 2 * np.pi / 3
    im = np.array([1.0, 1.0, 1.0]) * (1 / np.sqrt(3))
    q = Quaternion(a=t, imag=im)
    v = (1, 0, 0)
    test_result = q.rotate_vector(v)
    expected_result = (0, 1, 0)
    assert np.allclose(test_result, expected_result, atol=1e-15, equal_nan=False)


def test_rotateframe():
    """Test the rotate frame method"""
    t = 2 * np.pi / 3
    im = np.array([1.0, 1.0, 1.0]) * (1 / np.sqrt(3))
    q = Quaternion(a=t, imag=im)
    v = (0, 1, 0)
    test_result = q.rotate_frame(v)
    expected_result = (1, 0, 0)
    assert np.allclose(test_result, expected_result, atol=1e-15, equal_nan=False)
