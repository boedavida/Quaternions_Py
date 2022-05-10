"""Creates a quaternion class"""

# Quaternion construction:
# 1. q = Quaternion(a=1, b=2, c=3, d=4) # keyword arguments for the components
# 2. t = 2*np.pi/3
#    ax = np.array([1., 1., 1.])*(1/np.sqrt(3))
#    q = Quaternion(a = t, imag=ax) constructs a rotational quaternion with rotation
#        angle t (radians) and rotation axis ax (which is a numpy array of length 3)
import numpy as np


class Quaternion:
    """Quaternion class"""

    def __init__(self, **kwargs):

        if len(kwargs) == 4:
            # Construct a quaternion with its components
            self._a = kwargs["a"]
            self._b = kwargs["b"]
            self._c = kwargs["c"]
            self._d = kwargs["d"]

        elif len(kwargs) == 2:
            # Construct a rotational quaternion from the input rotation angle and axis of rotation
            # a = theta, rotation angle, [radians], theta such that q0 = cos(theta/2)
            # imag = axis, [1x3 numpy array]
            # Make sure that imag is a 1x3 unit vector
            t = kwargs["a"]
            ax = kwargs["imag"] / np.linalg.norm(kwargs["imag"])
            self._a = np.cos(t / 2)
            self._b = ax[0] * np.sin(t / 2)
            self._c = ax[1] * np.sin(t / 2)
            self._d = ax[2] * np.sin(t / 2)

        else:
            raise TypeError(
                f"Expected either 2 or 4 keyword arguments but got {len(kwargs)}"
            )

    # Getters. No setters as each quaternion is unique and immutable
    def a(self):
        """Getter for the real part"""
        return self._a

    def b(self):
        """Getter for the imaginary i part"""
        return self._b

    def c(self):
        """Getter for the imaginary j part"""
        return self._c

    def d(self):
        """Getter for the imaginary k part"""
        return self._d

    # String representation
    def __str__(self):
        return f"{self.a()} + {self.b()}i + {self.c()}j + {self.d()}k"

    # Printable representation
    def __repr__(self):
        return f"{self.a()} + {self.b()}i + {self.c()}j + {self.d()}k"

    def real(self):
        """The real part of the quaternion"""
        return self.a()

    def imaginary(self):
        """The imaginary part of the quaternion"""
        # returns a tuple, which is immutable
        return (self.b(), self.c(), self.d())

    def conj(self):
        """Conjugate of the quaternion"""
        return Quaternion(a=self.a(), b=-self.b(), c=-self.c(), d=-self.d())

    def norm(self):
        """Norm of the quaternion"""
        p = self.conj() * self
        return np.sqrt(p.a() + p.b() + p.c() + p.d())

    def __neg__(self):
        return Quaternion(a=-self.a(), b=-self.b(), c=-self.c(), d=-self.d())

    def __add__(self, other):
        return Quaternion(
            a=self.a() + other.a(),
            b=self.b() + other.b(),
            c=self.c() + other.c(),
            d=self.d() + other.d(),
        )

    def __sub__(self, other):
        return Quaternion(
            a=self.a() - other.a(),
            b=self.b() - other.b(),
            c=self.c() - other.c(),
            d=self.d() - other.d(),
        )

    def __mul__(self, other):
        """Multiplication of two quaternions"""
        # Quaternion multiplication is not commutative
        p0 = self.a()
        q0 = other.a()
        p = np.array([self.b(), self.c(), self.d()])
        q = np.array([other.b(), other.c(), other.d()])
        a0 = p0 * q0 - np.dot(p, q)
        bcd = p0 * q + q0 * p + np.cross(p, q)
        return Quaternion(a=a0, b=bcd[0], c=bcd[1], d=bcd[2])

    def angle(self):
        """Angle of the quaternion"""
        return 2.0 * np.arccos(self.a())

    def axis(self):
        """Axis of the quaternion"""
        # returns a tuple, which is immutable
        t = self.angle()
        q = np.array([self.b(), self.c(), self.d()])
        r = q / np.sin(t / 2.0)
        return (r[0], r[1], r[2])

    def inv(self):
        """Multiplicative inverse of the quaternion"""
        n = self.norm()
        if n**2 != 0.0:
            p = np.array([self.a(), -self.b(), -self.c(), -self.d()]) / n**2
        else:
            raise ZeroDivisionError("Division by zero is undefined")
        return Quaternion(a=p[0], b=p[1], c=p[2], d=p[3])

    def coef(self):
        """Coeficients of the quaternion"""
        # returns a tuple, which is immutable
        return (self.a(), self.b(), self.c(), self.d())

    def isequal(self, other):
        """Test of equality between two tuples"""
        return self.coef() == other.coef()

    def __eq__(self, other):
        """Test of equality between two tuples"""
        return self.coef() == other.coef()

    def rotate_vector(self, v):
        """Rotates a vector about an axis through an angle with respect to a coordinate system"""
        u = Quaternion(a=0, b=v[0], c=v[1], d=v[2])
        w = self * u * self.conj()
        return w.imaginary()

    def rotate_frame(self, v):
        """Rotates the coordinate system frame itself"""
        u = Quaternion(a=0, b=v[0], c=v[1], d=v[2])
        w = self.conj() * u * self
        return w.imaginary()


def main():
    """Main function"""
    # Construct two quaternions and multiply one by the other
    # Quaternion multiplication is not commutative
    x = Quaternion(a=3, b=1, c=-2, d=1)
    y = Quaternion(a=2, b=-1, c=2, d=3)
    z = x * y
    print("\nQuaternion multiplication:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"x * y = {z}")

    print("\nQuaternion division is acccomplished by multiplication with the inverse:")
    d = x * y.inv()
    print(f"x * y^-1 = {d}")

    # Vector rotation
    t = 2 * np.pi / 3
    ax = (1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3))
    v = (1, 0, 0)
    q = Quaternion(a=t, imag=ax)
    v_rotated = q.rotate_vector(v)
    print("\nRotation of a vector in a coordinate frame using a quaternion:")
    print(
        f"Vector coordinates before rotation are ({v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f})"
    )
    print(
        " ".join(
            [
                "Vector coordinates after rotation of 2*pi/3 about (1, 1, 1) are",
                f"({v_rotated[0]:.8f}, {v_rotated[1]:.8f}, {v_rotated[2]:.8f})",
            ]
        )
    )

    # Coordinate frame rotation
    v_frame_rotated = q.rotate_frame(v_rotated)
    print("\nRotation of the coordinate frame using a quaternion:")
    print(
        " ".join(
            [
                "Vector coordinates before rotation are",
                f"({v_rotated[0]:.8f}, {v_rotated[1]:.8f}, {v_rotated[2]:.8f})",
            ]
        )
    )
    print(
        " ".join(
            [
                "Vector coordinates after rotation of the frame of 2*pi/3 about (1, 1, 1) are",
                f"({v_frame_rotated[0]:.8f},",
                f"{v_frame_rotated[1]:.8f},",
                f"{v_frame_rotated[2]:.8f})\n\n",
            ]
        )
    )


if __name__ == "__main__":
    main()
