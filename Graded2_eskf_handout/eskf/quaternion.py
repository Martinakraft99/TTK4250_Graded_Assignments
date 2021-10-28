import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

from config import DEBUG

import solution
from cross_matrix import get_cross_matrix
import math


@dataclass
class RotationQuaterion:
    """Class representing a rotation quaternion (norm = 1). Has some useful
    methods for converting between rotation representations.

    Hint: You can implement all methods yourself, or use scipys Rotation class.
    scipys Rotation uses the xyzw notation for quats while the book uses wxyz
    (this i really annoying, I know).

    Args:
        real_part (float): eta (n) in the book, w in scipy notation
        vec_part (ndarray[3]): epsilon in the book, (x,y,z) in scipy notation
    """
    real_part: float
    vec_part: 'ndarray[3]'

    def __post_init__(self):
        if DEBUG:
            assert len(self.vec_part) == 3

        norm = np.sqrt(self.real_part**2 + sum(self.vec_part**2))
        if not np.allclose(norm, 1):
            self.real_part /= norm
            self.vec_part /= norm

        if self.real_part < 0:
            self.real_part *= -1
            self.vec_part *= -1

    def multiply(self, other: 'RotationQuaterion') -> 'RotationQuaterion':
        """Multiply two rotation quaternions
        Hint: see (10.33)

        As __matmul__ is implemented for this class, you can use:
        q1@q2 which is equivalent to q1.multiply(q2)

        Args:
            other (RotationQuaternion): the other quaternion    
        Returns:
            quaternion_product (RotationQuaternion): the product
        """
        #quaternion_product = self @ other
        #quaternion_product = self.multiply(other)
        
        #norm = np.sqrt(self.real_part**2 + other.real_part**2 + sum(self.vec_part**2) + sum(other.vec_part**2))
        
        real = self.real_part * other.real_part - self.vec_part.T @ other.vec_part
        
        vec = other.real_part * self.vec_part + self.real_part * other.vec_part + get_cross_matrix(self.vec_part) @ other.vec_part
    
        quaternion_product = RotationQuaterion(real, vec)        

        # TODO replace this with your own code
        #quaternion_product = solution.quaternion.RotationQuaterion.multiply(
        #    self, other)

        return quaternion_product

    def conjugate(self) -> 'RotationQuaterion':
        """Get the conjugate of the RotationQuaternion"""

        conj = RotationQuaterion(self.real_part, -self.vec_part)

        return conj

    def as_rotmat(self) -> 'ndarray[3,3]':
        """Get the rotation matrix representation of self

        Returns:
            R (ndarray[3,3]): rotation matrix
        """

        # By (10.37)    
        skew = get_cross_matrix(self.vec_part)
        R = np.identity(3) + 2 * self.real_part * skew + 2 * skew @ skew

        #R = solution.quaternion.RotationQuaterion.as_rotmat(self)

        return R

    @property
    def R(self) -> 'ndarray[3,3]':
        return self.as_rotmat()

    def as_euler(self) -> 'ndarray[3]':
        """Get the euler angle representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """

        # By (10.38)
        eta = self.real_part
        eps1 = self.vec_part[0]
        eps2 = self.vec_part[1]
        eps3 = self.vec_part[2]
        phi = math.atan2(2 * (eps3 * eps2 + eta * eps1), eta**2 - eps1**2 - eps2**2 + eps3**2)
        theta = math.asin(2 * (eta * eps2 - eps1 * eps3))
        psi = math.atan2(2 * (eps1 * eps2 + eta * eps3), eta**2 + eps1**2 - eps2**2 - eps3**2)
        euler = np.ndarray([3], buffer=np.array([phi, theta, psi]))
        
        # TODO replace this with your own code
        #euler = solution.quaternion.RotationQuaterion.as_euler(self)

        return euler

    def as_avec(self) -> 'ndarray[3]':
        """Get the angles vector representation of self

        Returns:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)
        """

        def normalize(vec):
            return math.sqrt(sum(vec**2))

        eta = self.real_part
        norm = normalize(self.vec_part)
        alpha = math.acos(eta**2 - norm**2)
        if (alpha == 0):
            n = np.zeros([3,1])
        else:
            n = self.vec_part / (math.sin(alpha / 2))

        avec = alpha*n/normalize(n)

        #avec = solution.quaternion.RotationQuaterion.as_avec(self)
    
        return avec

    @staticmethod
    def from_euler(euler: 'ndarray[3]') -> 'RotationQuaterion':
        """Get a rotation quaternion from euler angles
        usage: rquat = RotationQuaterion.from_euler(euler)

        Args:
            euler (ndarray[3]): extrinsic xyz euler angles (roll, pitch, yaw)

        Returns:
            rquat (RotationQuaternion): the rotation quaternion
        """
        scipy_quat = Rotation.from_euler('xyz', euler).as_quat()
        rquat = RotationQuaterion(scipy_quat[3], scipy_quat[:3])
        return rquat

    def _as_scipy_quat(self):
        """If you're using scipys Rotation class, this can be handy"""
        return np.append(self.vec_part, self.real_part)

    def __iter__(self):
        return iter([self.real_part, self.vec_part])

    def __matmul__(self, other) -> 'RotationQuaterion':
        """Lets u use the @ operator, q1@q2 == q1.multiply(q2)"""
        return self.multiply(other)
