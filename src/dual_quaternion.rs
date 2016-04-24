//! Dual Quaternion
#![deny(missing_docs)]

use float::Float;
use vector::{Vec3, Pos3};
use quaternion::Quat;
use std::ops::{Add, Sub, Mul, Neg};
use std::convert::From;

/// Dual Quaternion
///
/// # Definition
/// <div>$$
///   \mathbf{\sigma} = \mathbf{p} + \varepsilon \mathbf{q}, \\
///   \textrm{where } \mathbf{p}, \mathbf{q} \in \mathbb{H}
/// $$</div>
#[derive(Copy, Clone, Debug)]
pub struct DualQuat<T>(pub Quat<T>, pub Quat<T>);

impl<T: Float> Add for DualQuat<T> {
    type Output = DualQuat<T>;
    fn add(self, _rhs: DualQuat<T>) -> DualQuat<T> {
        DualQuat(self.0 + _rhs.0, self.1 + _rhs.1)
    }
}

impl<T: Float> Sub for DualQuat<T> {
    type Output = DualQuat<T>;
    fn sub(self, _rhs: DualQuat<T>) -> DualQuat<T> {
        DualQuat(self.0 - _rhs.0, self.1 - _rhs.1)
    }
}

impl<T: Float> Mul<T> for DualQuat<T> {
    type Output = DualQuat<T>;
    fn mul(self, s: T) -> DualQuat<T> {
        DualQuat(self.0 * s, self.1 * s)
    }
}

impl<T: Float> Mul for DualQuat<T> {
    type Output = DualQuat<T>;
    fn mul(self, _rhs: DualQuat<T>) -> DualQuat<T> {
        DualQuat(self.0 * _rhs.0, self.0 * _rhs.1 + self.1 * _rhs.0)
    }
}

impl<T: Float> Neg for DualQuat<T> {
    type Output = DualQuat<T>;
    fn neg(self) -> DualQuat<T> {
        DualQuat(-self.0, -self.1)
    }
}

impl<T: Float> DualQuat<T> {
    /// Squared Norm
    ///
    /// <div>$$
    ///   |\mathbf{\sigma}|^2
    ///   = \mathbf{\sigma} \mathbf{\sigma}^\ast
    ///   = \mathbf{p} \mathbf{p}^\ast
    ///     + \varepsilon ( \mathbf{p} \mathbf{q}^\ast + \mathbf{q} \mathbf{p}^\ast )
    /// $$</div>
    pub fn norm_squared(self) -> (T, T) {
        // (self.0 * !self.0, self.0 * !self.1 + self.1 * !self.0)
        let n0 = self.0.norm_squared();
        let n1 = self.0.dot(self.1) * T::two();
        (n0, n1)
    }

    /// Norm
    ///
    /// <div>$$
    ///   |\mathbf{\sigma}|
    ///   = \sqrt{\mathbf{\sigma} \mathbf{\sigma}^\ast}
    /// $$</div>
    pub fn norm(self) -> (T, T) {
        let (n0, n1) = self.norm_squared();
        (n0.sqrt(), n1.sqrt())
    }

    /// Create unit dual quaternion from rigit transformation
    ///
    /// # Parameters
    /// - `r`: Rotation Quaternion
    /// - `t`: Translation Vector
    ///
    /// # Definition
    /// <div>$$
    ///   \hat{\mathbf{\sigma}} = \mathbf{r} + \frac{\varepsilon}{2} \mathbf{t} \mathbf{r}, \\
    ///   \textrm{where } \mathbf{t} = 0 + t_x \mathbf{i} + t_y \mathbf{j} + t_z \mathbf{k}
    /// $$</div>
    pub fn from_transform(r: Quat<T>, t: Vec3<T>) -> DualQuat<T> {
        // real part = r
        // dual part = quat(xyz = t, w = 0) * r * 0.5
        DualQuat(r, Quat::from(t) * r * T::onehalf())
    }

    /// Get the rotation part
    pub fn orientation(self) -> Quat<T> {
        // yes, real part
        self.0
    }

    /// Get the translation part
    pub fn translation(self) -> Vec3<T> {
        // trans = (dual part) * (real part).inverse() * 2.0
        (self.1 * !self.0 * T::two()).xyz()
    }

    // TODO: transform a point
    // TODO: screw linear interpolation
}

impl<T: Float> From<Quat<T>> for DualQuat<T> {
    fn from(q: Quat<T>) -> DualQuat<T> {
        DualQuat(q, Quat::zero())
    }
}

impl<T: Float> From<Vec3<T>> for DualQuat<T> {
    fn from(v: Vec3<T>) -> DualQuat<T> {
        DualQuat(Quat::identity(), Quat::from(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vector::{Pos3, Vec3};
    use quaternion::Quat;

    #[test]
    fn test_add() {
        let q0 = Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
        let q1 = Quat { w: 5.0, x: 6.0, y: 7.0, z: 8.0 };
        let q2 = Quat { w: 9.0, x: 10.0, y: 11.0, z: 12.0 };
        let q3 = Quat { w: 13.0, x: 14.0, y: 15.0, z: 16.0 };
        let a = DualQuat(q0, q1);
        let b = DualQuat(q2, q3);
        let c = a + b;
        assert_eq!(c.0.w, 10.0);
        assert_eq!(c.0.x, 12.0);
        assert_eq!(c.0.y, 14.0);
        assert_eq!(c.0.z, 16.0);
        assert_eq!(c.1.w, 18.0);
        assert_eq!(c.1.x, 20.0);
        assert_eq!(c.1.y, 22.0);
        assert_eq!(c.1.z, 24.0);
    }

    #[test]
    fn test_sub() {
        let q0 = Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
        let q1 = Quat { w: 5.0, x: 6.0, y: 7.0, z: 8.0 };
        let q2 = Quat { w: 9.0, x: 10.0, y: 11.0, z: 12.0 };
        let q3 = Quat { w: 13.0, x: 14.0, y: 15.0, z: 16.0 };
        let a = DualQuat(q0, q1);
        let b = DualQuat(q2, q3);
        let c = a - b;
        assert_eq!(c.0.w, -8.0);
        assert_eq!(c.0.x, -8.0);
        assert_eq!(c.0.y, -8.0);
        assert_eq!(c.0.z, -8.0);
        assert_eq!(c.1.w, -8.0);
        assert_eq!(c.1.x, -8.0);
        assert_eq!(c.1.y, -8.0);
        assert_eq!(c.1.z, -8.0);
    }

    #[test]
    fn test_mul_scalar() {
        let q0 = Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
        let q1 = Quat { w: 5.0, x: 6.0, y: 7.0, z: 8.0 };
        let a = DualQuat(q0, q1);
        let b = a * 0.5;
        assert_eq!(b.0.w, 0.5);
        assert_eq!(b.0.x, 1.0);
        assert_eq!(b.0.y, 1.5);
        assert_eq!(b.0.z, 2.0);
        assert_eq!(b.1.w, 2.5);
        assert_eq!(b.1.x, 3.0);
        assert_eq!(b.1.y, 3.5);
        assert_eq!(b.1.z, 4.0);
    }

    #[test]
    fn test_mul_dualquat() {
        let q0 = Quat { w: 2.0, x: 2.0, y: 5.0, z: 7.0 };
        let q1 = Quat { w: 3.0, x: -3.0, y: 0.5, z: -1.0 };
        let a = DualQuat(q0, q1);
        let q0 = Quat { w: 1.5, x: 5.0, y: -1.0, z: 9.0 };
        let q1 = Quat { w: 0.25, x: 0.5, y: 6.0, z: 2.0 };
        let b = DualQuat(q0, q1);
        let c = a * b;
        assert_eq!(c.0.w, -65.0);
        assert_eq!(c.0.x, 65.0);
        assert_eq!(c.0.y, 22.5);
        assert_eq!(c.0.z, 1.5);
        assert_eq!(c.1.w, -15.5);
        assert_eq!(c.1.x, -16.5);
        assert_eq!(c.1.y, 32.5);
        assert_eq!(c.1.z, 41.25);
    }

    #[test]
    fn test_neg() {
        let q0 = Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
        let q1 = Quat { w: 5.0, x: 6.0, y: 7.0, z: 8.0 };
        let a = DualQuat(q0, q1);
        let b = -a;
        assert_eq!(b.0.w, -1.0);
        assert_eq!(b.0.x, -2.0);
        assert_eq!(b.0.y, -3.0);
        assert_eq!(b.0.z, -4.0);
        assert_eq!(b.1.w, -5.0);
        assert_eq!(b.1.x, -6.0);
        assert_eq!(b.1.y, -7.0);
        assert_eq!(b.1.z, -8.0);
    }

    #[test]
    fn test_norm2() {
        let q0 = Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
        let q1 = Quat { w: 5.0, x: 6.0, y: 7.0, z: 8.0 };
        let a = DualQuat(q0, q1);
        let (n0, n1) = a.norm_squared();
        assert_eq!(n0, 30.0);
        assert_eq!(n1, 140.0);
    }

    #[test]
    fn test_norm() {
        let q0 = Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 };
        let q1 = Quat { w: 5.0, x: 6.0, y: 7.0, z: 8.0 };
        let a = DualQuat(q0, q1);
        let (n0, n1) = a.norm();
        assert_approx_eq!(n0, 5.477225575051661);
        assert_approx_eq!(n1, 11.832159566199232);
    }

    #[test]
    fn test_from_tfm() {
        let r = Quat { w: 0.7071067811865476, x: 0.30860669992, y: -0.61721339984, z: 0.15430334996 };
        let t = Vec3 { x: -1.0, y: 2.0, z: 3.0 };
        let a = DualQuat::from_transform(r, t);
        assert_eq!(a.0.w, 0.7071067811865476);
        assert_eq!(a.0.x, 0.30860669992);
        assert_eq!(a.0.y, -0.61721339984);
        assert_eq!(a.0.z, 0.15430334996);
        assert_approx_eq!(a.1.w, 0.5400617248599999);
        assert_approx_eq!(a.1.x, 0.7265700591267262);
        assert_approx_eq!(a.1.y, 1.2471685060465476);
        assert_approx_eq!(a.1.z, 1.0606601717798214);
    }

    #[test]
    fn test_extract_orientation() {
        let q0 = Quat { w: 0.7071067811865476, x: 0.30860669992, y: -0.61721339984, z: 0.15430334996 };
        let q1 = Quat { w: 0.5400617248599999, x: 0.7265700591267262, y: 1.2471685060465476, z: 1.0606601717798214 };
        let a = DualQuat(q0, q1);
        let r = a.orientation();
        assert_eq!(r.w, 0.7071067811865476);
        assert_eq!(r.x, 0.30860669992);
        assert_eq!(r.y, -0.61721339984);
        assert_eq!(r.z, 0.15430334996);
    }

    #[test]
    fn test_extract_translation() {
        let q0 = Quat { w: 0.7071067811865476, x: 0.30860669992, y: -0.61721339984, z: 0.15430334996 };
        let q1 = Quat { w: 0.5400617248599999, x: 0.7265700591267262, y: 1.2471685060465476, z: 1.0606601717798214 };
        let a = DualQuat(q0, q1);
        let t = a.translation();
        assert_approx_eq!(t.x, -1.0);
        assert_approx_eq!(t.y, 2.0);
        assert_approx_eq!(t.z, 3.0);
    }

    #[test]
    fn test_quat_to_dualquat() {
        let q = Quat { w: 5.0, x: 10.0, y: 20.0, z: 30.0 };
        let dq = DualQuat::from(q);
        assert_eq!(dq.0.w, 5.0);
        assert_eq!(dq.0.x, 10.0);
        assert_eq!(dq.0.y, 20.0);
        assert_eq!(dq.0.z, 30.0);
        assert_eq!(dq.1.w, 0.0);
        assert_eq!(dq.1.x, 0.0);
        assert_eq!(dq.1.y, 0.0);
        assert_eq!(dq.1.z, 0.0);
    }

    #[test]
    fn test_vec3_to_dualquat() {
        let v = Vec3 { x: 10.0, y: 20.0, z: 30.0 };
        let q = DualQuat::from(v);
        assert_eq!(q.0.w, 1.0);
        assert_eq!(q.0.x, 0.0);
        assert_eq!(q.0.y, 0.0);
        assert_eq!(q.0.z, 0.0);
        assert_eq!(q.1.w, 0.0);
        assert_eq!(q.1.x, 10.0);
        assert_eq!(q.1.y, 20.0);
        assert_eq!(q.1.z, 30.0);
    }
}
