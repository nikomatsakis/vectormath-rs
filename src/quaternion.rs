#![deny(missing_docs)]
//! Quaternion

use float::{Float, Sel};
use vector::Vec3;
use matrix::Mat3;
use angle::Rad;
use std::ops::{Add, Sub, Mul, Neg, Not};
use std::convert::From;

/// Quaternion
#[derive(Debug, Copy, Clone)]
pub struct Quat<T> {
    /// real part (scalar part)
    pub w: T,
    /// 1st imaginary part (x of vector part)
    pub x: T,
    /// 2nd imaginary part (y of vector part)
    pub y: T,
    /// 3rd imaginary part (z of vector part)
    pub z: T
}

//
// Addition
//

impl<T> Add for Quat<T> where T: Float {
    type Output = Quat<T>;
    fn add(self, _rhs: Quat<T>) -> Quat<T> {
        Quat { w: (self.w + _rhs.w),
               x: (self.x + _rhs.x),
               y: (self.y + _rhs.y),
               z: (self.z + _rhs.z) }
    }
}

//
// Subtraction
//

impl<T> Sub for Quat<T> where T: Float {
    type Output = Quat<T>;
    fn sub(self, _rhs: Quat<T>) -> Quat<T> {
        Quat { w: (self.w - _rhs.w),
               x: (self.x - _rhs.x),
               y: (self.y - _rhs.y),
               z: (self.z - _rhs.z) }
    }
}

//
// Multiplication
//

impl<T> Mul<T> for Quat<T> where T: Float {
    type Output = Quat<T>;
    fn mul(self, s: T) -> Quat<T> {
        Quat { w: (self.w * s),
               x: (self.x * s),
               y: (self.y * s),
               z: (self.z * s) }
    }
}

impl<T> Mul<Quat<T>> for Quat<T> where T: Float {
    type Output = Quat<T>;
    fn mul(self, q: Quat<T>) -> Quat<T> {
        // let w0 = a.w, v0 = a.xyz, w1 = b.w, v1 = b.xyz
        // result.w = w0 * w1 - dot(v0, v1)
        let tw = self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z;
        // result.xyz = w0 * v1 + w1 * v0 + cross(v0, v1)
        let tx = self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y;
        let ty = self.w * q.y + self.y * q.w + self.z * q.x - self.x * q.z;
        let tz = self.w * q.z + self.z * q.w + self.x * q.y - self.y * q.x;
        Quat { w: tw, x: tx, y: ty, z: tz }
    }
}

//
// Negation
//

impl<T> Neg for Quat<T> where T: Float {
    type Output = Quat<T>;
    fn neg(self) -> Quat<T> {
        Quat { w: -self.w, x: -self.x, y: -self.y, z: -self.z }
    }
}

//
// Conjugate
//

impl<T> Not for Quat<T> where T: Float {
    type Output = Quat<T>;
    fn not(self) -> Quat<T> {
        Quat { w: self.w, x: -self.x, y: -self.y, z: -self.z }
    }
}

//
// Functions
//
impl<T: Float> Quat<T> {
    /// Magnitude
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    /// Squared Magnitude
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }

    /// Normalize
    pub fn normalize(self) -> Quat<T> {
        self * self.norm_squared().rsqrt()
    }

    /// Dot Product
    pub fn dot(self, rhs: Quat<T>) -> T {
        let mw = self.w * rhs.w;
        let mx = self.x * rhs.x;
        let my = self.y * rhs.y;
        let mz = self.z * rhs.z;
        (mw + mx + my + mz)
    }

    /// Transform a vector with the quaternion
    pub fn rotate(self, v: Vec3<T>) -> Vec3<T> {
        // let p = (xyz=v, w=0)
        // p' = q * p * inv(q)
        //    = q * p * ~q when q is unit
        // =>
        // tmpXYZ = q.w * v.xyz + cross(q.xyz, v.xyz)
        // tmpW = dot(q.xyz, v.xyz)
        // outXYZ = tmpW * q.xyz + tmpXYZ * q.w - cross(tmpXYZ, q.xyz)
        let tw = self.x * v.x + self.y * v.y + self.z * v.z;
        let tx = self.w * v.x + self.y * v.z - self.z * v.y;
        let ty = self.w * v.y + self.z * v.x - self.x * v.z;
        let tz = self.w * v.z + self.x * v.y - self.y * v.x;
        let rx = tw * self.x + self.w * tx - ty * self.z + tz * self.y;
        let ry = tw * self.y + self.w * ty - tz * self.x + tx * self.z;
        let rz = tw * self.z + self.w * tz - tx * self.y + ty * self.x;
        Vec3 { x: rx, y: ry, z: rz }
    }

    /// linear interpolation
    pub fn lerp(self, dest: Quat<T>, t: T) -> Quat<T> {
        self + (dest - self) * t
    }

    /// spherical linear interpolation
    pub fn slerp(self, dest: Quat<T>, t: T) -> Quat<T> {
        let cos_ht = self.dot(dest); // cos(theta/2)

        let case1 = self.lerp(dest, t);
        let case0 = {
            let ht = cos_ht.acos();
            let ra = ((T::one() - t) * ht).sin();
            let rb = ((t) * ht).sin();
            (self * ra + dest * rb) * ht.sin().recip()
        };

        let cond = cos_ht.gt(T::one() - T::epsilon());
        case0.sel(case1, cond)
    }
}

//
// Selection
//

impl<T: Sel> Quat<T> {
    fn sel(self, rhs: Quat<T>, cond: T::Bool) -> Quat<T> {
        Quat {
            w: self.w.sel(rhs.w, cond),
            x: self.x.sel(rhs.x, cond),
            y: self.y.sel(rhs.y, cond),
            z: self.z.sel(rhs.z, cond),
        }
    }
}

//
// Conversion
//

impl<T> From<Mat3<T>> for Quat<T> where T: Float {
    fn from(m: Mat3<T>) -> Quat<T> {
        let tr = m.0.x + m.1.y + m.2.z;

        let case1 = {
            let (sw, sv) = calc(tr);
            Quat { w: sw,
                   x: (m.1.z - m.2.y) * sv,
                   y: (m.2.x - m.0.z) * sv,
                   z: (m.0.y - m.1.x) * sv }
        };
        // else if
        let case2 = {
            let (sw, sv) = calc(m.0.x - m.1.y - m.2.z);
            Quat { w: (m.1.z - m.2.y) * sv,
                   x: sw,
                   y: (m.1.x + m.0.y) * sv,
                   z: (m.2.x + m.0.z) * sv }
        };
        // else if
        let case3 = {
            let (sw, sv) = calc(m.1.y - m.0.x - m.2.z);
            Quat { w: (m.2.x - m.0.z) * sv,
                   x: (m.1.x + m.0.y) * sv,
                   y: sw,
                   z: (m.2.y + m.1.z) * sv }
        };
        // else
        let case4 = {
            let (sw, sv) = calc(m.2.z - m.0.x - m.1.y);
            Quat { w: (m.0.y - m.1.x) * sv,
                   x: (m.2.x + m.0.z) * sv,
                   y: (m.2.y + m.1.z) * sv,
                   z: sw }
        };

        let case1_cond = tr.gt(T::zero());
        let case2_cond = m.0.x.gt(m.1.y) & m.0.x.gt(m.2.z);
        let case3_cond = m.1.y.gt(m.2.z);
        return case4.sel(case3, case3_cond).sel(case2, case2_cond).sel(case1, case1_cond);

        /// Returns (sw, sv)
        #[inline(always)]
        fn calc<T: Float>(tr: T) -> (T, T) {
            let t = tr + T::one();
            let s = t.rsqrt() * T::onehalf();
            (t * s, s)
        }
    }
}

impl<T> From<Quat<T>> for Mat3<T> where T: Float {
    fn from(q: Quat<T>) -> Mat3<T> {
        let (xx, xy, xz, xw) = (q.x * q.x, q.x * q.y, q.x * q.z, q.x * q.w);
        let (yy, yz, yw) = (q.y * q.y, q.y * q.z, q.y * q.w);
        let (zz, zw) = (q.z * q.z, q.z * q.w);
        let (xx2, xy2, xz2, xw2) = (xx + xx, xy + xy, xz + xz, xw + xw);
        let (yy2, yz2, yw2) = (yy + yy, yz + yz, yw + yw);
        let (zz2, zw2) = (zz + zz, zw + zw);
        let v0 = Vec3 { x: T::one() - yy2 - zz2, y: xy2 + zw2, z: xz2 - yw2 };
        let v1 = Vec3 { x: xy2 - zw2, y: T::one() - xx2 - zz2, z: yz2 + xw2 };
        let v2 = Vec3 { x: xz2 + yw2, y: yz2 - xw2, z: T::one() - xx2 - yy2 };
        Mat3(v0, v1, v2)
    }
}

/// Constructors
impl<T: Float> Quat<T> {
    /// Construct an identity quaternion
    pub fn identity() -> Quat<T> {
        Quat { w: T::one(), x: T::zero(), y: T::zero(), z: T::zero() }
    }

    /// Construct a quaternion that rotates by `angle` around `axis`
    pub fn from_angle_axis(angle: Rad<T>, axis: Vec3<T>) -> Quat<T> {
        let (s, c) = (angle.0 * T::onehalf()).sincos();
        let (sx, sy, sz) = (axis.x * s, axis.y * s, axis.z * s);
        Quat { w: c, x: sx, y: sy, z: sz }
    }

    /// Construct a quaternion that rotates from one vector to another
    pub fn from_vectors(v0: Vec3<T>, v1: Vec3<T>) -> Quat<T> {
        let sa = v0.cross(v1); // axis * sin(angle)
        let ca = v0.dot(v1); // cos(angle)
        // compute cos(angle/2) = sqrt((1+cos)/2)
        //         sin(angle/2) = sqrt((1-cos)/2)
        //      => sin(angle/2) / sin(angle) = 1/sqrt(2+2cos)
        let tmp0 = T::one() + ca;
        let tmp1 = (tmp0 + tmp0).rsqrt();
        let tmp2 = (tmp0 + tmp0) * tmp1 * T::onehalf();
        let (sx, sy, sz) = (sa.x * tmp1, sa.y * tmp1, sa.z * tmp1);
        Quat { w: tmp2, x: sx, y: sy, z: sz }
    }
}

#[cfg(test)]
mod tests_checktype {
    use super::*;
    use std::ops::{Add, Sub, Mul, Neg, Not};

    impl TQuat<f32> for Quat<f32> {}
    impl TQuat<f64> for Quat<f64> {}

    trait TQuat<T>:
        Copy
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<T, Output = Self>
        + Mul<Self, Output = Self>
        + Neg<Output = Self>
        + Not<Output = Self>
    {
    }
}

#[cfg(test)]
mod tests_quat {
    use super::Quat;

    #[test]
    fn test_add() {
        let a = Quat { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = Quat { x: 0.5, y: -3.5, z: 0.0, w: -6.0 };
        let c = a + b;
        assert_eq!(c.w, -2.0);
        assert_eq!(c.x, 1.5);
        assert_eq!(c.y, -1.5);
        assert_eq!(c.z, 3.0);
    }

    #[test]
    fn test_sub() {
        let a = Quat { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = Quat { x: 0.5, y: -3.5, z: 0.0, w: 5.0 };
        let c = a - b;
        assert_eq!(c.w, -1.0);
        assert_eq!(c.x, 0.5);
        assert_eq!(c.y, 5.5);
        assert_eq!(c.z, 3.0);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Quat { x: 1.0, y: -2.0, z: 3.5, w: -0.5 };
        let b = a * 2.0;
        assert_eq!(b.w, -1.0);
        assert_eq!(b.x, 2.0);
        assert_eq!(b.y, -4.0);
        assert_eq!(b.z, 7.0);
    }

    #[test]
    fn test_mul_quat() {
        let a = Quat { w: 1.0, x: 0.0, y: 1.0, z: 0.0 };
        let b = Quat { w: 1.0, x: 0.5, y: 0.5, z: 0.75 };
        let c = a * b;
        assert_eq!(c.w, 0.5);
        assert_eq!(c.x, 1.25);
        assert_eq!(c.y, 1.5);
        assert_eq!(c.z, 0.25);
    }

    #[test]
    fn test_neg() {
        let a = Quat { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = -a;
        assert_eq!(b.w, -4.0);
        assert_eq!(b.x, -1.0);
        assert_eq!(b.y, -2.0);
        assert_eq!(b.z, -3.0);
    }

    #[test]
    fn test_conj() {
        let a = Quat { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = !a;
        assert_eq!(b.w, 4.0);
        assert_eq!(b.x, -1.0);
        assert_eq!(b.y, -2.0);
        assert_eq!(b.z, -3.0);
    }

    #[test]
    fn test_norm() {
        let a = Quat { x: 1.0, y: 2.0, z: 2.0, w: 4.0 };
        let l = a.norm();
        assert_eq!(l, 5.0);
    }

    #[test]
    fn test_normsqr() {
        let a = Quat { x: 1.0, y: 2.0, z: 2.0, w: 4.0 };
        let l = a.norm_squared();
        assert_eq!(l, 25.0);
    }

    #[test]
    fn test_normalize() {
        let a = Quat { x: 1.0, y: 2.0, z: 2.0, w: 4.0 };
        let b = a.normalize();
        assert_eq!(b.x, 0.2);
        assert_eq!(b.y, 0.4);
        assert_eq!(b.z, 0.4);
        assert_eq!(b.w, 0.8);
    }

    #[test]
    fn test_dot_product() {
        let a = Quat { x: 1.0, y: 0.5, z: -1.5, w: 100.0 };
        let b = Quat { x: 2.0, y: -2.0, z: 1.0, w: 0.0 };
        let dp = a.dot(b);
        assert_eq!(dp, -0.5);
    }

    #[test]
    fn test_rotate() {
        use vector::Vec3;
        let c45 = 0.7071067811865476;
        let q = Quat { w: c45, x: 0.0, y: 0.0, z: c45 };
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let v2 = q.rotate(v1);
        assert_approx_eq!(v2.x, -2.0);
        assert_approx_eq!(v2.y, 1.0);
        assert_approx_eq!(v2.z, 3.0);
    }

    #[test]
    fn test_lerp() {
        let a = Quat { x: 1.0, y: 2.0, z: 3.0, w: -4.0 };
        let b = Quat { x: 0.0, y: -5.0, z: 7.0, w: -5.0 };
        let c = a.lerp(b, 0.25);
        assert_eq!(c.x, 0.75);
        assert_eq!(c.y, 0.25);
        assert_eq!(c.z, 4.0);
        assert_eq!(c.w, -4.25);
    }

    #[test]
    fn test_slerp() {
        let sqrt05 = 0.70710678118;
        let q0 = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };
        let q1 = Quat { w: sqrt05, x: 0.0, y: 0.0, z: sqrt05 };
        let qt = q0.slerp(q1, 0.33333);
        assert_approx_eq!(qt.w, 0.9659265038716348);
        assert_approx_eq!(qt.x, 0.0);
        assert_approx_eq!(qt.y, 0.0);
        assert_approx_eq!(qt.z, 0.25881651631192226);
    }

    #[test]
    fn test_sel() {
        let a = Quat { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
        let b = Quat { x: 0.5, y: -3.5, z: 0.0, w: 5.0 };
        let c = a.sel(b, false);
        assert_eq!(c.w, a.w);
        assert_eq!(c.x, a.x);
        assert_eq!(c.y, a.y);
        assert_eq!(c.z, a.z);
        let c = a.sel(b, true);
        assert_eq!(c.w, b.w);
        assert_eq!(c.x, b.x);
        assert_eq!(c.y, b.y);
        assert_eq!(c.z, b.z);
    }

    #[test]
    fn test_identity() {
        let q: Quat<f64> = Quat::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn test_from_angle_axis() {
        use angle::{Rad, Deg};
        use vector::Vec3;
        let q = Quat::from_angle_axis(Rad::from(Deg(30.0)), Vec3::x_axis());
        let v = q.rotate(Vec3 { x: 1.0, y: 2.0, z: 3.0 });
        assert_approx_eq!(v.x, 1.0);
        assert_approx_eq!(v.y, 0.232050808);
        assert_approx_eq!(v.z, 3.598076211);
        let q = Quat::from_angle_axis(Rad::from(Deg(30.0)), Vec3::y_axis());
        let v = q.rotate(Vec3 { x: 1.0, y: 2.0, z: 3.0 });
        assert_approx_eq!(v.x, 2.366025404);
        assert_approx_eq!(v.y, 2.0);
        assert_approx_eq!(v.z, 2.098076211);
        let q = Quat::from_angle_axis(Rad::from(Deg(30.0)), Vec3::z_axis());
        let v = q.rotate(Vec3 { x: 1.0, y: 2.0, z: 3.0 });
        assert_approx_eq!(v.x, -0.133974596);
        assert_approx_eq!(v.y, 2.232050808);
        assert_approx_eq!(v.z, 3.0);
    }

    #[test]
    fn test_from_vectors() {
        use vector::Vec3;
        let v1 = (Vec3 { x: 1.0, y: 1.0, z: 0.0 }).normalize();
        let v2 = (Vec3 { x: -1.0, y: 1.0, z: 0.0 }).normalize();
        let q = Quat::from_vectors(v1, v2);
        assert_approx_eq!(q.w, 0.70710678118);
        assert_approx_eq!(q.x, 0.0);
        assert_approx_eq!(q.y, 0.0);
        assert_approx_eq!(q.z, 0.70710678118);
    }
}

#[cfg(test)]
mod tests_conversion {
    use super::Quat;
    use matrix::Mat3;
    use vector::Vec3;
    use float::Float;

    #[test]
    fn test_quat_to_mat3() {
        let l3 = 0.577350269189626;
        let (x, y, z) = (l3, l3, l3); // axis
        // rotate 60 degree around v = normalize([1,1,1])
        let (s, c) = (0.5, 0.866025403784439); // sin(30), cos(30)
        let q = Quat { w: c, x: s * x, y: s * y, z: s * z };
        let m = Mat3::from(q);
        let (s, c) = (0.866025403784439, 0.5); // sin(60), cos(60)
        let t = 1.0 - c;
        assert_approx_eq!(m.0.x, t * x * x + c);
        assert_approx_eq!(m.0.y, t * x * y + z * s);
        assert_approx_eq!(m.0.z, t * x * z - y * s);
        assert_approx_eq!(m.1.x, t * x * y - z * s);
        assert_approx_eq!(m.1.y, t * y * y + c);
        assert_approx_eq!(m.1.z, t * y * z + x * s);
        assert_approx_eq!(m.2.x, t * x * z + y * s);
        assert_approx_eq!(m.2.y, t * y * z - x * s);
        assert_approx_eq!(m.2.z, t * z * z + c);
        let v1 = Vec3 { x: 1.0, y: 2.0, z: 3.0 };
        let vm = m * v1;
        let vq = q.rotate(v1);
        assert_approx_eq!(vm.x, vq.x);
        assert_approx_eq!(vm.y, vq.y);
        assert_approx_eq!(vm.z, vq.z);
    }

    #[test]
    fn test_mat3_to_quat_case1() {
        let m = Mat3(Vec3 { x: 0.0, y: 1.0, z: 0.0 },
                     Vec3 { x: -1.0, y: 0.0, z: 0.0 },
                     Vec3 { x: 0.0, y: 0.0, z: 1.0 });
        let q = Quat::from(m);
        let c45 = 0.707106781186548;
        assert_eq!(check_case(&m), 1);
        assert_approx_eq!(q.w, c45);
        assert_approx_eq!(q.x, 0.0);
        assert_approx_eq!(q.y, 0.0);
        assert_approx_eq!(q.z, c45);
    }

    #[test]
    fn test_mat3_to_quat_case2() {
        let m = Mat3(Vec3 { x: 1.0, y: 0.0, z: 0.0 },
                     Vec3 { x: 0.0, y: -1.0, z: 0.0 },
                     Vec3 { x: 0.0, y: 0.0, z: -1.0 });
        let q = Quat::from(m);
        assert_eq!(check_case(&m), 2);
        assert_eq!(q.w, 0.0);
        assert_eq!(q.x, 1.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn test_mat3_to_quat_case3() {
        let m = Mat3(Vec3 { x: -1.0, y: 0.0, z: 0.0 },
                     Vec3 { x: 0.0, y: 1.0, z: 0.0 },
                     Vec3 { x: 0.0, y: 0.0, z: -1.0 });
        let q = Quat::from(m);
        assert_eq!(check_case(&m), 3);
        assert_eq!(q.w, 0.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 1.0);
        assert_eq!(q.z, 0.0);
    }

    #[test]
    fn test_mat3_to_quat_case4() {
        let m = Mat3(Vec3 { x: -1.0, y: 0.0, z: 0.0 },
                     Vec3 { x: 0.0, y: -1.0, z: 0.0 },
                     Vec3 { x: 0.0, y: 0.0, z: 1.0 });
        let q = Quat::from(m);
        assert_eq!(check_case(&m), 4);
        assert_eq!(q.w, 0.0);
        assert_eq!(q.x, 0.0);
        assert_eq!(q.y, 0.0);
        assert_eq!(q.z, 1.0);
    }

    #[allow(dead_code)]
    fn check_case<T: Float + PartialOrd>(m: &Mat3<T>) -> i32 {
        if (m.0.x + m.1.y + m.2.z) > T::zero() {
            1
        } else if (m.0.x > m.1.y) && (m.0.x > m.2.z) {
            2
        } else if m.1.y > m.2.z {
            3
        } else {
            4
        }
    }
}
