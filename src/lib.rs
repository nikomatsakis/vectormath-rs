#[macro_export]
macro_rules! assert_approx_eq {
    ($lhs: expr, $rhs: expr) => ({
        assert_approx_eq!($lhs, $rhs, eps = 0.00001);
    });
    ($lhs: expr, $rhs: expr, eps = $eps: expr) => (
        match (&($lhs), &($rhs)) {
            (lhs_val, rhs_val) => {
                use $crate::float::Ops;
                assert!(Ops::abs(*lhs_val - *rhs_val) < ($eps),
                        "`{0}` should be approx equal to `{1}`, found: {0}: `{2:?}`, {1}: `{3:?}`",
                        stringify!($lhs), stringify!($rhs), lhs_val, rhs_val);
            }
        }
    )
}

pub mod float;
pub mod angle;
pub mod vector;
pub mod matrix;
pub mod quaternion;
pub mod plane;

pub use angle::{Rad, Deg};
pub use vector::{Vec2, Vec3, Vec4, Pos2, Pos3};
pub use matrix::{Mat2, Mat3, Mat4, Tfm3};
pub use quaternion::Quat;
pub use plane::Plane;

use float::{Min, Max};

#[inline(always)]
pub fn min<T: Min>(a1: T, a2: T) -> T {
    a1.min(a2)
}

#[inline(always)]
pub fn max<T: Max>(a1: T, a2: T) -> T {
    a1.max(a2)
}
