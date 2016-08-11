#![deny(missing_docs)]

//! Support module for floating-point types

use std::ops::{Add, Sub, Mul, Div, Rem, Neg, Not, BitAnd, BitOr, BitXor};
use std::{f32, f64};
use std::fmt::Debug;

/// Floating-point type trait
pub trait Float:
    Debug + Copy + Clone
    + Cmp + Sel
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + Neg<Output = Self>
    + Min + Max + Clamp + Ops + Trig
{
    /// Constant 1
    fn one() -> Self;

    /// Constant 0
    fn zero() -> Self;

    /// Constant 2
    fn two() -> Self;

    /// Constant 0.5
    fn onehalf() -> Self;

    /// Constant PI
    fn pi() -> Self;

    /// Constant PI / 180
    fn pi_by_c180() -> Self;

    /// Constant 180 / PI
    fn c180_by_pi() -> Self;

    /// Constant EPSILON
    fn epsilon() -> Self;
}

impl Float for f32 {
    #[inline(always)]
    fn one() -> f32 { 1.0_f32 }
    #[inline(always)]
    fn zero() -> f32 { 0.0_f32 }
    #[inline(always)]
    fn two() -> f32 { 2.0_f32 }
    #[inline(always)]
    fn onehalf() -> f32 { 0.5_f32 }

    #[inline(always)]
    fn pi() -> f32 { 3.14159265359_f32 }
    #[inline(always)]
    fn pi_by_c180() -> f32 { 3.14159265359_f32 / 180.0_f32 }
    #[inline(always)]
    fn c180_by_pi() -> f32 { 180.0_f32 / 3.14159265359_f32 }

    #[inline(always)]
    fn epsilon() -> f32 { 0.00001_f32 }
}

impl Float for f64 {
    #[inline(always)]
    fn one() -> f64 { 1.0_f64 }
    #[inline(always)]
    fn zero() -> f64 { 0.0_f64 }
    #[inline(always)]
    fn two() -> f64 { 2.0_f64 }
    #[inline(always)]
    fn onehalf() -> f64 { 0.5_f64 }

    #[inline(always)]
    fn pi() -> f64 { 3.141592653589793238462643383279502884197169399375105820974944592307816406286_f64 }
    #[inline(always)]
    fn pi_by_c180() -> f64 { Self::pi() / 180.0_f64 }
    #[inline(always)]
    fn c180_by_pi() -> f64 { 180.0_f64 / Self::pi() }

    #[inline(always)]
    fn epsilon() -> f64 { 0.00000001_f64 }
}

/// Comparison
pub trait Cmp {
    /// Corresponding boolean type
    type Bool: Copy
               + Not<Output = Self::Bool>
               + BitAnd<Self::Bool, Output = Self::Bool>
               + BitOr<Self::Bool, Output = Self::Bool>
               + BitXor<Self::Bool, Output = Self::Bool>;
    /// Test equality
    fn eq(self, rhs: Self) -> Self::Bool;
    /// Test inequality
    fn ne(self, rhs: Self) -> Self::Bool;
    /// Test greater than
    fn gt(self, rhs: Self) -> Self::Bool;
    /// Test less than
    fn lt(self, rhs: Self) -> Self::Bool;
    /// Test greater than or equal to
    fn ge(self, rhs: Self) -> Self::Bool;
    /// Test less than or equal to
    fn le(self, rhs: Self) -> Self::Bool;
}

/// Selection
pub trait Sel : Cmp {
    /// Element-wise selection
    /// Result[i] = (rhs[i] if cond[i] == true, self[i] otherwise)
    fn sel(self, rhs: Self, cond: Self::Bool) -> Self;
}

/// Trait of Operations
pub trait Ops {
    /// Absolute Value
    fn abs(self) -> Self;

    /// Reciprocal Value
    fn recip(self) -> Self;

    /// Square Root
    fn sqrt(self) -> Self;

    /// Reciprocal Square Root
    fn rsqrt(self) -> Self;
}

/// Trait of Trigonometry
pub trait Trig: Sized {
    /// Sine
    fn sin(self) -> Self;

    /// Cosine
    fn cos(self) -> Self;

    /// Returns (sin(x), cos(x))
    fn sincos(self) -> (Self, Self);

    /// Tangent
    fn tan(self) -> Self;

    /// Arccosine
    fn acos(self) -> Self;
}

/// Trait of Minimum
pub trait Min {
    /// Returns the minimum between values
    fn min(self, Self) -> Self;
}

/// Trait of Maximum
pub trait Max {
    /// Returns the maximum between values
    fn max(self, Self) -> Self;
}

/// Trait of Clamp
pub trait Clamp {
    /// Constrain a value to lie between two further values
    fn clamp(self, minval: Self, maxval: Self) -> Self;
}

impl Ops for f32 {
    #[inline(always)]
    fn abs(self) -> f32 { f32::abs(self) }

    #[inline(always)]
    fn recip(self) -> f32 { f32::recip(self) }

    #[inline(always)]
    fn sqrt(self) -> f32 { f32::sqrt(self) }

    #[inline(always)]
    fn rsqrt(self) -> f32 { f32::recip(f32::sqrt(self)) }
}

impl Ops for f64 {
    #[inline(always)]
    fn abs(self) -> f64 { f64::abs(self) }

    #[inline(always)]
    fn recip(self) -> f64 { f64::recip(self) }

    #[inline(always)]
    fn sqrt(self) -> f64 { f64::sqrt(self) }

    #[inline(always)]
    fn rsqrt(self) -> f64 { f64::recip(f64::sqrt(self)) }
}

impl Trig for f32 {
    #[inline(always)]
    fn sin(self) -> f32 { f32::sin(self) }

    #[inline(always)]
    fn cos(self) -> f32 { f32::cos(self) }

    #[inline(always)]
    fn sincos(self) -> (f32, f32) { f32::sin_cos(self) }

    #[inline(always)]
    fn tan(self) -> f32 { f32::tan(self) }

    #[inline(always)]
    fn acos(self) -> f32 { f32::acos(self) }
}

impl Trig for f64 {
    #[inline(always)]
    fn sin(self) -> f64 { f64::sin(self) }

    #[inline(always)]
    fn cos(self) -> f64 { f64::cos(self) }

    #[inline(always)]
    fn sincos(self) -> (f64, f64) { f64::sin_cos(self) }

    #[inline(always)]
    fn tan(self) -> f64 { f64::tan(self) }

    #[inline(always)]
    fn acos(self) -> f64 { f64::acos(self) }
}

impl Min for f32 {
    #[inline(always)]
    fn min(self, y: f32) -> f32 { f32::min(self, y) }
}

impl Max for f32 {
    #[inline(always)]
    fn max(self, y: f32) -> f32 { f32::max(self, y) }
}

impl Clamp for f32 {
    #[inline(always)]
    fn clamp(self, minval: f32, maxval: f32) -> f32 {
        f32::min(maxval, f32::max(minval, self))
    }
}

impl Min for f64 {
    #[inline(always)]
    fn min(self, y: f64) -> f64 { f64::min(self, y) }
}

impl Max for f64 {
    #[inline(always)]
    fn max(self, y: f64) -> f64 { f64::max(self, y) }
}

impl Clamp for f64 {
    #[inline(always)]
    fn clamp(self, minval: f64, maxval: f64) -> f64 {
        f64::min(maxval, f64::max(minval, self))
    }
}

impl Cmp for f32 {
    type Bool = bool;

    #[inline(always)]
    fn eq(self, rhs: Self) -> bool { self == rhs }

    #[inline(always)]
    fn ne(self, rhs: Self) -> bool { self != rhs }

    #[inline(always)]
    fn gt(self, rhs: Self) -> bool { self > rhs }

    #[inline(always)]
    fn lt(self, rhs: Self) -> bool { self < rhs }

    #[inline(always)]
    fn ge(self, rhs: Self) -> bool { self >= rhs }

    #[inline(always)]
    fn le(self, rhs: Self) -> bool { self <= rhs }
}

impl Cmp for f64 {
    type Bool = bool;

    #[inline(always)]
    fn eq(self, rhs: Self) -> bool { self == rhs }

    #[inline(always)]
    fn ne(self, rhs: Self) -> bool { self != rhs }

    #[inline(always)]
    fn gt(self, rhs: Self) -> bool { self > rhs }

    #[inline(always)]
    fn lt(self, rhs: Self) -> bool { self < rhs }

    #[inline(always)]
    fn ge(self, rhs: Self) -> bool { self >= rhs }

    #[inline(always)]
    fn le(self, rhs: Self) -> bool { self <= rhs }
}

impl Sel for f32 {
    #[inline(always)]
    fn sel(self, rhs: Self, cond: bool) -> Self {
        match cond {
            true => rhs,
            false => self,
        }
    }
}

impl Sel for f64 {
    #[inline(always)]
    fn sel(self, rhs: Self, cond: bool) -> Self {
        match cond {
            true => rhs,
            false => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_float<T: Float>() -> bool {
        true
    }

    #[test]
    fn test_type() {
        assert!(check_float::<f32>());
        assert!(check_float::<f64>());
    }

    #[test]
    fn test_one() {
        assert_eq!(<f32 as Float>::one(), 1.0_f32);
        assert_eq!(<f64 as Float>::one(), 1.0_f64);
    }

    #[test]
    fn test_zero() {
        assert_eq!(<f32 as Float>::zero(), 0.0_f32);
        assert_eq!(<f64 as Float>::zero(), 0.0_f64);
    }

    #[test]
    fn test_two() {
        assert_eq!(<f32 as Float>::two(), 2.0_f32);
        assert_eq!(<f64 as Float>::two(), 2.0_f64);
    }

    #[test]
    fn test_onehalf() {
        assert_eq!(<f32 as Float>::onehalf(), 0.5_f32);
        assert_eq!(<f64 as Float>::onehalf(), 0.5_f64);
    }

    #[test]
    fn test_pi() {
        assert_approx_eq!(<f32 as Float>::pi(), 3.14159265359);
        assert_approx_eq!(<f64 as Float>::pi(), 3.14159265359);
    }

    #[test]
    fn test_abs() {
        assert_approx_eq!(<f32 as Ops>::abs(-1.5), 1.5);
        assert_approx_eq!(<f64 as Ops>::abs(-1.5), 1.5);
    }

    #[test]
    fn test_recip() {
        assert_approx_eq!(<f32 as Ops>::recip(4.0), 0.25);
        assert_approx_eq!(<f64 as Ops>::recip(4.0), 0.25);
    }

    #[test]
    fn test_sqrt() {
        assert_approx_eq!(<f32 as Ops>::sqrt(4.0), 2.0);
        assert_approx_eq!(<f64 as Ops>::sqrt(4.0), 2.0);
    }

    #[test]
    fn test_rsqrt() {
        assert_approx_eq!(<f32 as Ops>::rsqrt(4.0), 0.5);
        assert_approx_eq!(<f64 as Ops>::rsqrt(4.0), 0.5);
    }

    #[test]
    fn test_sin() {
        assert_approx_eq!(<f32 as Trig>::sin(1.57079632679), 1.0);
        assert_approx_eq!(<f64 as Trig>::sin(1.57079632679), 1.0);
    }

    #[test]
    fn test_cos() {
        assert_approx_eq!(<f32 as Trig>::cos(1.57079632679), 0.0);
        assert_approx_eq!(<f64 as Trig>::cos(1.57079632679), 0.0);
    }

    #[test]
    fn test_sincos() {
        let (s, c) = <f32 as Trig>::sincos(0.0);
        assert_approx_eq!(s, 0.0);
        assert_approx_eq!(c, 1.0);

        let (s, c) = <f64 as Trig>::sincos(0.0);
        assert_approx_eq!(s, 0.0);
        assert_approx_eq!(c, 1.0);
    }

    #[test]
    fn test_tan() {
        assert_approx_eq!(<f32 as Trig>::tan(0.78539816339), 1.0);
        assert_approx_eq!(<f64 as Trig>::tan(0.78539816339), 1.0);
    }

    #[test]
    fn test_acos() {
        assert_approx_eq!(<f32 as Trig>::acos(1.0), 0.0);
        assert_approx_eq!(<f64 as Trig>::acos(1.0), 0.0);
    }

    #[test]
    fn test_min() {
        assert_eq!(<f32 as Min>::min(1.0, -1.0), -1.0);
        assert_eq!(<f64 as Min>::min(1.0, -1.0), -1.0);
    }

    #[test]
    fn test_max() {
        assert_eq!(<f32 as Max>::max(4.0, 2.0), 4.0);
        assert_eq!(<f64 as Max>::max(4.0, 2.0), 4.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(<f32 as Clamp>::clamp(-1.0, 2.0, 3.0), 2.0);
        assert_eq!(<f32 as Clamp>::clamp(4.0, 2.0, 3.0), 3.0);
        assert_eq!(<f32 as Clamp>::clamp(2.5, 2.0, 3.0), 2.5);

        assert_eq!(<f64 as Clamp>::clamp(-1.0, 2.0, 3.0), 2.0);
        assert_eq!(<f64 as Clamp>::clamp(4.0, 2.0, 3.0), 3.0);
        assert_eq!(<f64 as Clamp>::clamp(2.5, 2.0, 3.0), 2.5);
    }

    #[test]
    fn test_eq() {
        assert_eq!(<f32 as Cmp>::eq(1.0, 2.0), false);
        assert_eq!(<f32 as Cmp>::eq(-4.0, -4.0), true);

        assert_eq!(<f64 as Cmp>::eq(1.0, 2.0), false);
        assert_eq!(<f64 as Cmp>::eq(-4.0, -4.0), true);
    }

    #[test]
    fn test_ne() {
        assert_eq!(<f32 as Cmp>::ne(1.0, 2.0), true);
        assert_eq!(<f32 as Cmp>::ne(-4.0, -4.0), false);

        assert_eq!(<f64 as Cmp>::ne(1.0, 2.0), true);
        assert_eq!(<f64 as Cmp>::ne(-4.0, -4.0), false);
    }

    #[test]
    fn test_gt() {
        assert_eq!(<f32 as Cmp>::gt(1.0, 4.0), false);
        assert_eq!(<f32 as Cmp>::gt(3.0, -1.0), true);
        assert_eq!(<f32 as Cmp>::gt(2.5, 2.5), false);

        assert_eq!(<f64 as Cmp>::gt(1.0, 4.0), false);
        assert_eq!(<f64 as Cmp>::gt(3.0, -1.0), true);
        assert_eq!(<f64 as Cmp>::gt(2.5, 2.5), false);
    }

    #[test]
    fn test_lt() {
        assert_eq!(<f32 as Cmp>::lt(1.0, 4.0), true);
        assert_eq!(<f32 as Cmp>::lt(3.0, -1.0), false);
        assert_eq!(<f32 as Cmp>::lt(2.5, 2.5), false);

        assert_eq!(<f64 as Cmp>::lt(1.0, 4.0), true);
        assert_eq!(<f64 as Cmp>::lt(3.0, -1.0), false);
        assert_eq!(<f64 as Cmp>::lt(2.5, 2.5), false);
    }

    #[test]
    fn test_ge() {
        assert_eq!(<f32 as Cmp>::ge(1.0, 4.0), false);
        assert_eq!(<f32 as Cmp>::ge(3.0, -1.0), true);
        assert_eq!(<f32 as Cmp>::ge(2.5, 2.5), true);

        assert_eq!(<f64 as Cmp>::ge(1.0, 4.0), false);
        assert_eq!(<f64 as Cmp>::ge(3.0, -1.0), true);
        assert_eq!(<f64 as Cmp>::ge(2.5, 2.5), true);
    }

    #[test]
    fn test_le() {
        assert_eq!(<f32 as Cmp>::le(1.0, 4.0), true);
        assert_eq!(<f32 as Cmp>::le(3.0, -1.0), false);
        assert_eq!(<f32 as Cmp>::le(2.5, 2.5), true);

        assert_eq!(<f64 as Cmp>::le(1.0, 4.0), true);
        assert_eq!(<f64 as Cmp>::le(3.0, -1.0), false);
        assert_eq!(<f64 as Cmp>::le(2.5, 2.5), true);
    }

    #[test]
    fn test_sel() {
        assert_eq!(<f32 as Sel>::sel(-1.0, 2.5, false), -1.0);
        assert_eq!(<f32 as Sel>::sel(3.0, -0.5, true), -0.5);

        assert_eq!(<f64 as Sel>::sel(-1.0, 2.5, false), -1.0);
        assert_eq!(<f64 as Sel>::sel(3.0, -0.5, true), -0.5);
    }
}
