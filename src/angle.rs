#![deny(missing_docs)]

//! Radian and Degree

use float::Float;

/// Angle in radians
#[derive(Debug, Copy, Clone)]
pub struct Rad<T>(pub T);

/// Angle in degrees
#[derive(Debug, Copy, Clone)]
pub struct Deg<T>(pub T);

impl<T: Float> From<Rad<T>> for Deg<T> {
    #[inline(always)]
    fn from(r: Rad<T>) -> Deg<T> {
        Deg(r.0 * T::c180_by_pi())
    }
}

impl<T: Float> From<Deg<T>> for Rad<T> {
    #[inline(always)]
    fn from(d: Deg<T>) -> Rad<T> {
        Rad(d.0 * T::pi_by_c180())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rad2deg() {
        let a = Rad(1.0471975512);
        let b = Deg::from(a);
        assert_approx_eq!(b.0, 60.0);
    }

    #[test]
    fn test_deg2rad() {
        let a = Deg(30.0);
        let b = Rad::from(a);
        assert_approx_eq!(b.0, 0.52359877559);
    }
}
