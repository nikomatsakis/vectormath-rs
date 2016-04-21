//! Plane
#![deny(missing_docs)]

use float::Float;
use vector::{Pos3, Vec3};

/// Plane
///
/// # Equation
///
/// `ax + by + cz + d = 0` where `(a, b, c) = (normal.x, normal.y, normal.z)`
#[derive(Copy, Clone, Debug)]
pub struct Plane<T> {
    /// Normal vector (`a`, `b`, `c` coefficients)
    pub normal: Vec3<T>,
    /// `d` coefficient
    pub d: T,
}

impl<T: Float> Plane<T> {
    /// Signed distance to a point
    pub fn distance(self, pt: Pos3<T>) -> T {
        self.normal.dot(Vec3::from(pt)) + self.d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vector::{Pos3, Vec3};

    #[test]
    fn test_distance() {
        let pl = Plane { normal: Vec3 { x: 1.0, y: 2.0, z: 3.0 }, d: 4.0 };
        assert_eq!(pl.distance(Pos3 { x: 0.5, y: 2.5, z: 1.5 }), 14.0);
        assert_eq!(pl.distance(Pos3 { x: 1.0, y: 0.5, z: -2.0 }), 0.0);
        assert_eq!(pl.distance(Pos3 { x: -1.0, y: -1.0, z: -1.0 }), -2.0);
    }
}
